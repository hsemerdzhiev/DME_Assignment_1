"""Export both fixed ML contracts and analyses exclusively from Gold SQL."""
import json
from pathlib import Path
from importlib import resources
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row
from quantum_lake_student.connections import postgres_connection, minio_client
from quantum_lake_student.config import Settings
from quantum_lake_student.models import StageResult
from quantum_lake_student.ml import syndrome_data_split, google_data_split
from quantum_lake_student.ml_contracts import CONTRACTS, validate
from .prepare_data import RESULTS_ROOT, write_lake


def export(run_id, settings=None):
    settings = settings or Settings.from_environment()
    result = StageResult(stage='build_ml_tables', run_id=run_id)
    root = Path('ml')
    root.mkdir(exist_ok=True)
    part1 = RESULTS_ROOT / 'part1'
    analysis = part1 / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    queries = resources.files('quantum_lake_student.gold').joinpath('queries')
    counts, tables, traces = {}, {}, []
    source_trace = pq.read_table(part1 / 'source_trace.parquet')
    trace_ids = set(source_trace['source_record_id'].to_pylist())
    with postgres_connection(settings) as connection:
        connection.execute('DROP VIEW IF EXISTS gold.ml_syndrome_source, gold.ml_google_source')
        connection.execute(queries.joinpath('ml.sql').read_text())
        with connection.cursor(row_factory=dict_row) as cursor:
            for name, schema in CONTRACTS.items():
                view = 'ml_syndrome_source' if 'syndrome' in name else 'ml_google_source'
                rows = cursor.execute(f'SELECT * FROM gold.{view} ORDER BY example_id').fetchall()
                for row in rows:
                    gold_id = row.pop('gold_record_id')
                    if gold_id not in trace_ids:
                        raise ValueError(f'Gold record {gold_id} has no source trace')
                    row['data_split'] = syndrome_data_split(row['physical_fault_rate']) if 'syndrome' in name else google_data_split(row['shot_index'])
                    if row['data_split'] == 'test' and not any(t['ml_table'] == name for t in traces):
                        import pyarrow.compute as pc
                        evidence = source_trace.filter(pc.equal(source_trace['source_record_id'], gold_id)).to_pylist()
                        label_column = 'logical_error_label' if 'syndrome' in name else 'actual_observable_flip'
                        traces.append({'ml_table': name, 'example_id': row['example_id'], 'gold_record_id': gold_id,
                                       'data_split': 'test', 'label': bool(row[label_column]),
                                       'source_trace': evidence})
                table = pa.Table.from_pylist(rows, schema=schema)
                validate(table, name)
                expected_rows = 75_598 if 'syndrome' in name else 250_000
                if table.num_rows != expected_rows:
                    raise ValueError(f'{name}: expected {expected_rows} release rows, got {table.num_rows}')
                if 'syndrome' in name and sum(table['sample_weight'].to_pylist()) != 70_000_000:
                    raise ValueError('syndrome physical weights must sum to 70 million')
                tables[name] = table
                counts[f'ml.{name}'] = table.num_rows
            for name in ('syndrome', 'decoder', 'circuit'):
                rows = cursor.execute(queries.joinpath(f'{name}_analysis.sql').read_text()).fetchall()
                counts[f'analysis.{name}'] = len(rows)
                (analysis / f'{name}_analysis.json').write_text(json.dumps(rows, default=str, indent=2) + '\n')
    for name, table in tables.items():
        pq.write_table(table, root / f'{name}.parquet', compression='zstd')
        write_lake(minio_client(settings), settings.s3_bucket, f'ml/{name}.parquet', table)
    (part1 / 'trace_examples.json').write_text(json.dumps(traces, default=str, indent=2) + '\n')
    counts['results.trace_examples'] = len(traces)
    result.input_count = sum(table.num_rows for table in tables.values())
    result.output_count = result.input_count
    result.finish()
    return result, counts


def run(run_id):
    return export(run_id)[0]


def attach_predictions():
    path = RESULTS_ROOT / 'part1/trace_examples.json'
    traces = json.loads(path.read_text())
    by_id = {trace['example_id']: trace for trace in traces}
    for trace in traces:
        trace['predictions'] = []
    predictions = pq.ParquetFile(RESULTS_ROOT / 'part2/predictions.parquet')
    for batch in predictions.iter_batches():
        for row in batch.to_pylist():
            if row['example_id'] in by_id and row['split'] == 'test':
                by_id[row['example_id']]['predictions'].append(row)
    for trace in traces:
        if not trace['predictions']:
            raise ValueError(f"No saved test predictions for {trace['example_id']}")
    path.write_text(json.dumps(traces, indent=2) + '\n')
