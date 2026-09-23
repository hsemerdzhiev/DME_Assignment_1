"""Relational integration stage.

Builds the team-designed Gold model (``gold/schema.sql``) from the six Silver
tables and loads it as one all-or-nothing update:

1. inside a single transaction, create ``gold_build`` and run the DDL;
2. COPY every table in binary form, then run cross-table checks that
   constraints alone cannot express;
3. drop the previous ``gold`` schema and rename ``gold_build`` to ``gold``.

If any step fails the transaction rolls back and the previous ``gold`` stays
exactly as it was. Every key is a Silver ``source_record_id`` or a hash of
source facts, so a repeated load produces the same rows and no duplicates.
"""

from __future__ import annotations

import io
import json
from importlib import resources
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.gold import transform
from quantum_lake_student.models import StageResult
from quantum_lake_student.silver import schemas
from quantum_lake_student.stages.prepare_data import SILVER_PREFIX

STAGE = "load_postgres"
SCHEMA, BUILD_SCHEMA = "gold", "gold_build"

# table -> (columns, PostgreSQL types for binary COPY)
TABLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "experiment": (transform.EXPERIMENT_COLUMNS,
                   ("text", "text", "text", "text", "int4", "int4", "int4", "float8", "text", "int8", "int4", "int4",
                    "int4", "int4", "int4", "text")),
    "syndrome_pattern": (("pattern_id", "syndrome_bits", "round_count", "check_count", "fired_checks"),
                         ("text", "bytea", "int4", "int4", "int4")),
    "syndrome_observation": (("observation_id", "experiment_id", "pattern_id", "logical_error_label", "quantity"),
                             ("text", "text", "text", "bool", "int8")),
    "decoder": (("decoder_id", "description", "calibration"), ("text", "text", "text")),
    "shot": (("shot_id", "experiment_id", "shot_index", "measurement_bits", "sweep_bits", "detector_bits",
              "detector_event_count", "actual_observable_flip"),
             ("text", "text", "int8", "bytea", "bytea", "bytea", "int4", "bool")),
    "decoder_prediction": (("shot_id", "decoder_id", "predicted_flip"), ("text", "text", "bool")),
    "detector_summary": (("experiment_id", "detector_index", "fire_count"), ("text", "int4", "int8")),
    "circuit": (("circuit_id", "benchmark_name", "variant", "qubit_count", "measurement_count",
                 "two_qubit_gate_count", "source_record_id"),
                ("text", "text", "text", "int4", "int4", "int4", "text")),
    "circuit_register": (("circuit_id", "register_name", "kind", "size", "position"), ("text", "text", "text", "int4", "int4")),
    "stabilizer_check": (("check_id", "circuit_id", "check_label", "ancilla_qubit", "syndrome_bit"),
                         ("text", "text", "text", "text", "text")),
    "stabilizer_check_qubit": (("check_id", "data_qubit", "position"), ("text", "text", "int4")),
    "conditional_correction": (("correction_id", "circuit_id", "condition_register", "condition_value", "gate", "target_qubit"),
                               ("text", "text", "text", "int8", "text", "text")),
}

# Checks that span tables or need the whole load; each must return zero rows.
INTEGRITY_CHECKS = {
    "shot detector_event_count within detector_count":
        "SELECT 1 FROM shot s JOIN experiment e USING (experiment_id) WHERE s.detector_event_count > e.detector_count",
    "shot detector_bits has ceil(detector_count/8) bytes":
        "SELECT 1 FROM shot s JOIN experiment e USING (experiment_id) WHERE octet_length(s.detector_bits) <> (e.detector_count + 7) / 8",
    "shot measurement_bits has ceil(measurement_count/8) bytes":
        "SELECT 1 FROM shot s JOIN experiment e USING (experiment_id) WHERE octet_length(s.measurement_bits) <> (e.measurement_count + 7) / 8",
    "shots per experiment equal declared shots":
        "SELECT 1 FROM experiment e WHERE e.kind = 'hardware_memory' AND e.shots <> (SELECT count(*) FROM shot s WHERE s.experiment_id = e.experiment_id)",
    "every shot has every decoder's prediction":
        "SELECT 1 FROM shot s WHERE (SELECT count(*) FROM decoder_prediction p WHERE p.shot_id = s.shot_id) <> (SELECT count(*) FROM decoder)",
    "detector_summary fire counts equal summed shot event counts":
        "SELECT 1 FROM experiment e WHERE e.kind = 'hardware_memory' AND "
        "(SELECT coalesce(sum(fire_count), 0) FROM detector_summary d WHERE d.experiment_id = e.experiment_id) <> "
        "(SELECT coalesce(sum(detector_event_count), 0) FROM shot s WHERE s.experiment_id = e.experiment_id)",
    "every simulated experiment sums to ten million shots":
        "SELECT 1 FROM experiment e WHERE e.kind = 'simulated_syndrome' AND "
        "(SELECT sum(quantity) FROM syndrome_observation o WHERE o.experiment_id = e.experiment_id) <> 10000000",
    "every parity check has at least two data qubits":
        "SELECT 1 FROM stabilizer_check c WHERE (SELECT count(*) FROM stabilizer_check_qubit q WHERE q.check_id = c.check_id) < 2",
    "correction registers are declared classical registers of the circuit":
        "SELECT 1 FROM conditional_correction c WHERE NOT EXISTS (SELECT 1 FROM circuit_register r "
        "WHERE r.circuit_id = c.circuit_id AND r.register_name = c.condition_register AND r.kind = 'creg')",
}


def read_silver(client: Minio, bucket: str) -> dict[str, pa.Table]:
    tables = {}
    for name, path in schemas.SILVER_PATHS.items():
        response = client.get_object(bucket, SILVER_PREFIX + path)
        try:
            tables[name] = pq.read_table(io.BytesIO(response.read()))
        finally:
            response.close()
            response.release_conn()
    return tables


def build_rows(silver: dict[str, pa.Table]) -> dict[str, list[tuple]]:
    return {
        "experiment": transform.experiments(silver["syndrome_observation"], silver["experiment"]),
        "syndrome_pattern": transform.syndrome_patterns(silver["syndrome_observation"]),
        "syndrome_observation": transform.syndrome_observations(silver["syndrome_observation"]),
        "decoder": transform.decoders(),
        "shot": transform.shots(silver["shot"]),
        "decoder_prediction": transform.decoder_predictions(silver["shot"]),
        "detector_summary": transform.detector_summaries(silver["shot"], silver["experiment"]),
        "circuit": transform.circuits(silver["circuit"]),
        "circuit_register": transform.circuit_registers(silver["circuit"]),
        "stabilizer_check": transform.stabilizer_checks(silver["stabilizer_check"]),
        "stabilizer_check_qubit": transform.stabilizer_check_qubits(silver["stabilizer_check"]),
        "conditional_correction": transform.conditional_corrections(silver["conditional_correction"]),
    }


def load_gold(connection: psycopg.Connection, rows: dict[str, list[tuple]], run_id: str) -> dict[str, int]:
    """Build, fill, check, and swap the schema inside one transaction."""
    ddl = resources.files("quantum_lake_student.gold").joinpath("schema.sql").read_text(encoding="utf-8")
    with connection.transaction():
        connection.execute(f"DROP SCHEMA IF EXISTS {BUILD_SCHEMA} CASCADE")
        connection.execute(f"CREATE SCHEMA {BUILD_SCHEMA}")
        connection.execute(f"SET LOCAL search_path TO {BUILD_SCHEMA}")
        connection.execute(ddl)

        counts = {}
        with connection.cursor() as cursor:
            for table, (columns, types) in TABLES.items():
                with cursor.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN (FORMAT BINARY)") as copy:
                    copy.set_types(types)
                    for row in rows[table]:
                        copy.write_row(row)
                counts[table] = cursor.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for description, sql in INTEGRITY_CHECKS.items():
                if cursor.execute(f"{sql} LIMIT 1").fetchone():
                    raise RuntimeError(f"Gold integrity check failed: {description}")
            cursor.execute("INSERT INTO gold_load (run_id, row_counts) VALUES (%s, %s)", (run_id, json.dumps(counts)))

        connection.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        connection.execute(f"ALTER SCHEMA {BUILD_SCHEMA} RENAME TO {SCHEMA}")
    return counts


def load(run_id: str, settings: Settings | None = None) -> tuple[StageResult, dict[str, int]]:
    settings = settings or Settings.from_environment()
    result = StageResult(stage=STAGE, run_id=run_id)
    silver = read_silver(minio_client(settings), settings.s3_bucket)
    rows = build_rows(silver)
    with postgres_connection(settings) as connection:
        counts = load_gold(connection, rows, run_id)
    result.input_count = sum(table.num_rows for table in silver.values())
    result.output_count = sum(counts.values())
    result.finish()
    return result, {f"gold.{name}": count for name, count in counts.items()}


def run(run_id: str) -> StageResult:
    return load(run_id)[0]


if __name__ == "__main__":
    import uuid

    outcome, counts = load(str(uuid.uuid4()))
    for name, count in counts.items():
        print(f"{count:>10,}  {name}")
    print(f"loaded {outcome.output_count:,} Gold rows from {outcome.input_count:,} Silver rows")
