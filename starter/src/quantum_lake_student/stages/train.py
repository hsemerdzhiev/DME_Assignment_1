"""Task A: weighted prior and weighted logistic regression over ML Parquet."""
import hashlib
import json
import os
import pickle
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from quantum_lake_student.ml import syndrome_model_input, weighted_logical_error_rate
from quantum_lake_student.ml_contracts import CONTRACTS, validate
from quantum_lake_student.models import StageResult

SEED = 42
PREDICTIONS = pa.schema([('example_id', pa.string()), ('model_id', pa.string()), ('label', pa.bool_()),
                        ('prediction', pa.bool_()), ('probability', pa.float64()), ('split', pa.string())])


def evaluate(labels, probabilities, weights):
    predictions = probabilities >= 0.5
    return {'logical_error_rate': weighted_logical_error_rate(labels, predictions, weights),
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions, sample_weight=weights)),
            'brier_score': float(brier_score_loss(labels, probabilities, sample_weight=weights))}


def code_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.getenv('CODE_REVISION', 'unavailable')


def run(model_run_id, ml_root=Path('ml'), results_root=Path('results')):
    result = StageResult(stage='train_task_a', run_id=model_run_id)
    tables, hashes = {}, {}
    for name in CONTRACTS:
        path = ml_root / f'{name}.parquet'
        payload = path.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        table = pq.read_table(pa.BufferReader(payload))
        validate(table, name)
        if name == 'ml_syndrome_decoder_example':
            tables[name] = table
    rows = tables['ml_syndrome_decoder_example'].to_pylist()
    x = np.asarray([syndrome_model_input(r['syndrome_bits']) for r in rows], dtype=np.float64)
    y = np.asarray([r['logical_error_label'] for r in rows])
    w = np.asarray([r['sample_weight'] for r in rows], dtype=np.int64)
    splits = np.asarray([r['data_split'] for r in rows])
    train = splits == 'train'
    started = perf_counter()
    prior = float(np.average(y[train], weights=w[train]))
    prior_seconds = perf_counter() - started
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED, tol=1e-8)
    started = perf_counter()
    model.fit(x[train], y[train], sample_weight=w[train])
    fit_seconds = perf_counter() - started
    output = results_root / 'part2'
    output.mkdir(parents=True, exist_ok=True)
    models = output / 'models'
    models.mkdir(exist_ok=True)
    (models / 'task_a_logistic_regression.pkl').write_bytes(pickle.dumps(model))
    (models / 'task_a_weighted_prior.json').write_text(json.dumps({'probability': prior, 'threshold': 0.5}) + '\n')
    metrics, predictions = {}, []
    for model_id, fit_time in [('task_a_weighted_prior', prior_seconds), ('task_a_logistic_regression', fit_seconds)]:
        metrics[model_id] = {}
        for split in ('validation', 'test'):
            mask = splits == split
            started = perf_counter()
            probability = np.full(mask.sum(), prior) if model_id.endswith('prior') else model.predict_proba(x[mask])[:, 1]
            prediction_seconds = perf_counter() - started
            metrics[model_id][split] = {**evaluate(y[mask], probability, w[mask]),
                'training_time_seconds': fit_time, 'prediction_time_seconds': prediction_seconds,
                'aggregate_rows': int(mask.sum()), 'physical_observations': int(w[mask].sum())}
            for index, p in zip(np.flatnonzero(mask), probability, strict=True):
                predictions.append({'example_id': rows[index]['example_id'], 'model_id': model_id,
                    'label': bool(y[index]), 'prediction': bool(p >= 0.5), 'probability': float(p), 'split': split})
    pq.write_table(pa.Table.from_pylist(predictions, schema=PREDICTIONS), output / 'predictions.parquet')
    (output / 'metrics.json').write_text(json.dumps({'task_a': metrics}, indent=2) + '\n')
    record = {'run_id': model_run_id, 'tasks': ['A'], 'data_release': 'quantum-data-core', 'bundle_version': 3,
              'input_hashes': hashes, 'code_revision': code_revision(), 'random_seed': SEED,
              'dependency_versions': {n: version(n) for n in ('numpy', 'scipy', 'scikit-learn', 'pyarrow')},
              'feature_order': [f'round_{r}_check_{c}' for r in range(4) for c in range(4)],
              'split_rule': {'validation_fault_rate': 0.0005, 'test_fault_rate': 0.005, 'train': 'other five files'},
              'model_parameters': model.get_params(), 'threshold': 0.5,
              'started_at': result.started_at.isoformat(), 'finished_at': datetime.now(UTC).isoformat(),
              'timings': metrics}
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    lines = ['# Task A: weighted syndrome decoder', '',
        'Inputs are the 16 syndrome values in round-major order; the target is logical_error_label. '
        'Both ML table contracts are checked, and training reads only ML Parquet files.', '',
        'The prior is the training-weighted positive-label frequency. Logistic regression is a linear '
        'probabilistic classifier fitted with physical sample weights. C=1 and threshold=0.5 were fixed '
        'before evaluation; validation and test data were not used for fitting or selection.', '',
        '| Model | Test weighted LER | Test weighted balanced accuracy | Test weighted Brier |',
        '| --- | ---: | ---: | ---: |']
    for name, values in metrics.items():
        m = values['test']
        lines.append(f"| {name} | {m['logical_error_rate']:.6f} | {m['balanced_accuracy']:.6f} | {m['brier_score']:.6f} |")
    baseline_ler = metrics['task_a_weighted_prior']['test']['logical_error_rate']
    linear_ler = metrics['task_a_logistic_regression']['test']['logical_error_rate']
    lines += ['', f'The linear model has {"higher" if linear_ler > baseline_ler else "lower or equal"} '
              'test logical-error rate than the prior. Balanced accuracy and Brier score measure '
              'different aspects of the predictions; the table reports all three without choosing '
              'settings from the test results.']
    lines += ['', 'The flat vector preserves round order but does not explicitly encode check geometry or '
        'interactions. Identical syndromes may have either label. Held-out fault rates differ from training '
        'rates, so these results describe this supplied split rather than a causal effect of noise or '
        'general performance on hardware. Weights represent physical repetitions and are not features.', '',
        'This report covers Task A only. Tasks B and C remain to be implemented. '
        'Timings are in metrics.json; prediction IDs resolve through the Gold ML views.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    result.input_count, result.output_count = len(rows), len(predictions)
    result.finish()
    return result
