"""Part II: Task A syndromes, Task B decoders, and Task C detector bits."""
import hashlib
import json
import os
import pickle
import subprocess
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from quantum_lake_student.ml import (GOOGLE_META_PREDICTION_COLUMNS, google_meta_model_input,
                                    syndrome_model_input, unpack_little_endian_bits, weighted_logical_error_rate)
from quantum_lake_student.ml_contracts import CONTRACTS, validate
from quantum_lake_student.models import StageResult
from quantum_lake_student.stages.part2_report import write_report

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


def run_task_a(model_run_id, ml_root=Path('ml'), results_root=Path('results')):
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
              'timings': metrics, 'report_context': {'task_a': task_a_context(y, w, splits, rows)}}
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    write_report(output)
    result.input_count, result.output_count = len(rows), len(predictions)
    result.finish()
    return result


def task_a_context(y, w, splits, rows):
    """Weighted label rates per split and per fault rate, for the report."""
    rates = np.asarray([r['physical_fault_rate'] for r in rows], dtype=np.float64)
    return {'split_rows': {s: int((splits == s).sum()) for s in ('train', 'validation', 'test')},
            'physical_observations': {s: int(w[splits == s].sum()) for s in ('train', 'validation', 'test')},
            'weighted_positive_rate': {s: float(np.average(y[splits == s], weights=w[splits == s]))
                                       for s in ('train', 'validation', 'test') if (splits == s).any()},
            'weighted_positive_rate_by_fault_rate': [
                {'physical_fault_rate': float(rate), 'split': str(splits[rates == rate][0]),
                 'weighted_positive_rate': float(np.average(y[rates == rate], weights=w[rates == rate]))}
                for rate in sorted(set(rates.tolist()))]}


def _fit_google(model, x, y):
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x, y)
    return perf_counter() - started, sorted({str(item.message) for item in caught})


def _score_google(rows, x, y, models, priors, fit_times, supplied=False):
    splits = np.asarray([r['data_split'] for r in rows])
    columns = {f'task_b_d{rows[0]["distance"]}_{name.removesuffix("_prediction")}': name
               for name in GOOGLE_META_PREDICTION_COLUMNS} if supplied else {}
    metrics, predictions = {}, []
    for model_id in [*priors, *models, *columns]:
        metrics[model_id] = {}
        for split in ('validation', 'test'):
            mask = splits == split
            indices = np.flatnonzero(mask)
            if model_id in columns:
                probability, seconds = None, None
                predicted = np.asarray([rows[i][columns[model_id]] for i in indices])
                scores = {'logical_error_rate': float(np.mean(y[mask] != predicted)),
                          'balanced_accuracy': float(balanced_accuracy_score(y[mask], predicted)),
                          'brier_score': None}
            else:
                inputs = x[mask]
                started = perf_counter()
                probability = (np.full(len(indices), priors[model_id]) if model_id in priors
                               else models[model_id].predict_proba(inputs)[:, 1])
                seconds = perf_counter() - started
                predicted = probability >= 0.5
                scores = evaluate(y[mask], probability, np.ones(len(indices)))
            metrics[model_id][split] = {**scores, 'training_time_seconds': fit_times.get(model_id),
                                       'prediction_time_seconds': seconds, 'aggregate_rows': len(indices)}
            predictions.extend({'example_id': rows[index]['example_id'], 'model_id': model_id,
                                'label': bool(y[index]), 'prediction': bool(predicted[j]),
                                'probability': None if probability is None else float(probability[j]),
                                'split': split} for j, index in enumerate(indices))
    return {'models': models, 'priors': priors, 'metrics': metrics, 'predictions': predictions,
            'split_rows': {s: int((splits == s).sum()) for s in ('train', 'validation', 'test')}}


def decoder_overlap(rows):
    test = [r for r in rows if r['data_split'] == 'test']
    y = np.asarray([r['actual_observable_flip'] for r in test], dtype=bool)
    errors = {name: np.asarray([r[name] for r in test], dtype=bool) != y
              for name in GOOGLE_META_PREDICTION_COLUMNS}
    return [{'first': a, 'second': b, 'test_rows': len(test),
             'both_wrong': int((errors[a] & errors[b]).sum()),
             'only_first_wrong': int((errors[a] & ~errors[b]).sum()),
             'only_second_wrong': int((~errors[a] & errors[b]).sum()),
             'both_correct': int((~errors[a] & ~errors[b]).sum())}
            for a, b in combinations(GOOGLE_META_PREDICTION_COLUMNS, 2)]


def decoder_agreement(rows, predictions, model_id):
    """How many supplied decoders are wrong per test shot, and where the combined model helps."""
    test = [r for r in rows if r['data_split'] == 'test']
    y = np.asarray([r['actual_observable_flip'] for r in test], dtype=bool)
    errors = np.stack([np.asarray([r[c] for r in test], dtype=bool) != y
                       for c in GOOGLE_META_PREDICTION_COLUMNS]) if test else np.zeros((4, 0), dtype=bool)
    wrong = errors.sum(axis=0)
    combined = {p['example_id']: p['prediction'] for p in predictions
                if p['model_id'] == model_id and p['split'] == 'test'}
    combined_error = np.asarray([combined[r['example_id']] != r['actual_observable_flip'] for r in test], dtype=bool)
    best = int(np.argmin(errors.mean(axis=1))) if test else 0
    disagree = (wrong > 0) & (wrong < 4)
    return {'test_rows': len(test),
            'shots_by_decoders_wrong': {str(k): int((wrong == k).sum()) for k in range(5)},
            'combined_errors_by_decoders_wrong': {str(k): int((combined_error & (wrong == k)).sum()) for k in range(5)},
            'combined_error_rate_when_decoders_disagree': float(combined_error[disagree].mean()) if disagree.any() else None,
            'best_supplied_decoder': GOOGLE_META_PREDICTION_COLUMNS[best],
            'combined_right_best_wrong': int((~combined_error & errors[best]).sum()),
            'combined_wrong_best_right': int((combined_error & ~errors[best]).sum())}


def run_task_b(rows, distance):
    rows = [r for r in rows if r['distance'] == distance]
    x = np.asarray([google_meta_model_input(r) for r in rows], dtype=np.float64)
    y = np.asarray([r['actual_observable_flip'] for r in rows], dtype=bool)
    mask = np.asarray([r['data_split'] == 'train' for r in rows])
    started = perf_counter()
    prior = float(y[mask].mean())
    prior_seconds = perf_counter() - started
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED, tol=1e-8)
    seconds, messages = _fit_google(model, x[mask], y[mask])
    name, prior_name = f'task_b_d{distance}_logistic_regression', f'task_b_d{distance}_prior'
    result = _score_google(rows, x, y, {name: model}, {prior_name: prior},
                           {name: seconds, prior_name: prior_seconds}, supplied=True)
    feature_order = ['detector_event_density', *GOOGLE_META_PREDICTION_COLUMNS]
    result['settings'] = {name: {'parameters': model.get_params(), 'threshold': 0.5,
                                 'feature_order': feature_order, 'warnings': messages,
                                 'coefficients': dict(zip(feature_order, map(float, model.coef_[0]), strict=True)),
                                 'intercept': float(model.intercept_[0])}}
    result['decoder_overlap'] = decoder_overlap(rows)
    result['context'] = decoder_agreement(rows, result['predictions'], name)
    return result


def run_task_c(rows):
    rows = [r for r in rows if r['distance'] == 3 and r['shot_index'] < 12_500]
    x = np.asarray([unpack_little_endian_bits(r['detector_bits'], 200) for r in rows], dtype=np.float64)
    y = np.asarray([r['actual_observable_flip'] for r in rows], dtype=bool)
    mask = np.asarray([r['data_split'] == 'train' for r in rows])
    model = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', alpha=1e-4,
                          batch_size=256, learning_rate_init=0.001, max_iter=50, tol=1e-4,
                          n_iter_no_change=10, early_stopping=False, random_state=SEED)
    seconds, messages = _fit_google(model, x[mask], y[mask])
    name = 'task_c_d3_mlp'
    result = _score_google(rows, x, y, {name: model}, {}, {name: seconds})
    result['settings'] = {name: {'parameters': model.get_params(), 'threshold': 0.5,
                                 'feature_order': [f'detector_{i}' for i in range(200)],
                                 'warnings': messages, 'iterations': int(model.n_iter_)}}
    test = np.asarray([r['data_split'] == 'test' for r in rows])
    train_prior = float(y[mask].mean()) if mask.any() else 0.0
    result['context'] = {
        'test_positive_rate': float(y[test].mean()) if test.any() else None,
        'training_positive_rate': train_prior,
        'majority_baseline_test_logical_error_rate': float(np.mean(y[test] != (train_prior >= 0.5))) if test.any() else None,
        'supplied_decoder_test_logical_error_rate': {
            c: float(np.mean(np.asarray([r[c] for r in rows], dtype=bool)[test] != y[test])) if test.any() else None
            for c in GOOGLE_META_PREDICTION_COLUMNS}}
    return result


def run(model_run_id, ml_root=Path('ml'), results_root=Path('results')):
    with threadpool_limits(limits=1):
        result = run_task_a(model_run_id, ml_root, results_root)
        rows = pq.read_table(ml_root / 'ml_google_decoder_example.parquet').to_pylist()
        tasks = {'task_b_d3': run_task_b(rows, 3), 'task_b_d5': run_task_b(rows, 5),
                 'task_c': run_task_c(rows)}
    output = results_root / 'part2'
    predictions = pq.read_table(output / 'predictions.parquet').to_pylist()
    metrics = json.loads((output / 'metrics.json').read_text())
    record = json.loads((output / 'run.json').read_text())
    record['models'] = {'task_a_logistic_regression': {'parameters': record.pop('model_parameters'),
                        'feature_order': record.pop('feature_order'), 'threshold': record.pop('threshold')}}
    for task, values in tasks.items():
        metrics[task] = values['metrics']
        predictions.extend(values['predictions'])
        record['models'].update(values['settings'])
        for name, model in values['models'].items():
            (output / 'models' / f'{name}.pkl').write_bytes(pickle.dumps(model))
        for name, prior in values['priors'].items():
            (output / 'models' / f'{name}.json').write_text(json.dumps({'probability': prior, 'threshold': 0.5}) + '\n')
    predictions.sort(key=lambda row: (row['model_id'], row['split'], row['example_id']))
    pq.write_table(pa.Table.from_pylist(predictions, schema=PREDICTIONS), output / 'predictions.parquet')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    record.update(tasks=['A', 'B', 'C'], numerical_threads=1, timings=metrics,
                  finished_at=datetime.now(UTC).isoformat(),
                  split_rule={'task_a': record['split_rule'],
                              'google': {'test': 'shot_index % 2 == 1',
                                         'validation': 'shot_index % 10 == 8', 'train': 'other even shots'},
                              'task_c': 'distance == 3 and shot_index < 12500'},
                  split_rows={name: task['split_rows'] for name, task in tasks.items()},
                  decoder_overlap={str(d): tasks[f'task_b_d{d}']['decoder_overlap'] for d in (3, 5)})
    record['report_context'].update({name: task['context'] for name, task in tasks.items()})
    record['dependency_versions']['threadpoolctl'] = version('threadpoolctl')
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    write_report(output)
    result.stage = 'train'
    result.input_count += len(rows)
    result.output_count = len(predictions)
    result.finish()
    return result
