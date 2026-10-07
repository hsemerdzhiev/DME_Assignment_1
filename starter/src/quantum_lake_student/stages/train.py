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
    test_positive = float(np.average(y[splits == 'test'], weights=w[splits == 'test']))
    lines += ['', f'Positive labels account for {prior:.2%} of training observations and '
              f'{test_positive:.2%} of test observations, using physical weights. '
              'The prior always predicts the majority label. Its low error rate reflects this '
              'imbalance; its balanced accuracy is 0.5.']
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
        'Timings are in metrics.json; prediction IDs resolve through the Gold ML views.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    result.input_count, result.output_count = len(rows), len(predictions)
    result.finish()
    return result


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
    result['settings'] = {name: {'parameters': model.get_params(), 'threshold': 0.5,
                                 'feature_order': ['detector_event_density', *GOOGLE_META_PREDICTION_COLUMNS],
                                 'warnings': messages}}
    result['decoder_overlap'] = decoder_overlap(rows)
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
    return result


def _test_table(metrics):
    lines = ['| Model | Test rows | Logical-error rate | Balanced accuracy | Brier |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, scores in metrics.items():
        m = scores['test']
        brier = 'not applicable' if m['brier_score'] is None else f"{m['brier_score']:.6f}"
        lines.append(f"| {name} | {m['aggregate_rows']:,} | {m['logical_error_rate']:.6f} | "
                     f"{m['balanced_accuracy']:.6f} | {brier} |")
    return lines


def _google_report(tasks):
    lines = ['', '# Task B: supplied and combined Google decoders', '',
             'The target is actual_observable_flip. Each distance has its own training prior and '
             'logistic regression. The five inputs are detector-event density and the four supplied '
             'predictions, in the order recorded in run.json. The linear model learns a weight for '
             'each input. C=1 and threshold=0.5 are fixed; fitting uses only training shots. '
             'All six comparisons within a distance use the same test shots.']
    for distance in (3, 5):
        scores = tasks[f'task_b_d{distance}']['metrics']
        supplied = [scores[f'task_b_d{distance}_{c.removesuffix("_prediction")}']['test']['logical_error_rate']
                    for c in GOOGLE_META_PREDICTION_COLUMNS]
        delta = scores[f'task_b_d{distance}_logistic_regression']['test']['logical_error_rate'] - min(supplied)
        lines += ['', f'## Distance {distance}', '', *_test_table(scores), '',
                  f'The combined model changes logical-error rate by {delta:+.6f} relative to '
                  'the lowest-error supplied decoder on these test shots.']
    lines += ['', '## Decoder errors', '',
              '| Distance | First decoder | Second decoder | Both wrong | Only first wrong | Only second wrong |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for distance in (3, 5):
        for p in tasks[f'task_b_d{distance}']['decoder_overlap']:
            lines.append(f"| {distance} | {p['first']} | {p['second']} | {p['both_wrong']:,} | "
                         f"{p['only_first_wrong']:,} | {p['only_second_wrong']:,} |")
    lines += ['', 'The one-sided errors show cases where one decoder could correct another. '
              'The combined model needs inputs that distinguish those cases. Shared mistakes and '
              'a linear decision function limit what it can correct. Brier scores are not applicable '
              'to the supplied boolean predictions.', '',
              '# Task C: bounded raw-detector prototype', '',
              'One MLP with 32 ReLU hidden units predicts actual_observable_flip from 200 unpacked '
              'detector bits. It uses only distance-three shots with shot_index below 12,500. '
              'The inputs use little-endian bit order and are unscaled. The supplied split is retained; '
              'internal validation splitting is disabled. Adam runs for at most 50 iterations with seed 42.', '',
              *_test_table(tasks['task_c']['metrics']), '',
              'The hidden layer can learn combinations of detector bits. The flat vector does not '
              'explicitly describe detector coordinates, neighbouring detectors or changes over QEC '
              'rounds. The model must learn those relationships from the bit patterns. '
              'This test subset is smaller than Task B\'s, so the two result tables use different shots.', '',
              'metrics.json includes validation scores and training and prediction times. run.json '
              'records feature order, settings and fitting warnings. Elapsed times can vary even when '
              'predictions match. These results describe the supplied release and partitions.']
    for task in tasks.values():
        for name, settings in task['settings'].items():
            for message in settings['warnings']:
                lines += ['', f'{name}: {message}']
    return '\n'.join(lines) + '\n'


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
    record['dependency_versions']['threadpoolctl'] = version('threadpoolctl')
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    with (output / 'report.md').open('a') as report:
        report.write(_google_report(tasks))
    result.stage = 'train'
    result.input_count += len(rows)
    result.output_count = len(predictions)
    result.finish()
    return result
