"""Tasks B and C: Google inputs, held-out shots and saved results."""
import json
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.ml import (GOOGLE_META_PREDICTION_COLUMNS, google_data_split,
                                    google_meta_model_input, syndrome_model_input, unpack_little_endian_bits)
from quantum_lake_student.ml_contracts import GOOGLE
from quantum_lake_student.stages import train
from test_task_a import inputs as task_a_inputs


def inputs(root):
    task_a_inputs(root)
    rows = []
    shots = [(0, False), (2, True), (4, False), (6, True), (8, False), (18, True), (1, False), (3, True),
             (12500, False), (12502, True), (12508, False), (12518, True), (12501, False), (12503, True)]
    for distance in (3, 5):
        width = {3: 200, 5: 600}[distance]
        for index, label in shots:
            bits = bytes([129 if label else 1, int(label)]) + bytes(width // 8 - 2)
            rows.append(dict(example_id=f'd{distance}-{index}', experiment_id=f'hardware-{distance}',
                             shot_index=index, distance=distance, rounds=25, center_row=0, center_col=0,
                             detector_count=width, detector_event_count=3 if label else 1,
                             detector_bits=bits, actual_observable_flip=label, data_split=google_data_split(index),
                             **{name: label if i != 1 else not label
                                for i, name in enumerate(GOOGLE_META_PREDICTION_COLUMNS)}))
    pq.write_table(pa.Table.from_pylist(rows, schema=GOOGLE), root / 'ml_google_decoder_example.parquet')
    return rows


def test_task_b_uses_training_features_and_identical_test_shots(monkeypatch, tmp_path):
    rows = inputs(tmp_path / 'ml')
    observed = []
    original = train.LogisticRegression.fit

    def fit(self, x, y):
        observed.append((x.copy(), y.copy()))
        return original(self, x, y)

    monkeypatch.setattr(train.LogisticRegression, 'fit', fit)
    output = train.run_task_b(rows, 3)
    expected = [r for r in rows if r['distance'] == 3 and r['data_split'] == 'train']
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0][0], [google_meta_model_input(r) for r in expected])
    np.testing.assert_array_equal(observed[0][1], [r['actual_observable_flip'] for r in expected])
    expected_ids = {r['example_id'] for r in rows if r['distance'] == 3 and r['data_split'] == 'test'}
    assert len(output['metrics']) == 6
    for name in output['metrics']:
        assert {p['example_id'] for p in output['predictions'] if p['model_id'] == name and p['split'] == 'test'} == expected_ids
    for column in GOOGLE_META_PREDICTION_COLUMNS:
        name = 'task_b_d3_' + column.removesuffix('_prediction')
        scores = output['metrics'][name]['test']
        assert scores['brier_score'] is None
        assert scores['training_time_seconds'] is None
        assert scores['prediction_time_seconds'] is None
        assert all(p['probability'] is None for p in output['predictions'] if p['model_id'] == name)


def test_task_b_decoder_errors_partition_the_test_shots(tmp_path):
    pairs = train.decoder_overlap(inputs(tmp_path / 'ml'))
    assert len(pairs) == 6
    for pair in pairs:
        assert sum(pair[k] for k in ('both_wrong', 'only_first_wrong', 'only_second_wrong', 'both_correct')) == pair['test_rows']
    assert pairs[0]['only_second_wrong'] == pairs[0]['test_rows']
    assert pairs[0]['both_wrong'] == 0


def test_task_c_uses_the_prescribed_subset_and_bit_order(monkeypatch, tmp_path):
    observed = []
    original = train.MLPClassifier.fit

    def fit(self, x, y):
        observed.append((x.copy(), y.copy()))
        return original(self, x, y)

    monkeypatch.setattr(train.MLPClassifier, 'fit', fit)
    output = train.run_task_c(inputs(tmp_path / 'ml'))
    x, y = observed[0]
    assert x.shape == (4, 200)
    np.testing.assert_array_equal(y, [False, True, False, True])
    np.testing.assert_array_equal(x[1, :9], [1, 0, 0, 0, 0, 0, 0, 1, 1])
    assert {p['example_id'] for p in output['predictions']} == {'d3-8', 'd3-18', 'd3-1', 'd3-3'}
    settings = output['settings']['task_c_d3_mlp']
    assert settings['parameters']['early_stopping'] is False
    assert settings['iterations'] <= 50


def test_all_tasks_save_repeatable_predictions_and_models(tmp_path):
    ml = tmp_path / 'ml'
    inputs(ml)
    for directory in ('first', 'second'):
        train.run(directory, ml, tmp_path / directory)
    first, second = [tmp_path / d / 'part2' for d in ('first', 'second')]
    saved = pq.read_table(first / 'predictions.parquet')
    assert saved.equals(pq.read_table(second / 'predictions.parquet'))
    assert json.loads((first / 'run.json').read_text())['tasks'] == ['A', 'B', 'C']
    predictions = saved.to_pylist()
    assert len({p['model_id'] for p in predictions}) == 15
    examples = {r['example_id']: r for name in ('syndrome', 'google')
                for r in pq.read_table(ml / f'ml_{name}_decoder_example.parquet').to_pylist()}
    model_files = sorted((first / 'models').glob('*.pkl'))
    assert len(model_files) == 4
    for path in model_files:
        model = pickle.loads(path.read_bytes())
        records = [p for p in predictions if p['model_id'] == path.stem and p['split'] == 'test']
        rows = [examples[p['example_id']] for p in records]
        if path.stem.startswith('task_a'):
            x = [syndrome_model_input(r['syndrome_bits']) for r in rows]
        elif path.stem.startswith('task_b'):
            x = [google_meta_model_input(r) for r in rows]
        else:
            x = [unpack_little_endian_bits(r['detector_bits'], 200) for r in rows]
        np.testing.assert_allclose(model.predict_proba(x)[:, 1], [p['probability'] for p in records], rtol=0, atol=1e-12)
