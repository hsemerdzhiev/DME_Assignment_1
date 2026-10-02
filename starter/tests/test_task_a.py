"""Task A physical weights, input boundaries, and repeatable saved results."""
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from quantum_lake_student.ml_contracts import SYNDROME, GOOGLE, validate
from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS
from quantum_lake_student.stages.train import evaluate, run


def inputs(root):
    root.mkdir()
    syndrome = []
    for split, rate in [('train', .001), ('validation', .0005), ('test', .005)]:
        for label in (False, True):
            syndrome.append(dict(example_id=f'{split}-{label}', experiment_id=split, physical_fault_rate=rate,
                syndrome_bits=bytes([int(label)] * 16), round_count=4, check_count=4,
                logical_error_label=label, sample_weight=9 if not label else 1, data_split=split))
    google = []
    for split, index in [('train', 0), ('validation', 8), ('test', 1)]:
        google.append(dict(example_id=split, experiment_id='hardware', shot_index=index, distance=3,
            rounds=25, center_row=0, center_col=0, detector_count=200, detector_event_count=0,
            detector_bits=bytes(25), actual_observable_flip=False, data_split=split,
            **{name: False for name in GOOGLE_META_PREDICTION_COLUMNS}))
    for name, rows, schema in [('syndrome', syndrome, SYNDROME), ('google', google, GOOGLE)]:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / f'ml_{name}_decoder_example.parquet')
    return syndrome


def test_metrics_use_physical_weights():
    m = evaluate(np.array([False, True]), np.array([.1, .1]), np.array([9, 1]))
    assert m['logical_error_rate'] == .1
    assert m['balanced_accuracy'] == .5
    assert m['brier_score'] == pytest.approx(.09)


def test_contract_rejects_wrong_split_and_duplicate_ids(tmp_path):
    rows = inputs(tmp_path / 'ml')
    rows[0]['data_split'] = 'test'
    with pytest.raises(ValueError, match='split'):
        validate(pa.Table.from_pylist(rows, schema=SYNDROME), 'ml_syndrome_decoder_example')
    rows[0]['data_split'] = 'train'
    rows[1]['example_id'] = rows[0]['example_id']
    with pytest.raises(ValueError, match='duplicate'):
        validate(pa.Table.from_pylist(rows, schema=SYNDROME), 'ml_syndrome_decoder_example')


def test_repeatable_predictions_and_training_only_prior(tmp_path):
    ml = tmp_path / 'ml'
    inputs(ml)
    for directory in ('first', 'second'):
        run(directory, ml, tmp_path / directory)
    first, second = [tmp_path / d / 'part2' for d in ('first', 'second')]
    assert pq.read_table(first / 'predictions.parquet').equals(pq.read_table(second / 'predictions.parquet'))
    assert json.loads((first / 'models/task_a_weighted_prior.json').read_text())['probability'] == .1
    assert len(json.loads((first / 'run.json').read_text())['input_hashes']) == 2


def test_google_contract_rejects_event_count_and_feature_width(tmp_path):
    ml = tmp_path / 'ml'
    inputs(ml)
    name = 'ml_google_decoder_example'
    rows = pq.read_table(ml / f'{name}.parquet').to_pylist()
    rows[0]['detector_event_count'] = 1
    with pytest.raises(ValueError, match='event count'):
        validate(pa.Table.from_pylist(rows, schema=GOOGLE), name)
    rows[0]['detector_event_count'] = 0
    rows[0]['detector_bits'] = bytes(75)
    with pytest.raises(ValueError, match='byte length'):
        validate(pa.Table.from_pylist(rows, schema=GOOGLE), name)
