"""Exact ML hand-off contracts from assignment/required-ml-tables.md."""
import pyarrow as pa
from .ml import GOOGLE_META_PREDICTION_COLUMNS, syndrome_model_input, syndrome_data_split, google_data_split, unpack_little_endian_bits

SYNDROME = pa.schema([(n, t) for n, t in [
    ('example_id', pa.string()), ('experiment_id', pa.string()), ('physical_fault_rate', pa.float64()),
    ('syndrome_bits', pa.binary()), ('round_count', pa.int32()), ('check_count', pa.int32()),
    ('logical_error_label', pa.bool_()), ('sample_weight', pa.int64()), ('data_split', pa.string())]])
GOOGLE = pa.schema([
    ('example_id', pa.string()), ('experiment_id', pa.string()), ('shot_index', pa.int64()),
    *[(n, pa.int32()) for n in ('distance', 'rounds', 'center_row', 'center_col', 'detector_count', 'detector_event_count')],
    ('detector_bits', pa.binary()), *[(n, pa.bool_()) for n in GOOGLE_META_PREDICTION_COLUMNS],
    ('actual_observable_flip', pa.bool_()), ('data_split', pa.string())])
CONTRACTS = {'ml_syndrome_decoder_example': SYNDROME, 'ml_google_decoder_example': GOOGLE}


def validate(table, name):
    schema = CONTRACTS[name]
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f'{name}: unexpected columns or types: {table.schema}')
    if any(table[n].null_count for n in schema.names):
        raise ValueError(f'{name}: null values')
    ids = table['example_id'].to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError(f'{name}: duplicate example IDs')
    if set(table['data_split'].to_pylist()) != {'train', 'validation', 'test'}:
        raise ValueError(f'{name}: all three partitions must be present')
    for row in table.to_pylist():
        if name == 'ml_syndrome_decoder_example':
            syndrome_model_input(row['syndrome_bits'])
            if row['round_count'] != 4 or row['check_count'] != 4 or row['sample_weight'] <= 0:
                raise ValueError('invalid syndrome shape or weight')
            expected = syndrome_data_split(row['physical_fault_rate'])
        else:
            width = row['detector_count']
            if row['distance'] not in (3, 5) or width != {3: 200, 5: 600}[row['distance']] or row['rounds'] != 25:
                raise ValueError('invalid Google experiment dimensions')
            bits = unpack_little_endian_bits(row['detector_bits'], width)
            if sum(bits) != row['detector_event_count']:
                raise ValueError('detector event count mismatch')
            if width % 8 and row['detector_bits'][-1] >> (width % 8):
                raise ValueError('nonzero detector padding')
            expected = google_data_split(row['shot_index'])
        if row['data_split'] != expected:
            raise ValueError('incorrect course split')
