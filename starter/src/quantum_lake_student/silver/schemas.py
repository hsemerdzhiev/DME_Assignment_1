"""Arrow schemas for the six minimum Silver tables and the two results tables.

Column names and logical types follow ``assignment/silver-tables.md``.
"""

from __future__ import annotations

import pyarrow as pa

SYNDROME_OBSERVATION = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("physical_fault_rate", pa.float64()),
        ("syndrome_bits", pa.binary()),
        ("round_count", pa.int32()),
        ("check_count", pa.int32()),
        ("logical_error_label", pa.bool_()),
        ("quantity", pa.int64()),
    ]
)

GOOGLE_EXPERIMENT = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("basis", pa.string()),
        ("distance", pa.int32()),
        ("rounds", pa.int32()),
        ("shots", pa.int64()),
        ("center_row", pa.int32()),
        ("center_col", pa.int32()),
        ("measurement_count", pa.int32()),
        ("detector_count", pa.int32()),
    ]
)

GOOGLE_SHOT = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("experiment_id", pa.string()),
        ("shot_index", pa.int64()),
        ("measurement_bits", pa.binary()),
        ("sweep_bits", pa.binary()),
        ("detector_bits", pa.binary()),
        ("detector_event_count", pa.int32()),
        ("actual_observable_flip", pa.bool_()),
        ("belief_matching_prediction", pa.bool_()),
        ("correlated_matching_prediction", pa.bool_()),
        ("pymatching_prediction", pa.bool_()),
        ("tensor_network_contraction_prediction", pa.bool_()),
    ]
)

QASM_CIRCUIT = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("benchmark_name", pa.string()),
        ("variant", pa.string()),
        ("register_declarations", pa.string()),
        ("qubit_count", pa.int32()),
        ("measurement_count", pa.int32()),
        ("two_qubit_gate_count", pa.int32()),
    ]
)

QASM_STABILIZER_CHECK = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("check_id", pa.string()),
        ("ancilla_qubit", pa.string()),
        ("data_qubits", pa.list_(pa.string())),
        ("syndrome_bit", pa.string()),
    ]
)

QASM_CONDITIONAL_CORRECTION = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("circuit_id", pa.string()),
        ("condition_register", pa.string()),
        ("condition_value", pa.int64()),
        ("gate", pa.string()),
        ("target_qubit", pa.string()),
    ]
)

DATA_ISSUES = pa.schema(
    [
        ("issue_id", pa.string()),
        ("run_id", pa.string()),
        ("source_record_id", pa.string()),
        ("rule_id", pa.string()),
        ("severity", pa.string()),
        ("source_name", pa.string()),
        ("archive_member", pa.string()),
        ("record_locator", pa.string()),
        ("observed_value", pa.string()),
        ("action", pa.string()),
        ("reason", pa.string()),
    ]
)

SOURCE_TRACE = pa.schema(
    [
        ("source_record_id", pa.string()),
        ("source_name", pa.string()),
        ("bronze_object", pa.string()),
        ("archive_member", pa.string()),
        ("record_locator", pa.string()),
        ("input_sha256", pa.string()),
        ("silver_table", pa.string()),
    ]
)

# Lake path (below ``silver/``) for every minimum table.
SILVER_PATHS = {
    "syndrome_observation": "qec_syndromes/syndrome_observation.parquet",
    "experiment": "google_qec/experiment.parquet",
    "shot": "google_qec/shot.parquet",
    "circuit": "qasmbench/circuit.parquet",
    "stabilizer_check": "qasmbench/stabilizer_check.parquet",
    "conditional_correction": "qasmbench/conditional_correction.parquet",
}

SILVER_SCHEMAS = {
    "syndrome_observation": SYNDROME_OBSERVATION,
    "experiment": GOOGLE_EXPERIMENT,
    "shot": GOOGLE_SHOT,
    "circuit": QASM_CIRCUIT,
    "stabilizer_check": QASM_STABILIZER_CHECK,
    "conditional_correction": QASM_CONDITIONAL_CORRECTION,
}
