"""Unit tests for the Silver-to-Gold shaping functions on tiny Arrow tables."""

from __future__ import annotations

import pyarrow as pa

from quantum_lake_student.gold import transform
from quantum_lake_student.silver import schemas
from quantum_lake_student.silver.common import rows_to_table

ZERO, ONE = bytes(16), bytes([1] + [0] * 15)


def _syndromes() -> pa.Table:
    return rows_to_table([
        {"source_record_id": "s1", "experiment_id": "d-3_pfr-0.001000", "physical_fault_rate": 0.001, "syndrome_bits": ZERO,
         "round_count": 4, "check_count": 4, "logical_error_label": False, "quantity": 7},
        {"source_record_id": "s2", "experiment_id": "d-3_pfr-0.001000", "physical_fault_rate": 0.001, "syndrome_bits": ONE,
         "round_count": 4, "check_count": 4, "logical_error_label": True, "quantity": 3},
        {"source_record_id": "s3", "experiment_id": "d-3_pfr-0.005000", "physical_fault_rate": 0.005, "syndrome_bits": ONE,
         "round_count": 4, "check_count": 4, "logical_error_label": False, "quantity": 5},
    ], schemas.SYNDROME_OBSERVATION)


def _google_experiment() -> pa.Table:
    return rows_to_table([{
        "source_record_id": "e1", "experiment_id": "surface_code_bX_d3_r25_center_3_5", "basis": "X", "distance": 3,
        "rounds": 25, "shots": 2, "center_row": 3, "center_col": 5, "measurement_count": 209, "detector_count": 10,
    }], schemas.GOOGLE_EXPERIMENT)


def _shots() -> pa.Table:
    common = {"experiment_id": "surface_code_bX_d3_r25_center_3_5", "measurement_bits": bytes(27), "sweep_bits": bytes(2)}
    return rows_to_table([
        {**common, "source_record_id": "h0", "shot_index": 0, "detector_bits": bytes([0b00000101, 0b00000010]),  # bits 0,2,9
         "detector_event_count": 3, "actual_observable_flip": False, "belief_matching_prediction": False,
         "correlated_matching_prediction": True, "pymatching_prediction": False, "tensor_network_contraction_prediction": True},
        {**common, "source_record_id": "h1", "shot_index": 1, "detector_bits": bytes([0b00000001, 0b00000000]),  # bit 0
         "detector_event_count": 1, "actual_observable_flip": True, "belief_matching_prediction": True,
         "correlated_matching_prediction": True, "pymatching_prediction": True, "tensor_network_contraction_prediction": False},
    ], schemas.GOOGLE_SHOT)


def test_experiments_merge_both_sources_with_shared_vocabulary() -> None:
    rows = transform.experiments(_syndromes(), _google_experiment())
    by_id = {row[0]: dict(zip(transform.EXPERIMENT_COLUMNS, row)) for row in rows}
    assert set(by_id) == {"d-3_pfr-0.001000", "d-3_pfr-0.005000", "surface_code_bX_d3_r25_center_3_5"}
    simulated = by_id["d-3_pfr-0.001000"]
    assert (simulated["kind"], simulated["distance"], simulated["rounds"], simulated["check_count"]) == ("simulated_syndrome", 3, 4, 4)
    assert simulated["physical_fault_rate"] == 0.001 and simulated["shots"] is None
    hardware = by_id["surface_code_bX_d3_r25_center_3_5"]
    assert (hardware["kind"], hardware["rounds"], hardware["check_count"], hardware["shots"]) == ("hardware_memory", 25, 8, 2)
    assert hardware["sweep_bit_count"] == 209 - 25 * 8 == 9  # final data-qubit readout width
    assert hardware["physical_fault_rate"] is None


def test_patterns_are_distinct_across_experiments_and_count_fired_checks() -> None:
    patterns = transform.syndrome_patterns(_syndromes())
    assert len(patterns) == 2  # ONE appears in two experiments but is one pattern
    by_bits = {row[1]: row for row in patterns}
    assert by_bits[ZERO][4] == 0 and by_bits[ONE][4] == 1
    assert by_bits[ONE][0] == transform.pattern_id(ONE)


def test_observations_reference_pattern_and_keep_silver_id_as_key() -> None:
    rows = transform.syndrome_observations(_syndromes())
    assert [row[0] for row in rows] == ["s1", "s2", "s3"]
    assert rows[1][2] == rows[2][2] == transform.pattern_id(ONE)
    assert (rows[1][3], rows[1][4]) == (True, 3)


def test_predictions_are_unpivoted_one_row_per_decoder() -> None:
    rows = transform.decoder_predictions(_shots())
    assert len(rows) == 8
    assert {row[1] for row in rows} == {d[0] for d in transform.decoders()}
    assert ("h0", "correlated_matching", True) in rows and ("h1", "tensor_network_contraction", False) in rows


def test_detector_summaries_count_little_endian_bits_and_ignore_padding() -> None:
    rows = transform.detector_summaries(_shots(), _google_experiment())
    counts = {index: count for _, index, count in rows}
    assert len(counts) == 10                       # detector_count positions, padding bits 10..15 dropped
    assert counts[0] == 2 and counts[2] == 1 and counts[9] == 1
    assert sum(counts.values()) == 3 + 1           # equals summed detector_event_count


def test_circuit_registers_are_split_from_the_declaration_string() -> None:
    circuits = rows_to_table([{
        "source_record_id": "c1", "circuit_id": "qec_sm_n5/source", "benchmark_name": "qec_sm_n5", "variant": "source",
        "register_declarations": "qreg q[3];qreg a[2];creg c[3];creg syn[2]", "qubit_count": 5, "measurement_count": 5,
        "two_qubit_gate_count": 4,
    }], schemas.QASM_CIRCUIT)
    rows = transform.circuit_registers(circuits)
    assert rows == [("qec_sm_n5/source", "q", "qreg", 3, 0), ("qec_sm_n5/source", "a", "qreg", 2, 1),
                    ("qec_sm_n5/source", "c", "creg", 3, 2), ("qec_sm_n5/source", "syn", "creg", 2, 3)]


def test_check_qubits_keep_participation_order() -> None:
    checks = rows_to_table([{
        "source_record_id": "k1", "circuit_id": "qec_sm_n5/source", "check_id": "a[0]->syn[0]", "ancilla_qubit": "a[0]",
        "data_qubits": ["q[0]", "q[1]"], "syndrome_bit": "syn[0]",
    }], schemas.QASM_STABILIZER_CHECK)
    assert transform.stabilizer_check_qubits(checks) == [("k1", "q[0]", 0), ("k1", "q[1]", 1)]
    assert transform.stabilizer_checks(checks) == [("k1", "qec_sm_n5/source", "a[0]->syn[0]", "a[0]", "syn[0]")]
