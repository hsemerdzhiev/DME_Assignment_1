"""Unit tests for the Google experiment parser on a tiny synthetic experiment."""

from __future__ import annotations

import io
import zipfile

import pytest

from quantum_lake_student.silver import google
from quantum_lake_student.silver.common import BronzeObject, IssueLog, RunStop

DIR = "surface_code_bX_d3_r2_center_1_2"
PROPS = """type: surface_code_memory_experiment
basis: X
rounds: 2
distance: 3
shots: 3
center_data_qubit_row: 1
center_data_qubit_col: 2
circuit_measurements: 10
circuit_sweep_bits: 3
circuit_detectors: 9
"""
# 3 shots. measurements: 10 bits -> 2 bytes; sweep: 3 bits -> 1 byte; detectors: 9 bits -> 2 bytes.
MEASUREMENTS = bytes([0b10101010, 0b00000001, 0xFF, 0x03, 0x00, 0x00])
SWEEP = bytes([0b101, 0b000, 0b111])
DETECTORS = bytes([0b00000001, 0b00000001, 0x00, 0x00, 0xFF, 0x01])  # shot0: 2 events, shot1: 0, shot2: 9


def _files(**overrides: bytes) -> dict[str, bytes]:
    files = {
        "properties.yml": PROPS.encode(), "measurements.b8": MEASUREMENTS, "sweep.b8": SWEEP, "detection_events.b8": DETECTORS,
        "obs_flips_actual.01": b"0\n1\n1\n",
        "obs_flips_predicted_by_belief_matching.01": b"0\n1\n0\n",
        "obs_flips_predicted_by_correlated_matching.01": b"0\n0\n1\n",
        "obs_flips_predicted_by_pymatching.01": b"1\n1\n1\n",
        "obs_flips_predicted_by_tensor_network_contraction.01": b"0\n1\n1\n",
    }
    files.update(overrides)
    return files


def _bronze(files: dict[str, bytes], directory: str = DIR) -> BronzeObject:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README.txt", "readme")
        for name, data in files.items():
            archive.writestr(f"{directory}/{name}", data)
    return BronzeObject("google_qec", "bronze/source=google_qec/t.zip", "1" * 64, buffer.getvalue())


def test_aligned_experiment_yields_experiment_and_shot_rows() -> None:
    issues = IssueLog()
    result = google.prepare(_bronze(_files()), issues)

    assert result.experiments_read == 1 and len(result.experiments) == 1
    experiment = result.experiments[0]
    assert (experiment["experiment_id"], experiment["distance"], experiment["rounds"], experiment["shots"]) == (DIR, 3, 2, 3)
    assert (experiment["center_row"], experiment["center_col"]) == (1, 2)
    assert (experiment["measurement_count"], experiment["detector_count"]) == (10, 9)

    assert result.shots_read == 3 and len(result.shots) == 3
    first, second, third = result.shots
    assert [s["shot_index"] for s in result.shots] == [0, 1, 2]
    assert first["measurement_bits"] == MEASUREMENTS[:2] and first["sweep_bits"] == SWEEP[:1]
    assert first["detector_bits"] == DETECTORS[:2]
    assert [s["detector_event_count"] for s in result.shots] == [2, 0, 9]
    assert [s["actual_observable_flip"] for s in result.shots] == [False, True, True]
    assert (first["pymatching_prediction"], first["belief_matching_prediction"]) == (True, False)
    assert third["correlated_matching_prediction"] is True
    assert not issues.issues

    # One trace row for the experiment, then one per shot per companion file (3 b8 + 5 01 = 8).
    assert len(result.trace) == 1 + 3 * 8
    members = set(result.trace.archive_member)
    assert f"{DIR}/measurements.b8" in members and f"{DIR}/obs_flips_predicted_by_pymatching.01" in members
    assert result.trace.record_locator.count("shot 2") == 8


def test_non_zero_padding_bits_are_reported_but_kept() -> None:
    # detector row 2 = 0xFF 0x01 uses 9 bits; set bit 10 (a padding bit) in the second byte.
    detectors = DETECTORS[:5] + bytes([0x01 | 0b100])
    issues = IssueLog()
    result = google.prepare(_bronze(_files(**{"detection_events.b8": detectors})), issues)
    assert result.shots[2]["detector_bits"] == detectors[4:6]          # bytes unchanged
    assert result.shots[2]["detector_event_count"] == 9                # padding not counted
    [warning] = issues.issues
    assert warning.rule_id == "goog_b8_padding_zero" and warning.observed_value == "1"


def test_wrong_b8_length_rejects_the_whole_experiment() -> None:
    issues = IssueLog()
    result = google.prepare(_bronze(_files(**{"measurements.b8": MEASUREMENTS + b"\x00"})), issues)
    assert result.experiments == [] and result.shots == [] and result.experiments_read == 1
    rules = [i.rule_id for i in issues.issues]
    assert "goog_b8_length" in rules and "goog_companion_alignment" in rules


def test_01_file_with_wrong_row_count_or_value_is_rejected() -> None:
    issues = IssueLog()
    google.prepare(_bronze(_files(**{"obs_flips_actual.01": b"0\n2\n1\n"})), issues)
    assert any(i.rule_id == "goog_01_rows_binary" for i in issues.issues)


def test_directory_name_must_agree_with_properties() -> None:
    issues = IssueLog()
    result = google.prepare(_bronze(_files(), directory="surface_code_bZ_d3_r2_center_1_2"), issues)
    assert result.experiments == []
    assert issues.issues[0].rule_id == "goog_dirname_matches_properties"


def test_missing_companion_file_stops_the_run() -> None:
    files = _files()
    del files["sweep.b8"]
    issues = IssueLog()
    with pytest.raises(RunStop):
        google.prepare(_bronze(files), issues)
    assert issues.issues[-1].action == "stop" and "sweep.b8" in issues.issues[-1].observed_value
