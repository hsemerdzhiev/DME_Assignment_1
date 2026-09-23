"""Unit tests for the syndrome CSV parser on tiny in-memory archives."""

from __future__ import annotations

import io
import zipfile

import pytest

from quantum_lake_student.silver import syndromes
from quantum_lake_student.silver.common import BronzeObject, IssueLog

GOOD = "((0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, 1))"


def _bronze(files: dict[str, str]) -> BronzeObject:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return BronzeObject("qec_syndromes", "bronze/source=qec_syndromes/t.zip", "0" * 64, buffer.getvalue())


def test_parse_syndrome_is_round_major_one_byte_per_value() -> None:
    assert syndromes.parse_syndrome(GOOD) == bytes([0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])


@pytest.mark.parametrize(
    "text",
    ["((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))", "((0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))",
     "((0, 0, 0, 2), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))", "[0, 0, 0, 0]", "not a tuple"],
)
def test_parse_syndrome_rejects_wrong_shape_or_domain(text: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        syndromes.parse_syndrome(text)


def test_parse_filename_extracts_distance_rate_and_nominal_count() -> None:
    assert syndromes.parse_filename("d-3_pfr-0.000500_nb-10M.csv") == (3, 0.0005, 10_000_000)
    with pytest.raises(ValueError):
        syndromes.parse_filename("syndromes.csv")


def test_prepare_accepts_valid_rows_and_rejects_invalid_ones_with_reasons() -> None:
    csv = "\n".join([
        "labels,syndromes,quantity",
        f'0,"{GOOD}",6',                      # ok
        f'1,"{GOOD}",3',                      # ok: same syndrome, other label
        f'2,"{GOOD}",1',                      # bad label
        '0,"((0, 0, 0, 0), (0, 0, 0, 0))",1',  # bad shape
        f'0,"{GOOD}",0',                      # bad quantity
        "0,broken",                           # bad field count
    ])
    issues = IssueLog()
    result = syndromes.prepare(_bronze({"d-3_pfr-0.001000_nb-10.csv": csv}), issues)

    assert result.rows_read == 6 and len(result.rows) == 2 and issues.rejected == 4
    assert {i.rule_id for i in issues.issues if i.action == "reject"} == {
        "syn_label_binary", "syn_shape_4x4_binary", "syn_quantity_positive", "syn_field_count"}
    assert result.rows[0]["experiment_id"] == "d-3_pfr-0.001000"
    assert result.rows[0]["physical_fault_rate"] == 0.001
    assert (result.rows[0]["logical_error_label"], result.rows[1]["logical_error_label"]) == (False, True)
    assert result.rows[0]["source_record_id"] != result.rows[1]["source_record_id"]
    assert len(result.trace) == 2 and result.trace.record_locator == ["row 2", "row 3"]

    by_rule = {i.rule_id: i for i in issues.issues if i.action == "accept"}
    assert "syn_header_label_vs_labels" in by_rule
    assert by_rule["syn_syndrome_with_both_labels"].observed_value == "1"
    assert by_rule["syn_quantity_reconciles_to_filename"].observed_value == "9"  # 6 + 3 != nominal 10


def test_prepare_rejects_file_with_unexpected_header() -> None:
    issues = IssueLog()
    result = syndromes.prepare(_bronze({"d-3_pfr-0.001000_nb-10.csv": "a,b,c\n0,x,1\n"}), issues)
    assert result.rows == [] and result.rows_read == 0
    assert issues.issues[0].rule_id == "syn_header_unexpected"


def test_identifiers_are_stable_across_runs() -> None:
    files = {"d-3_pfr-0.001000_nb-10.csv": f'labels,syndromes,quantity\n0,"{GOOD}",10\n'}
    first = syndromes.prepare(_bronze(files), IssueLog()).rows[0]["source_record_id"]
    second = syndromes.prepare(_bronze(files), IssueLog()).rows[0]["source_record_id"]
    assert first == second
