"""End-to-end check of the Silver stage against the live course platform.

Runs the stage twice over the real Bronze release, reads the published Parquet
back from the object store, and checks the documented totals, trace coverage,
issue reconciliation, and identifier stability.
"""

from __future__ import annotations

import io
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.silver import schemas
from quantum_lake_student.stages import prepare_data

EXPECTED_ROWS = {
    "syndrome_observation": 75_598, "experiment": 5, "shot": 250_000,
    "circuit": 6, "stabilizer_check": 4, "conditional_correction": 6,
}


@pytest.fixture(scope="module")
def outcome(tmp_path_factory: pytest.TempPathFactory, monkeypatch_module):
    results = tmp_path_factory.mktemp("results")
    monkeypatch_module.setattr(prepare_data, "RESULTS_ROOT", results)
    settings = Settings.from_environment()
    first, counts = prepare_data.prepare(f"pytest-{uuid.uuid4()}", settings)
    client = minio_client(settings)
    tables = {name: _read(client, settings.s3_bucket, prepare_data.SILVER_PREFIX + path) for name, path in schemas.SILVER_PATHS.items()}
    return {"result": first, "counts": counts, "tables": tables, "settings": settings,
            "issues": pq.read_table(results / "part1/data_issues.parquet"),
            "trace": pq.read_table(results / "part1/source_trace.parquet")}


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patcher:
        yield patcher


def _read(client, bucket: str, key: str) -> pa.Table:
    response = client.get_object(bucket, key)
    try:
        return pq.read_table(io.BytesIO(response.read()))
    finally:
        response.close()
        response.release_conn()


def test_row_counts_match_the_release(outcome) -> None:
    assert {name: table.num_rows for name, table in outcome["tables"].items()} == EXPECTED_ROWS
    assert outcome["result"].output_count == sum(EXPECTED_ROWS.values())


def test_schemas_match_the_silver_contract(outcome) -> None:
    for name, table in outcome["tables"].items():
        assert table.schema.equals(schemas.SILVER_SCHEMAS[name], check_metadata=False), name


def test_syndrome_quantities_reconcile_to_seventy_million(outcome) -> None:
    table = outcome["tables"]["syndrome_observation"]
    assert pa.compute.sum(table["quantity"]).as_py() == 70_000_000
    assert set(pa.compute.unique(table["round_count"]).to_pylist()) == {4}
    assert all(len(v) == 16 for v in table["syndrome_bits"].slice(0, 500).to_pylist())
    assert len(pa.compute.unique(table["experiment_id"])) == 7


def test_google_shots_align_with_experiment_metadata(outcome) -> None:
    experiments = outcome["tables"]["experiment"].to_pylist()
    shots = outcome["tables"]["shot"]
    assert sum(e["shots"] for e in experiments) == shots.num_rows
    assert {e["distance"] for e in experiments} == {3, 5}
    widths = {e["experiment_id"]: (e["measurement_count"] + 7) // 8 for e in experiments}
    sample = shots.slice(0, 1000).to_pylist() + shots.slice(shots.num_rows - 1000).to_pylist()
    assert all(len(s["measurement_bits"]) == widths[s["experiment_id"]] for s in sample)
    assert all(0 <= s["detector_event_count"] <= 600 for s in sample)
    assert len(pa.compute.unique(shots["source_record_id"])) == shots.num_rows


def test_every_silver_row_is_traceable_to_bronze(outcome) -> None:
    trace = outcome["trace"]
    traced = set(trace["source_record_id"].to_pylist())
    for name, table in outcome["tables"].items():
        assert set(table["source_record_id"].to_pylist()) <= traced, name
    assert set(trace["source_name"].to_pylist()) == {"qec_syndromes", "google_qec", "qasmbench"}
    assert all(len(h) == 64 for h in pa.compute.unique(trace["input_sha256"]).to_pylist())
    assert all(len(i) == 32 for i in trace["source_record_id"].slice(0, 100).to_pylist())
    # A Google shot is assembled from eight companion members, so it has eight trace rows.
    shot_id = outcome["tables"]["shot"]["source_record_id"][0].as_py()
    mask = pa.compute.equal(trace["source_record_id"], shot_id)
    assert trace.filter(mask).num_rows == 8


def test_issues_reconcile_and_carry_required_columns(outcome) -> None:
    issues = outcome["issues"]
    assert {"issue_id", "run_id", "source_record_id", "rule_id", "severity", "observed_value", "action", "reason"} <= set(issues.column_names)
    assert issues.num_rows == outcome["result"].issue_count
    counts = outcome["counts"]
    for table in ("syndrome_observation", "experiment", "shot", "circuit"):
        assert counts[f"read.{table}"] == counts[f"silver.{table}"] + counts[f"rejected.{table}"], table
    assert issues.filter(pa.compute.equal(issues["action"], "reject")).num_rows == 0  # clean release
    assert set(issues.filter(pa.compute.equal(issues["rule_id"], "syn_header_label_vs_labels"))["archive_member"].to_pylist()) \
        == {f"d-3_pfr-{r}_nb-10M.csv" for r in ("0.000010", "0.000050", "0.000100", "0.000500", "0.001000", "0.005000", "0.010000")}


def test_second_run_produces_identical_identifiers(outcome) -> None:
    settings = outcome["settings"]
    second, counts = prepare_data.prepare(f"pytest-{uuid.uuid4()}", settings)
    assert counts == outcome["counts"]
    client = minio_client(settings)
    for name, path in schemas.SILVER_PATHS.items():
        again = _read(client, settings.s3_bucket, prepare_data.SILVER_PREFIX + path)
        assert again["source_record_id"].to_pylist() == outcome["tables"][name]["source_record_id"].to_pylist(), name
