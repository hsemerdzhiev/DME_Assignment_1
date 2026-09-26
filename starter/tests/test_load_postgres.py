"""Gold load against the live platform: counts, constraints, atomicity, reruns."""

from __future__ import annotations

import uuid

import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.stages import load_postgres


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_environment()


@pytest.fixture(scope="module")
def loaded(settings: Settings):
    """Load Gold once; remember the counts of the load that preceded it, if any."""
    with postgres_connection(settings) as connection:
        previous = connection.execute(
            "SELECT row_counts FROM gold.gold_load WHERE to_regclass('gold.gold_load') IS NOT NULL"
        ).fetchone() if _schema_exists(connection) else None
    run_id = f"pytest-{uuid.uuid4()}"
    result, counts = load_postgres.load(run_id, settings)
    return {"run_id": run_id, "result": result, "counts": counts, "previous": previous[0] if previous else None}


def _schema_exists(connection) -> bool:
    return connection.execute("SELECT 1 FROM pg_namespace WHERE nspname = 'gold'").fetchone() is not None


def _one(settings: Settings, sql: str, *params):
    with postgres_connection(settings) as connection:
        return connection.execute(sql, params).fetchone()


def test_row_counts_match_the_release(loaded) -> None:
    counts = loaded["counts"]
    assert counts["gold.experiment"] == 7 + 5
    assert counts["gold.syndrome_observation"] == 75_598
    assert counts["gold.shot"] == 250_000
    assert counts["gold.decoder_prediction"] == 4 * 250_000
    assert counts["gold.detector_summary"] == 4 * 200 + 600
    assert (counts["gold.circuit"], counts["gold.stabilizer_check"], counts["gold.conditional_correction"]) == (6, 4, 6)
    assert counts["gold.syndrome_pattern"] < counts["gold.syndrome_observation"]  # patterns are shared


def test_gold_holds_exactly_one_load_record_for_this_run(loaded, settings) -> None:
    assert _one(settings, "SELECT count(*), max(run_id) FROM gold.gold_load") == (1, loaded["run_id"])
    assert _one(settings, "SELECT count(*) FROM pg_namespace WHERE nspname = 'gold_build'") == (0,)


def test_weighted_totals_and_derived_view(settings) -> None:
    assert _one(settings, "SELECT sum(quantity) FROM gold.syndrome_observation") == (70_000_000,)
    assert _one(settings, "SELECT count(*) FROM gold.experiment WHERE kind = 'simulated_syndrome' AND check_count = 4 AND rounds = 4") == (7,)
    assert _one(settings, "SELECT count(*) FROM gold.experiment WHERE kind = 'hardware_memory' AND check_count IN (8, 24) AND rounds = 25") == (5,)
    total, errors = _one(settings, "SELECT count(*), count(*) FILTER (WHERE decoder_error) FROM gold.decoder_outcome")
    assert total == 1_000_000 and 0 < errors < total


def test_keys_constraints_and_indexes_exist(settings) -> None:
    kinds = dict(_rows(settings, "SELECT contype, count(*) FROM pg_constraint c JOIN pg_namespace n ON n.oid = c.connamespace "
                                 "WHERE n.nspname = 'gold' GROUP BY contype"))
    assert kinds["p"] == 13 and kinds["f"] >= 10 and kinds["c"] >= 25 and kinds["u"] >= 6
    assert _one(settings, "SELECT count(*) FROM pg_indexes WHERE schemaname = 'gold'")[0] >= 20


def test_gold_keys_resolve_to_silver_trace_ids(settings) -> None:
    # Gold keys equal Silver source_record_id for one-to-one grains: 32 hex characters.
    for table, key in (("shot", "shot_id"), ("syndrome_observation", "observation_id"), ("circuit", "source_record_id"),
                       ("stabilizer_check", "check_id"), ("conditional_correction", "correction_id")):
        assert _one(settings, f"SELECT count(*) FROM gold.{table} WHERE {key} !~ '^[0-9a-f]{{32}}$'") == (0,), table


def test_rerun_reproduces_the_previous_counts(loaded) -> None:
    if loaded["previous"] is None:
        pytest.skip("no earlier Gold load to compare against")
    assert {f"gold.{k}": v for k, v in loaded["previous"].items()} == loaded["counts"]


def test_failed_load_leaves_previous_gold_untouched(loaded, settings) -> None:
    silver = load_postgres.read_silver(minio_client(settings), settings.s3_bucket)
    rows = load_postgres.build_rows(silver)
    rows["decoder"] = rows["decoder"] + [("belief_matching", "duplicate key", "x")]  # violates the primary key
    with postgres_connection(settings) as connection, pytest.raises(Exception):
        load_postgres.load_gold(connection, rows, "pytest-should-fail")
    assert _one(settings, "SELECT run_id FROM gold.gold_load") == (loaded["run_id"],)
    assert _one(settings, "SELECT count(*) FROM pg_namespace WHERE nspname = 'gold_build'") == (0,)


def _rows(settings: Settings, sql: str):
    with postgres_connection(settings) as connection:
        return connection.execute(sql).fetchall()
