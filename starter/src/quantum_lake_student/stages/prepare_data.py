"""Parse, check, and connect the supplied QEC data.

Reads the three Bronze archives from the object store, runs the source-specific
parsers in ``quantum_lake_student.silver``, and publishes:

- the six minimum Silver tables under ``silver/`` in the lake bucket;
- ``results/part1/data_issues.parquet`` with every check outcome;
- ``results/part1/source_trace.parquet`` linking every Silver row to its
  Bronze object, archive member, record position, and input hash.

Each output is replaced whole, and every identifier derives from source facts
only, so a second run on unchanged input yields identical records.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult
from quantum_lake_student.silver import google, qasm, schemas, syndromes
from quantum_lake_student.silver.common import BronzeObject, IssueLog, TraceRows, rows_to_table
from quantum_lake_student.stages.register_sources import BRONZE_PREFIX, load_manifest

STAGE = "prepare_data"
SILVER_PREFIX = "silver/"
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "results"))


def read_bronze(client: Minio, bucket: str) -> list[BronzeObject]:
    """Fetch every manifest object; the hash is recomputed so the trace is self-contained."""
    objects = []
    for entry in load_manifest(client, bucket):
        key = BRONZE_PREFIX + entry.relative_key
        response = client.get_object(bucket, key)
        try:
            payload = response.read()
        finally:
            response.close()
            response.release_conn()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.sha256:
            raise RuntimeError(f"{key}: sha256 {digest} does not match the manifest; run register_sources first")
        objects.append(BronzeObject(entry.source, key, digest, payload))
    return objects


def build_tables(objects: list[BronzeObject], issues: IssueLog) -> tuple[dict[str, pa.Table], TraceRows, dict[str, int]]:
    """Run every parser and return typed Silver tables, trace rows, and rows-read counts."""
    tables: dict[str, list[dict]] = {name: [] for name in schemas.SILVER_SCHEMAS}
    trace, read = TraceRows(), {}
    for obj in objects:
        if obj.source_name == syndromes.SOURCE:
            result = syndromes.prepare(obj, issues)
            tables["syndrome_observation"] += result.rows
            read["syndrome_observation"] = result.rows_read
        elif obj.source_name == google.SOURCE:
            result = google.prepare(obj, issues)
            tables["experiment"] += result.experiments
            tables["shot"] += result.shots
            read["experiment"], read["shot"] = result.experiments_read, result.shots_read
        elif obj.source_name == qasm.SOURCE:
            result = qasm.prepare(obj, issues)
            tables["circuit"] += result.circuits
            tables["stabilizer_check"] += result.checks
            tables["conditional_correction"] += result.corrections
            read["circuit"] = result.circuits_read
        else:
            issues.warning("bronze_unknown_source", obj.source_name, obj.bronze_key, None, None, "no parser for this source")
            continue
        _merge_trace(trace, result.trace)
    typed = {name: rows_to_table(rows, schemas.SILVER_SCHEMAS[name]) for name, rows in tables.items()}
    return typed, trace, read


def _merge_trace(target: TraceRows, extra: TraceRows) -> None:
    for column in target.__dataclass_fields__:
        getattr(target, column).extend(getattr(extra, column))


def issues_table(issues: IssueLog, run_id: str) -> pa.Table:
    rows = [
        {
            "issue_id": issue.issue_id, "run_id": run_id, "source_record_id": issue.source_record_id,
            "rule_id": issue.rule_id, "severity": issue.severity.value, "source_name": issue.source_name,
            "archive_member": issue.archive_member, "record_locator": issue.record_locator,
            "observed_value": issue.observed_value, "action": issue.action, "reason": issue.reason,
        }
        for issue in issues.issues
    ]
    return rows_to_table(rows, schemas.DATA_ISSUES)


def trace_table(trace: TraceRows) -> pa.Table:
    return pa.table({name: pa.array(getattr(trace, name), type=schemas.SOURCE_TRACE.field(name).type)
                     for name in schemas.SOURCE_TRACE.names}, schema=schemas.SOURCE_TRACE)


def write_lake(client: Minio, bucket: str, key: str, table: pa.Table) -> None:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    client.put_object(bucket, key, io.BytesIO(buffer.getvalue()), buffer.tell(), content_type="application/octet-stream")


def prepare(run_id: str, settings: Settings | None = None) -> tuple[StageResult, dict[str, int]]:
    settings = settings or Settings.from_environment()
    client = minio_client(settings)
    result = StageResult(stage=STAGE, run_id=run_id)
    issues = IssueLog()

    objects = read_bronze(client, settings.s3_bucket)
    tables, trace, read = build_tables(objects, issues)  # raises RunStop on a blocking check

    accepted = {name: table.num_rows for name, table in tables.items()}
    rejected = reconcile(read, accepted, issues)
    traced = set(trace.source_record_id)
    for name, table in tables.items():
        untraced = set(table.column("source_record_id").to_pylist()) - traced
        if untraced:
            raise RuntimeError(f"{len(untraced)} {name} row(s) have no source_trace entry")

    for name, table in tables.items():
        write_lake(client, settings.s3_bucket, SILVER_PREFIX + schemas.SILVER_PATHS[name], table)
    part1 = RESULTS_ROOT / "part1"
    part1.mkdir(parents=True, exist_ok=True)
    pq.write_table(issues_table(issues, run_id), part1 / "data_issues.parquet")
    pq.write_table(trace_table(trace), part1 / "source_trace.parquet", compression="zstd")

    result.input_count = sum(read.values())
    result.output_count = sum(accepted.values())
    result.issue_count = len(issues.issues)
    result.finish()
    return result, {
        **{f"silver.{k}": v for k, v in accepted.items()},
        **{f"read.{k}": v for k, v in read.items()},
        **{f"rejected.{k}": v for k, v in rejected.items()},
        "results.data_issues": len(issues.issues),
        "results.source_trace": len(trace),
    }


# Which reject rules remove a record from which read table. The two derived
# QASM tables are extracted from accepted circuits, so nothing is "read" there.
REJECT_RULES = {
    "syndrome_observation": lambda i: i.rule_id.startswith("syn_") and (i.record_locator or "").startswith("row"),
    "experiment": lambda i: i.rule_id in {"goog_properties_complete", "goog_dirname_matches_properties", "goog_companion_alignment"},
    "shot": lambda i: False,  # shots of a rejected experiment are never read
    "circuit": lambda i: i.rule_id == "qasm_structure",
}


def reconcile(read: dict[str, int], accepted: dict[str, int], issues: IssueLog) -> dict[str, int]:
    """Assert rows read == rows accepted + rows rejected for every read table."""
    rejected = {}
    for table, matches in REJECT_RULES.items():
        rejected[table] = sum(1 for i in issues.issues if i.action == "reject" and matches(i))
        if read.get(table, 0) != accepted[table] + rejected[table]:
            raise RuntimeError(
                f"{table}: read {read.get(table, 0)} != accepted {accepted[table]} + rejected {rejected[table]}"
            )
    return rejected


def run(run_id: str) -> StageResult:
    return prepare(run_id)[0]


if __name__ == "__main__":
    import uuid

    outcome, counts = prepare(run_id=str(uuid.uuid4()))
    for name, count in counts.items():
        print(f"{count:>10,}  {name}")
    print(f"read {outcome.input_count:,} records, accepted {outcome.output_count:,}, {outcome.issue_count} issue(s)")
