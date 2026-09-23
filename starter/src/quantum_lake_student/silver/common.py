"""Shared building blocks for the Silver parsers: identifiers, issues, tracing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pyarrow as pa

from quantum_lake_student.models import Severity

ACCEPT, REJECT, STOP = "accept", "reject", "stop"


def record_id(*parts: object) -> str:
    """Stable identifier from source facts only: never from run time or row order.

    128 bits of SHA-256 (32 hex chars): collision-safe for millions of records
    and half the storage of the full digest in the two-million-row trace table.
    """
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class BronzeObject:
    """One supplied archive as read from the object store."""

    source_name: str
    bronze_key: str
    sha256: str
    payload: bytes

    @property
    def object_name(self) -> str:
        return self.bronze_key.rsplit("/", 1)[-1]


class RunStop(RuntimeError):
    """Raised when a check finds a condition that must stop the whole run."""


@dataclass
class Issue:
    rule_id: str
    severity: Severity
    source_name: str
    archive_member: str | None
    record_locator: str | None
    observed_value: str | None
    action: str
    reason: str
    source_record_id: str | None = None

    @property
    def issue_id(self) -> str:
        return record_id(
            self.rule_id, self.source_name, self.archive_member, self.record_locator, self.observed_value
        )


@dataclass
class IssueLog:
    """Collects data-quality findings for ``results/part1/data_issues.parquet``."""

    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> Issue:
        self.issues.append(issue)
        return issue

    def info(self, rule_id: str, source: str, member: str | None, locator: str | None, observed: object, reason: str) -> Issue:
        return self.add(Issue(rule_id, Severity.INFO, source, member, locator, _text(observed), ACCEPT, reason))

    def warning(self, rule_id: str, source: str, member: str | None, locator: str | None, observed: object, reason: str) -> Issue:
        return self.add(Issue(rule_id, Severity.WARNING, source, member, locator, _text(observed), ACCEPT, reason))

    def reject(self, rule_id: str, source: str, member: str | None, locator: str | None, observed: object, reason: str) -> Issue:
        return self.add(Issue(rule_id, Severity.ERROR, source, member, locator, _text(observed), REJECT, reason))

    def stop(self, rule_id: str, source: str, member: str | None, locator: str | None, observed: object, reason: str) -> RunStop:
        self.add(Issue(rule_id, Severity.ERROR, source, member, locator, _text(observed), STOP, reason))
        return RunStop(f"{rule_id}: {reason} ({member or source})")

    @property
    def rejected(self) -> int:
        return sum(issue.action == REJECT for issue in self.issues)


def _text(value: object, limit: int = 200) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class TraceRows:
    """Column-wise trace rows; cheaper than one dict per row for 2M+ entries."""

    source_record_id: list[str] = field(default_factory=list)
    source_name: list[str] = field(default_factory=list)
    bronze_object: list[str] = field(default_factory=list)
    archive_member: list[str | None] = field(default_factory=list)
    record_locator: list[str] = field(default_factory=list)
    input_sha256: list[str] = field(default_factory=list)
    silver_table: list[str] = field(default_factory=list)

    def add(self, source_record_id: str, obj: BronzeObject, member: str | None, locator: str, table: str) -> None:
        self.source_record_id.append(source_record_id)
        self.source_name.append(obj.source_name)
        self.bronze_object.append(obj.object_name)
        self.archive_member.append(member)
        self.record_locator.append(locator)
        self.input_sha256.append(obj.sha256)
        self.silver_table.append(table)

    def add_many(self, ids: list[str], obj: BronzeObject, member: str, locators: list[str], table: str) -> None:
        count = len(ids)
        self.source_record_id.extend(ids)
        self.source_name.extend([obj.source_name] * count)
        self.bronze_object.extend([obj.object_name] * count)
        self.archive_member.extend([member] * count)
        self.record_locator.extend(locators)
        self.input_sha256.extend([obj.sha256] * count)
        self.silver_table.extend([table] * count)

    def __len__(self) -> int:
        return len(self.source_record_id)


def rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Build a typed table from row dicts; an empty list still yields the schema."""
    columns = {name: [row[name] for row in rows] for name in schema.names}
    return pa.table({name: pa.array(values, type=schema.field(name).type) for name, values in columns.items()}, schema=schema)
