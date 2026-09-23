"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.

The release description is ``metadata/course-release/bundle-manifest.json``,
seeded into the bucket next to the three Bronze archives. Every object it lists
is checked (size, SHA-256, archive member names) without extracting anything;
everything under ``bronze/`` that it does not list is reported as unexpected.
One row per object goes to ``lake.source_object``, keyed by ``(source,
bronze_key)`` so reruns update in place. Failures are stored with
``verified = false`` and a reason before the run is stopped.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from minio import Minio
from minio.error import S3Error

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.models import StageResult

STAGE = "register_sources"
MANIFEST_KEY = "metadata/course-release/bundle-manifest.json"
BRONZE_PREFIX = "bronze/"
CHUNK_SIZE = 1024 * 1024

VERIFIED, FAILED, MISSING, UNEXPECTED = "verified", "failed", "missing", "unexpected"

DDL = """
CREATE SCHEMA IF NOT EXISTS lake;
CREATE TABLE IF NOT EXISTS lake.source_object (
    source           text        NOT NULL,
    bronze_key       text        NOT NULL,
    status           text        NOT NULL
        CHECK (status IN ('verified', 'failed', 'missing', 'unexpected')),
    verified         boolean     NOT NULL CHECK (verified = (status = 'verified')),
    mandatory        boolean     NOT NULL,
    expected_sha256  text,
    actual_sha256    text,
    expected_bytes   bigint,
    actual_bytes     bigint,
    member_count     integer,
    failure_reason   text,
    first_run_id     text        NOT NULL,
    last_run_id      text        NOT NULL,
    registered_at    timestamptz NOT NULL DEFAULT now(),
    last_verified_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, bronze_key)
);
"""

UPSERT = """
INSERT INTO lake.source_object (
    source, bronze_key, status, verified, mandatory, expected_sha256, actual_sha256,
    expected_bytes, actual_bytes, member_count, failure_reason, first_run_id, last_run_id
) VALUES (
    %(source)s, %(bronze_key)s, %(status)s, %(verified)s, %(mandatory)s, %(expected_sha256)s,
    %(actual_sha256)s, %(expected_bytes)s, %(actual_bytes)s, %(member_count)s,
    %(failure_reason)s, %(run_id)s, %(run_id)s
)
ON CONFLICT (source, bronze_key) DO UPDATE SET
    status = EXCLUDED.status, verified = EXCLUDED.verified, mandatory = EXCLUDED.mandatory,
    expected_sha256 = EXCLUDED.expected_sha256, actual_sha256 = EXCLUDED.actual_sha256,
    expected_bytes = EXCLUDED.expected_bytes, actual_bytes = EXCLUDED.actual_bytes,
    member_count = EXCLUDED.member_count, failure_reason = EXCLUDED.failure_reason,
    last_run_id = EXCLUDED.last_run_id, last_verified_at = now();
"""


@dataclass(frozen=True)
class ManifestObject:
    source: str
    path: str  # as written in the manifest, e.g. "raw/source=x/x.zip"
    sha256: str
    bytes: int
    mandatory: bool = True

    @property
    def relative_key(self) -> str:
        """Path below the data area: the seed mirrors ``raw/`` to ``bronze/``."""
        return self.path.removeprefix("raw/")


def parse_manifest(document: dict) -> list[ManifestObject]:
    objects = [
        ManifestObject(
            source=entry["source"],
            path=entry["path"],
            sha256=entry["sha256"].lower(),
            bytes=int(entry["bytes"]),
            mandatory=bool(entry.get("mandatory", True)),
        )
        for entry in document.get("objects", [])
    ]
    if not objects:
        raise ValueError("The bundle manifest lists no objects")
    return objects


def load_manifest(client: Minio, bucket: str, key: str = MANIFEST_KEY) -> list[ManifestObject]:
    response = client.get_object(bucket, key)
    try:
        return parse_manifest(json.loads(response.read()))
    finally:
        response.close()
        response.release_conn()


def is_safe_member(name: str) -> bool:
    """Reject names that could escape an extraction directory on any platform."""
    if not name or name.strip() != name or name.startswith(("/", "\\")):
        return False
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return False
    return ".." not in posix.parts and ".." not in windows.parts


class VerificationError(ValueError):
    """A supplied object does not match the release description."""


@dataclass
class SourceRecord:
    source: str
    bronze_key: str
    status: str
    mandatory: bool
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    expected_bytes: int | None = None
    actual_bytes: int | None = None
    member_count: int | None = None
    failure_reason: str | None = None

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    @property
    def blocks_run(self) -> bool:
        return self.mandatory and self.status in {FAILED, MISSING}

    def row(self, run_id: str) -> dict:
        return {**self.__dict__, "verified": self.verified, "run_id": run_id}


def verify_object(client: Minio, bucket: str, key: str, obj: ManifestObject) -> SourceRecord:
    """Check size, hash, and archive member names of one expected object."""
    record = SourceRecord(
        source=obj.source,
        bronze_key=key,
        status=FAILED,
        mandatory=obj.mandatory,
        expected_sha256=obj.sha256,
        expected_bytes=obj.bytes,
    )
    try:
        # Size first: it needs no download and gives the clearer message.
        record.actual_bytes = client.stat_object(bucket, key).size or 0
        if record.actual_bytes != obj.bytes:
            raise VerificationError(
                f"size mismatch (expected {obj.bytes}, got {record.actual_bytes})"
            )

        digest, payload = hashlib.sha256(), io.BytesIO()
        response = client.get_object(bucket, key)
        try:
            for chunk in response.stream(CHUNK_SIZE):
                digest.update(chunk)
                payload.write(chunk)
                if payload.tell() > obj.bytes:
                    raise VerificationError(f"object grew past the manifest size {obj.bytes}")
        finally:
            response.close()
            response.release_conn()
        record.actual_sha256 = digest.hexdigest()
        if record.actual_sha256 != obj.sha256:
            raise VerificationError(
                f"sha256 mismatch (expected {obj.sha256}, got {record.actual_sha256})"
            )

        with zipfile.ZipFile(payload) as archive:
            names = archive.namelist()
        if unsafe := [name for name in names if not is_safe_member(name)]:
            raise VerificationError(f"unsafe archive member paths: {unsafe[:5]}")
        record.member_count = len(names)
        record.status = VERIFIED
    except VerificationError as error:
        record.failure_reason = str(error)
    except zipfile.BadZipFile as error:
        record.failure_reason = f"not a readable ZIP archive: {error}"
    except S3Error as error:
        if error.code == "NoSuchKey":
            record.status = MISSING
            record.failure_reason = "listed in the manifest but absent from the Bronze area"
        else:
            record.failure_reason = f"object store error {error.code}: {error.message}"
    return record


def verify_sources(
    client: Minio, bucket: str, manifest: list[ManifestObject], prefix: str = BRONZE_PREFIX
) -> list[SourceRecord]:
    """Verify every expected object under ``prefix`` and report anything extra."""
    expected = {prefix + obj.relative_key: obj for obj in manifest}
    records = [verify_object(client, bucket, key, obj) for key, obj in expected.items()]
    for item in client.list_objects(bucket, prefix=prefix, recursive=True):
        if item.object_name in expected:
            continue
        head = item.object_name.removeprefix(prefix).split("/", 1)[0]
        records.append(
            SourceRecord(
                source=head.removeprefix("source=") if head.startswith("source=") else "unknown",
                bronze_key=item.object_name,
                status=UNEXPECTED,
                mandatory=False,
                actual_bytes=item.size or 0,
                failure_reason="present in the Bronze area but not listed in the manifest",
            )
        )
    return records


def register(run_id: str, settings: Settings | None = None) -> tuple[StageResult, list[SourceRecord]]:
    """Verify, then persist every record. Returns the result and the records."""
    settings = settings or Settings.from_environment()
    client = minio_client(settings)
    result = StageResult(stage=STAGE, run_id=run_id)

    # All verification happens before PostgreSQL is opened, so the transaction
    # is short and never waits on downloads.
    records = verify_sources(client, settings.s3_bucket, load_manifest(client, settings.s3_bucket))
    with postgres_connection(settings) as connection:  # commits on clean exit
        connection.execute(DDL)
        for record in records:
            connection.execute(UPSERT, record.row(run_id))

    result.input_count = sum(record.status != UNEXPECTED for record in records)
    result.output_count = sum(record.verified for record in records)
    result.issue_count = sum(not record.verified for record in records)
    result.finish()
    return result, records


def run(run_id: str) -> StageResult:
    result, records = register(run_id)
    if blocking := [record for record in records if record.blocks_run]:
        details = "; ".join(f"{r.bronze_key}: {r.failure_reason}" for r in blocking)
        raise RuntimeError(
            f"{STAGE}: {len(blocking)} mandatory source object(s) failed verification "
            f"(recorded in lake.source_object): {details}"
        )
    return result


if __name__ == "__main__":
    import uuid

    result, records = register(run_id=str(uuid.uuid4()))
    for record in records:
        print(f"{record.status:10s} {record.bronze_key}", record.failure_reason or "")
    print(f"verified {result.output_count}/{result.input_count}, {result.issue_count} issue(s)")
    raise SystemExit(1 if any(record.blocks_run for record in records) else 0)
