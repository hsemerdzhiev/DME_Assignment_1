"""Tests for the Bronze registration stage.

The verification tests run against the live course platform (MinIO and
PostgreSQL from ``make up``). Failure cases use a throw-away prefix in the
same bucket so the real ``bronze/`` area is never touched.
"""

from __future__ import annotations

import hashlib
import io
import uuid
import zipfile
from collections.abc import Iterator

import pytest
from minio import Minio
from minio.deleteobjects import DeleteObject

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client, postgres_connection
from quantum_lake_student.stages import register_sources as stage


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name), payload)  # ZipInfo keeps odd names as-is
    return buffer.getvalue()


def _manifest_object(source: str, filename: str, data: bytes, mandatory: bool = True) -> stage.ManifestObject:
    return stage.ManifestObject(
        source=source,
        path=f"raw/source={source}/{filename}",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        mandatory=mandatory,
    )


class Scratch:
    """A private prefix in the course bucket, emptied after each test."""

    def __init__(self, client: Minio, bucket: str) -> None:
        self.client, self.bucket = client, bucket
        self.prefix = f"_test/register_sources/{uuid.uuid4()}/"
        self.alpha = _zip_bytes({"a/one.txt": b"1", "a/two.txt": b"2"})
        self.beta = _zip_bytes({"b.csv": b"x,y\n"})
        self.put("source=alpha/alpha.zip", self.alpha)
        self.put("source=beta/beta.zip", self.beta)
        self.manifest = [
            _manifest_object("alpha", "alpha.zip", self.alpha),
            _manifest_object("beta", "beta.zip", self.beta),
        ]

    def put(self, relative_key: str, data: bytes) -> None:
        self.client.put_object(self.bucket, self.prefix + relative_key, io.BytesIO(data), len(data))

    def remove(self, relative_key: str) -> None:
        self.client.remove_object(self.bucket, self.prefix + relative_key)

    def verify(self, manifest: list[stage.ManifestObject] | None = None) -> dict[str, stage.SourceRecord]:
        records = stage.verify_sources(self.client, self.bucket, manifest or self.manifest, self.prefix)
        return {record.bronze_key.removeprefix(self.prefix): record for record in records}

    def cleanup(self) -> None:
        keys = [item.object_name for item in self.client.list_objects(self.bucket, self.prefix, recursive=True)]
        list(self.client.remove_objects(self.bucket, [DeleteObject(key) for key in keys]))


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_environment()


@pytest.fixture(scope="module")
def client(settings: Settings) -> Minio:
    return minio_client(settings)


@pytest.fixture()
def scratch(client: Minio, settings: Settings) -> Iterator[Scratch]:
    area = Scratch(client, settings.s3_bucket)
    yield area
    area.cleanup()


# --- pure checks ------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["small/qec_sm_n5/qec_sm_n5.qasm", "surface_code_bX_d3_r25_center_3_5/measurements.b8", "dir/", "a b.txt"],
)
def test_safe_member_accepts_relative_paths(name: str) -> None:
    assert stage.is_safe_member(name)


@pytest.mark.parametrize(
    "name",
    [
        "/etc/passwd", "\\windows\\x", "C:\\Windows\\evil.txt", "C:/evil.txt", "c:evil.txt",
        "//server/share/f", "\\\\server\\share\\f", "a/../b.txt", "..\\up.txt", "../up.txt", "..", "", " x.txt",
    ],
)
def test_safe_member_rejects_absolute_drive_and_parent_paths(name: str) -> None:
    assert not stage.is_safe_member(name)


def test_parse_manifest_defaults_mandatory_and_normalises() -> None:
    [obj] = stage.parse_manifest(
        {"objects": [{"source": "s", "path": "raw/source=s/s.zip", "sha256": "AB", "bytes": "3"}]}
    )
    assert obj.mandatory is True
    assert (obj.sha256, obj.bytes, obj.relative_key) == ("ab", 3, "source=s/s.zip")


def test_parse_manifest_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        stage.parse_manifest({"objects": []})


# --- against live MinIO -----------------------------------------------------


def test_matching_objects_verify(scratch: Scratch) -> None:
    records = scratch.verify()
    assert set(records) == {"source=alpha/alpha.zip", "source=beta/beta.zip"}
    alpha = records["source=alpha/alpha.zip"]
    assert alpha.status == stage.VERIFIED and alpha.verified and not alpha.blocks_run
    assert alpha.member_count == 2
    assert alpha.actual_sha256 == alpha.expected_sha256
    assert alpha.actual_bytes == alpha.expected_bytes == len(scratch.alpha)
    assert alpha.failure_reason is None


def test_missing_mandatory_object_is_reported_and_blocks(scratch: Scratch) -> None:
    scratch.remove("source=beta/beta.zip")
    records = scratch.verify()
    beta = records["source=beta/beta.zip"]
    assert beta.status == stage.MISSING and beta.blocks_run
    assert "absent" in beta.failure_reason
    assert records["source=alpha/alpha.zip"].verified  # one failure does not hide the rest


def test_missing_optional_object_does_not_block(scratch: Scratch) -> None:
    scratch.remove("source=beta/beta.zip")
    manifest = [scratch.manifest[0], _manifest_object("beta", "beta.zip", scratch.beta, mandatory=False)]
    beta = scratch.verify(manifest)["source=beta/beta.zip"]
    assert beta.status == stage.MISSING and not beta.blocks_run


def test_size_mismatch_is_caught_before_download(scratch: Scratch) -> None:
    scratch.put("source=beta/beta.zip", scratch.beta + b"extra")
    beta = scratch.verify()["source=beta/beta.zip"]
    assert beta.status == stage.FAILED and beta.blocks_run
    assert beta.failure_reason.startswith("size mismatch")
    assert beta.actual_sha256 is None


def test_hash_mismatch_with_same_size_is_caught(scratch: Scratch) -> None:
    corrupted = bytearray(scratch.beta)
    corrupted[-1] ^= 0xFF
    scratch.put("source=beta/beta.zip", bytes(corrupted))
    beta = scratch.verify()["source=beta/beta.zip"]
    assert beta.status == stage.FAILED
    assert beta.failure_reason.startswith("sha256 mismatch")


def test_unsafe_archive_member_fails(scratch: Scratch) -> None:
    evil = _zip_bytes({"../escape.txt": b"!"})
    scratch.put("source=beta/beta.zip", evil)
    manifest = [scratch.manifest[0], _manifest_object("beta", "beta.zip", evil)]
    beta = scratch.verify(manifest)["source=beta/beta.zip"]
    assert beta.status == stage.FAILED
    assert "unsafe archive member" in beta.failure_reason and "../escape.txt" in beta.failure_reason


def test_unexpected_objects_are_reported_without_blocking(scratch: Scratch) -> None:
    scratch.put("source=alpha/alpha/unpacked.txt", b"should not be here")
    extra = scratch.verify()["source=alpha/alpha/unpacked.txt"]
    assert extra.status == stage.UNEXPECTED
    assert extra.source == "alpha"
    assert extra.actual_bytes == len(b"should not be here")
    assert not extra.verified and not extra.blocks_run


def test_supplied_bronze_matches_release_manifest(client: Minio, settings: Settings) -> None:
    manifest = stage.load_manifest(client, settings.s3_bucket)
    records = stage.verify_sources(client, settings.s3_bucket, manifest)
    assert {r.source for r in records if r.verified} == {"qasmbench", "qec_syndromes", "google_qec"}
    assert not [r for r in records if r.blocks_run]


# --- against live PostgreSQL ------------------------------------------------


def test_second_run_updates_in_place_without_duplicates(settings: Settings) -> None:
    first, second = f"pytest-{uuid.uuid4()}", f"pytest-{uuid.uuid4()}"
    stage.register(first, settings)
    result, _ = stage.register(second, settings)
    assert result.output_count == 3

    with postgres_connection(settings) as connection:
        rows = connection.execute(
            "SELECT count(*), count(DISTINCT (source, bronze_key)),"
            "       bool_and(first_run_id <> %s), bool_and(last_run_id = %s),"
            "       bool_and(registered_at <= last_verified_at)"
            " FROM lake.source_object WHERE status = 'verified'",
            (second, second),
        ).fetchone()
    total, distinct, first_kept, last_moved, timestamps_ordered = rows
    assert total == distinct == 3
    assert first_kept and last_moved and timestamps_ordered
