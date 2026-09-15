"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.
"""


from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass

import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.models import StageResult

# Mounted read-only into the workspace container by compose.yaml.
MANIFEST_PATH = "/course-data/metadata/bundle-manifest.json"

DDL = """
CREATE SCHEMA IF NOT EXISTS lake;

CREATE TABLE IF NOT EXISTS lake.source_object (
    source text NOT NULL,
    bronze_key text NOT NULL,
    sha256 text NOT NULL,
    bytes bigint NOT NULL,
    member_count integer NOT NULL,
    verified boolean NOT NULL,
    last_run_id text NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, bronze_key)
);
"""

UPSERT = """
INSERT INTO lake.source_object
    (source, bronze_key, sha256, bytes, member_count, verified, last_run_id)
VALUES (%(source)s, %(bronze_key)s, %(sha256)s, %(bytes)s,
        %(member_count)s, %(verified)s, %(run_id)s)
ON CONFLICT (source, bronze_key) DO UPDATE SET
    sha256 = EXCLUDED.sha256,
    bytes = EXCLUDED.bytes,
    member_count = EXCLUDED.member_count,
    verified = EXCLUDED.verified,
    last_run_id = EXCLUDED.last_run_id,
    registered_at = now();
"""


@dataclass(frozen=True)
class ManifestObject:
    source: str
    raw_path: str
    sha256: str
    bytes: int
    mandatory: bool

    @property
    def bronze_key(self) -> str:
        # init/minio/seed.sh mirrors /seed/raw -> bronze/, so the manifest's
        # "raw/..." path becomes "bronze/..." in the object store.
        return "bronze/" + self.raw_path.removeprefix("raw/")


def load_manifest() -> list[ManifestObject]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [
        ManifestObject(
            source=entry["source"],
            raw_path=entry["path"],
            sha256=entry["sha256"],
            bytes=entry["bytes"],
            mandatory=entry["mandatory"],
        )
        for entry in manifest["objects"]
    ]


def _is_safe_member(name: str) -> bool:
    if name.startswith("/") or name.startswith("\\"):
        return False

    parts = name.replace("\\", "/").split("/")
    return ".." not in parts


def fetch_and_verify(settings: Settings, obj: ManifestObject) -> tuple[str, int, int]:
    """Download one Bronze object, hash it, and safely count archive members.

    Returns (sha256_hex, byte_length, member_count). Raises ValueError on any
    mismatch against the manifest so a broken run fails loudly.
    """
    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, obj.bronze_key)
    try:
        payload = response.read()
    finally:
        response.close()
        response.release_conn()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != obj.sha256:
        raise ValueError(
            f"{obj.bronze_key}: sha256 mismatch "
            f"(expected {obj.sha256}, got {digest})"
        )
    if len(payload) != obj.bytes:
        raise ValueError(
            f"{obj.bronze_key}: size mismatch "
            f"(expected {obj.bytes}, got {len(payload)})"
        )

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        unsafe = [name for name in names if not _is_safe_member(name)]
        if unsafe:
            raise ValueError(f"{obj.bronze_key}: unsafe archive member paths: {unsafe}")
        member_count = len(names)

    return digest, len(payload), member_count


def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    manifest = load_manifest()

    result = StageResult(stage="register_sources", run_id=run_id)
    result.input_count = len(manifest)

    with psycopg.connect(settings.postgres_dsn) as connection:
        connection.execute(DDL)
        for obj in manifest:
            digest, size, member_count = fetch_and_verify(settings, obj)
            connection.execute(
                UPSERT,
                {
                    "source": obj.source,
                    "bronze_key": obj.bronze_key,
                    "sha256": digest,
                    "bytes": size,
                    "member_count": member_count,
                    "verified": True,
                    "run_id": run_id,
                },
            )
            result.output_count += 1
        connection.commit()

    result.finish()
    return result


if __name__ == "__main__":
    import uuid

    outcome = run(run_id=str(uuid.uuid4()))
    print(f"Registered {outcome.output_count}/{outcome.input_count} sources.")

