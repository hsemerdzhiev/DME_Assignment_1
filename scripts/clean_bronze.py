#!/usr/bin/env python3
"""Remove everything under bronze/ that the release manifest does not list.

Only ./starter is mounted into the workspace container, so pipe this file in
from the repo root:

    docker compose exec -T workspace sh -c 'cd /workspace && python -' < scripts/clean_bronze.py

Bronze must hold only the supplied bytes. Extracted archive members belong in
Silver (or a scratch area), never next to the source zips.
"""

from __future__ import annotations

import sys

from minio.deleteobjects import DeleteObject

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.stages.register_sources import BRONZE_PREFIX, load_manifest


def main() -> int:
    settings = Settings.from_environment()
    client = minio_client(settings)
    keep = {
        BRONZE_PREFIX + obj.relative_key
        for obj in load_manifest(client, settings.s3_bucket)
    }
    stray = sorted(
        item.object_name
        for item in client.list_objects(settings.s3_bucket, prefix="bronze/", recursive=True)
        if item.object_name not in keep
    )
    print("keeping:")
    for key in sorted(keep):
        print("  ", key)
    if not stray:
        print("nothing to remove; Bronze already matches the manifest")
        return 0
    print(f"removing {len(stray)} object(s) not in the manifest")
    errors = list(client.remove_objects(settings.s3_bucket, [DeleteObject(k) for k in stray]))
    for error in errors:
        print("ERROR", error)
    remaining = [
        item.object_name
        for item in client.list_objects(settings.s3_bucket, prefix="bronze/", recursive=True)
    ]
    print(f"remaining under bronze/: {len(remaining)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
