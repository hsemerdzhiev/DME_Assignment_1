"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .stages import build_ml_tables, load_postgres, prepare_data, register_sources, train


console = Console()


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


def command_run(_: Settings) -> int:
    """Part I: register Bronze, build Silver, load Gold. The ML export follows."""
    import uuid

    run_id = str(uuid.uuid4())
    table = Table(title=f"Part I run {run_id}")
    for column in ("stage", "read", "accepted", "issues", "seconds"):
        table.add_column(column, justify="right" if column != "stage" else "left")
    import json
    from pathlib import Path
    from datetime import UTC, datetime
    from .connections import minio_client
    counts, stages = {}, []
    result = register_sources.run(run_id)
    stages.append(result)
    counts['bronze.verified_objects'] = result.output_count
    result, stage_counts = prepare_data.prepare(run_id)
    stages.append(result)
    counts.update(stage_counts)
    result, stage_counts = load_postgres.load(run_id)
    stages.append(result)
    counts.update(stage_counts)
    counts['gold.gold_load'] = 1
    result, stage_counts = build_ml_tables.export(run_id)
    stages.append(result)
    counts.update(stage_counts)
    part1 = prepare_data.RESULTS_ROOT / 'part1'
    (part1 / 'row_counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    settings = Settings.from_environment()
    manifest = register_sources.load_manifest(minio_client(settings), settings.s3_bucket)
    record = {'run_id': run_id, 'code_revision': train.code_revision(),
              'data_release': 'quantum-data-core', 'bundle_version': 3,
              'input_hashes': {entry.relative_key: entry.sha256 for entry in manifest},
              'started_at': stages[0].started_at.isoformat(), 'finished_at': datetime.now(UTC).isoformat(),
              'output_counts': counts}
    (part1 / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    for result in stages:
        seconds = (result.finished_at - result.started_at).total_seconds()
        table.add_row(result.stage, f"{result.input_count:,}", f"{result.output_count:,}", str(result.issue_count), f"{seconds:.1f}")
    console.print(table)
    return 0


def command_train(_: Settings) -> int:
    import uuid
    result = train.run(str(uuid.uuid4()))
    console.print(f"Task A: wrote {result.output_count:,} predictions under results/part2/")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train"),
        help="Action to perform",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "run": command_run,
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
