"""Parse and check the simulated surface-code syndrome CSV files.

One Silver row is one original aggregate CSV row: a 4-round by 4-check
syndrome pattern, its logical-error label, and the number of simulated shots
(``quantity``) it stands for. Rows are never expanded.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import zipfile
from dataclasses import dataclass, field

from quantum_lake_student.silver.common import BronzeObject, IssueLog, TraceRows, record_id

SOURCE = "qec_syndromes"
TABLE = "syndrome_observation"
EXPECTED_HEADER = ["labels", "syndromes", "quantity"]
DOCUMENTED_HEADER = ["label", "syndromes", "quantity"]
ROUNDS, CHECKS = 4, 4
FILENAME = re.compile(r"^d-(?P<distance>\d+)_pfr-(?P<pfr>[0-9.]+)_nb-(?P<nb>\d+)(?P<unit>[KM]?)\.csv$")
SCALE = {"": 1, "K": 1_000, "M": 1_000_000}


@dataclass
class SyndromeResult:
    rows: list[dict] = field(default_factory=list)
    trace: TraceRows = field(default_factory=TraceRows)
    rows_read: int = 0


def parse_syndrome(text: str) -> bytes:
    """Turn ``((0, 0, 0, 0), ...)`` into 16 one-byte values, round-major."""
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or len(value) != ROUNDS:
        raise ValueError(f"expected {ROUNDS} rounds")
    bits = bytearray()
    for round_values in value:
        if not isinstance(round_values, tuple) or len(round_values) != CHECKS:
            raise ValueError(f"expected {CHECKS} checks per round")
        for bit in round_values:
            if bit not in (0, 1) or isinstance(bit, bool):
                raise ValueError("syndrome values must be 0 or 1")
            bits.append(bit)
    return bytes(bits)


def parse_filename(name: str) -> tuple[int, float, int]:
    """Return (distance, physical fault rate, nominal sample count) from the file name."""
    match = FILENAME.match(name.rsplit("/", 1)[-1])
    if not match:
        raise ValueError(f"unrecognised syndrome file name {name!r}")
    nominal = int(match["nb"]) * SCALE[match["unit"]]
    return int(match["distance"]), float(match["pfr"]), nominal


def prepare(obj: BronzeObject, issues: IssueLog) -> SyndromeResult:
    result = SyndromeResult()
    with zipfile.ZipFile(io.BytesIO(obj.payload)) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        if not members:
            raise issues.stop("syn_no_csv_members", SOURCE, None, None, archive.namelist(), "archive holds no CSV files")
        for member in members:
            _prepare_member(obj, member, archive.read(member).decode("utf-8"), issues, result)
    return result


def _prepare_member(obj: BronzeObject, member: str, text: str, issues: IssueLog, result: SyndromeResult) -> None:
    try:
        distance, fault_rate, nominal = parse_filename(member)
    except ValueError as error:
        issues.reject("syn_filename_pattern", SOURCE, member, None, member, str(error))
        return
    experiment_id = f"d-{distance}_pfr-{fault_rate:.6f}"

    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    if header != EXPECTED_HEADER:
        issues.reject("syn_header_unexpected", SOURCE, member, "row 1", header, f"expected header {EXPECTED_HEADER}")
        return
    if header != DOCUMENTED_HEADER:
        issues.info(
            "syn_header_label_vs_labels", SOURCE, member, "row 1", header,
            "README documents column 'label'; the file uses 'labels'. Read as documented.",
        )

    accepted, quantity_total = 0, 0
    labels_by_syndrome: dict[bytes, set[bool]] = {}
    for line_number, fields in enumerate(reader, start=2):
        result.rows_read += 1
        locator = f"row {line_number}"
        if len(fields) != 3:
            issues.reject("syn_field_count", SOURCE, member, locator, fields, "expected exactly three fields")
            continue
        label_text, syndrome_text, quantity_text = fields
        if label_text not in ("0", "1"):
            issues.reject("syn_label_binary", SOURCE, member, locator, label_text, "label must be 0 or 1")
            continue
        try:
            bits = parse_syndrome(syndrome_text)
        except (ValueError, SyntaxError) as error:
            issues.reject("syn_shape_4x4_binary", SOURCE, member, locator, syndrome_text, str(error))
            continue
        if not quantity_text.isdigit() or int(quantity_text) <= 0:
            issues.reject("syn_quantity_positive", SOURCE, member, locator, quantity_text, "quantity must be a positive integer")
            continue

        label = label_text == "1"
        quantity = int(quantity_text)
        source_record_id = record_id(SOURCE, member, line_number)
        result.rows.append(
            {
                "source_record_id": source_record_id,
                "experiment_id": experiment_id,
                "physical_fault_rate": fault_rate,
                "syndrome_bits": bits,
                "round_count": ROUNDS,
                "check_count": CHECKS,
                "logical_error_label": label,
                "quantity": quantity,
            }
        )
        result.trace.add(source_record_id, obj, member, locator, TABLE)
        accepted += 1
        quantity_total += quantity
        labels_by_syndrome.setdefault(bits, set()).add(label)

    if quantity_total != nominal:
        issues.warning(
            "syn_quantity_reconciles_to_filename", SOURCE, member, None, quantity_total,
            f"accepted quantities sum to {quantity_total}, file name states {nominal}",
        )
    both = sum(len(labels) == 2 for labels in labels_by_syndrome.values())
    issues.info(
        "syn_syndrome_with_both_labels", SOURCE, member, None, both,
        f"{both} syndrome pattern(s) occur with both labels in {accepted} rows; valid, key is (syndrome, label)",
    )
