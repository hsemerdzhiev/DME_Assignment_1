"""Turn the six Silver tables into Gold rows.

Pure functions over Arrow tables: no I/O, so each shape decision is testable
on a handful of rows. Every returned list is ordered deterministically so a
rerun yields identical rows in identical order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
import pyarrow as pa

from quantum_lake_student.silver.common import record_id

DECODERS = [
    ("belief_matching", "belief propagation combined with minimum-weight perfect matching",
     "pij_from_even_for_odd.dem for odd shots, pij_from_odd_for_even.dem for even shots"),
    ("correlated_matching", "minimum-weight perfect matching with X/Z correlation reweighting",
     "circuit_detector_error_model.dem for all shots"),
    ("pymatching", "PyMatching minimum-weight perfect matching",
     "circuit_detector_error_model.dem for all shots"),
    ("tensor_network_contraction", "tensor-network approximation of maximum-likelihood decoding",
     "pij_from_even_for_odd.dem for odd shots, pij_from_odd_for_even.dem for even shots"),
]
PREDICTION_COLUMNS = {f"{decoder_id}_prediction": decoder_id for decoder_id, _, _ in DECODERS}
REGISTER = re.compile(r"^(qreg|creg) (\w+)\[(\d+)\]$")


def _rows(table: pa.Table) -> Iterator[dict]:
    for batch in table.to_batches():
        yield from batch.to_pylist()


# --- experiments (both sources) ------------------------------------------------


def experiments(syndromes: pa.Table, google: pa.Table) -> list[tuple]:
    """One row per simulated fault-rate sweep and per hardware experiment."""
    rows = []
    seen: dict[str, float] = {}
    for row in _rows(syndromes.select(["experiment_id", "physical_fault_rate", "round_count", "check_count"])):
        if row["experiment_id"] not in seen:
            seen[row["experiment_id"]] = row["physical_fault_rate"]
            rows.append((row["experiment_id"], "qec_syndromes", "simulated_syndrome", "surface_code", 3,
                         row["round_count"], row["check_count"], row["physical_fault_rate"],
                         None, None, None, None, None, None, None, None))
    for row in _rows(google):
        rows.append((row["experiment_id"], "google_qec", "hardware_memory", "surface_code", row["distance"],
                     row["rounds"], row["distance"] ** 2 - 1, None, row["basis"], row["shots"],
                     row["center_row"], row["center_col"], row["measurement_count"],
                     (row["measurement_count"] - row["rounds"] * (row["distance"] ** 2 - 1)),  # final data-qubit readout = sweep width
                     row["detector_count"], row["source_record_id"]))
    return sorted(rows)


EXPERIMENT_COLUMNS = ("experiment_id", "source_name", "kind", "code_family", "distance", "rounds", "check_count",
                      "physical_fault_rate", "basis", "shots", "center_row", "center_col", "measurement_count",
                      "sweep_bit_count", "detector_count", "source_record_id")


# --- syndromes -----------------------------------------------------------------


def pattern_id(bits: bytes) -> str:
    return record_id("syndrome_pattern", bits.hex())


def syndrome_patterns(syndromes: pa.Table) -> list[tuple]:
    """Distinct 16-value patterns across every fault rate."""
    distinct = {bits.as_py() for bits in syndromes["syndrome_bits"]}
    return sorted((pattern_id(bits), bits, 4, 4, sum(bits)) for bits in distinct)


def syndrome_observations(syndromes: pa.Table) -> list[tuple]:
    return sorted(
        (row["source_record_id"], row["experiment_id"], pattern_id(row["syndrome_bits"]),
         row["logical_error_label"], row["quantity"])
        for row in _rows(syndromes)
    )


# --- Google --------------------------------------------------------------------


def decoders() -> list[tuple]:
    return list(DECODERS)


def shots(shot_table: pa.Table) -> list[tuple]:
    columns = ["source_record_id", "experiment_id", "shot_index", "measurement_bits", "sweep_bits",
               "detector_bits", "detector_event_count", "actual_observable_flip"]
    return [tuple(row[c] for c in columns) for row in _rows(shot_table.select(columns))]


def decoder_predictions(shot_table: pa.Table) -> list[tuple]:
    """Unpivot the four prediction columns into (shot, decoder, prediction) rows."""
    rows = []
    for row in _rows(shot_table.select(["source_record_id", *PREDICTION_COLUMNS])):
        rows.extend((row["source_record_id"], decoder_id, row[column]) for column, decoder_id in PREDICTION_COLUMNS.items())
    return rows


def detector_summaries(shot_table: pa.Table, google: pa.Table) -> list[tuple]:
    """Per-experiment fire count of every detector position, from the packed bits."""
    widths = {row["experiment_id"]: row["detector_count"] for row in _rows(google)}
    rows = []
    for experiment_id, width in sorted(widths.items()):
        subset = shot_table.filter(pa.compute.equal(shot_table["experiment_id"], experiment_id))
        if subset.num_rows == 0:
            continue
        packed = np.frombuffer(b"".join(subset["detector_bits"].to_pylist()), dtype=np.uint8)
        matrix = np.unpackbits(packed.reshape(subset.num_rows, -1), axis=1, bitorder="little")[:, :width]
        rows.extend((experiment_id, index, int(count)) for index, count in enumerate(matrix.sum(axis=0)))
    return rows


# --- QASMBench -----------------------------------------------------------------


def circuits(circuit_table: pa.Table) -> list[tuple]:
    columns = ["circuit_id", "benchmark_name", "variant", "qubit_count", "measurement_count",
               "two_qubit_gate_count", "source_record_id"]
    return sorted(tuple(row[c] for c in columns) for row in _rows(circuit_table.select(columns)))


def circuit_registers(circuit_table: pa.Table) -> list[tuple]:
    """Split the serialized ``qreg q[3];creg c[3]`` declaration into rows."""
    rows = []
    for row in _rows(circuit_table.select(["circuit_id", "register_declarations"])):
        for position, declaration in enumerate(row["register_declarations"].split(";")):
            match = REGISTER.match(declaration)
            if not match:
                raise ValueError(f"{row['circuit_id']}: cannot parse register {declaration!r}")
            kind, name, size = match.groups()
            rows.append((row["circuit_id"], name, kind, int(size), position))
    return rows


def stabilizer_checks(check_table: pa.Table) -> list[tuple]:
    return sorted((row["source_record_id"], row["circuit_id"], row["check_id"], row["ancilla_qubit"], row["syndrome_bit"])
                  for row in _rows(check_table))


def stabilizer_check_qubits(check_table: pa.Table) -> list[tuple]:
    rows = []
    for row in _rows(check_table.select(["source_record_id", "data_qubits"])):
        rows.extend((row["source_record_id"], qubit, position) for position, qubit in enumerate(row["data_qubits"]))
    return rows


def conditional_corrections(correction_table: pa.Table) -> list[tuple]:
    columns = ["source_record_id", "circuit_id", "condition_register", "condition_value", "gate", "target_qubit"]
    return sorted(tuple(row[c] for c in columns) for row in _rows(correction_table.select(columns)))
