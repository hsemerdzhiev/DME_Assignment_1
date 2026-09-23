"""Parse and check the Google surface-code hardware experiments.

One experiment row is one experiment directory. One shot row is one hardware
shot assembled from the aligned companion files: packed measurements, sweep
bits, packed detector events, the actual logical flip, and four decoder
predictions. Measurements, detector events, actual outcomes, and predictions
stay separate columns; a decoder mistake is derived later, never stored here.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

import yaml

from quantum_lake_student.formats import b8_record_bytes
from quantum_lake_student.silver.common import BronzeObject, IssueLog, TraceRows, record_id

SOURCE = "google_qec"
EXPERIMENT_TABLE, SHOT_TABLE = "experiment", "shot"
DIRNAME = re.compile(r"^(?P<code>[a-z_]+)_b(?P<basis>[XZ])_d(?P<distance>\d+)_r(?P<rounds>\d+)_center_(?P<row>\d+)_(?P<col>\d+)$")

PREDICTIONS = {
    "obs_flips_predicted_by_belief_matching.01": "belief_matching_prediction",
    "obs_flips_predicted_by_correlated_matching.01": "correlated_matching_prediction",
    "obs_flips_predicted_by_pymatching.01": "pymatching_prediction",
    "obs_flips_predicted_by_tensor_network_contraction.01": "tensor_network_contraction_prediction",
}
B8_FILES = {"measurements.b8": "circuit_measurements", "sweep.b8": "circuit_sweep_bits", "detection_events.b8": "circuit_detectors"}
REQUIRED = {"properties.yml", "obs_flips_actual.01", *B8_FILES, *PREDICTIONS}
PROPERTY_KEYS = ("basis", "rounds", "distance", "shots", "center_data_qubit_row", "center_data_qubit_col",
                 "circuit_measurements", "circuit_sweep_bits", "circuit_detectors")


@dataclass
class GoogleResult:
    experiments: list[dict] = field(default_factory=list)
    shots: list[dict] = field(default_factory=list)
    trace: TraceRows = field(default_factory=TraceRows)
    experiments_read: int = 0
    shots_read: int = 0


def prepare(obj: BronzeObject, issues: IssueLog) -> GoogleResult:
    result = GoogleResult()
    with zipfile.ZipFile(io.BytesIO(obj.payload)) as archive:
        names = set(archive.namelist())
        directories = sorted({name.split("/", 1)[0] for name in names if "/" in name and DIRNAME.match(name.split("/", 1)[0])})
        if not directories:
            raise issues.stop("goog_no_experiments", SOURCE, None, None, None, "archive holds no experiment directories")
        for directory in directories:
            result.experiments_read += 1
            missing = sorted(f for f in REQUIRED if f"{directory}/{f}" not in names)
            if missing:
                raise issues.stop("goog_required_companion_files", SOURCE, directory, None, missing,
                                  "required companion file(s) are missing; the run must stop")
            files = {f: archive.read(f"{directory}/{f}") for f in REQUIRED}
            _prepare_experiment(obj, directory, files, issues, result)
    return result


def _prepare_experiment(obj: BronzeObject, directory: str, files: dict[str, bytes], issues: IssueLog, result: GoogleResult) -> None:
    member = f"{directory}/properties.yml"
    props = yaml.safe_load(files["properties.yml"]) or {}
    if missing := [key for key in PROPERTY_KEYS if key not in props]:
        issues.reject("goog_properties_complete", SOURCE, member, None, missing, "properties.yml lacks required keys")
        return

    named = DIRNAME.match(directory).groupdict()
    declared = {"basis": props["basis"], "distance": props["distance"], "rounds": props["rounds"],
                "row": props["center_data_qubit_row"], "col": props["center_data_qubit_col"]}
    from_name = {"basis": named["basis"], "distance": int(named["distance"]), "rounds": int(named["rounds"]),
                 "row": int(named["row"]), "col": int(named["col"])}
    if declared != from_name:
        issues.reject("goog_dirname_matches_properties", SOURCE, member, None, {"name": from_name, "properties": declared},
                      "directory name and properties.yml disagree")
        return

    shots = int(props["shots"])
    widths = {name: int(props[key]) for name, key in B8_FILES.items()}
    ok = True
    for name, bits in widths.items():
        expected = shots * b8_record_bytes(bits)
        if len(files[name]) != expected:
            issues.reject("goog_b8_length", SOURCE, f"{directory}/{name}", None, len(files[name]),
                          f"expected {shots} shots x {b8_record_bytes(bits)} bytes = {expected} bytes for {bits} bits")
            ok = False
    lines: dict[str, list[bytes]] = {}
    for name in ("obs_flips_actual.01", *PREDICTIONS):
        values = files[name].splitlines()
        if len(values) != shots or any(v not in (b"0", b"1") for v in values):
            issues.reject("goog_01_rows_binary", SOURCE, f"{directory}/{name}", None, len(values),
                          f"expected {shots} lines of 0 or 1")
            ok = False
        lines[name] = values
    if not ok:
        issues.reject("goog_companion_alignment", SOURCE, directory, None, shots,
                      "companion files are not aligned; experiment and its shots are excluded")
        return

    experiment_id = directory
    experiment_record_id = record_id(SOURCE, member)
    result.experiments.append({
        "source_record_id": experiment_record_id, "experiment_id": experiment_id, "basis": props["basis"],
        "distance": int(props["distance"]), "rounds": int(props["rounds"]), "shots": shots,
        "center_row": int(props["center_data_qubit_row"]), "center_col": int(props["center_data_qubit_col"]),
        "measurement_count": widths["measurements.b8"], "detector_count": widths["detection_events.b8"],
    })
    result.trace.add(experiment_record_id, obj, member, "file", EXPERIMENT_TABLE)
    _prepare_shots(obj, directory, experiment_id, shots, widths, files, lines, issues, result)


def _prepare_shots(obj: BronzeObject, directory: str, experiment_id: str, shots: int, widths: dict[str, int],
                   files: dict[str, bytes], lines: dict[str, list[bytes]], issues: IssueLog, result: GoogleResult) -> None:
    sizes = {name: b8_record_bytes(bits) for name, bits in widths.items()}
    detector_bits = widths["detection_events.b8"]
    detector_mask = (1 << detector_bits) - 1
    padding: dict[str, int] = {name: 0 for name in B8_FILES}
    ids, locators = [], []

    for index in range(shots):
        result.shots_read += 1
        packed = {name: files[name][index * size:(index + 1) * size] for name, size in sizes.items()}
        for name, bits in widths.items():
            if int.from_bytes(packed[name], "little") >> bits:
                padding[name] += 1
        detectors = int.from_bytes(packed["detection_events.b8"], "little") & detector_mask
        shot_id = record_id(SOURCE, experiment_id, index)
        result.shots.append({
            "source_record_id": shot_id, "experiment_id": experiment_id, "shot_index": index,
            "measurement_bits": packed["measurements.b8"], "sweep_bits": packed["sweep.b8"],
            "detector_bits": packed["detection_events.b8"], "detector_event_count": detectors.bit_count(),
            "actual_observable_flip": lines["obs_flips_actual.01"][index] == b"1",
            **{column: lines[name][index] == b"1" for name, column in PREDICTIONS.items()},
        })
        ids.append(shot_id)
        locators.append(f"shot {index}")

    for name, count in padding.items():
        if count:
            issues.warning("goog_b8_padding_zero", SOURCE, f"{directory}/{name}", None, count,
                           f"{count} shot(s) carry non-zero padding bits beyond {widths[name]} bits; bytes kept unchanged")
    for name in (*B8_FILES, "obs_flips_actual.01", *PREDICTIONS):
        result.trace.add_many(ids, obj, f"{directory}/{name}", locators, SHOT_TABLE)
