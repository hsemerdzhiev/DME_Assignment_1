"""Parse the QASMBench QEC circuits into circuits, parity checks, and corrections.

A deliberately small OpenQASM 2.0 reader: registers, standard ``qelib1``
gates, user gate definitions (expanded only where they are *called*), register
broadcasts (``measure a -> syn`` becomes one measurement per index), barriers,
resets, and ``if(creg==value) op`` conditionals. Nothing is simulated.

A parity/stabilizer check is recognised structurally: a qubit that, between
the start of the circuit (or its previous measurement/reset) and its own
measurement, is touched only as the *target* of CX gates from two or more
distinct control qubits. Those controls are the data qubits and the measured
classical bit is the syndrome bit. Circuits that spread parity through
Hadamards or measure data qubits directly yield no checks, which is the
honest result for the encoder and five-qubit-code benchmarks.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field

from quantum_lake_student.silver.common import BronzeObject, IssueLog, TraceRows, record_id

SOURCE = "qasmbench"
CIRCUIT_TABLE, CHECK_TABLE, CORRECTION_TABLE = "circuit", "stabilizer_check", "conditional_correction"

# Qubit arity of every gate in qelib1.inc (parameters ignored).
QELIB_ARITY = {
    **dict.fromkeys(["u3", "u2", "u1", "u0", "u", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "sx", "sxdg", "p"], 1),
    **dict.fromkeys(["cx", "cz", "cy", "ch", "swap", "crx", "cry", "crz", "cu1", "cu3", "cu", "rxx", "rzz", "cp"], 2),
    **dict.fromkeys(["ccx", "cswap", "rccx"], 3),
    "rc3x": 4, "c3x": 4, "c3sqrtx": 4, "c4x": 5,
}
GATE_DEF = re.compile(r"gate\s+(?P<name>\w+)\s*(?:\((?P<params>[^)]*)\))?\s*(?P<args>[^{]*?)\s*\{(?P<body>[^}]*)\}", re.S)
REG_DECL = re.compile(r"^(?P<kind>qreg|creg)\s+(?P<name>\w+)\s*\[\s*(?P<size>\d+)\s*\]$")
IF_STMT = re.compile(r"^if\s*\(\s*(?P<reg>\w+)\s*==\s*(?P<value>\d+)\s*\)\s*(?P<op>.+)$", re.S)
APPLY = re.compile(r"^(?P<name>\w+)\s*(?:\((?P<params>[^)]*)\))?\s*(?P<args>[^()]*)$", re.S)
OPERAND = re.compile(r"^(?P<reg>\w+)(?:\[(?P<index>\d+)\])?$")


@dataclass(frozen=True)
class Op:
    """One executed operation after register broadcast and gate expansion."""

    line: int
    name: str            # gate name, "measure", "barrier", or "reset"
    qubits: tuple[str, ...]
    cbits: tuple[str, ...] = ()
    condition: tuple[str, int] | None = None


@dataclass
class Circuit:
    member: str
    registers: list[tuple[str, str, int]] = field(default_factory=list)  # (kind, name, size)
    ops: list[Op] = field(default_factory=list)

    @property
    def qubit_count(self) -> int:
        return sum(size for kind, _, size in self.registers if kind == "qreg")

    @property
    def register_declarations(self) -> str:
        return ";".join(f"{kind} {name}[{size}]" for kind, name, size in self.registers)


class QasmError(ValueError):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


def parse_qasm(text: str, member: str = "<text>") -> Circuit:
    """Parse one OpenQASM 2.0 program into declared registers and executed ops."""
    lines = [line.split("//", 1)[0] for line in text.splitlines()]
    source = "\n".join(lines)

    gates: dict[str, tuple[list[str], list[tuple[int, str]]]] = {}
    for match in GATE_DEF.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        formals = [arg.strip() for arg in match["args"].split(",") if arg.strip()]
        body = [(line + match["body"].count("\n", 0, offset), stmt) for offset, stmt in _statements(match["body"])]
        gates[match["name"]] = (formals, body)
    source = GATE_DEF.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), source)  # keep line numbers intact

    circuit = Circuit(member=member)
    sizes: dict[str, tuple[str, int]] = {}
    saw_header = False
    for offset, stmt in _statements(source):
        line = source.count("\n", 0, offset) + 1
        if stmt.startswith("OPENQASM"):
            saw_header = True
            continue
        if stmt.startswith("include"):
            continue
        if decl := REG_DECL.match(stmt):
            circuit.registers.append((decl["kind"], decl["name"], int(decl["size"])))
            sizes[decl["name"]] = (decl["kind"], int(decl["size"]))
            continue
        condition = None
        if cond := IF_STMT.match(stmt):
            condition = (cond["reg"], int(cond["value"]))
            if cond["reg"] not in sizes or sizes[cond["reg"]][0] != "creg":
                raise QasmError(line, f"condition register {cond['reg']!r} is not a declared creg")
            stmt = cond["op"].strip()
        circuit.ops.extend(_expand(stmt, line, sizes, gates, condition, {}))
    if not saw_header:
        raise QasmError(1, "missing OPENQASM header")
    if not any(kind == "qreg" for kind, _, _ in circuit.registers):
        raise QasmError(1, "no quantum register declared")
    return circuit


def _statements(text: str) -> list[tuple[int, str]]:
    """Return (offset, statement) pairs; offset points at the statement's first character."""
    result, start = [], 0
    for index, char in enumerate(text):
        if char == ";":
            raw = text[start:index]
            stripped = raw.strip()
            if stripped:
                result.append((start + (len(raw) - len(raw.lstrip())), stripped))
            start = index + 1
    if text[start:].strip():
        raise QasmError(text.count("\n") + 1, f"statement without terminating ';': {text[start:].strip()[:40]!r}")
    return result


def _expand(stmt: str, line: int, sizes: dict[str, tuple[str, int]], gates: dict, condition, bound: dict[str, str]) -> list[Op]:
    if stmt.startswith("measure"):
        parts = stmt[len("measure"):].split("->")
        if len(parts) != 2:
            raise QasmError(line, "measure needs 'qubit -> bit'")
        return [Op(line, "measure", (q,), (c,), condition) for q, c in _broadcast([parts[0].strip(), parts[1].strip()], line, sizes, bound)]
    if stmt.startswith(("barrier", "reset")):
        name, _, args = stmt.partition(" ")
        return [Op(line, name, tuple(qs), (), condition) for qs in _broadcast(_split_args(args), line, sizes, bound)] if name == "reset" \
            else [Op(line, "barrier", tuple(q for group in _broadcast(_split_args(args), line, sizes, bound) for q in group))]
    match = APPLY.match(stmt)
    if not match:
        raise QasmError(line, f"unrecognised statement {stmt!r}")
    name, args = match["name"], _split_args(match["args"])
    if name in gates:
        formals, body = gates[name]
        if len(formals) != len(args):
            raise QasmError(line, f"gate {name} expects {len(formals)} qubit(s), got {len(args)}")
        ops: list[Op] = []
        for actuals in _broadcast(args, line, sizes, bound):
            binding = dict(zip(formals, actuals))
            for _, body_stmt in body:
                ops.extend(Op(line, op.name, op.qubits, op.cbits, condition) for op in _expand(body_stmt, line, sizes, gates, None, binding))
        return ops
    if name not in QELIB_ARITY:
        raise QasmError(line, f"unknown gate {name!r}")
    if len(args) != QELIB_ARITY[name]:
        raise QasmError(line, f"gate {name} expects {QELIB_ARITY[name]} qubit(s), got {len(args)}")
    return [Op(line, name, tuple(qs), (), condition) for qs in _broadcast(args, line, sizes, bound)]


def _split_args(text: str) -> list[str]:
    return [arg.strip() for arg in text.split(",") if arg.strip()]


def _broadcast(args: list[str], line: int, sizes: dict[str, tuple[str, int]], bound: dict[str, str]) -> list[tuple[str, ...]]:
    """Expand whole-register operands to one operand tuple per index."""
    resolved: list[list[str]] = []
    for arg in args:
        if arg in bound:  # formal parameter inside a gate body
            resolved.append([bound[arg]])
            continue
        match = OPERAND.match(arg)
        if not match or match["reg"] not in sizes:
            raise QasmError(line, f"unknown operand {arg!r}")
        kind, size = sizes[match["reg"]]
        if match["index"] is None:
            resolved.append([f"{match['reg']}[{i}]" for i in range(size)])
        elif int(match["index"]) >= size:
            raise QasmError(line, f"index {arg} is out of range for {kind} of size {size}")
        else:
            resolved.append([arg.replace(" ", "")])
    widths = {len(values) for values in resolved if len(values) > 1}
    if len(widths) > 1:
        raise QasmError(line, "broadcast registers have different sizes")
    width = widths.pop() if widths else 1
    return [tuple(values[i] if len(values) > 1 else values[0] for values in resolved) for i in range(width)]


def find_parity_checks(circuit: Circuit) -> list[dict]:
    """Return {ancilla, data_qubits, syndrome_bit, lines} for every structural parity check."""
    checks = []
    for index, op in enumerate(circuit.ops):
        if op.name != "measure":
            continue
        ancilla, cbit = op.qubits[0], op.cbits[0]
        controls: list[str] = []
        lines = {op.line}
        pure = True
        for earlier in reversed(circuit.ops[:index]):
            if ancilla not in earlier.qubits or earlier.name == "barrier":
                continue
            if earlier.name in ("measure", "reset"):
                break
            if earlier.name == "cx" and earlier.qubits[1] == ancilla and earlier.condition is None:
                controls.append(earlier.qubits[0])
                lines.add(earlier.line)
            else:
                pure = False
                break
        data = list(dict.fromkeys(reversed(controls)))
        if pure and len(data) >= 2:
            checks.append({"ancilla": ancilla, "data_qubits": data, "syndrome_bit": cbit, "lines": sorted(lines)})
    return checks


@dataclass
class QasmResult:
    circuits: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    trace: TraceRows = field(default_factory=TraceRows)
    circuits_read: int = 0


def prepare(obj: BronzeObject, issues: IssueLog) -> QasmResult:
    result = QasmResult()
    with zipfile.ZipFile(io.BytesIO(obj.payload)) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".qasm"))
        if not members:
            raise issues.stop("qasm_no_members", SOURCE, None, None, None, "archive holds no .qasm files")
        benchmarks = {m.rsplit("/", 1)[0] for m in members}
        for directory in sorted(benchmarks):
            name = directory.rsplit("/", 1)[-1]
            expected = {f"{directory}/{name}.qasm", f"{directory}/{name}_transpiled.qasm"}
            if missing := sorted(expected - set(members)):
                raise issues.stop("qasm_required_companion_files", SOURCE, directory, None, missing,
                                  "source and transpiled variants are both required; the run must stop")
        for member in members:
            result.circuits_read += 1
            _prepare_member(obj, member, archive.read(member).decode("utf-8"), issues, result)
    return result


def _prepare_member(obj: BronzeObject, member: str, text: str, issues: IssueLog, result: QasmResult) -> None:
    filename = member.rsplit("/", 1)[-1]
    benchmark = member.rsplit("/", 2)[-2]
    variant = "transpiled" if filename.endswith("_transpiled.qasm") else "source"
    try:
        circuit = parse_qasm(text, member)
    except QasmError as error:
        issues.reject("qasm_structure", SOURCE, member, f"line {error.line}", None, str(error))
        return

    circuit_id = f"{benchmark}/{variant}"
    circuit_record_id = record_id(SOURCE, member)
    executed = [op for op in circuit.ops if op.name != "barrier"]
    result.circuits.append({
        "source_record_id": circuit_record_id, "circuit_id": circuit_id, "benchmark_name": benchmark, "variant": variant,
        "register_declarations": circuit.register_declarations, "qubit_count": circuit.qubit_count,
        "measurement_count": sum(op.name == "measure" for op in executed),
        "two_qubit_gate_count": sum(op.name != "measure" and len(op.qubits) == 2 for op in executed),
    })
    result.trace.add(circuit_record_id, obj, member, "file", CIRCUIT_TABLE)

    for check in find_parity_checks(circuit):
        locator = f"lines {check['lines'][0]}-{check['lines'][-1]}"
        check_record_id = record_id(SOURCE, member, "check", check["ancilla"], check["syndrome_bit"], *check["lines"])
        result.checks.append({
            "source_record_id": check_record_id, "circuit_id": circuit_id,
            "check_id": f"{check['ancilla']}->{check['syndrome_bit']}", "ancilla_qubit": check["ancilla"],
            "data_qubits": check["data_qubits"], "syndrome_bit": check["syndrome_bit"],
        })
        result.trace.add(check_record_id, obj, member, locator, CHECK_TABLE)

    for position, op in enumerate(op for op in executed if op.condition is not None):
        if op.name == "measure" or len(op.qubits) != 1:
            issues.warning("qasm_conditional_not_single_qubit_gate", SOURCE, member, f"line {op.line}", op.name,
                           "conditional operation is not a single-qubit recovery gate; not recorded as a correction")
            continue
        correction_record_id = record_id(SOURCE, member, "if", op.line, position)
        result.corrections.append({
            "source_record_id": correction_record_id, "circuit_id": circuit_id,
            "condition_register": op.condition[0], "condition_value": op.condition[1],
            "gate": op.name, "target_qubit": op.qubits[0],
        })
        result.trace.add(correction_record_id, obj, member, f"line {op.line}", CORRECTION_TABLE)

    if not any(op.name == "measure" for op in executed):
        issues.warning("qasm_no_measurement", SOURCE, member, None, None, "circuit executes no measurement")
