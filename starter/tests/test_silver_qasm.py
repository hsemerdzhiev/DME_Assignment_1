"""Unit tests for the OpenQASM reader and the parity-check / correction extraction."""

from __future__ import annotations

import io
import zipfile

import pytest

from quantum_lake_student.silver import qasm
from quantum_lake_student.silver.common import BronzeObject, IssueLog, RunStop

QEC_SM = """// Repetition code syndrome measurement
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2
{
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
x q[0]; // error
barrier q;
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;
"""


def test_gate_bodies_are_definitions_and_calls_are_expanded() -> None:
    circuit = qasm.parse_qasm(QEC_SM)
    assert circuit.register_declarations == "qreg q[3];qreg a[2];creg c[3];creg syn[2]"
    assert circuit.qubit_count == 5
    names = [op.name for op in circuit.ops]
    assert names.count("cx") == 4                      # the syndrome body executed once
    assert names.count("measure") == 5                 # 'measure a -> syn' (2) + 'measure q -> c' (3)
    assert names.count("barrier") == 1
    conditional = [op for op in circuit.ops if op.condition]
    assert [(op.condition, op.qubits[0]) for op in conditional] == [(("syn", 1), "q[0]"), (("syn", 2), "q[2]"), (("syn", 3), "q[1]")]
    assert all(op.line == 15 for op in circuit.ops if op.name == "cx")   # expanded ops keep the call-site line


def test_register_broadcast_expands_per_index_and_keeps_register_names() -> None:
    circuit = qasm.parse_qasm("OPENQASM 2.0;\nqreg q[2];\nqreg a[2];\ncreg c[2];\ncx q,a;\nmeasure a -> c;\n")
    assert [op.qubits for op in circuit.ops if op.name == "cx"] == [("q[0]", "a[0]"), ("q[1]", "a[1]")]
    assert [(op.qubits[0], op.cbits[0]) for op in circuit.ops if op.name == "measure"] == [("a[0]", "c[0]"), ("a[1]", "c[1]")]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n", "unknown gate"),
        ("OPENQASM 2.0;\nqreg q[1];\nx q[3];\n", "out of range"),
        ("OPENQASM 2.0;\nqreg q[1];\ncx q[0];\n", "expects 2"),
        ("qreg q[1];\nx q[0];\n", "OPENQASM header"),
        ("OPENQASM 2.0;\nqreg q[1];\nx q[0]\n", "terminating"),
        ("OPENQASM 2.0;\nqreg q[2];\nqreg a[3];\ncx q,a;\n", "different sizes"),
    ],
)
def test_structural_errors_are_reported_with_a_line(text: str, fragment: str) -> None:
    with pytest.raises(qasm.QasmError, match=fragment):
        qasm.parse_qasm(text)


def test_parity_checks_are_cx_targets_measured_without_other_gates() -> None:
    checks = qasm.find_parity_checks(qasm.parse_qasm(QEC_SM))
    assert [(c["ancilla"], c["data_qubits"], c["syndrome_bit"]) for c in checks] == [
        ("a[0]", ["q[0]", "q[1]"], "syn[0]"),
        ("a[1]", ["q[1]", "q[2]"], "syn[1]"),
    ]
    assert checks[0]["lines"] == [15, 16]


def test_hadamard_on_the_measured_qubit_is_not_a_parity_check() -> None:
    text = "OPENQASM 2.0;\nqreg q[3];\ncreg c[1];\ncx q[0],q[2];\nh q[2];\ncx q[1],q[2];\nmeasure q[2] -> c[0];\n"
    assert qasm.find_parity_checks(qasm.parse_qasm(text)) == []


def _bronze(files: dict[str, str]) -> BronzeObject:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return BronzeObject("qasmbench", "bronze/source=qasmbench/t.zip", "2" * 64, buffer.getvalue())


def test_prepare_builds_all_three_tables_for_both_variants() -> None:
    issues = IssueLog()
    result = qasm.prepare(_bronze({"small/qec_sm_n5/qec_sm_n5.qasm": QEC_SM, "small/qec_sm_n5/qec_sm_n5_transpiled.qasm": QEC_SM}), issues)
    assert result.circuits_read == 2
    assert [(c["circuit_id"], c["variant"], c["measurement_count"], c["two_qubit_gate_count"]) for c in result.circuits] == [
        ("qec_sm_n5/source", "source", 5, 4), ("qec_sm_n5/transpiled", "transpiled", 5, 4)]
    assert len(result.checks) == 4 and len(result.corrections) == 6
    assert {c["check_id"] for c in result.checks} == {"a[0]->syn[0]", "a[1]->syn[1]"}
    assert [(c["condition_value"], c["gate"], c["target_qubit"]) for c in result.corrections[:3]] == [(1, "x", "q[0]"), (2, "x", "q[2]"), (3, "x", "q[1]")]
    assert len({r for r in result.trace.source_record_id}) == 2 + 4 + 6
    assert not issues.issues


def test_unparseable_circuit_is_rejected_not_fatal() -> None:
    issues = IssueLog()
    result = qasm.prepare(_bronze({"small/b/b.qasm": "OPENQASM 2.0;\nqreg q[1];\nzz q[0];\n", "small/b/b_transpiled.qasm": QEC_SM}), issues)
    assert len(result.circuits) == 1
    [issue] = issues.issues
    assert issue.rule_id == "qasm_structure" and issue.record_locator == "line 3"


def test_missing_transpiled_variant_stops_the_run() -> None:
    with pytest.raises(RunStop):
        qasm.prepare(_bronze({"small/qec_sm_n5/qec_sm_n5.qasm": QEC_SM}), IssueLog())
