SELECT c.benchmark_name, c.variant, k.check_label, q.data_qubit, k.ancilla_qubit,
       k.syndrome_bit, r.condition_register, r.condition_value, r.gate, r.target_qubit
FROM gold.circuit c JOIN gold.stabilizer_check k USING (circuit_id)
JOIN gold.stabilizer_check_qubit q USING (check_id)
LEFT JOIN gold.conditional_correction r ON r.circuit_id = c.circuit_id
 AND r.condition_register = split_part(k.syndrome_bit, '[', 1)
ORDER BY c.benchmark_name, c.variant, k.check_label, q.position, r.condition_value;
