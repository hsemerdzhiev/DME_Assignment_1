SELECT e.physical_fault_rate, encode(p.syndrome_bits, 'hex') AS syndrome,
       sum(o.quantity) AS observations,
       sum(o.quantity::numeric * o.logical_error_label::int) / sum(o.quantity) AS logical_error_rate,
       sum(o.quantity)::numeric / sum(sum(o.quantity)) OVER (PARTITION BY e.experiment_id) AS weighted_frequency
FROM gold.syndrome_observation o JOIN gold.experiment e USING (experiment_id)
JOIN gold.syndrome_pattern p USING (pattern_id)
GROUP BY e.experiment_id, e.physical_fault_rate, p.syndrome_bits
ORDER BY e.physical_fault_rate, syndrome;
