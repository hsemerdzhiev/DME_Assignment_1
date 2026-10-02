CREATE VIEW gold.ml_syndrome_source AS
SELECT md5(e.experiment_id || ':' || encode(p.syndrome_bits, 'hex') || ':' || o.logical_error_label::text) AS example_id,
       e.experiment_id, e.physical_fault_rate, p.syndrome_bits, p.round_count, p.check_count,
       o.logical_error_label, o.quantity AS sample_weight, o.observation_id AS gold_record_id
FROM gold.syndrome_observation o
JOIN gold.experiment e USING (experiment_id)
JOIN gold.syndrome_pattern p USING (pattern_id);

CREATE VIEW gold.ml_google_source AS
SELECT md5(s.experiment_id || ':' || s.shot_index::text) AS example_id,
       s.experiment_id, s.shot_index, e.distance, e.rounds, e.center_row, e.center_col,
       e.detector_count, s.detector_event_count, s.detector_bits,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_id = 'belief_matching') AS belief_matching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_id = 'correlated_matching') AS correlated_matching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_id = 'pymatching') AS pymatching_prediction,
       bool_or(p.predicted_flip) FILTER (WHERE p.decoder_id = 'tensor_network_contraction') AS tensor_network_contraction_prediction,
       s.actual_observable_flip, s.shot_id AS gold_record_id
FROM gold.shot s JOIN gold.experiment e USING (experiment_id)
JOIN gold.decoder_prediction p USING (shot_id)
GROUP BY s.shot_id, e.experiment_id;
