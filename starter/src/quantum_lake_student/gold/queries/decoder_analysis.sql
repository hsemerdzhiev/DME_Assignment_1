SELECT e.distance, e.center_row, e.center_col, d.decoder_id,
       count(*) AS shots, avg(d.decoder_error::int) AS logical_error_rate
FROM gold.decoder_outcome d JOIN gold.experiment e USING (experiment_id)
GROUP BY e.distance, e.center_row, e.center_col, d.decoder_id
ORDER BY e.distance, e.center_row, e.center_col, d.decoder_id;
