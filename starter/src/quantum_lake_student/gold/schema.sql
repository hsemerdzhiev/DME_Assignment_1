-- Gold: relational QEC model. Executed with search_path set to the build
-- schema, which is renamed to "gold" once every table is loaded and checked.
-- One sentence per table states what one row represents.

-- One row: one experiment from either source. Simulated fault-rate sweeps and
-- hardware memory experiments share distance/rounds vocabulary but stay
-- distinct rows; nothing links a simulated row to a hardware row.
CREATE TABLE experiment (
    experiment_id        text PRIMARY KEY,
    source_name          text NOT NULL CHECK (source_name IN ('qec_syndromes', 'google_qec')),
    kind                 text NOT NULL CHECK (kind IN ('simulated_syndrome', 'hardware_memory')),
    code_family          text NOT NULL DEFAULT 'surface_code',
    distance             integer NOT NULL CHECK (distance >= 3 AND distance % 2 = 1),
    rounds               integer NOT NULL CHECK (rounds > 0),
    check_count          integer NOT NULL CHECK (check_count > 0),   -- stabilizer checks measured per round
    physical_fault_rate  double precision CHECK (physical_fault_rate > 0 AND physical_fault_rate < 1),
    basis                text CHECK (basis IN ('X', 'Z')),
    shots                bigint CHECK (shots > 0),
    center_row           integer,
    center_col           integer,
    measurement_count    integer CHECK (measurement_count > 0),
    sweep_bit_count      integer CHECK (sweep_bit_count >= 0),
    detector_count       integer CHECK (detector_count > 0),
    source_record_id     text,
    CHECK (kind <> 'simulated_syndrome' OR (physical_fault_rate IS NOT NULL AND shots IS NULL)),
    CHECK (kind <> 'hardware_memory' OR (basis IS NOT NULL AND shots IS NOT NULL AND center_row IS NOT NULL
           AND center_col IS NOT NULL AND measurement_count IS NOT NULL AND detector_count IS NOT NULL
           AND source_record_id IS NOT NULL))
);
CREATE UNIQUE INDEX experiment_simulated_rate ON experiment (physical_fault_rate) WHERE kind = 'simulated_syndrome';

-- One row: one distinct 4-round by 4-check syndrome pattern, shared by every
-- fault-rate experiment in which it occurs. Bits are round-major, one byte each.
CREATE TABLE syndrome_pattern (
    pattern_id     text PRIMARY KEY,
    syndrome_bits  bytea NOT NULL UNIQUE CHECK (octet_length(syndrome_bits) = 16),
    round_count    integer NOT NULL CHECK (round_count = 4),
    check_count    integer NOT NULL CHECK (check_count = 4),
    fired_checks   integer NOT NULL CHECK (fired_checks BETWEEN 0 AND 16)
);

-- One row: one aggregate simulated observation; quantity is the number of
-- physical shots it stands for. The same pattern may appear with both labels.
CREATE TABLE syndrome_observation (
    observation_id       text PRIMARY KEY,                 -- Silver source_record_id
    experiment_id        text NOT NULL REFERENCES experiment,
    pattern_id           text NOT NULL REFERENCES syndrome_pattern,
    logical_error_label  boolean NOT NULL,
    quantity             bigint NOT NULL CHECK (quantity > 0),
    UNIQUE (experiment_id, pattern_id, logical_error_label)
);
CREATE INDEX syndrome_observation_pattern ON syndrome_observation (pattern_id);

-- One row: one supplied decoder whose predictions are stored per shot.
CREATE TABLE decoder (
    decoder_id   text PRIMARY KEY,
    description  text NOT NULL,
    calibration  text NOT NULL
);

-- One row: one aligned hardware shot with its packed bit rows and actual outcome.
CREATE TABLE shot (
    shot_id                 text PRIMARY KEY,              -- Silver source_record_id
    experiment_id           text NOT NULL REFERENCES experiment,
    shot_index              bigint NOT NULL CHECK (shot_index >= 0),
    measurement_bits        bytea NOT NULL,
    sweep_bits              bytea NOT NULL,
    detector_bits           bytea NOT NULL,
    detector_event_count    integer NOT NULL CHECK (detector_event_count >= 0),
    actual_observable_flip  boolean NOT NULL,
    UNIQUE (experiment_id, shot_index)
);

-- One row: one decoder's prediction for one shot. A decoder error is
-- predicted_flip <> shot.actual_observable_flip and is derived, never stored.
CREATE TABLE decoder_prediction (
    shot_id         text NOT NULL REFERENCES shot,
    decoder_id      text NOT NULL REFERENCES decoder,
    predicted_flip  boolean NOT NULL,
    PRIMARY KEY (shot_id, decoder_id)
);
CREATE INDEX decoder_prediction_decoder ON decoder_prediction (decoder_id);

-- One row: how often one detector position fired across all shots of one
-- experiment. Chosen over a one-row-per-fired-detector event table; see
-- docs/gold-design.md for the size comparison.
CREATE TABLE detector_summary (
    experiment_id   text NOT NULL REFERENCES experiment,
    detector_index  integer NOT NULL CHECK (detector_index >= 0),
    fire_count      bigint NOT NULL CHECK (fire_count >= 0),
    PRIMARY KEY (experiment_id, detector_index)
);

-- One row: one parsed circuit variant (source or transpiled) of a benchmark.
CREATE TABLE circuit (
    circuit_id             text PRIMARY KEY,
    benchmark_name         text NOT NULL,
    variant                text NOT NULL CHECK (variant IN ('source', 'transpiled')),
    qubit_count            integer NOT NULL CHECK (qubit_count > 0),
    measurement_count      integer NOT NULL CHECK (measurement_count >= 0),
    two_qubit_gate_count   integer NOT NULL CHECK (two_qubit_gate_count >= 0),
    source_record_id       text NOT NULL UNIQUE,
    UNIQUE (benchmark_name, variant)
);

-- One row: one declared quantum or classical register of a circuit.
CREATE TABLE circuit_register (
    circuit_id     text NOT NULL REFERENCES circuit,
    register_name  text NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('qreg', 'creg')),
    size           integer NOT NULL CHECK (size > 0),
    position       integer NOT NULL,
    PRIMARY KEY (circuit_id, register_name)
);

-- One row: one parity check: an ancilla collecting data-qubit parity into a
-- syndrome bit. Register-qualified qubit names ("a[0]") are never bare indices.
CREATE TABLE stabilizer_check (
    check_id       text PRIMARY KEY,                       -- Silver source_record_id
    circuit_id     text NOT NULL REFERENCES circuit,
    check_label    text NOT NULL,
    ancilla_qubit  text NOT NULL,
    syndrome_bit   text NOT NULL,
    UNIQUE (circuit_id, check_label)
);

-- One row: one data qubit participating in one parity check.
CREATE TABLE stabilizer_check_qubit (
    check_id    text NOT NULL REFERENCES stabilizer_check,
    data_qubit  text NOT NULL,
    position    integer NOT NULL CHECK (position >= 0),
    PRIMARY KEY (check_id, data_qubit)
);

-- One row: one recovery gate applied when the syndrome register equals a value.
CREATE TABLE conditional_correction (
    correction_id       text PRIMARY KEY,                  -- Silver source_record_id
    circuit_id          text NOT NULL REFERENCES circuit,
    condition_register  text NOT NULL,
    condition_value     bigint NOT NULL CHECK (condition_value >= 0),
    gate                text NOT NULL,
    target_qubit        text NOT NULL,
    UNIQUE (circuit_id, condition_register, condition_value, target_qubit)
);

-- One row: the load that produced this schema version.
CREATE TABLE gold_load (
    run_id      text PRIMARY KEY,
    loaded_at   timestamptz NOT NULL DEFAULT now(),
    row_counts  jsonb NOT NULL
);

-- Derived: one row per shot per decoder, with the mistake made explicit.
CREATE VIEW decoder_outcome AS
SELECT p.shot_id, s.experiment_id, s.shot_index, p.decoder_id, p.predicted_flip,
       s.actual_observable_flip, p.predicted_flip <> s.actual_observable_flip AS decoder_error
FROM decoder_prediction p
JOIN shot s USING (shot_id);
