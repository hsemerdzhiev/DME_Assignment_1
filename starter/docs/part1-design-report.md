# Part I design report: QEC data pipeline

This report explains the data-management half of the assignment: what the three sources contain, how the
pipeline turns them into checked Silver tables, a PostgreSQL Gold model and two ML tables, and the evidence
that the result is correct and traceable. Detailed size measurements and per-table notes are in
`docs/gold-design.md`; machine-readable evidence stays in `results/part1/`. All counts below are from the
run recorded in `results/part1/run.json` (run `a62b9db5`, code revision `f158c5e`).

## 1. Source discovery and row meanings

All three supplied archives are used. QASMBench cannot be joined to the other two (Section 4).

| Source and Bronze object | Members and formats | What one record is | Identifier we use |
| --- | --- | --- | --- |
| `qec_syndromes`: `syndromes_dataset.zip` | 7 CSV files `d-3_pfr-<rate>_nb-10M.csv`, header `labels,syndromes,quantity` | one aggregate observation: a 4-round by 4-check syndrome pattern, a logical-error label and the number of simulated shots (`quantity`) it stands for | file + CSV row; business key (fault rate, pattern, label) |
| `google_qec`: `google-surface-code-curated.zip` | 5 experiment directories `surface_code_bX_d{3,5}_r25_center_<row>_<col>/`, each with `properties.yml`, three Stim `b8` files, `obs_flips_actual.01` and four `obs_flips_predicted_by_*.01` files | one hardware shot: the same position in eight aligned companion files | experiment directory + zero-based `shot_index` |
| `qasmbench`: `qasmbench-qec.zip` | `error_correctiond3_n5`, `qec_en_n5`, `qec_sm_n5`, each as source and transpiled OpenQASM 2.0 | one circuit variant; inside it, registers, executed operations, parity checks and conditional corrections | member path; check and correction by circuit + statement |

Findings that shaped the design:

- **Syndromes.** The 7 files hold 75,598 rows that represent 70,000,000 shots (10,000,000 per file, matching the
  `nb-10M` in each name). The `syndromes` string parses to 4 rounds of 4 binary values. A syndrome alone is not a
  key: the same pattern occurs with both labels in 6 of 7 files (13 patterns at fault rate 0.00005 up to 18,210 at
  0.01). Across files there are only 31,941 distinct patterns, and 14,187 occur at more than one fault rate, so a
  pattern is a shared entity rather than an attribute of one row. Every aggregate must be weighted by `quantity`.
- **Google.** Four distance-three experiments at different processor centres and one distance-five experiment,
  all X basis, 25 rounds and 50,000 shots, giving 250,000 shots. Per shot there are packed measurements, sweep
  bits and detector events (200 detector bits at d=3, 25 rounds of 8 stabilizers; 600 at d=5), one actual logical
  flip and four decoder predictions. Records in `b8` files are byte-aligned and little-endian within a byte; the
  expected length is `shots * ceil(bits / 8)` from `properties.yml`. Measurements, detector events (changes
  between rounds), the actual flip, a decoder prediction and a decoder error (prediction differs from the actual
  flip) are five different things and are kept apart. The Stim circuits, layouts and detector error models are
  descriptive and not needed by any required output.
- **QASMBench.** Six circuit variants with 16 declared registers in total. Register-local indices are qualified
  (`q[0]` and `a[0]` are different qubits), gate definitions are expanded only where called, and register
  broadcasts count as one executed operation per index. Only `qec_sm_n5` measures explicit parity checks: two
  ancillas each collect the parity of two adjacent data qubits into `syn`, and `if(syn==v) x ...` applies the
  recovery. In the other two benchmarks the parser finds no ancilla that collects the parity of two data qubits
  before being measured, so they contribute circuits and registers but no checks.

## 2. Architecture and reproduction

We use the course architecture without changes.

```text
bronze/source=<source>/   supplied archives, read-only, verified against the release manifest
  -> stages/register_sources.py   size, SHA-256, safe member names, missing/unexpected objects (lake.source_object)
  -> stages/prepare_data.py       parse + check per source -> six Silver Parquet tables,
                                  results/part1/data_issues.parquet, results/part1/source_trace.parquet
silver/<source>/<table>.parquet
  -> stages/load_postgres.py      build gold_build, COPY, integrity checks, swap to gold in one transaction
gold (PostgreSQL)
  -> stages/build_ml_tables.py    committed views in gold/queries/ml.sql + supplied split helpers
ml/ml_syndrome_decoder_example.parquet, ml/ml_google_decoder_example.parquet
```

Commands, from the repository root after `make bootstrap`:

| Purpose | Command |
| --- | --- |
| Part I (Bronze to ML, plus all `results/part1/` files) | `make pipeline` |
| Part II | `make train` |
| Tests | `docker compose exec -T workspace make test` |
| Attach saved predictions to the trace examples (after Part II) | `docker compose exec -T workspace python -m quantum_lake_student.cli trace` |

**Repeated runs.** Every identifier is a hash of source facts only (`record_id(source, member, row or shot)`),
never of run time or load order. Bronze registration upserts on `(source, bronze_key)`; Silver files and the
results files are replaced whole; Gold is rebuilt in a fresh schema and swapped in. A second run on unchanged
input therefore yields the same identifiers and no duplicate business records, which
`tests/test_load_postgres.py::test_rerun_reproduces_the_previous_counts` checks.

## 3. Gold design

Each Gold table, with what one row represents (keys and constraints are in `gold/schema.sql`):

| Table | One row is | Key | Rows |
| --- | --- | --- | ---: |
| `experiment` | one simulated fault-rate sweep or one hardware memory experiment | `experiment_id` | 12 |
| `syndrome_pattern` | one distinct 16-value syndrome pattern, shared by all fault rates where it occurs | `pattern_id`; `syndrome_bits` unique | 31,941 |
| `syndrome_observation` | one aggregate simulated observation: pattern, label and `quantity` | Silver `source_record_id`; unique (experiment, pattern, label) | 75,598 |
| `decoder` | one supplied decoder | `decoder_id` | 4 |
| `shot` | one aligned hardware shot with packed measurement, sweep and detector bytes and the actual flip | Silver `source_record_id`; unique (experiment, shot_index) | 250,000 |
| `decoder_prediction` | one decoder's prediction for one shot | (shot_id, decoder_id) | 1,000,000 |
| `detector_summary` | how often one detector position fired across one experiment | (experiment_id, detector_index) | 1,400 |
| `circuit` | one parsed circuit variant | `circuit_id`; unique (benchmark, variant) | 6 |
| `circuit_register` | one declared quantum or classical register | (circuit_id, register_name) | 16 |
| `stabilizer_check` | one parity check: ancilla, syndrome bit | Silver `source_record_id`; unique (circuit, label) | 4 |
| `stabilizer_check_qubit` | one data qubit in one parity check | (check_id, data_qubit) | 8 |
| `conditional_correction` | one recovery gate applied when a syndrome register equals a value | Silver `source_record_id` | 6 |
| `gold_load` | the load that produced the current Gold version | `run_id` | 1 |
| `decoder_outcome` (view) | one prediction next to the actual flip, with `decoder_error` derived | | 1,000,000 |

Relationships: experiment 1:n syndrome_observation and 1:n shot; syndrome_pattern 1:n syndrome_observation;
shot 1:n decoder_prediction (exactly 4, checked); decoder 1:n decoder_prediction; experiment 1:n
detector_summary; circuit 1:n register, check and correction; check 1:n check_qubit. The schema has 13 primary
keys, 10 foreign keys, 7 unique constraints, 30 CHECK constraints (odd distance, 16-byte patterns, positive
quantity, kind-specific required columns, and others), plus secondary indexes on
`syndrome_observation.pattern_id` and `decoder_prediction.decoder_id` and a partial unique index on the simulated
fault rate.

The ML views `gold.ml_syndrome_source` and `gold.ml_google_source` (`gold/queries/ml.sql`) build every ML column,
including a deterministic `example_id` (md5 of experiment + pattern + label, or experiment + shot index) and
`gold_record_id`, which is the resolution relation from an ML example to its Gold record.

## 4. Important decisions and the rejected relationship

| Decision | Why | Cost (measured) |
| --- | --- | --- |
| Syndrome patterns are their own entity | 31,941 patterns are reused across fault rates; frequency per rate becomes a join, and the (experiment, pattern, label) key allows the valid both-label case | one extra table (about 6 MiB) and one join in the ML export |
| Decoder predictions stored long, not as four columns | decoders are compared as things (`GROUP BY decoder_id`); decoder errors are derived in a view, never stored | 1,000,000 rows, 184 MiB with indexes |
| Detector events stay packed on `shot`, with per-position `detector_summary` | the packed bytes are exactly what the Google ML table needs, so the export is a projection (`s.detector_bits`) and never re-reads Bronze or Silver | 83.6 MiB, against 1,341.8 MiB for one row per fired detector (10,446,925 rows) and 9,127.6 MiB for one row per position per shot; per-shot, per-position filtering in SQL is given up |
| Shared `experiment` table for simulated and hardware experiments | common vocabulary (distance, rounds, checks) with a `kind` column and CHECK rules per kind; no column links the two kinds | some nullable kind-specific columns |
| Gold keys reuse Silver `source_record_id` where the grain matches | one join from any Gold row to `source_trace.parquet` | 32-character text keys |
| All-or-nothing load | `gold_build` is created, filled with binary COPY, checked by 9 cross-table integrity queries, then swapped for `gold` in one transaction; any failure rolls back and the previous Gold stays | about 21 s per full load |

**Investigated and rejected: joining simulated syndromes to hardware shots.** Both sources are distance-three
surface codes with a logical-error label, and "syndrome" and "detector event" sound alike, so a join on distance
is tempting. The evidence in `gold.experiment` rules it out:

| kind | distance | rounds | checks per round | experiments |
| --- | ---: | ---: | ---: | ---: |
| simulated_syndrome | 3 | 4 | 4 | 7 |
| hardware_memory | 3 | 25 | 8 | 4 |
| hardware_memory | 5 | 25 | 24 | 1 |

A simulated observation has 16 raw syndrome values; a hardware d=3 shot has 200 derived detector events. There
is no shared experiment, shot or qubit identifier, and the simulated fault rate has no hardware counterpart. A
join would be a fabricated match on `distance = 3` alone, so the two stay separate and are queried separately.
For the same reason QASMBench circuits have no key to any experiment: similar code-family names describe context,
not identity.

## 5. Quality findings and count reconciliation

Every check is implemented in the Silver parsers (`silver/*.py`) or the Gold load. Invalid records would be
excluded from Silver and written to `results/part1/data_issues.parquet` with value, rule, severity, action and
reason; a missing companion file or unsafe member stops the run.

| Check | Rule id | Outcome in this run |
| --- | --- | --- |
| Bronze size and SHA-256 against the manifest | `register_sources` | 3 of 3 objects verified |
| Safe archive member names, missing or unexpected objects | `register_sources` | all members safe; none missing or unexpected |
| Documented `label` versus actual `labels` header | `syn_header_label_vs_labels` | found in all 7 files; recorded as info and read by position |
| 4 rounds of 4 binary values; binary label; positive integer `quantity` | `syn_shape_4x4_binary`, `syn_label_binary`, `syn_quantity_positive` | 0 rows rejected |
| Weighted total equals 10,000,000 per file | `syn_quantity_reconciles_to_filename` (+ Gold integrity check) | all 7 files reconcile |
| Same syndrome with both labels | `syn_syndrome_with_both_labels` | valid; 0 / 13 / 46 / 179 / 467 / 6,244 / 18,210 patterns from lowest to highest rate, kept in Silver |
| Directory name agrees with `properties.yml` | `goog_dirname_matches_properties` | 5 of 5 agree |
| Required companion files present | `goog_required_companion_files`, `qasm_required_companion_files` | all present (run did not stop) |
| `b8` length = shots x byte-aligned bits per shot | `goog_b8_length` | 15 of 15 files correct |
| Padding bits beyond the declared width are zero | `goog_b8_padding_zero` | no non-zero padding found |
| `01` files: one binary line per shot | `goog_01_rows_binary` | 25 of 25 files correct; actual and predicted flips aligned by shot |
| QASM registers, operations, checks, measurements, conditionals | `qasm_structure`, `qasm_no_measurement`, `qasm_conditional_not_single_qubit_gate` | 6 of 6 circuits parsed; no warnings |
| Cross-table rules in Gold (event counts, byte widths, shots per experiment, 4 predictions per shot, summary totals, correction registers) | `INTEGRITY_CHECKS` in `stages/load_postgres.py` | all pass |

`data_issues.parquet` holds 14 rows, all severity `info` with action `accept` (the header note and the both-label
note per syndrome file). No record was invalid in this release, so nothing was rejected. Counts reconcile at every
step (`results/part1/row_counts.json`):

| Data | Read | Rejected | Silver | Gold | ML |
| --- | ---: | ---: | ---: | ---: | ---: |
| Syndrome observations | 75,598 | 0 | 75,598 | 75,598 observations (31,941 patterns) | 75,598 |
| Google experiments | 5 | 0 | 5 | 5 hardware + 7 simulated = 12 | |
| Google shots | 250,000 | 0 | 250,000 | 250,000 shots, 1,000,000 predictions | 250,000 |
| QASM circuits | 6 | 0 | 6 | 6 circuits, 16 registers | |
| Stabilizer checks | | | 4 | 4 checks, 8 check qubits | |
| Conditional corrections | | | 6 | 6 | |

`prepare_data.reconcile` stops the run if read is not equal to accepted plus rejected for any table, and the
export stops if an ML table does not have the release row count or the syndrome weights do not sum to 70,000,000.
`source_trace.parquet` has 2,075,619 rows: one per Silver record, and eight per shot (one per aligned member).

## 6. Example traces

Both examples are test rows and are stored in `results/part1/trace_examples.json`, which the export writes and
`cli trace` later extends with the saved Part II predictions.

**Syndrome prediction.**
`ml_syndrome_decoder_example.example_id = 000150fd917812873edc5a55b5f17c2b`
-> `gold.ml_syndrome_source.gold_record_id` = `gold.syndrome_observation.observation_id = 287465856fecd1d80ece7024d09af352`
-> Silver `syndrome_observation.source_record_id` (same value)
-> `source_trace`: `syndromes_dataset.zip`, member `d-3_pfr-0.005000_nb-10M.csv`, `row 18267`, SHA-256 `bdfce36a...`.
Label `false`. Saved predictions: `task_a_weighted_prior` false (probability 0.043), `task_a_logistic_regression`
true (probability 0.947).

**Google prediction.**
`ml_google_decoder_example.example_id = 00011e9603e37b3c0c2c75a87d1a0745`
-> `gold.shot.shot_id = 30e5405aac0076b64b124a24558a04f5` (experiment `surface_code_bX_d3_r25_center_5_7`, shot 18681)
-> Silver `shot.source_record_id` (same value)
-> `source_trace`: eight rows in `google-surface-code-curated.zip` (SHA-256 `5d6a24f8...`), `shot 18681` of
`measurements.b8`, `sweep.b8`, `detection_events.b8`, `obs_flips_actual.01` and the four
`obs_flips_predicted_by_*.01` files. Actual flip `true`. Saved Task B predictions: belief matching, correlated
matching, PyMatching and the combined logistic regression predict a flip; tensor network contraction and the prior
do not.

## 7. SQL analyses

The SQL is committed in `src/quantum_lake_student/gold/queries/` and run by the Part I export; results are
written to `results/part1/analysis/` as JSON.

**Q1. Weighted syndrome frequency and logical-error labels by physical fault rate** (`syndrome_analysis.sql`,
joins `syndrome_observation`, `experiment`, `syndrome_pattern`; 50,439 rows, one per fault rate and pattern).
Summarised per fault rate:

| Fault rate | Distinct patterns | Weight of all-zero syndrome | Weighted logical-error rate |
| ---: | ---: | ---: | ---: |
| 0.00001 | 68 | 99.87% | 0.023% |
| 0.00005 | 202 | 99.36% | 0.115% |
| 0.0001 | 445 | 98.74% | 0.229% |
| 0.0005 | 1,228 | 93.84% | 1.144% |
| 0.001 | 2,387 | 88.06% | 2.270% |
| 0.005 | 14,643 | 53.10% | 10.395% |
| 0.01 | 31,466 | 28.40% | 18.653% |

As the simulated fault rate rises, weight moves away from the all-zero syndrome to many more distinct patterns,
and the weighted share of logical-error labels rises from 0.023% to 18.7%. The all-zero syndrome almost never
carries a logical error (at most 0.04% of its weight, at 0.01). These are associations within the simulation;
the data do not show anything about hardware noise.

**Q2. Supplied decoder logical-error rates by distance and processor location** (`decoder_analysis.sql`, joins
the `decoder_outcome` view over `decoder_prediction` and `shot` with `experiment`; 50,000 shots per cell).

| Distance | Centre (row, col) | Belief matching | Correlated matching | PyMatching | Tensor network |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | (3, 5) | 0.4025 | 0.4194 | 0.4342 | 0.4023 |
| 3 | (5, 3) | 0.4159 | 0.4252 | 0.4432 | 0.4130 |
| 3 | (5, 7) | 0.3920 | 0.4182 | 0.4306 | 0.3839 |
| 3 | (7, 5) | 0.3884 | 0.4130 | 0.4299 | 0.3887 |
| 5 | (5, 5) | 0.4015 | 0.4257 | 0.4510 | 0.3955 |

The decoder ranking is the same at every location: tensor network contraction and belief matching are lowest,
then correlated matching, and PyMatching is highest. The d=3 locations differ by up to about 3 percentage points
for the same decoder, and the single d=5 experiment is not lower than the best d=3 locations. With one d=5
experiment at one location this cannot be separated from location effects, so we do not draw a conclusion about
distance. All rates are high (0.38 to 0.45) after 25 rounds; the analysis describes the supplied predictions,
not decoder quality in general.

**Q3. Repetition-code mapping from data qubits to checks and corrections** (`circuit_analysis.sql`, joins
`circuit`, `stabilizer_check`, `stabilizer_check_qubit` and `conditional_correction`; 24 rows because each check
row is listed with every correction on the same register).

| Check | Data qubits | Ancilla -> syndrome bit | Correction |
| --- | --- | --- | --- |
| `a[0]->syn[0]` | `q[0]`, `q[1]` | `a[0]` -> `syn[0]` | `syn == 1`: `x q[0]` |
| `a[1]->syn[1]` | `q[1]`, `q[2]` | `a[1]` -> `syn[1]` | `syn == 2`: `x q[2]`; `syn == 3`: `x q[1]` |

The source and transpiled variants of `qec_sm_n5` give the same mapping. Reading `syn` as `syn[0] + 2 * syn[1]`,
value 1 means only the `q[0]`/`q[1]` check fired, so `q[0]` is flipped; value 2 means only the `q[1]`/`q[2]` check
fired, so `q[2]` is flipped; value 3 means both fired, pointing to the shared qubit `q[1]`. This is the usual
three-qubit repetition-code decoding table, recovered from the circuit structure without simulation. The other
two benchmarks contribute circuits and registers but no explicit parity checks.

## Known limitations

- The parity-check rule is structural and finds checks only where an ancilla is the CX target of two or more data
  qubits before its measurement; it reports none for the encoder and five-qubit-code benchmarks.
- Passing Google and QASM checks are not written to `data_issues.parquet` as rows; their outcome follows from the
  absence of reject or warning rows and the reconciled counts above.
- `detector_summary` supports per-position questions per experiment, but per-shot, per-position filtering needs
  unpacking outside SQL.
