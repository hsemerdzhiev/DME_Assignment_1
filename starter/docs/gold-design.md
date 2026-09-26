# Gold design

Gold is the team-designed PostgreSQL model. It is rebuilt from the six Silver
tables by `stages/load_postgres.py` using the DDL in `gold/schema.sql`. This
note records what each row means, the decisions behind the shape, their
measured costs, and the relationship we investigated and rejected.

## Row meanings

| Table | One row is |
| --- | --- |
| `experiment` | one experiment from either source: a simulated fault-rate sweep or a hardware memory experiment |
| `syndrome_pattern` | one distinct 4-round by 4-check syndrome pattern, shared by every fault rate in which it occurs |
| `syndrome_observation` | one aggregate simulated observation: pattern, label, and the number of shots it stands for |
| `decoder` | one supplied decoder |
| `shot` | one aligned hardware shot with its packed measurement, sweep, and detector rows and the actual logical flip |
| `decoder_prediction` | one decoder's prediction for one shot |
| `detector_summary` | how often one detector position fired across all shots of one experiment |
| `circuit` | one parsed circuit variant (source or transpiled) of a benchmark |
| `circuit_register` | one declared quantum or classical register of a circuit |
| `stabilizer_check` | one parity check: an ancilla collecting data-qubit parity into a syndrome bit |
| `stabilizer_check_qubit` | one data qubit taking part in one parity check |
| `conditional_correction` | one recovery gate applied when a syndrome register equals a value |
| `gold_load` | the load that produced the current schema version |
| `decoder_outcome` (view) | one decoder prediction next to the actual flip, with `decoder_error` derived |

## Shared vocabulary without invented matches

Both data sources describe distance-3 surface-code experiments, so they share
one `experiment` table with common `distance`, `rounds`, and `check_count`
columns and a `kind` column that says which they are. CHECK constraints make
the kind-specific columns mandatory for one kind and forbidden for the other.
No column, key, or view relates a simulated row to a hardware row.

QASMBench circuits are not experiments and have no row in `experiment`. They
live in their own tables with register-qualified qubit names such as `a[0]`,
because index zero of the data register is not the qubit at index zero of the
ancilla register.

## Decisions and their costs

**Syndrome patterns are an entity.** The seven CSV files hold 75,598 rows but
only 31,941 distinct 16-bit patterns; 14,187 patterns occur at more than one
fault rate. `syndrome_pattern` stores each once with its fired-check count;
`syndrome_observation` references it. This makes "how does the frequency of
this pattern change with fault rate" a join instead of a byte comparison and
lets the same pattern legitimately carry both labels within one experiment
(the unique key is experiment, pattern, label). Cost: a 6 MB table and one
join in the ML export.

**Decoder predictions are long, not wide.** Silver has four boolean columns.
Gold has a `decoder` dimension and one `decoder_prediction` row per shot per
decoder. Queries such as "compare decoders by distance" become a GROUP BY on
`decoder_id`, a fifth decoder is a new row rather than a new column, and a
decoder mistake is computed in the `decoder_outcome` view rather than stored.
Cost: 1,000,000 rows and 184 MB including indexes, versus roughly 20 MB for
four columns on `shot`. We accepted this because the analysis questions and
Part II Task B are all about comparing decoders as things.

**Detector bits stay packed; positions are summarised.** Three shapes were
measured for the 10,446,925 fired detector events in the release:

| Shape | Rows | Size |
| --- | --- | --- |
| Packed `bytea` on `shot` plus `detector_summary` (chosen) | 250,000 + 1,400 | 83 MB + 248 kB |
| One row per fired detector | 10,446,925 | about 700 MB before indexes |
| One row per detector position per shot | 110,000,000 | not attempted |

The packed row is exactly what the ML export must reproduce, so keeping it
means the export is a projection rather than a re-packing. `detector_summary`
gives per-position fire counts for the "which detectors fire most" style of
question without unpacking bits in SQL. What we give up is per-shot,
per-position filtering in SQL; Part II unpacks the bits in Python using the
supplied helper, which is where that work belongs.

**Gold keys are Silver identifiers where the grain matches.** `shot_id`,
`observation_id`, `check_id`, `correction_id`, and `circuit.source_record_id`
are the Silver `source_record_id` values. Resolving a Gold row to its Bronze
bytes is one join to `results/part1/source_trace.parquet`. Derived tables use
composite natural keys (`shot_id, decoder_id`; `experiment_id, detector_index`;
`check_id, data_qubit`), so nothing depends on insertion order or run time.

**Constraints do the checking the loader cannot skip.** 13 primary keys, 10
foreign keys, 7 unique constraints, and 30 CHECK constraints, including
domain rules such as odd distance, 16-byte patterns, positive quantity, and
the kind-specific column rules. Nine cross-table checks that constraints
cannot express run inside the load transaction: byte widths against the
experiment's bit counts, event counts within `detector_count`, shots per
experiment equal to the declared count, every shot having every decoder's
prediction, per-position fire counts summing to per-shot event counts, every
simulated experiment summing to ten million shots, and correction registers
being declared classical registers.

**All-or-nothing load.** The loader creates `gold_build`, runs the DDL, COPYs
every table in binary form, runs the checks, drops `gold`, and renames
`gold_build` to `gold`, all inside one transaction. PostgreSQL DDL is
transactional, so a failure anywhere rolls the whole thing back and the
previous `gold` remains readable throughout. A full load takes about 21
seconds. `tests/test_load_postgres.py` proves the rollback by loading a
duplicate decoder key and checking the earlier `gold_load` row is still the
only one.

## Investigated and rejected: joining simulated syndromes to hardware shots

Both sources are distance-3 surface codes with a logical-error label, and the
plain-language names line up: "syndrome" in one, "detector event" in the
other. It is tempting to treat the simulated sweep as a model of the hardware
experiments and join them on distance.

The `experiment` table itself is the evidence against it:

```sql
SELECT kind, distance, rounds, check_count, count(*) FROM gold.experiment GROUP BY 1, 2, 3, 4;
```

| kind | distance | rounds | check_count | experiments |
| --- | --- | --- | --- | --- |
| simulated_syndrome | 3 | 4 | 4 | 7 |
| hardware_memory | 3 | 25 | 8 | 4 |
| hardware_memory | 5 | 25 | 24 | 1 |

A simulated observation has 4 rounds of 4 checks: 16 values in total. A
hardware distance-3 shot has 25 rounds of 8 stabilizers: 200 detector bits.
The simulated data also measures raw syndromes, while the hardware data stores
derived detector events, which are changes between rounds. There is no shared
experiment, shot, or qubit identifier, and the simulated fault rate has no
hardware counterpart. Any join would be a fabricated match on the single
column `distance = 3`. We keep both in `experiment` for vocabulary only and
answer questions about each source separately.

## Tracing path

Bronze object and member → `results/part1/source_trace.parquet` →
`source_record_id` → Gold key (`shot_id`, `observation_id`, ...) → Gold row.
The ML export will hash Gold keys into `example_id` so the chain continues
into Part II.
