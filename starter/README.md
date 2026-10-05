# Student pipeline workspace

This is the starting point for your implementation. The platform supplies the
services, dependencies, source data, configuration, connection helpers, and
simple stage templates. Your team supplies the data engineering.

Read these documents in order:

1. `../assignment/getting-started.md`
2. `../assignment/brief.md`
3. `../assignment/plain-language-guide.md`
4. `../assignment/quantum-data-primer.md`
5. `../assignment/data-sources.md`
6. [Parquet in this assignment](../assignment/plain-language-guide.md#parquet-in-this-assignment)
7. `../assignment/silver-tables.md`
8. `../assignment/required-ml-tables.md`
9. `../assignment/part-2-ai-ml.md`
10. `../assignment/rubric.md`

The implementation includes source registration, the six Silver tables, the
PostgreSQL Gold model, both ML exports, and Tasks A, B and C. `make check`
verifies connections, `make inventory` lists Bronze objects, `make run` builds
the Part I outputs, and `make train` runs Tasks A, B and C.

Suggested source layout:

```text
src/quantum_lake_student/
├── config.py
├── connections.py
├── formats.py
├── ml.py
├── models.py
├── cli.py
└── stages/
    ├── register_sources.py
    ├── prepare_data.py
    ├── build_ml_tables.py
    ├── load_postgres.py
    └── train.py
```

`formats.py` provides representation-level readers for the supplied Stim `b8`
and `01` files. It does not decide how parity measurements, detector events,
shots, or decoder outputs should be modeled; those decisions remain part of
the assignment.

`ml.py` provides bit unpacking, course split assignment, the documented model
inputs, partition loading, and weighted LER. It does not select models, perform
training, or interpret results. Those remain required Part II work.

Generated data, reports, credentials, and notebook outputs should not be
committed to version control. Write required run evidence and outputs under the
`results/part1/` and `results/part2/` layout described in the brief.

The AI/ML stage is a downstream consumer check. A particular model score or an
improvement over a supplied decoder is not part of the grade.

## Implemented pipeline and Tasks A, B and C

From the repository root, start and verify the platform with `make bootstrap`.
Then run:

```bash
make pipeline
make train
docker compose exec -T workspace make test
```

`make run` now registers Bronze, builds Silver, loads Gold, exports both
prescribed `ml/ml_*_decoder_example.parquet` tables, and writes Part I run,
count, trace, and SQL-analysis evidence. Generated ML files are ignored by Git.
`make train` runs Task A's weighted prior and weighted logistic regression,
Task B's Google decoder comparisons, and Task C's raw-detector MLP. It
regenerates models and the prescribed files under `results/part2/`. Training
requires both ML files for contract validation and hash recording. It never reads Gold,
Silver, or Bronze. The saved Gold ML views resolve prediction IDs to source
records. Timings vary across runs; predictions and metrics are deterministic.

In `stages/train.py`, the task functions are `run_task_a`, `run_task_b` and
`run_task_c`. The report uses the same task headings.

After training, run `docker compose exec -T workspace python -m quantum_lake_student.cli trace`
to attach saved test predictions to the two source examples.
