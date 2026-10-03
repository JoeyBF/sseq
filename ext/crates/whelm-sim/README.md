# whelm-sim

Offline simulation of Nassau scheduling against the placement policies of the sibling crate
[`whelm`](../whelm). Every simulator drives the real `whelm` policies through their
message-driven API (`Policy::handle` / `Policy::poll`), so a result here is what the library would
do given the modelled service times.

The crate is standalone (its own Cargo workspace and lock file). Run the binaries from this
directory:

```sh
# Replay a scheduling trace (standalone JSONL or a `whelm::log::JsonlSink` event log, optionally
# gzipped) against several policies, open or closed loop.
cargo run --release --bin whelm-sim -- --trace sched_trace.jsonl.gz --json results.json
cargo run --release --bin whelm-sim -- --trace sched_trace.jsonl.gz --closed --rank

# Simulate a whole resolution, built a priori from census CSVs, under several dispatch plans.
cargo run --release --bin whelm-whole -- --trace sched_trace.jsonl.gz --census a.csv --census b.csv

# Device-aware admission on a synthetic small-card scenario.
cargo run --release --bin whelm-device

# Compare two plans on many small instances, typically and adversarially (PISA). Plan suffixes
# include +fast, +eft, +spec (speculative second attempts) and +age<seconds>.
cargo run --release --bin whelm-pisa -- --a rank-oracle+fast --b group+fast typical --samples 2000
cargo run --release --bin whelm-pisa -- --a rank-oracle+fast --b group+fast anneal --restarts 8

# The same on tiny instances (`small::TINY_JOBS` jobs), with each plan's gap to the exact optimum.
cargo run --release --bin whelm-pisa -- --a heft-oracle+fast --b group+fast --family tiny typical
```

Plans order jobs by group (oldest bidegree first), by upward rank, or, in `whelm-pisa`, by an
offline HEFT schedule (`heft`, `heft-oracle`: each job's planned start becomes its
`JobSpec::priority`) or by Smith's rule (`wspt`). `--timing` picks the scheduler's machine model
in `whelm-pisa` and `whelm-whole`.

The exact oracle (`exact::solve`) is a branch and bound for Q|prec|Cmax with known durations,
each worker slot a machine at its worker's speed. It reports whether it proved its makespan
optimal or stopped at its node or time limit (`--nodes`, `--time-limit`) with the best schedule
found; `--exact` runs it on other families too, where it mostly stops at the limit.

Each binary's `--help` lists its options. `cargo test` runs the unit tests and the round trip of
a logged run through the trace reader and the replay.

- `RESULTS.md`: what the simulations found, with the commands that produced each table.
- `LITERATURE.md`: the scheduling literature the policies draw on.
- `research/`: notes extracted from other schedulers and simulators, and the plan they fed.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
