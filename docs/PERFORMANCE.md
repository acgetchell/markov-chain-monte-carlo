# Benchmark Performance

> [!WARNING]
> **Legacy, non-reproducible report.** This pre-promotion working-tree snapshot is retained only as historical context. Repository-owned CSV measurements,
> JSON provenance with exact commands, Cargo-lock and combined-source digests, a concrete CPU model, and native Criterion sample archives are unavailable.
> Treat the rows below as legacy observations, not reproducible release evidence. The next committed `just performance-release` promotion will replace this
> report with one linked to its tracked evidence; do not regenerate or promote it from an unrelated dirty tree.

**markov-chain-monte-carlo** v0.4.1 working tree · `56d2cb5`
**Statistic**: median

Comparison against baseline **v0.4.0**:

Positive time reduction means the current duration is lower (faster); negative means it is higher (slower).
The relative-performance column states how many times the current version is faster or slower.
Shown confidence intervals are Criterion's marginal intervals; they are not a paired significance test.

## Measurement Context

- Source mode: same-host isolated worktrees; current `HEAD` with tracked and untracked working-tree changes applied.
- Host: `macOS-26.6-arm64-arm-64bit-Mach-O` on `arm64`.
- Current commit: `56d2cb5a6a8e9e5909fe2c6da2c76704f4f81004`; rustc: `rustc 1.97.1 (8bab26f4f 2026-07-14)`; Criterion: `0.8.2`.
- Baseline commit: `e88fc901fa3b02c34e6c67d785d897231b75754d`; rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`; Criterion: `0.8.2`.
- Benchmark harness SHA-256 prefixes: current `7763d9a61040`; baseline `77e1fc0e4212`.
- Benchmark harness hashes differ; verify that every shared name retains the same workload contract.

## Results

| Benchmark | Baseline | Current | Time reduction | Current vs baseline |
|:----------|---------:|--------:|---------------:|--------------------:|
| `chain_step_by_value` | 15.92 ns (15.87 ns - 15.99 ns) | 16.33 ns (16.28 ns - 16.38 ns) | -2.53% | 1.03x slower |
| `chain_step_delayed_accept_commit` | 8.05 ns (8.03 ns - 8.07 ns) | 8.06 ns (8.03 ns - 8.08 ns) | -0.20% | 1.002x slower |
| `chain_step_delayed_no_plan` | 1.14 ns (1.13 ns - 1.14 ns) | 1.12 ns (1.12 ns - 1.13 ns) | +1.37% | 1.01x faster |
| `chain_step_delayed_reject_plan` | 8.13 ns (8.10 ns - 8.18 ns) | 8.19 ns (8.16 ns - 8.20 ns) | -0.78% | 1.008x slower |
| `chain_step_mut_accept` | 12.19 ns (12.16 ns - 12.24 ns) | 13.05 ns (13.00 ns - 13.11 ns) | -7.06% | 1.07x slower |
| `chain_step_mut_reject_rollback` | 105.57 ns (105.14 ns - 105.76 ns) | 109.02 ns (108.69 ns - 109.19 ns) | -3.27% | 1.03x slower |
| `observing_manual_online_sum_100` | 980.30 ns (976.83 ns - 983.27 ns) | 964.58 ns (964.13 ns - 964.91 ns) | +1.60% | 1.02x faster |
| `observing_run_observing_buffer_100` | 1.61 µs (1.61 µs - 1.62 µs) | 1.62 µs (1.62 µs - 1.62 µs) | -0.47% | 1.005x slower |
| `observing_run_observing_into_binning_100` | 2.06 µs (2.05 µs - 2.06 µs) | 2.04 µs (2.03 µs - 2.04 µs) | +1.02% | 1.01x faster |
| `observing_run_observing_into_online_stats_100` | 1.30 µs (1.30 µs - 1.31 µs) | 1.30 µs (1.30 µs - 1.31 µs) | -0.02% | 1.0002x slower |
| `sampler_run_by_value_100` | 1.58 µs (1.57 µs - 1.58 µs) | 1.58 µs (1.58 µs - 1.58 µs) | +0.05% | 1.0005x faster |
| `sampler_run_delayed_100` | 932.73 ns (927.50 ns - 935.80 ns) | 949.96 ns (946.69 ns - 952.84 ns) | -1.85% | 1.02x slower |
| `sampler_run_mut_100` | 1.08 µs (1.08 µs - 1.08 µs) | 1.09 µs (1.08 µs - 1.09 µs) | -0.24% | 1.002x slower |

## How to Update

```bash
just performance-local
just performance-github-assets
just performance-release
just performance-rerender
just performance-release <current-tag> <baseline-tag>
```

Generated Markdown, CSV measurements, and JSON provenance live under `target/bench-reports/`.
The curated release report is `docs/PERFORMANCE.md`.
Older curated reports are indexed under `docs/archive/performance/`.

See `docs/BENCHMARKING.md` for command semantics and reproducibility limits.
