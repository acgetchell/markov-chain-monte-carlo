# Benchmark Performance

**markov-chain-monte-carlo** v0.4.2 working tree · `ef21cc6`
**Statistic**: median

Comparison against baseline **v0.4.1**:

Positive time reduction means the current duration is lower (faster); negative means it is higher (slower).
The relative-performance column states how many times the current version is faster or slower.
Shown confidence intervals are Criterion's marginal intervals; they are not a paired significance test.

## Measurement Context

- Source mode: same-host isolated worktrees; current `HEAD` with tracked and untracked working-tree changes applied.
- Host: `macOS-26.6.2-arm64-arm-64bit-Mach-O` on `arm64`; CPU: `Apple M4 Max`.
- Current commit: `ef21cc6430329cefa64497f7eed1ffa8e669bea3`; rustc: `rustc 1.98.0 (88d9e12ae 2026-08-18)`; Criterion: `0.8.2`.
- Baseline commit: `b0d93a1b386aaec7222866fef24feeef6cf13475`; rustc: `rustc 1.97.1 (8bab26f4f 2026-07-14)`; Criterion: `0.8.2`.
- Benchmark harness SHA-256 prefixes: current `823f21999027`; baseline `7763d9a61040`.
- Benchmark harness hashes differ; verify that every shared name retains the same workload contract.

## Results

| Benchmark | Baseline | Current | Time reduction | Current vs baseline |
|:----------|---------:|--------:|---------------:|--------------------:|
| `chain_step_by_value` | 15.80 ns (15.74 ns - 15.87 ns) | 16.91 ns (16.90 ns - 16.93 ns) | -7.04% | 1.07x slower |
| `chain_step_delayed_no_plan` | 1.03 ns (1.03 ns - 1.03 ns) | 0.73 ns (0.73 ns - 0.73 ns) | +29.46% | 1.42x faster |
| `chain_step_mut_accept` | 12.68 ns (12.66 ns - 12.70 ns) | 13.69 ns (13.65 ns - 13.72 ns) | -7.96% | 1.08x slower |
| `chain_step_mut_reject_rollback` | 102.78 ns (102.21 ns - 103.47 ns) | 198.01 ns (197.44 ns - 198.35 ns) | -92.66% | 1.93x slower |
| `observing_manual_online_sum_100` | 934.48 ns (933.26 ns - 935.27 ns) | 1.28 µs (1.27 µs - 1.28 µs) | -36.45% | 1.36x slower |
| `observing_run_observing_buffer_100` | 1.56 µs (1.56 µs - 1.56 µs) | 1.78 µs (1.77 µs - 1.78 µs) | -14.02% | 1.14x slower |
| `observing_run_observing_into_binning_100` | 1.97 µs (1.96 µs - 1.97 µs) | 2.23 µs (2.23 µs - 2.23 µs) | -13.48% | 1.13x slower |
| `observing_run_observing_into_online_stats_100` | 1.25 µs (1.24 µs - 1.25 µs) | 1.60 µs (1.60 µs - 1.60 µs) | -28.42% | 1.28x slower |
| `sampler_run_by_value_100` | 1.50 µs (1.50 µs - 1.51 µs) | 1.54 µs (1.53 µs - 1.54 µs) | -2.15% | 1.02x slower |
| `sampler_run_mut_100` | 1.04 µs (1.04 µs - 1.04 µs) | 1.28 µs (1.28 µs - 1.28 µs) | -22.89% | 1.23x slower |

## Coverage Notes

Current-only rows without a saved baseline:

- `chain_step_delayed_accept_reflection`
- `chain_step_delayed_reject_reflection`
- `sampler_run_by_value_thinned_100/1`
- `sampler_run_by_value_thinned_100/16`
- `sampler_run_by_value_thinned_100/2`
- `sampler_run_delayed_reflection_100`
- `sampler_run_delayed_thinned_100/1`
- `sampler_run_delayed_thinned_100/16`
- `sampler_run_delayed_thinned_100/2`
- `sampler_run_mut_thinned_100/1`
- `sampler_run_mut_thinned_100/16`
- `sampler_run_mut_thinned_100/2`

Baseline-only rows without a current sample:

- `chain_step_delayed_accept_commit`
- `chain_step_delayed_reject_plan`
- `sampler_run_delayed_100`

## How to Update

```bash
just performance-local
just performance-github-assets
just performance-release
just performance-doc
just performance-readme
just performance-release <current-tag> <baseline-tag>
```

Generated Markdown, CSV measurements, and JSON provenance live under `target/bench-reports/`.
The curated release report is `docs/PERFORMANCE.md`.
Older curated reports are indexed under `docs/archive/performance/`.

See `docs/BENCHMARKING.md` for command semantics and reproducibility limits.

## Reproducibility Evidence

- [CSV measurements](archive/performance/v0.4.2-vs-v0.4.1.csv)
- [JSON provenance](archive/performance/v0.4.2-vs-v0.4.1.provenance.json)
