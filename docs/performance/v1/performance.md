<!-- research-repo-tools/performance-report/v1 current=v0.5.1 baseline=v0.5.0 -->
# MCMC benchmark timings

Statistic: median. Unit: ns.

Ratios describe timings; they do not establish statistical significance or scientific acceptance.

| Benchmark | Coverage | Baseline | Current | Baseline/current | Time reduction (%) |
| --- | --- | --- | --- | --- | --- |
| chain&#95;step&#95;by&#95;value | common | 16.841 ns [16.8248, 16.8776] (0.95 confidence) | 18.1579 ns [17.6604, 18.426] (0.95 confidence) | 0.927479 | -7.8192 |
| chain&#95;step&#95;delayed&#95;accept&#95;reflection | common | 11.3523 ns [11.1183, 11.4854] (0.95 confidence) | 11.1298 ns [10.9418, 11.2921] (0.95 confidence) | 1.01999 | 1.96005 |
| chain&#95;step&#95;delayed&#95;no&#95;plan | common | 0.778675 ns [0.768478, 0.800562] (0.95 confidence) | 0.779942 ns [0.765882, 0.793459] (0.95 confidence) | 0.998375 | -0.162723 |
| chain&#95;step&#95;delayed&#95;reject&#95;reflection | common | 11.358 ns [11.2565, 11.625] (0.95 confidence) | 11.4297 ns [11.1473, 11.5677] (0.95 confidence) | 0.993729 | -0.631097 |
| chain&#95;step&#95;mut&#95;accept | common | 13.7307 ns [13.6853, 13.7789] (0.95 confidence) | 14.8717 ns [14.3751, 15.1055] (0.95 confidence) | 0.923274 | -8.31024 |
| chain&#95;step&#95;mut&#95;reject&#95;rollback | common | 211.765 ns [207.992, 214.468] (0.95 confidence) | 217.209 ns [211.211, 220.952] (0.95 confidence) | 0.974937 | -2.57077 |
| observing&#95;manual&#95;online&#95;sum&#95;100 | common | 1382.36 ns [1363.68, 1401.71] (0.95 confidence) | 1363.82 ns [1334.29, 1400.46] (0.95 confidence) | 1.01359 | 1.34074 |
| observing&#95;run&#95;observing&#95;buffer&#95;100 | common | 1905.93 ns [1879.32, 1935.43] (0.95 confidence) | 1902.02 ns [1868.84, 1938.52] (0.95 confidence) | 1.00206 | 0.20542 |
| observing&#95;run&#95;observing&#95;into&#95;binning&#95;100 | common | 2481.99 ns [2424.85, 2531.32] (0.95 confidence) | 2420.43 ns [2367.64, 2478.18] (0.95 confidence) | 1.02543 | 2.48007 |
| observing&#95;run&#95;observing&#95;into&#95;online&#95;stats&#95;100 | common | 1693.05 ns [1658.57, 1730.43] (0.95 confidence) | 1643.84 ns [1638.81, 1653.2] (0.95 confidence) | 1.02994 | 2.90687 |
| sampler&#95;run&#95;by&#95;value&#95;100 | common | 1649.59 ns [1610.45, 1695.04] (0.95 confidence) | 1662.36 ns [1632.45, 1689.78] (0.95 confidence) | 0.99232 | -0.773988 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/1 | common | 1879.09 ns [1842.49, 1917.6] (0.95 confidence) | 1843.99 ns [1817.7, 1873.73] (0.95 confidence) | 1.01903 | 1.86794 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/16 | common | 1877.66 ns [1842.35, 1913.66] (0.95 confidence) | 1864.25 ns [1851.63, 1908.02] (0.95 confidence) | 1.00719 | 0.714262 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/2 | common | 1894.25 ns [1866.28, 1928.52] (0.95 confidence) | 1866.06 ns [1858.39, 1878.48] (0.95 confidence) | 1.0151 | 1.48782 |
| sampler&#95;run&#95;delayed&#95;reflection&#95;100 | common | 944.39 ns [922.456, 960.272] (0.95 confidence) | 902.976 ns [884.981, 917.641] (0.95 confidence) | 1.04586 | 4.38523 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/1 | common | 1123.69 ns [1106.05, 1139.05] (0.95 confidence) | 1126.74 ns [1110.58, 1144.36] (0.95 confidence) | 0.997294 | -0.271356 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/16 | common | 1119.91 ns [1103.33, 1133.92] (0.95 confidence) | 1113.17 ns [1092, 1127.18] (0.95 confidence) | 1.00605 | 0.601552 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/2 | common | 1109.2 ns [1088.1, 1131.86] (0.95 confidence) | 1117.84 ns [1092.01, 1148.16] (0.95 confidence) | 0.992266 | -0.779393 |
| sampler&#95;run&#95;mut&#95;100 | common | 1355.18 ns [1340.85, 1384.39] (0.95 confidence) | 1298 ns [1292.99, 1306.54] (0.95 confidence) | 1.04405 | 4.21942 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/1 | common | 1142.56 ns [1120.16, 1155.01] (0.95 confidence) | 1119.91 ns [1108.27, 1130.51] (0.95 confidence) | 1.02023 | 1.98251 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/16 | common | 1116.99 ns [1103.56, 1142.26] (0.95 confidence) | 1094.35 ns [1081.53, 1119.63] (0.95 confidence) | 1.02069 | 2.02709 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/2 | common | 1130.42 ns [1108.39, 1154.45] (0.95 confidence) | 1117.33 ns [1102.34, 1132.15] (0.95 confidence) | 1.01171 | 1.15756 |

Current: **v0.5.1**. Baseline: **v0.5.0**.

Recorded intervals are marginal timing intervals, not paired ratio intervals. Missing provenance stays unknown.

<!-- rumdl-disable MD041 -->

These fixed-seed `stepping` workloads measure transition and observation overhead.
They do not establish convergence, mixing, effective sample size, or scientific efficiency.
Ratios are baseline time divided by current time; values above one mean lower current time.
Marginal timing bounds do not establish a paired ratio interval or statistical significance.

Review common names against the lifecycle contracts in `docs/BENCHMARKING.md`, especially
when harness fingerprints differ. Added and removed names are coverage changes.
Local measurements require matching known host identities. Release-asset comparisons
need separate hardware and workload review; GitHub runners can change between releases.
Source and harness fingerprints come from the shared measurement workflow.

## Baseline provenance

- Revision: `83f729c62dd959dc9afa2ca9c4146b0515465fae`
- Source fingerprint: `6979500b7e7e8b0d9d174654c8f118cbfddc1acbd30efa5957b69566adfca104`
- Harness fingerprint: `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`
- architecture: arm64
- command: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;
- cpu: Apple M4 Max (arm64)
- dependency.criterion: 0.8.2
- fingerprint-schema: research-repo-tools/files/v1
- harness-inventory: &#91;&quot;benches/stepping.rs&quot;&#93;
- mode: tag
- os: macOS-27.0.1-arm64-arm-64bit-Mach-O
- release: v0.5.0
- scope: release-signal
- source-inventory: &#91;&quot;Cargo.lock&quot;, &quot;Cargo.toml&quot;, &quot;benches/stepping.rs&quot;, &quot;rust-toolchain.toml&quot;, &quot;src/adaptive.rs&quot;, &quot;src/autocorrelation.rs&quot;, &quot;src/benchmarks.rs&quot;, &quot;src/chain.rs&quot;, &quot;src/continuous&#95;testing.rs&quot;, &quot;src/convergence.rs&quot;, &quot;src/diagnostics.rs&quot;, &quot;src/error.rs&quot;, &quot;src/lib.rs&quot;, &quot;src/numerics.rs&quot;, &quot;src/observable.rs&quot;, &quot;src/sampler.rs&quot;, &quot;src/statistics.rs&quot;, &quot;src/testing.rs&quot;, &quot;src/traits.rs&quot;&#93;
- suite: stepping
- tool.rustc: &quot;rustc 1.98.1 (48a229cea 2026-09-01)&#92;nbinary: rustc&#92;ncommit-hash: 48a229ceaefd4985c50990b14116b6d856af0985&#92;ncommit-date: 2026-09-01&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.98.1&#92;nLLVM version: 22.1.8&quot;

## Current provenance

- Revision: `9fc2479167aaa47c7bbe3ef5e43eb76393c9dc89`
- Source fingerprint: `43bcf4fd3e302cb234ad2327c83340c5ea648a3e9bbc784e772625f1979c8e27`
- Harness fingerprint: `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`
- architecture: arm64
- command: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;
- cpu: Apple M4 Max (arm64)
- dependency.criterion: 0.8.2
- fingerprint-schema: research-repo-tools/files/v1
- harness-inventory: &#91;&quot;benches/stepping.rs&quot;&#93;
- mode: working-tree
- os: macOS-27.0.1-arm64-arm-64bit-Mach-O
- release: v0.5.1
- scope: release-signal
- source-inventory: &#91;&quot;Cargo.lock&quot;, &quot;Cargo.toml&quot;, &quot;benches/stepping.rs&quot;, &quot;rust-toolchain.toml&quot;, &quot;src/adaptive.rs&quot;, &quot;src/autocorrelation.rs&quot;, &quot;src/benchmarks.rs&quot;, &quot;src/chain.rs&quot;, &quot;src/continuous&#95;testing.rs&quot;, &quot;src/convergence.rs&quot;, &quot;src/diagnostics.rs&quot;, &quot;src/error.rs&quot;, &quot;src/ess.rs&quot;, &quot;src/lib.rs&quot;, &quot;src/numerics.rs&quot;, &quot;src/observable.rs&quot;, &quot;src/ranks.rs&quot;, &quot;src/sampler.rs&quot;, &quot;src/statistics.rs&quot;, &quot;src/testing.rs&quot;, &quot;src/traits.rs&quot;&#93;
- suite: stepping
- tool.rustc: &quot;rustc 1.98.1 (48a229cea 2026-09-01)&#92;nbinary: rustc&#92;ncommit-hash: 48a229ceaefd4985c50990b14116b6d856af0985&#92;ncommit-date: 2026-09-01&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.98.1&#92;nLLVM version: 22.1.8&quot;

- [Comparison evidence](v0.5.1-vs-v0.5.0.comparison.json)
- [Provenance](v0.5.1-vs-v0.5.0.evidence.json)
- [CSV export](v0.5.1-vs-v0.5.0.csv)
