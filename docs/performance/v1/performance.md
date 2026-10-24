<!-- research-repo-tools/performance-report/v1 current=v0.5.2 baseline=v0.5.1 -->
# MCMC benchmark timings

Statistic: median. Unit: ns.

Ratios describe timings; they do not establish statistical significance or scientific acceptance.

| Benchmark | Coverage | Baseline | Current | Baseline/current | Time reduction (%) |
| --- | --- | --- | --- | --- | --- |
| chain&#95;step&#95;by&#95;value | common | 16.7305 ns [16.6881, 16.7601] (0.95 confidence) | 16.8223 ns [16.8063, 16.8472] (0.95 confidence) | 0.994541 | -0.548916 |
| chain&#95;step&#95;delayed&#95;accept&#95;reflection | common | 10.5007 ns [10.4679, 10.5179] (0.95 confidence) | 10.4465 ns [10.4261, 10.4882] (0.95 confidence) | 1.00519 | 0.516515 |
| chain&#95;step&#95;delayed&#95;no&#95;plan | common | 0.726462 ns [0.726029, 0.727028] (0.95 confidence) | 0.725055 ns [0.724612, 0.725611] (0.95 confidence) | 1.00194 | 0.193617 |
| chain&#95;step&#95;delayed&#95;reject&#95;reflection | common | 10.7442 ns [10.7239, 10.7782] (0.95 confidence) | 10.7075 ns [10.6851, 10.7249] (0.95 confidence) | 1.00343 | 0.341801 |
| chain&#95;step&#95;mut&#95;accept | common | 13.7211 ns [13.6794, 13.7666] (0.95 confidence) | 13.8022 ns [13.7574, 13.847] (0.95 confidence) | 0.994117 | -0.591771 |
| chain&#95;step&#95;mut&#95;reject&#95;rollback | common | 194.155 ns [193.632, 194.8] (0.95 confidence) | 198.85 ns [197.611, 199.265] (0.95 confidence) | 0.976388 | -2.41826 |
| observing&#95;manual&#95;online&#95;sum&#95;100 | common | 1275.03 ns [1271.02, 1277.33] (0.95 confidence) | 1270.34 ns [1267.26, 1273.32] (0.95 confidence) | 1.00369 | 0.368075 |
| observing&#95;run&#95;observing&#95;buffer&#95;100 | common | 1720.73 ns [1718.93, 1726.78] (0.95 confidence) | 1716.39 ns [1713.72, 1720.61] (0.95 confidence) | 1.00253 | 0.252168 |
| observing&#95;run&#95;observing&#95;into&#95;binning&#95;100 | common | 2288.59 ns [2283.05, 2291.33] (0.95 confidence) | 2293.59 ns [2290.89, 2299] (0.95 confidence) | 0.997821 | -0.218398 |
| observing&#95;run&#95;observing&#95;into&#95;online&#95;stats&#95;100 | common | 1456.04 ns [1453.31, 1460.85] (0.95 confidence) | 1464.02 ns [1461.02, 1467.01] (0.95 confidence) | 0.994546 | -0.54838 |
| sampler&#95;run&#95;by&#95;value&#95;100 | common | 1670.7 ns [1668.59, 1671.87] (0.95 confidence) | 1680.13 ns [1678.13, 1681.6] (0.95 confidence) | 0.994388 | -0.564403 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/1 | common | 1732.28 ns [1728.33, 1740.27] (0.95 confidence) | 1717.99 ns [1714.32, 1721.54] (0.95 confidence) | 1.00832 | 0.825035 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/16 | common | 1718.25 ns [1714.13, 1723.24] (0.95 confidence) | 1719.69 ns [1714.18, 1722.68] (0.95 confidence) | 0.999159 | -0.0841673 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/2 | common | 1722.37 ns [1719.71, 1724.48] (0.95 confidence) | 1724.98 ns [1720.44, 1728.08] (0.95 confidence) | 0.99849 | -0.151262 |
| sampler&#95;run&#95;delayed&#95;reflection&#95;100 | common | 876.103 ns [873.82, 877.716] (0.95 confidence) | 877.842 ns [876.725, 879.467] (0.95 confidence) | 0.998018 | -0.198553 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/1 | common | 1052.38 ns [1050.76, 1053.02] (0.95 confidence) | 1048.86 ns [1047.73, 1050.42] (0.95 confidence) | 1.00336 | 0.334476 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/16 | common | 1044.62 ns [1043.77, 1045.35] (0.95 confidence) | 1043.82 ns [1041.96, 1044.97] (0.95 confidence) | 1.00077 | 0.0766807 |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/2 | common | 1045.05 ns [1043.04, 1047.66] (0.95 confidence) | 1053.92 ns [1050.52, 1057.98] (0.95 confidence) | 0.991587 | -0.848457 |
| sampler&#95;run&#95;mut&#95;100 | common | 1302.16 ns [1300.06, 1304.63] (0.95 confidence) | 1307.07 ns [1305.38, 1308.89] (0.95 confidence) | 0.996246 | -0.37681 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/1 | common | 1054.8 ns [1053.57, 1056.64] (0.95 confidence) | 1054.83 ns [1053.87, 1055.66] (0.95 confidence) | 0.999971 | -0.00288052 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/16 | common | 1037.26 ns [1035.98, 1040.76] (0.95 confidence) | 1036.54 ns [1035.63, 1037.52] (0.95 confidence) | 1.0007 | 0.0698236 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/2 | common | 1057.21 ns [1054.98, 1060.2] (0.95 confidence) | 1042.29 ns [1041.57, 1043.48] (0.95 confidence) | 1.01432 | 1.4113 |

Current: **v0.5.2**. Baseline: **v0.5.1**.

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

- Revision: `21c77b32163581ccb830c4ff79c5ec25fdacc69a`
- Source fingerprint: `43bcf4fd3e302cb234ad2327c83340c5ea648a3e9bbc784e772625f1979c8e27`
- Harness fingerprint: `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`
- architecture: arm64
- command: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;
- cpu: Apple M4 Max (arm64)
- dependency.criterion: 0.8.2
- fingerprint-schema: research-repo-tools/files/v1
- harness-inventory: &#91;&quot;benches/stepping.rs&quot;&#93;
- host: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;cpu&quot;: &quot;Apple M4 Max (arm64)&quot;, &quot;logical&#95;threads&quot;: 16, &quot;memory&#95;bytes&quot;: 68719476736, &quot;os&quot;: &quot;macOS-27.0.1-arm64-arm-64bit-Mach-O&quot;, &quot;physical&#95;cores&quot;: 16, &quot;schema&quot;: &quot;research-repo-tools/host/v1&quot;, &quot;tools&quot;: {&quot;rustc&quot;: &quot;rustc 1.99.1 (58cf8f9c3 2026-10-13)&#92;nbinary: rustc&#92;ncommit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67&#92;ncommit-date: 2026-10-13&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.99.1&#92;nLLVM version: 23.1.1&quot;}}
- host.logical&#95;threads: 16
- host.memory&#95;bytes: 68719476736
- host.physical&#95;cores: 16
- mode: tag
- os: macOS-27.0.1-arm64-arm-64bit-Mach-O
- release: v0.5.1
- scope: release-signal
- source-inventory: &#91;&quot;Cargo.lock&quot;, &quot;Cargo.toml&quot;, &quot;benches/stepping.rs&quot;, &quot;rust-toolchain.toml&quot;, &quot;src/adaptive.rs&quot;, &quot;src/autocorrelation.rs&quot;, &quot;src/benchmarks.rs&quot;, &quot;src/chain.rs&quot;, &quot;src/continuous&#95;testing.rs&quot;, &quot;src/convergence.rs&quot;, &quot;src/diagnostics.rs&quot;, &quot;src/error.rs&quot;, &quot;src/ess.rs&quot;, &quot;src/lib.rs&quot;, &quot;src/numerics.rs&quot;, &quot;src/observable.rs&quot;, &quot;src/ranks.rs&quot;, &quot;src/sampler.rs&quot;, &quot;src/statistics.rs&quot;, &quot;src/testing.rs&quot;, &quot;src/traits.rs&quot;&#93;
- suite: stepping
- tool.rustc: &quot;rustc 1.99.1 (58cf8f9c3 2026-10-13)&#92;nbinary: rustc&#92;ncommit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67&#92;ncommit-date: 2026-10-13&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.99.1&#92;nLLVM version: 23.1.1&quot;

## Current provenance

- Revision: `69ece452919c88ed4369d3e3abc4bfd4264d0f17`
- Source fingerprint: `f1c62ac99e6f8b5b6a36ffdde278fa036057cc867ab6913a5c7f4b32b5146307`
- Harness fingerprint: `a5fdc42d4aba98ad977398bfc01fae799cdd6f0d19dbc1adc0e969b298650b9c`
- architecture: arm64
- command: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;
- cpu: Apple M4 Max (arm64)
- dependency.criterion: 0.8.2
- fingerprint-schema: research-repo-tools/files/v1
- harness-inventory: &#91;&quot;benches/stepping.rs&quot;&#93;
- host: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;cpu&quot;: &quot;Apple M4 Max (arm64)&quot;, &quot;logical&#95;threads&quot;: 16, &quot;memory&#95;bytes&quot;: 68719476736, &quot;os&quot;: &quot;macOS-27.0.1-arm64-arm-64bit-Mach-O&quot;, &quot;physical&#95;cores&quot;: 16, &quot;schema&quot;: &quot;research-repo-tools/host/v1&quot;, &quot;tools&quot;: {&quot;rustc&quot;: &quot;rustc 1.99.1 (58cf8f9c3 2026-10-13)&#92;nbinary: rustc&#92;ncommit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67&#92;ncommit-date: 2026-10-13&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.99.1&#92;nLLVM version: 23.1.1&quot;}}
- host.logical&#95;threads: 16
- host.memory&#95;bytes: 68719476736
- host.physical&#95;cores: 16
- mode: working-tree
- os: macOS-27.0.1-arm64-arm-64bit-Mach-O
- release: v0.5.2
- scope: release-signal
- source-inventory: &#91;&quot;Cargo.lock&quot;, &quot;Cargo.toml&quot;, &quot;benches/stepping.rs&quot;, &quot;rust-toolchain.toml&quot;, &quot;src/adaptive.rs&quot;, &quot;src/autocorrelation.rs&quot;, &quot;src/benchmarks.rs&quot;, &quot;src/chain.rs&quot;, &quot;src/continuous&#95;testing.rs&quot;, &quot;src/convergence.rs&quot;, &quot;src/diagnostics.rs&quot;, &quot;src/error.rs&quot;, &quot;src/ess.rs&quot;, &quot;src/lib.rs&quot;, &quot;src/numerics.rs&quot;, &quot;src/observable.rs&quot;, &quot;src/ranks.rs&quot;, &quot;src/sampler.rs&quot;, &quot;src/statistics.rs&quot;, &quot;src/testing.rs&quot;, &quot;src/traits.rs&quot;&#93;
- suite: stepping
- tool.rustc: &quot;rustc 1.99.1 (58cf8f9c3 2026-10-13)&#92;nbinary: rustc&#92;ncommit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67&#92;ncommit-date: 2026-10-13&#92;nhost: aarch64-apple-darwin&#92;nrelease: 1.99.1&#92;nLLVM version: 23.1.1&quot;

- [Comparison evidence](v0.5.2-vs-v0.5.1.comparison.json)
- [Provenance](v0.5.2-vs-v0.5.1.evidence.json)
- [CSV export](v0.5.2-vs-v0.5.1.csv)
