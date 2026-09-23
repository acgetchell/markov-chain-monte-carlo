<!-- research-repo-tools/performance-report/v1 current=v0.4.2 baseline=v0.4.1 -->
# MCMC benchmark timings

Statistic: median. Unit: ns.

Ratios describe timings; they do not establish statistical significance or scientific acceptance.

| Benchmark | Coverage | Baseline | Current | Baseline/current | Time reduction (%) |
| --- | --- | --- | --- | --- | --- |
| chain&#95;step&#95;by&#95;value | common | 15.7995 ns [15.7427, 15.8675] | 16.9117 ns [16.8967, 16.9283] | 0.934236 | -7.03929 |
| chain&#95;step&#95;delayed&#95;accept&#95;commit | missing | 7.7726 ns [7.75692, 7.77912] | — | — | — |
| chain&#95;step&#95;delayed&#95;accept&#95;reflection | added | — | 10.462 ns [10.4354, 10.4732] | — | — |
| chain&#95;step&#95;delayed&#95;no&#95;plan | common | 1.02911 ns [1.02716, 1.03094] | 0.725958 ns [0.725128, 0.727456] | 1.41759 | 29.4579 |
| chain&#95;step&#95;delayed&#95;reject&#95;plan | missing | 7.84949 ns [7.83786, 7.85866] | — | — | — |
| chain&#95;step&#95;delayed&#95;reject&#95;reflection | added | — | 10.5335 ns [10.5256, 10.5486] | — | — |
| chain&#95;step&#95;mut&#95;accept | common | 12.6828 ns [12.6572, 12.6963] | 13.6922 ns [13.6499, 13.723] | 0.926277 | -7.95906 |
| chain&#95;step&#95;mut&#95;reject&#95;rollback | common | 102.776 ns [102.209, 103.466] | 198.012 ns [197.44, 198.349] | 0.519041 | -92.6629 |
| observing&#95;manual&#95;online&#95;sum&#95;100 | common | 934.479 ns [933.262, 935.265] | 1275.09 ns [1272.17, 1278.6] | 0.732875 | -36.449 |
| observing&#95;run&#95;observing&#95;buffer&#95;100 | common | 1558.29 ns [1556.63, 1560.17] | 1776.71 ns [1774.8, 1778.86] | 0.877066 | -14.0165 |
| observing&#95;run&#95;observing&#95;into&#95;binning&#95;100 | common | 1965.72 ns [1962.92, 1970.09] | 2230.77 ns [2228.33, 2233.73] | 0.881186 | -13.4835 |
| observing&#95;run&#95;observing&#95;into&#95;online&#95;stats&#95;100 | common | 1245.66 ns [1243.69, 1246.62] | 1599.62 ns [1597.39, 1601.5] | 0.77872 | -28.4159 |
| sampler&#95;run&#95;by&#95;value&#95;100 | common | 1502.86 ns [1500.48, 1506.02] | 1535.17 ns [1531.44, 1538.02] | 0.97895 | -2.15021 |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/1 | added | — | 1778.65 ns [1777.3, 1781.36] | — | — |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/16 | added | — | 1779.81 ns [1776.6, 1781.51] | — | — |
| sampler&#95;run&#95;by&#95;value&#95;thinned&#95;100/2 | added | — | 1792.87 ns [1789.06, 1799.81] | — | — |
| sampler&#95;run&#95;delayed&#95;100 | missing | 918.554 ns [917.93, 919.296] | — | — | — |
| sampler&#95;run&#95;delayed&#95;reflection&#95;100 | added | — | 862.45 ns [861.502, 863.049] | — | — |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/1 | added | — | 1049.5 ns [1048.75, 1050.01] | — | — |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/16 | added | — | 1051.15 ns [1047.79, 1055.7] | — | — |
| sampler&#95;run&#95;delayed&#95;thinned&#95;100/2 | added | — | 1048.91 ns [1047.36, 1050.05] | — | — |
| sampler&#95;run&#95;mut&#95;100 | common | 1038.86 ns [1036.02, 1040.4] | 1276.7 ns [1276.11, 1278.05] | 0.813708 | -22.8942 |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/1 | added | — | 1048.26 ns [1046.97, 1049.46] | — | — |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/16 | added | — | 1044.76 ns [1043.23, 1045.82] | — | — |
| sampler&#95;run&#95;mut&#95;thinned&#95;100/2 | added | — | 1044.56 ns [1043.78, 1045.5] | — | — |

Current: **v0.4.2**. Baseline: **v0.4.1**.

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
Unknown historical provenance stays unknown. Legacy fingerprints retain their original
meaning in the evidence's `legacy.*` context and are not shared-format fingerprints.

## Baseline provenance

- Revision: `b0d93a1b386aaec7222866fef24feeef6cf13475`
- Source fingerprint: `unknown`
- Harness fingerprint: `unknown`
- legacy.configuration-sha256: b06d52c6c86f8bca2ef2ffd44e94293a3355848d2a5995df5451b0e0d1e7bd35
- legacy.manifest: {&quot;csv&#95;schema&quot;: &quot;criterion-comparison/v1&quot;, &quot;csv&#95;sha256&quot;: &quot;4a8dbd2e6c647446dea2a677f6d04b771dee5b68516f9b408dbb5be39c9dc26c&quot;, &quot;measurement&quot;: {&quot;baseline&quot;: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;7763d9a61040885ae7b8092526c79d0a83609c775d73bae0495aa2e6cd404f0c&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;c1d15fbfc462f9fc15453f315581c037fb75c44c721e2735e01772591b215882&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;, &quot;--&quot;, &quot;--save-baseline&quot;, &quot;v0.4.1&quot;&#93;, &quot;commit&quot;: &quot;b0d93a1b386aaec7222866fef24feeef6cf13475&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.97.1 (8bab26f4f 2026-07-14)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5665918d221294cf42b842f5395724a7aaca107bc249d3340cecb716bfa832ee&quot;, &quot;tag&quot;: &quot;v0.4.1&quot;}, &quot;current&quot;: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;823f21999027643203fd69e8dd9401efc1d8ca4794de07883fee407419e3b9eb&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;6ecf887d50b749f6becdc268a9c97ddde7bfb7eeaf5b093c1e0ec60fcde0e0de&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;, &quot;commit&quot;: &quot;ef21cc6430329cefa64497f7eed1ffa8e669bea3&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.98.0 (88d9e12ae 2026-08-18)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5e8feb078b9185bce3511ef0ec1c12ae1b90deb8f23a89c548ec47728a48e502&quot;, &quot;tag&quot;: &quot;v0.4.2&quot;}, &quot;mode&quot;: &quot;local-isolated-worktrees&quot;, &quot;working&#95;tree&#95;applied&quot;: true}, &quot;release&quot;: {&quot;baseline&#95;tag&quot;: &quot;v0.4.1&quot;, &quot;current&#95;tag&quot;: &quot;v0.4.2&quot;}, &quot;report&quot;: {&quot;baseline&#95;label&quot;: &quot;v0.4.1&quot;, &quot;current&#95;label&quot;: &quot;v0.4.2 working tree&quot;, &quot;measurement&#95;context&quot;: &#91;&quot;Source mode: same-host isolated worktrees; current &#96;HEAD&#96; with tracked and untracked working-tree changes applied.&quot;, &quot;Host: &#96;macOS-26.6.2-arm64-arm-64bit-Mach-O&#96; on &#96;arm64&#96;; CPU: &#96;Apple M4 Max&#96;.&quot;, &quot;Current commit: &#96;ef21cc6430329cefa64497f7eed1ffa8e669bea3&#96;; rustc: &#96;rustc 1.98.0 (88d9e12ae 2026-08-18)&#96;; Criterion: &#96;0.8.2&#96;.&quot;, &quot;Baseline commit: &#96;b0d93a1b386aaec7222866fef24feeef6cf13475&#96;; rustc: &#96;rustc 1.97.1 (8bab26f4f 2026-07-14)&#96;; Criterion: &#96;0.8.2&#96;.&quot;, &quot;Benchmark harness SHA-256 prefixes: current &#96;823f21999027&#96;; baseline &#96;7763d9a61040&#96;.&quot;, &quot;Benchmark harness hashes differ; verify that every shared name retains the same workload contract.&quot;&#93;, &quot;revision&quot;: &quot;ef21cc6&quot;, &quot;statistic&quot;: &quot;median&quot;}, &quot;schema&quot;: &quot;mcmc-performance-provenance/v2&quot;}
- legacy.manifest-sha256: e9de4d014742791e22a9d9a020ca46c06fb72bcb73d052f689ed8885b738f4f8
- legacy.payload-sha256: 4a8dbd2e6c647446dea2a677f6d04b771dee5b68516f9b408dbb5be39c9dc26c
- legacy.record: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;7763d9a61040885ae7b8092526c79d0a83609c775d73bae0495aa2e6cd404f0c&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;c1d15fbfc462f9fc15453f315581c037fb75c44c721e2735e01772591b215882&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;, &quot;--&quot;, &quot;--save-baseline&quot;, &quot;v0.4.1&quot;&#93;, &quot;commit&quot;: &quot;b0d93a1b386aaec7222866fef24feeef6cf13475&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.97.1 (8bab26f4f 2026-07-14)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5665918d221294cf42b842f5395724a7aaca107bc249d3340cecb716bfa832ee&quot;, &quot;tag&quot;: &quot;v0.4.1&quot;}
- legacy.schema: mcmc-performance-provenance/v2
- release: v0.4.1

## Current provenance

- Revision: `ef21cc6430329cefa64497f7eed1ffa8e669bea3`
- Source fingerprint: `unknown`
- Harness fingerprint: `unknown`
- legacy.configuration-sha256: b06d52c6c86f8bca2ef2ffd44e94293a3355848d2a5995df5451b0e0d1e7bd35
- legacy.manifest: {&quot;csv&#95;schema&quot;: &quot;criterion-comparison/v1&quot;, &quot;csv&#95;sha256&quot;: &quot;4a8dbd2e6c647446dea2a677f6d04b771dee5b68516f9b408dbb5be39c9dc26c&quot;, &quot;measurement&quot;: {&quot;baseline&quot;: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;7763d9a61040885ae7b8092526c79d0a83609c775d73bae0495aa2e6cd404f0c&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;c1d15fbfc462f9fc15453f315581c037fb75c44c721e2735e01772591b215882&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;, &quot;--&quot;, &quot;--save-baseline&quot;, &quot;v0.4.1&quot;&#93;, &quot;commit&quot;: &quot;b0d93a1b386aaec7222866fef24feeef6cf13475&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.97.1 (8bab26f4f 2026-07-14)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5665918d221294cf42b842f5395724a7aaca107bc249d3340cecb716bfa832ee&quot;, &quot;tag&quot;: &quot;v0.4.1&quot;}, &quot;current&quot;: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;823f21999027643203fd69e8dd9401efc1d8ca4794de07883fee407419e3b9eb&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;6ecf887d50b749f6becdc268a9c97ddde7bfb7eeaf5b093c1e0ec60fcde0e0de&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;, &quot;commit&quot;: &quot;ef21cc6430329cefa64497f7eed1ffa8e669bea3&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.98.0 (88d9e12ae 2026-08-18)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5e8feb078b9185bce3511ef0ec1c12ae1b90deb8f23a89c548ec47728a48e502&quot;, &quot;tag&quot;: &quot;v0.4.2&quot;}, &quot;mode&quot;: &quot;local-isolated-worktrees&quot;, &quot;working&#95;tree&#95;applied&quot;: true}, &quot;release&quot;: {&quot;baseline&#95;tag&quot;: &quot;v0.4.1&quot;, &quot;current&#95;tag&quot;: &quot;v0.4.2&quot;}, &quot;report&quot;: {&quot;baseline&#95;label&quot;: &quot;v0.4.1&quot;, &quot;current&#95;label&quot;: &quot;v0.4.2 working tree&quot;, &quot;measurement&#95;context&quot;: &#91;&quot;Source mode: same-host isolated worktrees; current &#96;HEAD&#96; with tracked and untracked working-tree changes applied.&quot;, &quot;Host: &#96;macOS-26.6.2-arm64-arm-64bit-Mach-O&#96; on &#96;arm64&#96;; CPU: &#96;Apple M4 Max&#96;.&quot;, &quot;Current commit: &#96;ef21cc6430329cefa64497f7eed1ffa8e669bea3&#96;; rustc: &#96;rustc 1.98.0 (88d9e12ae 2026-08-18)&#96;; Criterion: &#96;0.8.2&#96;.&quot;, &quot;Baseline commit: &#96;b0d93a1b386aaec7222866fef24feeef6cf13475&#96;; rustc: &#96;rustc 1.97.1 (8bab26f4f 2026-07-14)&#96;; Criterion: &#96;0.8.2&#96;.&quot;, &quot;Benchmark harness SHA-256 prefixes: current &#96;823f21999027&#96;; baseline &#96;7763d9a61040&#96;.&quot;, &quot;Benchmark harness hashes differ; verify that every shared name retains the same workload contract.&quot;&#93;, &quot;revision&quot;: &quot;ef21cc6&quot;, &quot;statistic&quot;: &quot;median&quot;}, &quot;schema&quot;: &quot;mcmc-performance-provenance/v2&quot;}
- legacy.manifest-sha256: e9de4d014742791e22a9d9a020ca46c06fb72bcb73d052f689ed8885b738f4f8
- legacy.payload-sha256: 4a8dbd2e6c647446dea2a677f6d04b771dee5b68516f9b408dbb5be39c9dc26c
- legacy.record: {&quot;architecture&quot;: &quot;arm64&quot;, &quot;benchmark&#95;harness&#95;sha256&quot;: &quot;823f21999027643203fd69e8dd9401efc1d8ca4794de07883fee407419e3b9eb&quot;, &quot;cargo&#95;lock&#95;sha256&quot;: &quot;6ecf887d50b749f6becdc268a9c97ddde7bfb7eeaf5b093c1e0ec60fcde0e0de&quot;, &quot;command&quot;: &#91;&quot;cargo&quot;, &quot;bench&quot;, &quot;--locked&quot;, &quot;--bench&quot;, &quot;stepping&quot;&#93;, &quot;commit&quot;: &quot;ef21cc6430329cefa64497f7eed1ffa8e669bea3&quot;, &quot;cpu&#95;model&quot;: &quot;Apple M4 Max&quot;, &quot;criterion&#95;version&quot;: &quot;0.8.2&quot;, &quot;operating&#95;system&quot;: &quot;macOS-26.6.2-arm64-arm-64bit-Mach-O&quot;, &quot;rustc&quot;: &quot;rustc 1.98.0 (88d9e12ae 2026-08-18)&quot;, &quot;source&#95;digest&#95;sha256&quot;: &quot;5e8feb078b9185bce3511ef0ec1c12ae1b90deb8f23a89c548ec47728a48e502&quot;, &quot;tag&quot;: &quot;v0.4.2&quot;}
- legacy.schema: mcmc-performance-provenance/v2
- release: v0.4.2

- [Comparison evidence](v0.4.2-vs-v0.4.1.comparison.json)
- [Provenance](v0.4.2-vs-v0.4.1.evidence.json)
- [CSV export](v0.4.2-vs-v0.4.1.csv)
