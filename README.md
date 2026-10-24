# markov-chain-monte-carlo

[![DOI](https://badgen.net/badge/DOI/10.5281%2Fzenodo.20033111/blue)](https://doi.org/10.5281/zenodo.20033111)
[![Crates.io](https://badgen.net/crates/v/markov-chain-monte-carlo)](https://crates.io/crates/markov-chain-monte-carlo)
[![Downloads](https://badgen.net/crates/d/markov-chain-monte-carlo)](https://crates.io/crates/markov-chain-monte-carlo)
[![License](https://badgen.net/github/license/acgetchell/markov-chain-monte-carlo)](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/LICENSE)
[![Docs.rs](https://docs.rs/markov-chain-monte-carlo/badge.svg)](https://docs.rs/markov-chain-monte-carlo)
[![CI](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/ci.yml)
[![CodeQL](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/codeql.yml)
[![zizmor](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/zizmor.yml)
[![rust-clippy analyze](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo/graph/badge.svg)](https://codecov.io/gh/acgetchell/markov-chain-monte-carlo)
[![Audit dependencies](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/audit.yml)
[![OSV-Scanner](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/osv.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/osv.yml)
[![Gitleaks](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/acgetchell/markov-chain-monte-carlo/actions/workflows/gitleaks.yml)

![Ising energy trace](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/baef8747db4d9afa4f77dd3229f9548fdd7faa30/docs/assets/ising_energy_trace.png)

_Single-chain illustration: open-boundary 1-D Ising with 50 spins, β = 0.5, J = 1, seed 42, 5,000 burn-in steps, and 20,000 recorded steps. The current
example extends this to four chains. `just notebook-check`
regenerates `target/ising_1d_trace.csv`, the executed notebook, and the PNG under `target/notebooks/`; `just notebook-ising-figure` promotes that exact PNG to
the tracked image above._

Research-oriented Metropolis-Hastings tools in Rust for ordinary numeric states, large combinatorial state spaces, and proposal implementations that need
rollback-safe mutation or delayed commits.

## Contents

- [Introduction](#-introduction)
- [Use this crate when](#-use-this-crate-when)
- [Quick start](#-quick-start)
- [API at a glance](#-api-at-a-glance)
- [Features](#-features)
- [Cargo features](#-cargo-features)
- [Scientific basis](#-scientific-basis)
- [Validation model](#-validation-model)
- [Documentation map](#-documentation-map)
- [Examples](#-examples)
- [Benchmarking](#-benchmarking)
- [Ecosystem](#-ecosystem)
- [Limitations and roadmap](#-limitations-and-roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [References](#-references)
- [AI Agents](#-ai-agents)
- [License](#-license)

## 📐 Introduction

This library implements composable Metropolis-Hastings sampling in Rust for application-specific state spaces, proposals, and observables. Domain code
owns the model and proposal kernel; the sampler owns transition bookkeeping, acceptance decisions, and measurement workflows.

Targets return unnormalized natural log weights. Proposals describe the same concrete transition they generate, with proposal asymmetry in the Hastings
correction. Numeric examples, spin systems, and triangulation moves use this same contract.

🚧 **Pre-release (0.x)** — This is research software under active development. APIs may change before 1.0.
API links point to the latest published documentation. Run `just doc` for the API reference matching this checkout.

## ✅ Use this crate when

- Application-specific states and proposals need reusable Metropolis-Hastings transition mechanics.
- Composable log weights represent model, bias, energy/action, or externally supplied regularizer terms.
- Large states benefit from in-place rollback or delayed commits.
- Long runs need streaming statistics, numeric traces, thinning, or validated checkpoints.
- Proposal development needs explicit Hastings ratios and empirical transition checks.

Proposal correctness, irreducibility, aperiodicity, equilibration, and scientific interpretation remain application responsibilities.

## 🚀 Quick start

Add the library and `rand` 0.10 to your crate:

```bash
cargo add markov-chain-monte-carlo
cargo add rand@0.10
```

The example imports `StdRng` and its traits from `rand`, so your application needs
to declare `rand` directly. You create and seed the generator, then pass it to the
sampler to control its randomness.

Enable checkpoint serialization when needed:

```bash
cargo add markov-chain-monte-carlo --features serde
```

Rust 1.99.1 or newer is required.

Minimal by-value Metropolis-Hastings sampler. This example demonstrates the transition mechanics; convergence assessment remains a separate analysis step.

```rust
use markov_chain_monte_carlo::prelude::by_value::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

#[derive(Clone)]
struct Scalar(f64);

struct Normal;
impl Target<Scalar> for Normal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

struct RandomWalk {
    width: f64,
}
impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta = rng.random_range(-self.width..self.width);
        Scalar(current.0 + delta)
    }
}

fn main() -> Result<(), McmcError> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut chain = Chain::new(Scalar(0.0), &Normal)?;
    let proposal = RandomWalk { width: 1.0 };

    for _ in 0..1000 {
        let _ = chain.step(&Normal, &proposal, &mut rng)?;
    }

    assert!(chain.acceptance_rate() > 0.0);
    Ok(())
}
```

To run the repository's examples, follow the [environment setup guidance](#-contributing), then run these commands from the checkout:

```bash
just example ising_1d  # Run the one-dimensional Ising example
just examples         # Run all examples
just                  # List all public recipes, arguments, and descriptions
```

<a id="-choosing-an-api"></a>

## 🧭 API at a glance

| Need | Start here |
| --- | --- |
| Small states returned by value | [`Proposal`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/trait.Proposal.html) and `Chain::step` |
| Expensive state copies, with reliable rollback | [`ProposalMut`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/trait.ProposalMut.html) and `Chain::step_mut` |
| Score a concrete move before mutation | [`DelayedProposal`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/trait.DelayedProposal.html) and `Chain::step_delayed` |
| Model and bias log weights | [`AdditiveTarget`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.AdditiveTarget.html) |
| Repeated runs, chunks, thinning, or observation | [`Sampler`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.Sampler.html) |
| Resume against a checked target | [`ChainCheckpoint`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.ChainCheckpoint.html) |
| Retained numeric observations and CSV | [`TraceRecorder`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.TraceRecorder.html) |
| Statistics without retaining every draw | [`OnlineStats`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.OnlineStats.html) and [`BinningAnalysis`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/struct.BinningAnalysis.html) |
| Scalar proposal tuning during warmup | `AdaptiveScale`, `TunableProposal`, and `Sampler::warm_up*` |
| Scalar correlation, precision, or R-hat | `Autocorrelation`, `EssEstimate` (multi-chain mean/bulk/quantile), `TailEss`, `MeanMcse`, `QuantileMcse`, and `CombinedRhat`; see [analyzing chains][analyzing-chains] |
| Original-chain rank plots | `PooledRanks::from_chains` and borrowed per-chain rank slices; deterministic average ties, with all supplied draws retained |
| Independent proposal validation | [`verify_detailed_balance*`](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/fn.verify_detailed_balance.html), density, and bin checks |

Use explicit step calls when every transition needs metadata. Bulk in-place runs skip observational telemetry hooks; delayed chunk observation can retain
per-step telemetry and post-step state. See the [proposal testing workflow][validating-proposals] and the generated API contracts for each method.
The checkout's crate-level **API migration** section records signature, telemetry, thinning, and checkpoint changes since earlier APIs.

## ✨ Features

- Additive target composition keeps model/bias weights separate from proposal corrections.
- By-value, in-place rollback, and delayed-commit proposals share log-space acceptance with typed invalid-value errors.
- Checkpoints recompute cached log weights against the resumed target; optional `serde` support uses a canonical portable shape.
- Detailed-balance diagnostics compare representative discrete transition flows, continuous density ratios, and sampled proposal bins.
- Fixed two-dimensional reference distributions with analytical moments are available behind `benchmarks`.
- Repeated and resumable runs support iterator sampling, observation, counter resets after burn-in, and positive validated thinning intervals.
- Scalar adaptive warmup tunes a bounded proposal scale; production keeps the final scale fixed.
- Scalar autocorrelation, single-chain and multi-chain ESS, measured ESS/second, mean/quantile MCSE, and R-hat have typed unavailable results.
  Multi-chain ESS accounts for between-chain disagreement; MCSE reports uncertainty in estimated summaries in original observable units.
- Streaming statistics and binning summaries avoid retaining every sample.
- Pooled rank data preserve original chain and draw order for downstream plots without a rendering dependency.
- Trace recording retains chain IDs, acceptance metadata, and target log weights for CSV export.

## 📦 Cargo features

No Cargo features are enabled by default. Scalar diagnostics need no optional feature.

| Feature | Capability |
| --- | --- |
| `benchmarks` | Fixed Rosenbrock, Neal's funnel, Gaussian mixture, and banana `BenchmarkTarget` presets |
| `serde` | Serialize chains/samplers as checkpoints and deserialize `ChainCheckpoint` for validated resume |
| `tracing` | DEBUG loop spans and TRACE events for completed transitions |

<a id="tracing-setup"></a>

For `tracing`, install an application-owned subscriber before sampling. The checkout's crate-level **Tracing contract** includes setup, event fields,
timing costs, and counter semantics. Build it with `just doc`; the library never installs a subscriber.

## 🧪 Scientific basis

The Metropolis-Hastings acceptance probability is

```text
alpha(x, y) = min(1, exp(log pi(y) - log pi(x) + log q(x | y) - log q(y | x)))
```

`Target<S>` supplies the unnormalized log weight; the proposal supplies the reverse/forward log ratio for that same move. Additive target terms change
the sampled distribution, while proposal corrections account for how moves are generated. See [scientific basis and scope][scientific-basis] for
assumptions, arithmetic conventions, method definitions, and diagnostic limitations, and the [method-to-source index][method-sources] for attribution.

## ✅ Validation model

The crate checks local transition mechanics: invalid floating-point values, acceptance, counters, cached weights, checkpoint restoration, and proposal
rollback/commit contracts. Deterministic and property tests exercise these contracts; representative empirical checks help proposal authors detect errors.

Independent fixtures cross-check rank-normalized R-hat, multi-chain ESS, and MCSE against pinned ArviZ results, and pooled ranks against SciPy.
The [analysis guide][analyzing-chains] documents versions, numerical conventions, and commands to verify exported Rust results. The plotting notebook
renders those results; ArviZ and SciPy are isolated validation dependencies, not Rust library dependencies.

These checks do not establish irreducibility, aperiodicity, equilibration, or convergence. Correlated uncertainty and observable-specific diagnostics need
scientific assessment. The [reviewer guide][reviewer-guide] maps claims to evidence and reproducible checks.

<a id="-documentation"></a>

## 🗺️ Documentation map

| Question | Owner |
| --- | --- |
| Which API and caller contract? | [Published API reference](https://docs.rs/markov-chain-monte-carlo/latest/markov_chain_monte_carlo/); `just doc` for this checkout |
| Which assumptions, methods, and limitations? | [Scientific basis][scientific-basis] |
| Which literature supports each method? | [References and source index][method-sources] |
| How do I validate a proposal? | [Validating proposals][validating-proposals] |
| How do I analyze traces and unavailable estimates? | [Analyzing chains][analyzing-chains] |
| How are benchmark targets defined? | [Reference distributions][benchmark-distributions] |
| How do I assess the crate's evidence? | [Reviewer guide][reviewer-guide] |
| Where does implementation or documentation belong? | [Code organization][code-organization] |
| How do I develop, benchmark, or release? | [Developing][developing], [benchmarking][benchmarking], and [releasing][releasing] |
| What changed or remains planned? | [Changelog](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CHANGELOG.md) and [roadmap][roadmap] |
| How do I report a vulnerability? | [Security policy](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/SECURITY.md) |

Repository guides and source examples follow `main`; API links use docs.rs `latest`. Run `just doc` for the API reference matching this checkout.

## 🧪 Examples

Example workflows live in [`examples/`](https://github.com/acgetchell/markov-chain-monte-carlo/tree/main/examples):

| Example | Workflow |
| --- | --- |
| `adaptive_normal` | Tune a bounded proposal scale during warmup, then freeze it for production |
| `additive_target_bias` | Compose model and bias log weights |
| `benchmark_distributions` | Compare seeded sampling against reference distributions with analytical moments |
| `delayed_chunked_telemetry` | Resume chunks while recording delayed-step telemetry |
| `detailed_balance` | Check by-value, in-place, delayed, and batch transition flows |
| `ising_1d` | Compare baseline and cold spin chains using energy/magnetization traces, ranks, ACF, ESS, MCSE, R-hat, and prefix efficiency |
| `iterator_sampling` | Drive a sampler as an iterator |
| `normal_1d` | Sample a normal target with a by-value random walk |

Run `just examples` for all validated examples or `just example NAME` for one. The reference-distribution example requires `--features benchmarks`
when run directly with Cargo.

Run `just diagnostic-plots` for original-chain rank overlays, bulk/tail ESS and relative-ESS curves, traces, ACF, and saved numeric reports.
The Ising example compares two temperatures using four chains per temperature, with 5,000 warmup steps and 20,000 production draws per chain.
`just notebook-check` executes both this workflow and the Ising trace-analysis notebook under `target/notebooks/`.
See [analyzing chains][analyzing-chains] for input selection, timing scope, exports, and error handling; these demonstrations do not certify convergence.
This crate is a library: the examples and notebook are contributor workflows, with no installed CLI.

<a id="performance"></a>

## 📈 Benchmarking

[Benchmarking][benchmarking] owns workload contracts, reproduction commands, and the shared `research-repo-tools` publication workflow.
Use `just bench-save-baseline NAME` to save this checkout's local stepping measurements, then compare later runs with `just bench-compare NAME`.

<!-- PERFORMANCE:BEGIN -->

![Baseline/current timing point ratios](https://raw.githubusercontent.com/acgetchell/markov-chain-monte-carlo/v0.5.2/docs/performance/v1/v0.5.2-vs-v0.5.1.svg)

Statistic: median. Unit: ns.

| Benchmark | v0.5.1 | v0.5.2 (prepared tree) | Baseline/current | Time reduction (%) |
| --- | --- | --- | --- | --- |
| By-value step | 16.7305 ns [16.6881, 16.7601] (0.95 confidence) | 16.8223 ns [16.8063, 16.8472] (0.95 confidence) | 0.994541 | -0.548916 |
| In-place accepted step | 13.7211 ns [13.6794, 13.7666] (0.95 confidence) | 13.8022 ns [13.7574, 13.847] (0.95 confidence) | 0.994117 | -0.591771 |
| In-place rollback step | 194.155 ns [193.632, 194.8] (0.95 confidence) | 198.85 ns [197.611, 199.265] (0.95 confidence) | 0.976388 | -2.41826 |
| Delayed accepted step | 10.5007 ns [10.4679, 10.5179] (0.95 confidence) | 10.4465 ns [10.4261, 10.4882] (0.95 confidence) | 1.00519 | 0.516515 |
| Delayed rejected step | 10.7442 ns [10.7239, 10.7782] (0.95 confidence) | 10.7075 ns [10.6851, 10.7249] (0.95 confidence) | 1.00343 | 0.341801 |
| Delayed no-plan step | 0.726462 ns [0.726029, 0.727028] (0.95 confidence) | 0.725055 ns [0.724612, 0.725611] (0.95 confidence) | 1.00194 | 0.193617 |
| By-value sampling (100 steps) | 1670.7 ns [1668.59, 1671.87] (0.95 confidence) | 1680.13 ns [1678.13, 1681.6] (0.95 confidence) | 0.994388 | -0.564403 |
| In-place sampling (100 steps) | 1302.16 ns [1300.06, 1304.63] (0.95 confidence) | 1307.07 ns [1305.38, 1308.89] (0.95 confidence) | 0.996246 | -0.37681 |
| Delayed sampling (100 steps) | 876.103 ns [873.82, 877.716] (0.95 confidence) | 877.842 ns [876.725, 879.467] (0.95 confidence) | 0.998018 | -0.198553 |
| Buffered observation (100 steps) | 1720.73 ns [1718.93, 1726.78] (0.95 confidence) | 1716.39 ns [1713.72, 1720.61] (0.95 confidence) | 1.00253 | 0.252168 |
| Manual online sum (100 steps) | 1275.03 ns [1271.02, 1277.33] (0.95 confidence) | 1270.34 ns [1267.26, 1273.32] (0.95 confidence) | 1.00369 | 0.368075 |
| Online statistics (100 steps) | 1456.04 ns [1453.31, 1460.85] (0.95 confidence) | 1464.02 ns [1461.02, 1467.01] (0.95 confidence) | 0.994546 | -0.54838 |
| Binning analysis (100 steps) | 2288.59 ns [2283.05, 2291.33] (0.95 confidence) | 2293.59 ns [2290.89, 2299] (0.95 confidence) | 0.997821 | -0.218398 |

Coverage: 13 selected of 22 comparable; 0 current-only; 0 baseline-only benchmarks.

Ratios are point estimates, not significance tests or scientific acceptance.

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

- [Full timing report](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.5.2/docs/performance/v1/performance.md)
- [Comparison evidence](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.5.2/docs/performance/v1/v0.5.2-vs-v0.5.1.comparison.json)
- [Provenance](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.5.2/docs/performance/v1/v0.5.2-vs-v0.5.1.evidence.json)
- [CSV export](https://github.com/acgetchell/markov-chain-monte-carlo/blob/v0.5.2/docs/performance/v1/v0.5.2-vs-v0.5.1.csv)

<!-- PERFORMANCE:END -->

## 🧩 Ecosystem

This crate is part of a broader Rust ecosystem for computational geometry and simulation:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations) — CDT physics and simulation
- [`delaunay`](https://crates.io/crates/delaunay) — geometric primitives and triangulations
- [`la-stack`](https://crates.io/crates/la-stack) — fixed-size linear algebra

The long-term architecture separates:

- **Geometry**: triangulations and geometric predicates
- **Sampling**: this crate
- **Physics**: CDT actions, observables, and domain-specific dynamics

<a id="-reviewer-guide"></a>

## 🛣️ Limitations and roadmap

The crate supplies sampling mechanics and empirical diagnostics. Domain code owns model choice, valid proposals, reproducible random streams,
equilibration, and precision thresholds. Classical split R-hat is not rank-normalized or folded; ESS and MCSE are observable-specific.
Mean MCSE requires finite population moments; finite heavy-tail output does not establish their existence. Tied quantiles can leave MCSE unavailable.

Sequential multi-chain workflows and cross-chain diagnostics are available today. Built-in parallel-chain orchestration, tempering, and dedicated
learned-proposal integrations remain roadmap work. Externally supplied learned log weights can already be composed as target terms; training energy models
or proposal policies remains outside this crate's scope. See the [roadmap][roadmap] and [reviewer guide][reviewer-guide].

## 🤝 Contributing

See [CONTRIBUTING.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CONTRIBUTING.md) for the full contributor guide (project layout,
development workflow, code style, testing, documentation layout, performance/benchmarking, and the release process). Community expectations live in
[`CODE_OF_CONDUCT.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CODE_OF_CONDUCT.md). AI assistants should follow
[`AGENTS.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/AGENTS.md).

## 📚 Citation

If you use this software in academic work or downstream research software, cite the Zenodo DOI and
include the software metadata from [CITATION.cff][citation-metadata].

- DOI: <https://doi.org/10.5281/zenodo.20033111>
- Citation metadata: [CITATION.cff][citation-metadata]

```bibtex
@software{getchell_markov_chain_monte_carlo,
  author = {Adam Getchell},
  title = {markov-chain-monte-carlo: A composable MCMC framework for Rust},
  doi = {10.5281/zenodo.20033111},
  url = {https://github.com/acgetchell/markov-chain-monte-carlo}
}
```

For release-specific fields such as version, release date, and ORCID, prefer [CITATION.cff][citation-metadata].

[citation-metadata]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CITATION.cff

## 🔎 References

For canonical background references for Metropolis-Hastings, MCMC, and the example models, see
[bibliographic records and method-to-source index][method-sources].

## 🤖 AI Agents

AI coding assistants should read [`AGENTS.md`](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/AGENTS.md) before proposing or applying
changes. See [CONTRIBUTING.md](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/CONTRIBUTING.md#ai-assisted-development) for the repository's
AI-assisted development note.

## 📜 License

This project is licensed under the [BSD 3-Clause License](https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/LICENSE).

[analyzing-chains]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/ANALYZING_CHAINS.md
[benchmark-distributions]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/benchmark_distributions.md
[benchmarking]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/BENCHMARKING.md
[code-organization]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/code_organization.md
[developing]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/dev/DEVELOPING.md
[method-sources]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/REFERENCES.md
[releasing]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/RELEASING.md
[reviewer-guide]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/reviewer_guide.md
[roadmap]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/roadmap.md
[scientific-basis]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/scientific_basis.md
[validating-proposals]: https://github.com/acgetchell/markov-chain-monte-carlo/blob/main/docs/VALIDATING_PROPOSALS.md
