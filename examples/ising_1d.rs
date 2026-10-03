//! Open-boundary 1-D Ising model sampled with `ProposalMut` (in-place mutation + rollback).
//!
//! Demonstrates [`Sampler`] with `run_mut` for burn-in, `step_mut` for
//! per-sample collection on a discrete, non-Clone state space, and
//! [`TraceRecorder`] for CSV trace export. Four separately seeded chains also
//! demonstrate [`Autocorrelation`] for mean ESS and measured ESS per second,
//! and [`SplitRhat`] for classical split convergence diagnostics. Energy and
//! magnetization also receive pooled means, multi-chain mean/bulk/tail ESS,
//! mean MCSE, and [`CombinedRhat`]. The companion JSON report records estimator
//! names, chain identities, timing scope, and unavailable component results.
//! The same workflow runs at beta=0.5 and beta=2; the colder run illustrates
//! slow magnetization exploration. Both runs supply original-chain rank and
//! prefix-efficiency plots through `ising_1d/diagnostics.rs`.
//! The example is intentionally small:
//! it shows the sampler contract for a familiar statistical-physics model,
//! not a finite-size scaling study.  Parameters use dimensionless units with
//! Boltzmann's constant absorbed into the inverse temperature `beta`.
//!
//! Run from the repository root with `just example ising_1d`. Configuration is
//! fixed in this source; there is no command-line configuration interface.
//! Each run replaces the example's CSV and JSON files under `target/`, relative
//! to the working directory. Console output summarizes progress and results;
//! downstream analysis should consume the CSV and JSON artifacts.

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::time::{Duration, Instant};

use markov_chain_monte_carlo::prelude::in_place::*;
use markov_chain_monte_carlo::prelude::{
    Autocorrelation, AutocorrelationError, ChainId, CombinedRhat, DiagnosticTimingError,
    EssEstimate, EssEstimator, MeanMcse, MonteCarloError, OnlineStats, PooledRankError, SplitRhat,
    SplitRhatError, StatisticsError, TailEss, Trace, TraceError, TraceRecorder, TraceStepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use serde_json::{Error as JsonError, Value, json};

#[path = "ising_1d/diagnostics.rs"]
mod diagnostics;

/// Errors from the Ising trace example.
#[derive(Debug)]
enum ExampleError {
    /// MCMC transition or initialization failed.
    Mcmc(McmcError),
    /// Trace recording or observable selection failed.
    Trace(TraceError),
    /// Scalar diagnostics failed for a recorded observable.
    Autocorrelation {
        /// Chain whose observable was being analyzed.
        chain_id: ChainId,
        /// Observable's exact trace-column name.
        observable: String,
        /// Original typed estimator failure.
        source: AutocorrelationError,
    },
    /// Accumulating an observable's pooled mean failed.
    Statistics(StatisticsError),
    /// Invalid inputs for the multi-chain report.
    Rhat(SplitRhatError),
    /// Pooled plot ranks could not be constructed.
    Ranks(PooledRankError),
    /// Production timing could not be paired with diagnostic inputs.
    Timing(DiagnosticTimingError),
    /// Structurally invalid multi-chain ESS/MCSE inputs.
    Precision(MonteCarloError),
    /// Output directory creation or CSV/JSON artifact I/O failed.
    Io(io::Error),
    /// JSON diagnostic export failed.
    Json(JsonError),
}

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mcmc(err) => write!(f, "{err}"),
            Self::Trace(err) => write!(f, "{err}"),
            Self::Autocorrelation {
                chain_id,
                observable,
                source,
            } => write!(
                f,
                "autocorrelation for chain {chain_id}, observable {observable:?}: {source}"
            ),
            Self::Statistics(err) => write!(f, "{err}"),
            Self::Rhat(err) => write!(f, "{err}"),
            Self::Ranks(err) => write!(f, "{err}"),
            Self::Timing(err) => write!(f, "{err}"),
            Self::Precision(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
        }
    }
}

impl Error for ExampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mcmc(err) => Some(err),
            Self::Trace(err) => Some(err),
            Self::Autocorrelation { source, .. } => Some(source),
            Self::Statistics(err) => Some(err),
            Self::Rhat(err) => Some(err),
            Self::Ranks(err) => Some(err),
            Self::Timing(err) => Some(err),
            Self::Precision(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
        }
    }
}

impl From<McmcError> for ExampleError {
    fn from(err: McmcError) -> Self {
        Self::Mcmc(err)
    }
}

impl From<TraceError> for ExampleError {
    fn from(err: TraceError) -> Self {
        Self::Trace(err)
    }
}

impl From<StatisticsError> for ExampleError {
    fn from(err: StatisticsError) -> Self {
        Self::Statistics(err)
    }
}

impl From<SplitRhatError> for ExampleError {
    fn from(err: SplitRhatError) -> Self {
        Self::Rhat(err)
    }
}

impl From<PooledRankError> for ExampleError {
    fn from(err: PooledRankError) -> Self {
        Self::Ranks(err)
    }
}

impl From<DiagnosticTimingError> for ExampleError {
    fn from(err: DiagnosticTimingError) -> Self {
        Self::Timing(err)
    }
}

impl From<MonteCarloError> for ExampleError {
    fn from(err: MonteCarloError) -> Self {
        Self::Precision(err)
    }
}

impl From<io::Error> for ExampleError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<JsonError> for ExampleError {
    fn from(err: JsonError) -> Self {
        Self::Json(err)
    }
}

// --- State: a chain of ±1 spins (intentionally not Clone) ---

/// A one-dimensional chain of Ising spins.
struct SpinChain {
    spins: Vec<i8>,
}

impl SpinChain {
    /// Create a uniform +1 spin chain of the given length.
    fn all_up(n: usize) -> Self {
        Self { spins: vec![1; n] }
    }

    /// Magnetization per spin: m = (1/N) Σ `s_i`.
    fn magnetization(&self) -> f64 {
        let sum: i32 = self.spins.iter().map(|&s| i32::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "spin chain length won't exceed 2^52"
        )]
        let n = self.spins.len() as f64;
        f64::from(sum) / n
    }
}

// --- Target: nearest-neighbour Ising energy at inverse temperature β ---

/// Open-boundary nearest-neighbour Ising Hamiltonian:
/// H = −J Σ_{i=0}^{N−2} `s_i` · `s_{i+1}`.
///
/// The first and last spins are not coupled. `log_prob = −β H`.
struct Ising {
    /// Dimensionless coupling constant (positive = ferromagnetic).
    coupling: f64,
    /// Dimensionless inverse temperature.
    beta: f64,
}

impl Ising {
    /// Energy of the nearest-neighbour spin chain.
    fn energy(&self, state: &SpinChain) -> f64 {
        let interaction: f64 = state
            .spins
            .windows(2)
            .map(|w| f64::from(w[0]) * f64::from(w[1]))
            .sum();
        -self.coupling * interaction
    }
}

impl Target<SpinChain> for Ising {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        -self.beta * self.energy(state)
    }
}

// --- Proposal: flip one random spin, symmetric over sites for non-empty chains ---

/// Single-site spin flip.  Undo token is the flipped site index.
struct SpinFlip;

impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;
    type Info = usize;

    fn propose_mut<R: Rng + ?Sized>(
        &mut self,
        state: &mut SpinChain,
        rng: &mut R,
    ) -> Option<usize> {
        if state.spins.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }

    fn info(&self, _state: &SpinChain, idx: &usize) -> usize {
        *idx
    }

    fn undo(&mut self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1; // flipping twice = identity
    }
}

/// Unrecorded warmup transitions per chain, excluded from production timing.
const BURN_IN: usize = 5_000;
/// Retained production draws per chain, with one record after every transition.
const SAMPLES: u32 = 20_000;

/// Record one separately seeded Ising chain after discarding warmup.
///
/// The caller supplies indices 0..4 to choose uniform up/down or alternating
/// starts. Resetting counters before recording keeps the trace and acceptance
/// statistics on the same production scope. Time only those production steps,
/// including observation and recording, so the returned duration can be paired
/// with the entire returned trace for ESS rates; export and analysis are excluded.
fn sample_chain(
    chain_index: usize,
    seed: u64,
    target: &Ising,
) -> Result<(Trace, Duration), ExampleError> {
    let mut rng = StdRng::seed_from_u64(seed);

    let n_spins = 50_usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "example spin count is small enough to represent exactly"
    )]
    let n_spins_f64 = n_spins as f64;
    let Ising { beta, coupling } = *target;
    let mut initial = SpinChain::all_up(n_spins);
    // Uniform up/down and the two alternating configurations provide
    // dispersed starts, with a distinct RNG seed for each original chain.
    for (index, spin) in initial.spins.iter_mut().enumerate() {
        if (chain_index == 1) || (chain_index >= 2 && index % 2 == usize::from(chain_index == 3)) {
            *spin = -1;
        }
    }
    let chain = Chain::new(initial, target)?;
    let mut sampler = Sampler::new(chain, target, SpinFlip, &mut rng)?;

    println!("1-D Ising model ({n_spins} spins, β={beta}, J={coupling}, seed={seed})");
    println!(
        "Initial magnetization: {:.3}",
        sampler.chain_ref().state().magnetization()
    );

    // Burn-in
    sampler.run_mut(BURN_IN)?;
    println!(
        "After {BURN_IN} burn-in steps: m = {:.3}",
        sampler.chain_ref().state().magnetization()
    );

    // Reset counters so acceptance rate reflects production only
    sampler.reset_counters();

    // Collect samples and export a reusable trace for downstream diagnostics.
    let mut mag_sum = 0.0;
    let mut mag_sq_sum = 0.0;
    let chain_id = ChainId::new(chain_index);
    let mut trace = TraceRecorder::new(chain_id, ["energy", "magnetization"])?;
    let started = Instant::now();
    for _ in 0..SAMPLES {
        let step = sampler.step_mut()?;
        let chain = sampler.chain_ref();
        let state = chain.state();
        let energy = target.energy(state);
        let m = state.magnetization();
        trace.record(chain, TraceStepOutcome::from(&step), [energy, m])?;
        mag_sum += m;
        mag_sq_sum = m.mul_add(m, mag_sq_sum);
    }
    let elapsed = started.elapsed();
    let mean_mag = mag_sum / f64::from(SAMPLES);
    let mean_mag_sq = mag_sq_sum / f64::from(SAMPLES);
    // Finite-sample fluctuation estimate, not a thermodynamic-limit claim.
    let susceptibility = beta * n_spins_f64 * (mean_mag_sq - mean_mag * mean_mag);
    println!("\nChain {chain_id} results ({SAMPLES} samples):");
    println!("  <m>:             {mean_mag:+.4}");
    println!("  <m²>:            {mean_mag_sq:.4}");
    println!("  susceptibility:  {susceptibility:.2}");
    println!(
        "  acceptance rate: {:.1}%",
        sampler.chain_ref().acceptance_rate() * 100.0
    );
    Ok((trace.into_trace(), elapsed))
}

/// Run four chains sequentially and export reusable traces and diagnostic metadata.
fn main() -> Result<(), ExampleError> {
    let target = Ising {
        coupling: 1.0,
        beta: 0.5,
    };
    let (trace, timings) = sample_chains(&target)?;
    fs::create_dir_all("target")?;
    let trace_path = "target/ising_1d_trace.csv";
    let mut trace_csv = BufWriter::new(File::create(trace_path)?);
    trace.write_csv(&mut trace_csv)?;
    // Flush explicitly: dropping a buffered writer cannot report I/O errors.
    trace_csv.flush()?;

    let chain_ids: Vec<_> = timings.iter().map(|&(id, _)| id).collect();
    let diagnostics = json!({
        "schema_version": 1, "trace_file": "ising_1d_trace.csv",
        "ess_method": "single_chain_mean_geyer_initial_monotone",
        "max_lag": 2000,
        "warmup_policy": "discard_before_recording",
        "timing_scope": "production_including_observation_excluding_warmup_export_and_analysis",
        "execution": "sequential_chains",
        "chains": write_scalar_diagnostics(&trace, &timings)?,
        "rhat": rhat_diagnostics(&trace, &chain_ids)?,
        "multi_chain": precision_diagnostics(&trace, &chain_ids)?,
    });
    let mut diagnostics_file = BufWriter::new(File::create("target/ising_1d_diagnostics.json")?);
    serde_json::to_writer_pretty(&mut diagnostics_file, &diagnostics)?;
    writeln!(diagnostics_file)?;
    diagnostics_file.flush()?;
    println!("  trace CSV:        {trace_path}");
    println!("  diagnostics JSON: target/ising_1d_diagnostics.json");

    // Same model and observation policy, colder temperature. Compare actual
    // sampling difficulty without shifting or rescaling recorded observations.
    let cold_target = Ising {
        coupling: 1.0,
        beta: 2.0,
    };
    let (cold_trace, cold_timings) = sample_chains(&cold_target)?;
    let mut cold_csv = BufWriter::new(File::create("target/ising_1d_cold_trace.csv")?);
    cold_trace.write_csv(&mut cold_csv)?;
    cold_csv.flush()?;
    diagnostics::export(&[
        diagnostics::Case {
            label: "baseline",
            trace: &trace,
            timings: &timings,
            target: &target,
        },
        diagnostics::Case {
            label: "cold",
            trace: &cold_trace,
            timings: &cold_timings,
            target: &cold_target,
        },
    ])?;
    Ok(())
}

/// Run the same four starts, seeds, warmup, and recording policy for one target.
fn sample_chains(target: &Ising) -> Result<(Trace, Vec<(ChainId, Duration)>), ExampleError> {
    let mut trace = Trace::new(["energy", "magnetization"])?;
    let mut timings = Vec::new();
    for (chain_index, seed) in (42..46).enumerate() {
        let (chain_trace, elapsed) = sample_chain(chain_index, seed, target)?;
        trace.extend(chain_trace)?;
        timings.push((ChainId::new(chain_index), elapsed));
    }
    Ok((trace, timings))
}

/// Analyze each physical observable while preserving original chain boundaries.
///
/// Pool observations only for the sample mean. Precision and convergence APIs
/// receive separate ordered columns. Discrete tails or folded ranks can be
/// unavailable; retain both components and their errors in the exported report.
fn precision_diagnostics(trace: &Trace, chain_ids: &[ChainId]) -> Result<Vec<Value>, ExampleError> {
    let mut reports = Vec::new();
    for name in trace.observable_names() {
        let columns: Vec<Vec<f64>> = chain_ids
            .iter()
            .map(|&id| {
                trace
                    .observable_values(id, name)
                    .map(|values| values.copied().collect())
            })
            .collect::<Result<_, _>>()?;
        let chains: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let summary = OnlineStats::try_from_iter(columns.iter().flatten().copied())?;
        let mean_error = MeanMcse::estimate(&chains);
        let bulk = EssEstimate::estimate(&chains, EssEstimator::Bulk);
        let tails = TailEss::estimate(&chains);
        let rhat = CombinedRhat::estimate(&chains);

        println!("\nMulti-chain {name}: pooled mean={:?}", summary.mean());
        println!("  {name} mean MCSE: {:?}", mean_error.map(MeanMcse::value));
        println!(
            "  {name} multi-chain mean ESS: {:?}",
            mean_error.map(|error| error.effective_sample_size().value())
        );
        println!("  {name} bulk ESS: {:?}", bulk.map(EssEstimate::value));
        println!("  {name} tail ESS: {:?}", tails.map(|tail| tail.value()));
        println!(
            "  {name} combined R-hat: {:?}",
            rhat.map(CombinedRhat::value)
        );

        let tail_report = match tails {
            Ok(tail) => json!({
                "status": if tail.value().is_some() { "estimated" } else { "unavailable" },
                "method": "minimum_quantile_ess_0.05_0.95",
                "ess": tail.value(), "relative_ess": tail.relative(),
                "lower": estimate_json(tail.lower(), ess_json),
                "upper": estimate_json(tail.upper(), ess_json),
            }),
            Err(error) => estimate_json::<TailEss>(Err(error), |_| Value::Null),
        };
        let rhat_report = match rhat {
            Ok(rhat) => json!({
                "status": if rhat.value().is_some() { "estimated" } else { "unavailable" },
                "method": "maximum_rank_normalized_and_folded_split_rhat",
                "rhat": rhat.value(),
                "rank_normalized": estimate_json(rhat.rank_normalized(), |part| json!(part.value())),
                "folded": estimate_json(rhat.folded(), |part| json!(part.value())),
            }),
            Err(error) => estimate_json::<CombinedRhat>(Err(error), |_| Value::Null),
        };
        reports.push(json!({
            "observable": name,
            "units": if name == "energy" { "energy (J=1)" } else { "magnetization per spin" },
            "chain_ids": chain_ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
            "original_sample_count": summary.count(), "mean": summary.mean(),
            "mean_mcse": estimate_json(mean_error, |error| json!({
                "method": "pooled_sample_sd_over_sqrt_raw_mean_ess",
                "mcse": error.value(), "mean_ess": ess_json(*error.effective_sample_size()),
            })),
            "bulk_ess": estimate_json(bulk, ess_json),
            "tail_ess": tail_report, "combined_rhat": rhat_report,
        }));
    }
    Ok(reports)
}

/// Preserve an estimate or its unavailable reason without numeric sentinels.
fn estimate_json<T>(
    estimate: Result<T, impl fmt::Display>,
    value: impl FnOnce(T) -> Value,
) -> Value {
    match estimate {
        Ok(estimate) => json!({ "status": "estimated", "result": value(estimate) }),
        Err(error) => {
            json!({ "status": "unavailable", "result": null, "error": error.to_string() })
        }
    }
}

/// Describe the estimator and both original and retained counts for an ESS.
fn ess_json(ess: EssEstimate) -> Value {
    json!({
        "method": format!("{:?}", ess.estimator()),
        "ess": ess.value(), "relative_ess": ess.relative(),
        "original_sample_count": ess.original_sample_count(),
        "retained_sample_count": ess.sample_count(),
        "chain_count": ess.chain_count(), "samples_per_chain": ess.samples_per_chain(),
        "samples_per_split_chain": ess.samples_per_split_chain(),
        "regularized": ess.is_regularized(),
    })
}

/// Export per-chain ACF/time CSVs and return observable ESS/rate JSON records.
///
/// Each timing must cover the full production trace for its identifier, using
/// this example's fixed warmup, sample-count, and recording-interval settings.
/// Estimating each named column separately preserves chain boundaries. ACF or
/// I/O failures stop export; unavailable integrated-time or rate estimates are
/// represented explicitly instead of substituting numeric sentinel values.
fn write_scalar_diagnostics(
    trace: &Trace,
    timings: &[(ChainId, Duration)],
) -> Result<Vec<Value>, ExampleError> {
    // Analyze production rows, retaining rejections and no-proposal self-loops.
    // The notebook independently calculates these diagnostics from the trace;
    // these companion CSVs also let downstream tools consume Rust results.
    let mut acf_csv = BufWriter::new(File::create("target/ising_1d_acf.csv")?);
    let mut time_csv = BufWriter::new(File::create("target/ising_1d_autocorrelation_time.csv")?);
    writeln!(acf_csv, "chain_id,observable,lag,acf")?;
    writeln!(time_csv, "chain_id,observable,samples,tau,window")?;
    let mut chain_diagnostics = Vec::new();
    for &(chain_id, elapsed) in timings {
        let mut observables = Vec::new();
        for name in trace.observable_names() {
            let values: Vec<_> = trace.observable_values(chain_id, name)?.copied().collect();
            let acf = Autocorrelation::estimate(&values, 2_000).map_err(|source| {
                ExampleError::Autocorrelation {
                    chain_id,
                    observable: name.clone(),
                    source,
                }
            })?;
            for (lag, value) in acf.values().iter().enumerate() {
                writeln!(acf_csv, "{chain_id},{name},{lag},{value}")?;
            }
            match acf.integrated_time() {
                Ok(time) => {
                    writeln!(
                        time_csv,
                        "{chain_id},{name},{},{},{}",
                        time.sample_count(),
                        time.estimate(),
                        time.window()
                    )?;
                    println!(
                        "  {name} autocorrelation time: {:.2} recorded steps (window {})",
                        time.estimate(),
                        time.window()
                    );
                    let rate = time.effective_sample_size_per_second(elapsed);
                    println!(
                        "  {name} ESS: {:.2}; ESS/second: {rate:?}",
                        time.effective_sample_size()
                    );
                    observables.push(json!({
                        "observable": name,
                        "status": "estimated",
                        "tau": time.estimate(),
                        "window": time.window(),
                        "ess": time.effective_sample_size(),
                        "ess_per_second": rate.as_ref().ok(),
                        "rate_error": rate.err().map(|error| error.to_string()),
                    }));
                }
                Err(err) => {
                    println!("  {name} autocorrelation time unavailable: {err}");
                    observables.push(json!({
                        "observable": name, "status": "unavailable", "error": err.to_string(),
                        "tau": null, "window": null, "ess": null, "ess_per_second": null,
                    }));
                }
            }
        }
        chain_diagnostics.push(json!({
            "chain_id": chain_id.get(), "seed": 42 + chain_id.get(),
            "samples": SAMPLES, "discarded_warmup_steps": BURN_IN,
            "recording_interval_steps": 1, "elapsed_seconds": elapsed.as_secs_f64(),
            "observables": observables,
        }));
    }

    acf_csv.flush()?;
    time_csv.flush()?;
    Ok(chain_diagnostics)
}

/// Compare the same observable across all original chains and describe the result.
///
/// The selected identifiers establish chain order independently of any timing
/// metadata. Borrowed columns preserve those original chain boundaries. The
/// successful estimator owns the exported split counts; failures retain input
/// shape information but leave result-dependent counts unavailable.
fn rhat_diagnostics(trace: &Trace, chain_ids: &[ChainId]) -> Result<Vec<Value>, TraceError> {
    let mut reports = Vec::new();
    for name in trace.observable_names() {
        let columns: Result<Vec<Vec<_>>, TraceError> = chain_ids
            .iter()
            .map(|&id| {
                trace
                    .observable_values(id, name)
                    .map(|values| values.copied().collect())
            })
            .collect();
        let columns = columns?;
        let slices: Vec<_> = columns.iter().map(Vec::as_slice).collect();
        let common_samples = columns
            .first()
            .map(Vec::len)
            .filter(|&count| columns.iter().all(|chain| chain.len() == count));
        let mut result = json!({
            "observable": name, "method": "classical_split_rhat",
            "chain_ids": chain_ids.iter().map(|id| id.get()).collect::<Vec<_>>(),
            "chain_count": chain_ids.len(), "samples_per_chain": common_samples,
            "samples_per_split_chain": null, "omitted_middle_draws_per_chain": null,
        });
        match SplitRhat::estimate(&slices) {
            Ok(rhat) => {
                println!("  {name} split R-hat: {:.5}", rhat.value());
                result["status"] = json!("estimated");
                result["rhat"] = json!(rhat.value());
                result["chain_count"] = json!(rhat.chain_count());
                result["samples_per_chain"] = json!(rhat.samples_per_chain());
                result["samples_per_split_chain"] = json!(rhat.samples_per_split_chain());
                result["omitted_middle_draws_per_chain"] =
                    json!(rhat.samples_per_chain() - 2 * rhat.samples_per_split_chain());
            }
            Err(err) => {
                println!("  {name} split R-hat unavailable: {err}");
                result["status"] = json!("unavailable");
                result["rhat"] = Value::Null;
                result["error"] = json!(err.to_string());
            }
        }
        reports.push(result);
    }
    Ok(reports)
}
