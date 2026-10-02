//! ACF, multi-chain precision, and classical/rank-based R-hat for four chains.
//!
//! Run with: `cargo run --release --example diagnostics`
//! Chains run sequentially with distinct seeds and dispersed starts. Diagnostics
//! need comparable traces, not parallel execution. No optional feature is needed.

use std::{
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, BufWriter, Write},
    time::{Duration, Instant},
};

use markov_chain_monte_carlo::prelude::by_value::*;
use markov_chain_monte_carlo::prelude::{
    Autocorrelation, CombinedRhat, DiagnosticTiming, DiagnosticTimingError, EssEstimate,
    EssEstimator, FoldedRankNormalizedSplitRhat, MeanMcse, MonteCarloError, QuantileMcse,
    RankNormalizedSplitRhat, SplitRhat, SplitRhatError, TailEss,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};

const WARMUP: usize = 2_000;

#[derive(Debug)]
enum DiagnosticsError {
    Mcmc(McmcError),
    Rhat(SplitRhatError),
    Io(io::Error),
    Json(serde_json::Error),
    Timing(DiagnosticTimingError),
    Precision(MonteCarloError),
}

impl fmt::Display for DiagnosticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mcmc(error) => error.fmt(f),
            Self::Rhat(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::Timing(error) => error.fmt(f),
            Self::Precision(error) => error.fmt(f),
        }
    }
}

impl Error for DiagnosticsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mcmc(error) => Some(error),
            Self::Rhat(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Timing(error) => Some(error),
            Self::Precision(error) => Some(error),
        }
    }
}

impl From<McmcError> for DiagnosticsError {
    fn from(error: McmcError) -> Self {
        Self::Mcmc(error)
    }
}

impl From<SplitRhatError> for DiagnosticsError {
    fn from(error: SplitRhatError) -> Self {
        Self::Rhat(error)
    }
}

impl From<DiagnosticTimingError> for DiagnosticsError {
    fn from(error: DiagnosticTimingError) -> Self {
        Self::Timing(error)
    }
}

impl From<MonteCarloError> for DiagnosticsError {
    fn from(error: MonteCarloError) -> Self {
        Self::Precision(error)
    }
}

impl From<io::Error> for DiagnosticsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for DiagnosticsError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

struct StandardNormal;

impl Target<f64> for StandardNormal {
    fn log_prob(&self, state: &f64) -> f64 {
        -0.5 * state * state
    }
}

struct RandomWalk;

impl Proposal<f64> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &f64, rng: &mut R) -> f64 {
        current + rng.random_range(-2.0..2.0)
    }
}

fn main() -> Result<(), DiagnosticsError> {
    const DRAWS: usize = 10_000;
    const MAX_LAG: usize = 511;

    println!("Scalar diagnostics: four sequential N(0,1) chains");
    println!("warmup={WARMUP}, draws per chain={DRAWS}, recording interval=1, max lag={MAX_LAG}");
    println!("Observable: position; fixed uniform random-walk half-width=2.");
    let mut traces = Vec::with_capacity(4);
    let mut production_time = Duration::ZERO;
    for (id, (start, seed)) in [(-6.0, 40), (-2.0, 41), (2.0, 42), (6.0, 43)]
        .into_iter()
        .enumerate()
    {
        let target = StandardNormal;
        let proposal = RandomWalk;
        let mut rng = StdRng::seed_from_u64(seed);
        let chain = Chain::new(start, &target)?;
        let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)?;
        sampler.run(WARMUP)?;
        sampler.reset_counters();
        let mut draws = Vec::with_capacity(DRAWS);
        let started = Instant::now();
        for _ in 0..DRAWS {
            let _ = sampler.step()?;
            // Record every post-step state, including repeated rejected states.
            draws.push(*sampler.chain_ref().state());
        }
        // Sum nonoverlapping wall-clock segments because these chains run
        // sequentially. Parallel chains require the concurrent workload's wall time.
        production_time += started.elapsed();
        println!(
            "chain {id}: seed={seed}, start={start:+.1}, acceptance={:.3}",
            sampler.chain_ref().acceptance_rate()
        );
        match Autocorrelation::estimate(&draws, MAX_LAG) {
            Ok(acf) => {
                println!("chain {id}: ACF[1]={:.4}", acf.values()[1]);
                match acf.integrated_time() {
                    Ok(time) => println!(
                        "chain {id}: mean ESS={:.1}, integrated time={:.3}, window={}",
                        time.effective_sample_size(),
                        time.estimate(),
                        time.window()
                    ),
                    Err(error) => println!("chain {id}: mean ESS unavailable: {error}"),
                }
            }
            Err(error) => println!("chain {id}: ACF and mean ESS unavailable: {error}"),
        }
        traces.push(draws);
    }

    // Borrow separate chains: concatenating them would corrupt ACF/ESS ordering.
    let chains: Vec<&[f64]> = traces.iter().map(Vec::as_slice).collect();
    let timing = DiagnosticTiming::try_new(production_time, chains.len(), DRAWS)?;
    print_precision(&chains, &timing)?;
    let classical = SplitRhat::estimate(&chains);
    match classical {
        Ok(rhat) => println!("Classical split R-hat: {:.4}", rhat.value()),
        Err(error) => println!("Classical split R-hat unavailable: {error}"),
    }
    let report = CombinedRhat::estimate(&chains)?;
    match report.rank_normalized() {
        Ok(rhat) => println!("Rank-normalized split R-hat component: {:.4}", rhat.value()),
        Err(error) => println!("Rank-normalized split R-hat component unavailable: {error}"),
    }
    match report.folded() {
        Ok(rhat) => println!(
            "Folded rank-normalized split R-hat component: {:.4}",
            rhat.value()
        ),
        Err(error) => println!("Folded rank-normalized split R-hat component unavailable: {error}"),
    }
    match report.value() {
        Some(value) => println!("Combined R-hat: {value:.4}"),
        None => println!("Combined R-hat unavailable: inspect both component results"),
    }
    export_diagnostics(&traces, classical, &report)?;
    println!(
        "Single-chain mean ESS and multi-chain mean/bulk/tail ESS have distinct estimator contracts."
    );
    println!("Combined R-hat is available only when both rank-based components are available.");
    println!("These estimates do not certify convergence or exploration of all modes.");
    Ok(())
}

fn print_precision(chains: &[&[f64]], timing: &DiagnosticTiming) -> Result<(), DiagnosticsError> {
    for (label, method) in [("mean", EssEstimator::Mean), ("bulk", EssEstimator::Bulk)] {
        let ess = EssEstimate::estimate(chains, method)?;
        println!(
            "Multi-chain {label} ESS: {:.1}, ESS/S={:.3}, ESS/second={:.1}, regularized={}",
            ess.value(),
            ess.relative(),
            ess.per_second(Some(timing))?,
            ess.is_regularized()
        );
    }
    let tail = TailEss::estimate(chains)?;
    let lower = tail.lower()?;
    let upper = tail.upper()?;
    println!(
        "Tail ESS: {:?}, q05={:.1}, q95={:.1}, ESS/S={:?}, ESS/second={:?}",
        tail.value(),
        lower.value(),
        upper.value(),
        tail.relative(),
        tail.per_second(Some(timing))?
    );
    let mean = MeanMcse::estimate(chains)?;
    println!("Mean MCSE: {:.5} position units", mean.value());
    for p in [0.05, 0.5, 0.95] {
        let error = QuantileMcse::estimate(chains, p)?;
        println!(
            "Quantile p={p}: estimate={:.4}, MCSE={:.5} position units, uncertainty bounds={:?}, ESS={:.1}",
            error.quantile(),
            error.value(),
            error.interval(),
            error.effective_sample_size().value()
        );
    }
    println!(
        "Target standard deviation is 1 position unit; MCSE measures error in estimated summaries, not target spread."
    );
    println!(
        "Timing: sequential production sampling and recording; excludes allocation, warmup, diagnostics, and export."
    );
    Ok(())
}

fn export_diagnostics(
    traces: &[Vec<f64>],
    classical: Result<SplitRhat, SplitRhatError>,
    report: &CombinedRhat,
) -> Result<(), DiagnosticsError> {
    fs::create_dir_all("target")?;
    let mut csv = BufWriter::new(File::create("target/diagnostics_trace.csv")?);
    writeln!(csv, "chain_id,step,position")?;
    for (id, draws) in traces.iter().enumerate() {
        for (step, value) in draws.iter().enumerate() {
            writeln!(csv, "{id},{},{value}", step + 1)?;
        }
    }
    csv.flush()?;
    let output = json!({
        "schema_version": 1,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "observable": "position",
        "target": "standard_normal",
        "proposal": "uniform_random_walk_half_width_2",
        "rng": "rand::rngs::StdRng",
        "chain_ids": [0, 1, 2, 3],
        "seeds": [40, 41, 42, 43],
        "starts": [-6, -2, 2, 6],
        "warmup_per_chain": WARMUP,
        "recording_interval": 1,
        "chain_count": report.chain_count(),
        "samples_per_chain": report.samples_per_chain(),
        "samples_per_split_chain": report.samples_per_split_chain(),
        "median_scope": "all_original_draws_before_splitting",
        "split": "first_and_last_floor_N_over_2",
        "trace": "diagnostics_trace.csv",
        "classical_split": component_json(classical.map(SplitRhat::value)),
        "rank_normalized_split": component_json(report.rank_normalized().map(RankNormalizedSplitRhat::value)),
        "folded_rank_normalized_split": component_json(report.folded().map(FoldedRankNormalizedSplitRhat::value)),
        "combined_rank_normalized_maximum": {
            "status": if report.value().is_some() { "estimated" } else { "unavailable" },
            "value": report.value()
        }
    });
    let mut file = BufWriter::new(File::create("target/diagnostics.json")?);
    serde_json::to_writer_pretty(&mut file, &output)?;
    writeln!(file)?;
    file.flush()?;
    println!("diagnostics JSON: target/diagnostics.json; trace CSV: target/diagnostics_trace.csv");
    Ok(())
}

fn component_json(result: Result<f64, SplitRhatError>) -> Value {
    match result {
        Ok(value) => json!({"status": "estimated", "value": value}),
        Err(error) => json!({"status": "unavailable", "value": null, "error": error.to_string()}),
    }
}
