//! Consumer-owned report assembly; every numerical diagnostic comes from public APIs.

use std::{
    fmt::{Debug, Display},
    fs::{self, File},
    io::{BufWriter, Write},
    time::Duration,
};

use markov_chain_monte_carlo::{
    Autocorrelation, BinningAnalysis, ChainId, CombinedRhat, DiagnosticTiming, EssEstimate,
    EssEstimator, FoldedRankNormalizedSplitRhat, IntegratedAutocorrelationTime, MeanMcse,
    MonteCarloError, OnlineStats, PooledRanks, QuantileMcse, RankNormalizedSplitRhat, SplitRhat,
    TailEss, Trace,
};
use serde_json::{Value, json};

use super::{BURN_IN, ExampleError, Ising, SAMPLES};

const PREFIXES: [usize; 8] = [64, 127, 256, 512, 1_024, 4_096, 8_192, SAMPLES as usize];

pub struct Case<'a> {
    pub label: &'static str,
    pub trace: &'a Trace,
    pub timings: &'a [(ChainId, Duration)],
    pub target: &'a Ising,
}

struct ObservableRun<'a> {
    case: &'a Case<'a>,
    observable: &'a str,
    units: &'static str,
    draws: &'a [Vec<f64>],
    times: &'a [Duration],
}

pub fn export(cases: &[Case<'_>]) -> Result<(), ExampleError> {
    let mut runs = Vec::new();
    for case in cases {
        let times: Vec<_> = case.timings.iter().map(|&(_, elapsed)| elapsed).collect();
        for name in case.trace.observable_names() {
            let draws: Vec<Vec<f64>> = case
                .timings
                .iter()
                .map(|&(id, _)| {
                    case.trace
                        .observable_values(id, name)
                        .map(|values| values.copied().collect())
                })
                .collect::<Result<_, _>>()?;
            runs.push(observable_json(&ObservableRun {
                case,
                observable: name,
                units: if name == "energy" {
                    "energy (J=1)"
                } else {
                    "magnetization per spin"
                },
                draws: &draws,
                times: &times,
            })?);
        }
    }
    write_report(&runs, cases)
}

fn write_report(runs: &[Value], cases: &[Case<'_>]) -> Result<(), ExampleError> {
    let output = json!({
        "schema_version": 2,
        "workflow": "rank_ess_efficiency_v1",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "source_revision": option_env!("MCMC_SOURCE_REVISION"),
        "source_dirty": option_env!("MCMC_SOURCE_DIRTY"),
        "cargo_lock": include_str!("../../Cargo.lock"),
        "rust_msrv": env!("CARGO_PKG_RUST_VERSION"),
        "estimator_reference": "ArviZ 0.22.0; Vehtari et al. (2021)",
        "rank_convention": "one_based_average_ties_all_original_prefix_draws_signed_zero_equal",
        "split_convention": "first_and_last_floor_N_over_2",
        "quantile_convention": "linear_type_7_all_original_prefix_draws",
        "timing_scope": "sequential_production_sampling_observation_recording_and_summaries_excluding_warmup_analysis_export",
        "prefixes": PREFIXES,
        "runs": runs,
    });
    fs::create_dir_all("target")?;
    let mut file = BufWriter::new(File::create("target/diagnostics.json")?);
    serde_json::to_writer_pretty(&mut file, &output)?;
    writeln!(file)?;
    file.flush()?;
    let mut csv = BufWriter::new(File::create("target/diagnostics_trace.csv")?);
    writeln!(csv, "run_id,observable,chain_id,draw,value")?;
    for case in cases {
        for name in case.trace.observable_names() {
            for &(id, _) in case.timings {
                for (draw, value) in case.trace.observable_values(id, name)?.enumerate() {
                    writeln!(
                        csv,
                        "ising_1d-{}-{name}-seeds42-45,{name},chain-{},{},{value}",
                        case.label,
                        id.get(),
                        draw + 1
                    )?;
                }
            }
        }
    }
    csv.flush()?;
    println!("diagnostics JSON: target/diagnostics.json; trace CSV: target/diagnostics_trace.csv");
    println!(
        "Ising rank/ESS-efficiency data: energy and magnetization, {} reranked prefixes each",
        PREFIXES.len()
    );
    Ok(())
}

fn observable_json(run: &ObservableRun<'_>) -> Result<Value, ExampleError> {
    let target = run.case.target;
    let prefixes: Vec<_> = PREFIXES
        .into_iter()
        .map(|length| prefix_json(run, length))
        .collect::<Result<_, _>>()?;
    let chains: Vec<_> = run.draws.iter().enumerate().map(|(index, draws)| json!({
        "chain_id": format!("chain-{index}"), "seed": 42 + index,
        "start": (["all_up", "all_down", "alternating_down_up", "alternating_up_down"][index]),
        "draws": draws, "elapsed_seconds": run.times[index].as_secs_f64(),
    })).collect();
    Ok(json!({
        "run_id": format!("ising_1d-{}-{}-seeds42-45", run.case.label, run.observable),
        "scenario": format!("ising_{}_{}", run.case.label, run.observable), "observable": run.observable, "units": run.units,
        "target_before_transform": "open_boundary_zero_field_1d_ising", "transform": "identity",
        "model": {"spins": 50, "coupling": target.coupling, "beta": target.beta, "boundary": "open", "field": 0},
        "proposal": "uniform_single_spin_flip",
        "rng": "rand::rngs::StdRng; exact version in cargo_lock", "warmup_per_chain": BURN_IN,
        "recording_interval": 1, "chain_identity": "original_unsplit",
        "chains": chains, "prefixes": prefixes,
    }))
}

fn prefix_json(run: &ObservableRun<'_>, length: usize) -> Result<Value, ExampleError> {
    let chains: Vec<_> = run.draws.iter().map(|draws| &draws[..length]).collect();
    let ranks = PooledRanks::from_chains(&chains)?;
    let by_chain: Vec<_> = (0..ranks.chain_count())
        .map(|index| {
            json!({
                "chain_id": format!("chain-{index}"), "ranks": ranks.chain(index),
            })
        })
        .collect();
    let times = (length == SAMPLES as usize).then_some(run.times);
    let timing = times
        .map(|times| DiagnosticTiming::try_new(times.iter().copied().sum(), chains.len(), length))
        .transpose()?;
    let timing_reason = "prefix_not_timed";
    let rhat = CombinedRhat::estimate(&chains)?;
    let tail = TailEss::estimate(&chains)?;
    let quantiles: Vec<_> = [0.05, 0.5, 0.95].into_iter().map(|probability| {
        let mcse = QuantileMcse::estimate(&chains, probability);
        json!({
            "probability": probability,
            "ess": ess_json(EssEstimate::estimate(&chains, EssEstimator::Quantile(probability)), timing.as_ref(), timing_reason),
            "mcse": metric(mcse.map(QuantileMcse::value)),
            "quantile": mcse.ok().map(QuantileMcse::quantile),
            "interval": mcse.ok().map(QuantileMcse::interval),
        })
    }).collect();
    let single: Vec<_> = chains
        .iter()
        .enumerate()
        .map(|(index, draws)| {
            single_chain_json(index, draws, times.map(|times| times[index]), timing_reason)
        })
        .collect();
    let spread = OnlineStats::try_from_iter(chains.iter().flat_map(|chain| chain.iter().copied()));
    Ok(json!({
        "samples_per_chain": length, "chain_count": chains.len(),
        "original_draws": ranks.sample_count(), "used_split_draws": chains.len() * 2 * (length / 2),
        "samples_per_split_chain": length / 2, "omitted_middle_per_chain": length % 2,
        "timing": {"status": if timing.is_some() { "measured" } else { "unavailable" },
            "reason": if timing.is_some() { None } else { Some(timing_reason) },
            "elapsed_seconds": timing.map(|timing| timing.elapsed().as_secs_f64())},
        "ranks": by_chain,
        "rhat": {
            "classical": metric(SplitRhat::estimate(&chains).map(SplitRhat::value)),
            "rank_normalized": metric(rhat.rank_normalized().map(RankNormalizedSplitRhat::value)),
            "folded": metric(rhat.folded().map(FoldedRankNormalizedSplitRhat::value)),
            "combined": optional_metric(rhat.value(), "component_unavailable"),
        },
        "ess": {
            "mean": ess_json(EssEstimate::estimate(&chains, EssEstimator::Mean), timing.as_ref(), timing_reason),
            "bulk": ess_json(EssEstimate::estimate(&chains, EssEstimator::Bulk), timing.as_ref(), timing_reason),
            "tail": {"estimate": optional_metric(tail.value(), "component_unavailable"),
                "relative": tail.relative(), "rate": timing.as_ref().map_or_else(
                    || unavailable(timing_reason),
                    |timing| match tail.per_second(Some(timing)) {
                        Ok(value) => optional_metric(value, "component_unavailable"),
                        Err(error) => failure(&error),
                    },
                ),
                "lower": ess_json(tail.lower(), timing.as_ref(), timing_reason),
                "upper": ess_json(tail.upper(), timing.as_ref(), timing_reason)},
        },
        "mean_mcse": metric(MeanMcse::estimate(&chains).map(MeanMcse::value)),
        "pooled_sample_sd": match spread { Ok(stats) => optional_metric(stats.sample_std_dev(), "insufficient_samples"), Err(error) => failure(&error) },
        "quantiles": quantiles, "single_chain": single,
    }))
}

fn ess_json(
    ess: Result<EssEstimate, MonteCarloError>,
    timing: Option<&DiagnosticTiming>,
    timing_reason: &str,
) -> Value {
    match ess {
        Ok(ess) => json!({
            "estimate": metric(Ok::<_, MonteCarloError>(ess.value())),
            "estimator": format!("{:?}", ess.estimator()), "relative": ess.relative(),
            "regularized": ess.is_regularized(),
            "rate": if timing.is_some() { metric(ess.per_second(timing)) } else { unavailable(timing_reason) },
        }),
        Err(error) => {
            json!({"estimate": failure(&error), "relative": null, "regularized": null, "rate": failure(&error)})
        }
    }
}

fn single_chain_json(
    index: usize,
    draws: &[f64],
    elapsed: Option<Duration>,
    timing_reason: &str,
) -> Value {
    let acf = Autocorrelation::estimate(draws, (draws.len() - 1).min(511));
    let time = acf
        .as_ref()
        .map_err(|error| *error)
        .and_then(Autocorrelation::integrated_time);
    let binning = BinningAnalysis::try_from_iter(draws.iter().copied());
    json!({
        "chain_id": format!("chain-{index}"),
        "acf": match &acf { Ok(acf) => json!({"status": "estimated", "values": acf.values()}), Err(error) => failure(error) },
        "integrated_time": metric(time.map(IntegratedAutocorrelationTime::estimate)),
        "mean_ess": metric(time.map(IntegratedAutocorrelationTime::effective_sample_size)),
        "mean_ess_rate": match (time, elapsed) {
            (Ok(time), Some(elapsed)) => metric(time.effective_sample_size_per_second(elapsed)),
            (Err(error), _) => failure(&error),
            (_, None) => unavailable(timing_reason),
        },
        "blocked_mean_error": match binning {
            Ok(binning) => json!({"status": "estimated", "levels": binning.estimates().map(|estimate| json!({
                "block_size": estimate.block_size(), "block_count": estimate.block_count(),
                "used_draws": estimate.block_size() * estimate.block_count(),
                "standard_error": optional_metric(estimate.standard_error(), "insufficient_blocks"),
            })).collect::<Vec<_>>() }),
            Err(error) => failure(&error),
        },
    })
}

fn metric<E: Display + Debug>(result: Result<f64, E>) -> Value {
    match result {
        Ok(value) => json!({"status": "estimated", "value": value}),
        Err(error) => failure(&error),
    }
}

fn failure(error: &(impl Display + Debug)) -> Value {
    json!({"status": "unavailable", "value": null, "reason": format!("{error:?}"), "message": error.to_string()})
}

fn unavailable(reason: &str) -> Value {
    json!({"status": "unavailable", "value": null, "reason": reason})
}

fn optional_metric(value: Option<f64>, reason: &str) -> Value {
    value.map_or_else(
        || unavailable(reason),
        |value| json!({"status": "estimated", "value": value}),
    )
}
