//! Consumer-owned report assembly; every numerical diagnostic comes from public APIs.

use std::{
    fmt::{Debug, Display},
    fs::{self, File},
    io::{BufWriter, Write},
    time::Duration,
};

use markov_chain_monte_carlo::{
    Autocorrelation, BinningAnalysis, CombinedRhat, DiagnosticTiming, EssEstimate, EssEstimator,
    FoldedRankNormalizedSplitRhat, IntegratedAutocorrelationTime, MeanMcse, MonteCarloError,
    OnlineStats, PooledRanks, QuantileMcse, RankNormalizedSplitRhat, SplitRhat, TailEss,
};
use serde_json::{Value, json};

use super::{DRAWS, DiagnosticsError, SEEDS, STARTS, WARMUP};

const PREFIXES: [usize; 5] = [64, 127, 256, 512, DRAWS];

struct Scenario<'a> {
    name: &'static str,
    transform: &'static str,
    observable: &'static str,
    units: &'static str,
    half_width: f64,
    draws: &'a [Vec<f64>],
    times: Option<&'a [Duration]>,
}

pub fn export(
    baseline: &[Vec<f64>],
    baseline_times: &[Duration],
    slow: &[Vec<f64>],
    slow_times: &[Duration],
) -> Result<(), DiagnosticsError> {
    let shifted: Vec<Vec<_>> = baseline
        .iter()
        .enumerate()
        .map(|(chain, draws)| {
            draws
                .iter()
                .map(|&value| if chain == 3 { value + 4.0 } else { value })
                .collect()
        })
        .collect();
    let rescaled: Vec<Vec<_>> = baseline
        .iter()
        .enumerate()
        .map(|(chain, draws)| {
            draws
                .iter()
                .map(|&value| if chain == 3 { value * 4.0 } else { value })
                .collect()
        })
        .collect();
    let counts: Vec<Vec<_>> = baseline
        .iter()
        .map(|draws| draws.iter().map(|value| value.abs().floor()).collect())
        .collect();
    let scenarios = [
        Scenario {
            name: "well_behaved",
            transform: "identity",
            observable: "position",
            units: "position",
            half_width: 2.0,
            draws: baseline,
            times: Some(baseline_times),
        },
        Scenario {
            name: "location_disagreement",
            transform: "add 4 to original chain 3 after sampling",
            observable: "position",
            units: "position",
            half_width: 2.0,
            draws: &shifted,
            times: None,
        },
        Scenario {
            name: "scale_disagreement",
            transform: "multiply original chain 3 by 4 after sampling",
            observable: "position",
            units: "position",
            half_width: 2.0,
            draws: &rescaled,
            times: None,
        },
        Scenario {
            name: "slow_mixing",
            transform: "identity",
            observable: "position",
            units: "position",
            half_width: 0.1,
            draws: slow,
            times: Some(slow_times),
        },
        Scenario {
            name: "discrete_counts",
            transform: "floor(abs(position)) after sampling",
            observable: "count",
            units: "counts",
            half_width: 2.0,
            draws: &counts,
            times: None,
        },
    ];
    write_report(&scenarios)
}

fn write_report(scenarios: &[Scenario<'_>]) -> Result<(), DiagnosticsError> {
    let runs: Vec<_> = scenarios
        .iter()
        .map(scenario_json)
        .collect::<Result<_, _>>()?;
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
        "timing_scope": "sequential_production_sampling_and_recording_excluding_allocation_warmup_analysis_export",
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
    for scenario in scenarios {
        for (chain, draws) in scenario.draws.iter().enumerate() {
            for (draw, value) in draws.iter().enumerate() {
                writeln!(
                    csv,
                    "{}-seeds40-43,{},chain-{chain},{},{value}",
                    scenario.name,
                    scenario.observable,
                    draw + 1
                )?;
            }
        }
    }
    csv.flush()?;
    println!("diagnostics JSON: target/diagnostics.json; trace CSV: target/diagnostics_trace.csv");
    println!(
        "Rank/ESS-efficiency data: five scenarios, {} reranked prefixes each",
        PREFIXES.len()
    );
    Ok(())
}

fn scenario_json(scenario: &Scenario<'_>) -> Result<Value, DiagnosticsError> {
    let prefixes: Vec<_> = PREFIXES
        .into_iter()
        .map(|length| prefix_json(scenario, length))
        .collect::<Result<_, _>>()?;
    let chains: Vec<_> = scenario.draws.iter().enumerate().map(|(index, draws)| json!({
        "chain_id": format!("chain-{index}"), "seed": SEEDS[index], "start": STARTS[index],
        "draws": draws, "elapsed_seconds": scenario.times.map(|times| times[index].as_secs_f64()),
    })).collect();
    Ok(json!({
        "run_id": format!("{}-seeds40-43", scenario.name),
        "scenario": scenario.name, "observable": scenario.observable, "units": scenario.units,
        "target_before_transform": "standard_normal", "transform": scenario.transform,
        "proposal": "uniform_random_walk", "proposal_half_width": scenario.half_width,
        "rng": "rand::rngs::StdRng; exact version in cargo_lock", "warmup_per_chain": WARMUP,
        "recording_interval": 1, "chain_identity": "original_unsplit",
        "chains": chains, "prefixes": prefixes,
    }))
}

fn prefix_json(scenario: &Scenario<'_>, length: usize) -> Result<Value, DiagnosticsError> {
    let chains: Vec<_> = scenario
        .draws
        .iter()
        .map(|draws| &draws[..length])
        .collect();
    let ranks = PooledRanks::from_chains(&chains)?;
    let by_chain: Vec<_> = (0..ranks.chain_count())
        .map(|index| {
            json!({
                "chain_id": format!("chain-{index}"), "ranks": ranks.chain(index),
            })
        })
        .collect();
    let times = scenario.times.filter(|_| length == DRAWS);
    let timing = times
        .map(|times| DiagnosticTiming::try_new(times.iter().copied().sum(), chains.len(), length))
        .transpose()?;
    let timing_reason = if length < DRAWS {
        "prefix_not_timed"
    } else {
        "derived_observable_not_timed"
    };
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
