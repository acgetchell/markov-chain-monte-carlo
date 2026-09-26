//! Subscriber-visible sampling contracts for the optional tracing integration.

#![cfg(feature = "tracing")]

use core::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use approx::assert_relative_eq;
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::DelayedProposal;
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{
    AdaptiveScale, Chain, OnlineStats, Sampler, Target, ThinningInterval, TunableProposal,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};

#[derive(Default, Debug)]
struct Fields(BTreeMap<String, String>);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
}

#[derive(Debug)]
struct Record {
    name: &'static str,
    fields: Fields,
    scope: Vec<&'static str>,
}

#[derive(Default)]
struct Records {
    events: Vec<Record>,
    spans: Vec<Record>,
    closed: usize,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Records>>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, _: &Id, _: Context<'_, S>) {
        assert_eq!(attrs.metadata().target(), "markov_chain_monte_carlo");
        assert_eq!(*attrs.metadata().level(), Level::DEBUG);
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        self.0.lock().unwrap().spans.push(Record {
            name: attrs.metadata().name(),
            fields,
            scope: Vec::new(),
        });
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        assert_eq!(event.metadata().target(), "markov_chain_monte_carlo");
        assert_eq!(*event.metadata().level(), Level::TRACE);
        let mut fields = Fields::default();
        event.record(&mut fields);
        let scope = ctx.event_scope(event).map_or_else(Vec::new, |scope| {
            scope.from_root().map(|span| span.name()).collect()
        });
        self.0.lock().unwrap().events.push(Record {
            name: event.metadata().name(),
            fields,
            scope,
        });
    }

    fn on_close(&self, _: Id, _: Context<'_, S>) {
        self.0.lock().unwrap().closed += 1;
    }
}

fn capture(run: impl FnOnce()) -> Records {
    let capture = Capture::default();
    tracing::subscriber::with_default(Registry::default().with(capture.clone()), run);
    Arc::try_unwrap(capture.0)
        .ok()
        .unwrap()
        .into_inner()
        .unwrap()
}

#[derive(Default)]
struct Score(Cell<usize>);

impl Target<i32> for Score {
    fn log_prob(&self, state: &i32) -> f64 {
        self.0.set(self.0.get() + 1);
        -f64::from(*state)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Fault {
    #[default]
    None,
    Plan,
    Score,
    Ratio,
    Commit,
    Mismatch,
}

struct Move {
    ratio: f64,
    absent: bool,
    fault: Fault,
    metadata: Cell<usize>,
}

impl Default for Move {
    fn default() -> Self {
        Self {
            ratio: 1.0,
            absent: false,
            fault: Fault::None,
            metadata: Cell::new(0),
        }
    }
}

impl Proposal<i32> for Move {
    fn propose<R: Rng + ?Sized>(&self, current: &i32, _: &mut R) -> i32 {
        current + 1
    }

    fn log_q_ratio(&self, _: &i32, _: &i32) -> f64 {
        self.ratio
    }
}

impl ProposalMut<i32> for Move {
    type Undo = i32;
    type Info = ();

    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut i32, _: &mut R) -> Option<i32> {
        if self.absent {
            return None;
        }
        let old = *state;
        *state += 1;
        Some(old)
    }

    fn info(&self, _: &i32, _: &i32) {
        self.metadata.set(self.metadata.get() + 1);
    }

    fn undo(&mut self, state: &mut i32, token: i32) {
        *state = token;
    }

    fn log_q_ratio(&self, _: &i32, _: &i32) -> f64 {
        self.ratio
    }
}

impl DelayedProposal<i32> for Move {
    type Plan = i32;
    type Info = ();
    type Error = &'static str;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &i32,
        _: &mut R,
    ) -> Result<Option<i32>, Self::Error> {
        if self.fault == Fault::Plan {
            return Err("plan");
        }
        Ok((!self.absent).then_some(state + 1))
    }

    fn proposed_log_prob<T: Target<i32> + ?Sized>(
        &self,
        _: &i32,
        plan: &i32,
        target: &T,
    ) -> Result<f64, Self::Error> {
        if self.fault == Fault::Score {
            return Err("score");
        }
        Ok(target.log_prob(plan))
    }

    fn log_q_ratio(&self, _: &i32, _: &i32) -> Result<f64, Self::Error> {
        if self.fault == Fault::Ratio {
            return Err("ratio");
        }
        Ok(self.ratio)
    }

    fn info(&self, _: &i32) {
        self.metadata.set(self.metadata.get() + 1);
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut i32,
        plan: i32,
        _: &mut R,
    ) -> Result<(), Self::Error> {
        if self.fault == Fault::Commit {
            return Err("commit");
        }
        *state = plan + i32::from(self.fault == Fault::Mismatch);
        Ok(())
    }
}

impl TunableProposal for Move {
    fn set_scale(&mut self, _: f64) {}
}

#[test]
fn completed_steps_report_post_transition_metrics_for_every_kernel() {
    for kernel in ["by_value", "in_place", "delayed", "delayed_checked"] {
        let records = capture(|| {
            let target = Score::default();
            let mut chain = Chain::new(0, &target).unwrap();
            let mut proposal = Move::default();
            let mut rng = StdRng::seed_from_u64(21);
            for round in 0..if kernel == "by_value" { 2 } else { 3 } {
                proposal.ratio = if round == 0 { 1.0 } else { f64::NEG_INFINITY };
                proposal.absent = round == 2;
                let _step = match kernel {
                    "by_value" => chain.step(&target, &proposal, &mut rng).unwrap(),
                    "in_place" => chain.step_mut(&target, &mut proposal, &mut rng).unwrap(),
                    "delayed" => chain
                        .step_delayed(&target, &mut proposal, &mut rng)
                        .unwrap(),
                    _ => chain
                        .step_delayed_checked(&target, &mut proposal, &mut rng)
                        .unwrap(),
                };
                assert_eq!(*chain.state(), 1);
            }
        });
        assert_eq!(
            records.events.len(),
            if kernel == "by_value" { 2 } else { 3 }
        );
        for (index, event) in records.events.iter().enumerate() {
            let fields = &event.fields.0;
            assert_eq!(fields["kernel"], kernel);
            assert_eq!(fields["step"], (index + 1).to_string());
            assert_eq!(fields["accepted"], (index == 0).to_string());
            assert_eq!(fields["proposed"], (index != 2).to_string());
            assert_eq!(fields["log_prob"], "-1.0");
            let expected_rate = [1.0, 0.5, 1.0 / 3.0][index];
            assert_relative_eq!(
                fields["acceptance_rate"].parse::<f64>().unwrap(),
                expected_rate
            );
            assert!(event.scope.is_empty());
        }
    }
}

#[test]
fn failed_steps_do_not_publish_completed_transitions() {
    let records = capture(|| {
        let target = Score::default();
        let mut chain = Chain::new(0, &target).unwrap();
        let mut rng = StdRng::seed_from_u64(21);
        let mut proposal = Move {
            ratio: f64::NAN,
            ..Move::default()
        };
        assert!(chain.step(&target, &proposal, &mut rng).is_err());
        assert!(chain.step_mut(&target, &mut proposal, &mut rng).is_err());
        assert!(
            chain
                .step_delayed(&target, &mut proposal, &mut rng)
                .is_err()
        );
        assert!(
            chain
                .step_delayed_checked(&target, &mut proposal, &mut rng)
                .is_err()
        );
        proposal.ratio = 1.0;
        for fault in [Fault::Plan, Fault::Score, Fault::Ratio, Fault::Commit] {
            proposal.fault = fault;
            assert!(
                chain
                    .step_delayed(&target, &mut proposal, &mut rng)
                    .is_err()
            );
            assert!(
                chain
                    .step_delayed_checked(&target, &mut proposal, &mut rng)
                    .is_err()
            );
        }
        proposal.fault = Fault::Mismatch;
        assert!(
            chain
                .step_delayed_checked(&target, &mut proposal, &mut rng)
                .is_err()
        );
        assert_eq!(*chain.state(), 0);
        assert_eq!(chain.total_steps(), 0);
    });
    assert!(records.events.is_empty());
}

type TestSampler<'a> = Sampler<'a, i32, Score, Move, StdRng>;

fn assert_run(span_name: &str, run: impl FnOnce(&mut TestSampler<'_>)) {
    let records = capture(|| {
        let target = Score::default();
        let mut rng = StdRng::seed_from_u64(21);
        let mut sampler = Sampler::from_state(0, &target, Move::default(), &mut rng).unwrap();
        run(&mut sampler);
        assert_eq!(sampler.chain_ref().total_steps(), 3);
        assert_eq!(*sampler.chain_ref().state(), 3);
        assert_eq!(sampler.proposal_ref().metadata.get(), 0);
    });
    assert_eq!(records.spans.len(), 1);
    assert_eq!(records.closed, 1);
    assert_eq!(records.spans[0].name, span_name);
    assert_eq!(records.spans[0].fields.0["steps"], "3");
    assert_eq!(records.spans[0].fields.0["start_step"], "0");
    assert_eq!(records.events.len(), 3);
    for event in &records.events {
        assert_eq!(event.scope, [span_name]);
    }
}

#[test]
fn bulk_runs_and_observation_variants_scope_every_transition() {
    let thin = ThinningInterval::new(2).unwrap();
    // Each family shares the thinned loop, but owns its unthinned loops.
    macro_rules! check_family {
        ($run:ident, $observe:ident, $stream:ident, $try_observe:ident, $try_stream:ident, $thinned:ident) => {
            assert_run(stringify!($run), |s| s.$run(3).unwrap());
            assert_run(stringify!($observe), |s| {
                let _ = s.$observe(3, &mut |x: &i32| f64::from(*x)).unwrap();
            });
            assert_run(stringify!($stream), |s| {
                s.$stream(3, &mut |x: &i32| f64::from(*x), &mut OnlineStats::new())
                    .unwrap();
            });
            assert_run(stringify!($try_observe), |s| {
                let _ = s
                    .$try_observe(3, &mut |x: &i32| Ok::<_, &'static str>(f64::from(*x)))
                    .unwrap();
            });
            assert_run(stringify!($try_stream), |s| {
                s.$try_stream(
                    3,
                    &mut |x: &i32| Ok::<_, &'static str>(f64::from(*x)),
                    &mut OnlineStats::new(),
                )
                .unwrap();
            });
            assert_run("run_thinning_loop", |s| {
                let samples = s.$thinned(3, thin).unwrap();
                assert_eq!(samples.as_slice(), [2]);
            });
        };
    }
    check_family!(
        run,
        run_observing,
        run_observing_into,
        try_run_observing,
        try_run_observing_into,
        run_with_thinning
    );
    check_family!(
        run_mut,
        run_mut_observing,
        run_mut_observing_into,
        try_run_mut_observing,
        try_run_mut_observing_into,
        run_mut_with_thinning
    );
    check_family!(
        run_delayed,
        run_delayed_observing,
        run_delayed_observing_into,
        try_run_delayed_observing,
        try_run_delayed_observing_into,
        run_delayed_with_thinning
    );
}

#[test]
fn spans_close_on_error_and_empty_runs_without_leaking_context() {
    let records = capture(|| {
        let target = Score::default();
        let mut rng = StdRng::seed_from_u64(21);
        let mut sampler = Sampler::from_state(0, &target, Move::default(), &mut rng).unwrap();
        sampler.run(0).unwrap();
        let mut observable = |_: &i32| Err::<f64, _>("observation");
        assert!(sampler.try_run_observing(5, &mut observable).is_err());
        sampler.proposal_mut().fault = Fault::Commit;
        assert!(sampler.run_delayed(5).is_err());
        sampler.proposal_mut().fault = Fault::None;
        let _ = sampler.step().unwrap();
    });
    assert_eq!(records.closed, 3);
    assert_eq!(records.events.len(), 2);
    assert_eq!(records.events[0].scope, ["try_run_observing"]);
    assert!(records.events[1].scope.is_empty());
    assert_eq!(records.events[1].fields.0["step"], "2");
}

#[test]
fn chunks_resets_and_warmup_use_current_chain_counters() {
    let records = capture(|| {
        let target = Score::default();
        let mut rng = StdRng::seed_from_u64(21);
        let mut sampler = Sampler::from_state(0, &target, Move::default(), &mut rng).unwrap();
        let _ = sampler.run_chunk(2).unwrap();
        let _ = sampler.run_mut_chunk(1).unwrap();
        let _ = sampler.run_delayed_chunk(1).unwrap();
        let _ = sampler
            .run_delayed_chunk_observing(1, |step, state| {
                assert!(step.outcome().is_accepted());
                assert_eq!(*state, 5);
            })
            .unwrap();
        sampler.reset_counters();
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
        sampler.warm_up(1, &mut tuning).unwrap();
        sampler.warm_up_mut(1, &mut tuning).unwrap();
        sampler.warm_up_delayed(1, &mut tuning).unwrap();
    });
    let steps: Vec<_> = records
        .events
        .iter()
        .map(|e| e.fields.0["step"].as_str())
        .collect();
    assert_eq!(steps, ["1", "2", "3", "4", "5", "1", "2", "3"]);
    assert_eq!(records.closed, 7);
    assert_eq!(records.spans[1].fields.0["start_step"], "2");
    assert_eq!(records.spans[4].fields.0["start_step"], "0");
    assert_eq!(records.events[4].scope, ["run_delayed_chunk_observing"]);
    assert_eq!(records.events[5].scope, ["warm_up"]);
}

#[test]
fn subscription_preserves_seeded_results_rng_and_target_evaluations() {
    fn run() -> (i32, usize, usize, usize, u64) {
        let target = Score::default();
        let mut rng = StdRng::seed_from_u64(21);
        let mut proposal = Move {
            ratio: 0.0,
            ..Move::default()
        };
        let mut chain = Chain::new(0, &target).unwrap();
        for _ in 0..100 {
            let _ = chain.step(&target, &proposal, &mut rng).unwrap();
            let _ = chain.step_mut(&target, &mut proposal, &mut rng).unwrap();
            let _ = chain
                .step_delayed(&target, &mut proposal, &mut rng)
                .unwrap();
            let _ = chain
                .step_delayed_checked(&target, &mut proposal, &mut rng)
                .unwrap();
        }
        (
            *chain.state(),
            chain.accepted(),
            chain.rejected(),
            target.0.get(),
            rng.random(),
        )
    }
    let baseline =
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), run);
    let records = capture(|| assert_eq!(run(), baseline));
    assert_eq!(records.events.len(), 400);
    let filtered = Capture::default();
    let subscriber = Registry::default().with(
        filtered
            .clone()
            .with_filter(tracing_subscriber::filter::LevelFilter::INFO),
    );
    tracing::subscriber::with_default(subscriber, || assert_eq!(run(), baseline));
    assert!(filtered.0.lock().unwrap().events.is_empty());
}
