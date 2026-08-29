//! Downstream compile contract for crate-root and scoped-prelude exports.

use core::convert::Infallible;

use markov_chain_monte_carlo::prelude::{self, by_value, delayed, in_place, testing};
use markov_chain_monte_carlo::{
    AdditiveTarget, BinningAnalysis, BinningEstimate, Chain, ChainCheckpoint, ChainId, DelayedStep,
    DetailedBalanceBatchReport, DetailedBalanceConfig, DetailedBalanceDelayedTransition,
    DetailedBalanceDirection, DetailedBalanceError, DetailedBalanceFailure, DetailedBalanceReport,
    DetailedBalanceState, InvalidThinningInterval, McmcError, Observable, ObservedDelayedStep,
    ObservedMutStep, OnlineStats, Proposal, ProposalMut, SampleBuffer, Sampler, StatisticsError,
    Step, StepOutcome, StepRejectionReason, Target, ThinningInterval, Trace, TraceError,
    TraceRecord, TraceRecorder, TraceStepOutcome, TryObservedMutStepResult,
};
use rand::{Rng, rngs::StdRng};

struct Smoke;

impl Target<f64> for Smoke {
    fn log_prob(&self, _: &f64) -> f64 {
        0.0
    }
}

impl Proposal<f64> for Smoke {
    fn propose<R: Rng + ?Sized>(&self, current: &f64, _: &mut R) -> f64 {
        *current
    }
}

impl ProposalMut<f64> for Smoke {
    type Undo = ();
    type Info = ();

    fn propose_mut<R: Rng + ?Sized>(&mut self, _: &mut f64, _: &mut R) -> Option<Self::Undo> {
        Some(())
    }

    fn info(&self, _: &f64, _token: &Self::Undo) {}

    fn undo(&mut self, _: &mut f64, (): Self::Undo) {}
}

impl delayed::DelayedProposal<f64> for Smoke {
    type Plan = ();
    type Info = ();
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        _: &f64,
        _: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error> {
        Ok(Some(()))
    }

    fn proposed_log_prob<T: delayed::Target<f64>>(
        &self,
        state: &f64,
        (): &Self::Plan,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(state))
    }

    fn info(&self, (): &Self::Plan) -> Self::Info {}

    fn commit<R: Rng + ?Sized>(
        &mut self,
        _: &mut f64,
        (): Self::Plan,
        _: &mut R,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[test]
fn downstream_exports_compile() {
    fn needs_target<T: prelude::Target<f64>>() {}
    fn needs_by_value<P: by_value::Proposal<f64>>() {}
    fn needs_in_place<P: in_place::ProposalMut<f64>>() {}
    fn needs_delayed<P: delayed::DelayedProposal<f64>>() {}
    fn needs_testing_target<T: testing::Target<f64>>() {}
    fn needs_testing_proposal<P: testing::Proposal<f64>>() {}
    fn needs_observable<O: Observable<f64, Output = f64>>(_: &mut O) {}

    needs_target::<Smoke>();
    needs_by_value::<Smoke>();
    needs_in_place::<Smoke>();
    needs_delayed::<Smoke>();
    needs_testing_target::<Smoke>();
    needs_testing_proposal::<Smoke>();

    let mut observable = |state: &f64| *state;
    needs_observable(&mut observable);

    let _: Option<AdditiveTarget<Smoke, Smoke>> = None;
    let _: Option<Chain<f64>> = None;
    let _: Option<ChainCheckpoint<f64>> = None;
    let _: Option<Step<()>> = None;
    let _: Option<DelayedStep<()>> = None;
    let _: Option<StepOutcome> = None;
    let _: Option<StepRejectionReason> = None;
    let _: Option<ChainId> = None;
    let _: Option<Trace> = None;
    let _: Option<TraceError> = None;
    let _: Option<TraceRecord> = None;
    let _: Option<TraceRecorder> = None;
    let _: Option<TraceStepOutcome> = None;
    let _: Option<McmcError> = None;
    let _: Option<SampleBuffer<f64>> = None;
    let _: Option<Sampler<'_, f64, Smoke, Smoke, StdRng>> = None;
    let _: Option<ThinningInterval> = None;
    let _: Option<InvalidThinningInterval> = None;
    let _: Option<ObservedDelayedStep<(), f64>> = None;
    let _: Option<ObservedMutStep<(), f64>> = None;
    let _: Option<TryObservedMutStepResult<(), f64, Infallible>> = None;
    let _: Option<BinningAnalysis> = None;
    let _: Option<BinningEstimate> = None;
    let _: Option<OnlineStats> = None;
    let _: Option<StatisticsError> = None;
    let _: Option<DetailedBalanceConfig> = None;
    let _: Option<DetailedBalanceDirection> = None;
    let _: Option<DetailedBalanceError> = None;
    let _: Option<DetailedBalanceFailure> = None;
    let _: Option<DetailedBalanceDelayedTransition<'_, f64, ()>> = None;
    let _: Option<DetailedBalanceBatchReport> = None;
    let _: Option<DetailedBalanceReport> = None;
    let _: Option<DetailedBalanceState> = None;

    let _: Option<prelude::TraceRecorder> = None;
    let _: Option<prelude::OnlineStats> = None;
    let _: Option<by_value::Step<()>> = None;
    let _: Option<by_value::ObservedStep<f64>> = None;
    let _: Option<by_value::DiscreteProposalRatio> = None;
    let _: Option<in_place::Step<()>> = None;
    let _: Option<in_place::ObservedMutStep<(), f64>> = None;
    let _: Option<in_place::DiscreteProposalRatioError> = None;
    let _: Option<delayed::DelayedStep<()>> = None;
    let _: Option<delayed::ObservedDelayedStep<(), f64>> = None;
    let _: Option<delayed::DelayedStepError<Infallible>> = None;
    let _: Option<testing::DetailedBalanceConfig> = None;
    let _: Option<testing::DiscreteProposalRatio> = None;
}
