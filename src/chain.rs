//! MCMC chain implementation.

use core::{cmp::Ordering, hint::cold_path};

use std::error::Error;
use std::fmt;

use rand::distr::Open01;
use rand::{Rng, RngExt};
#[cfg(feature = "serde")]
use serde::{Serialize, Serializer};

use crate::{DelayedProposal, McmcError, Proposal, ProposalMut, Target};

/// Decide acceptance from a precomputed `log(u)`.
///
/// This helper keeps the edge-case policy in one place: nonnegative
/// log-acceptance ratios always accept, negative ratios compare in log space,
/// and NaN acceptance ratios are treated as ordinary rejections.
fn accept_from_log_uniform(log_alpha: f64, log_u: f64) -> bool {
    // NaN can arise from valid inputs such as -inf - (-inf); treat that as a
    // zero acceptance probability instead of relying on float comparison quirks.
    if log_alpha.is_nan() {
        cold_path();
        false
    } else if log_alpha >= 0.0 {
        true
    } else {
        log_u < log_alpha
    }
}

/// Draw a log-uniform variate and apply Metropolis-Hastings acceptance.
///
/// This exists so all step implementations use the same stable log-space
/// acceptance rule without exponentiating tiny probabilities.
fn accept_log_alpha<R: Rng + ?Sized>(log_alpha: f64, rng: &mut R) -> bool {
    let log_u: f64 = rng.sample::<f64, _>(Open01).ln();
    accept_from_log_uniform(log_alpha, log_u)
}

/// Check a proposed state's log-probability.
///
/// This centralizes the crate's numeric contract for proposed states:
/// finite values and `-inf` are allowed, while NaN and `+inf` are reported
/// with the proposal-specific `McmcError` variants.
fn check_proposed_log_prob(log_prob: f64) -> Result<(), McmcError> {
    if log_prob.is_nan() {
        cold_path();
        return Err(McmcError::NanProposedLogProb);
    }
    if log_prob == f64::INFINITY {
        cold_path();
        return Err(McmcError::InfiniteProposedLogProb);
    }
    Ok(())
}

/// Check a proposal log q-ratio.
///
/// This keeps by-value, in-place, and delayed proposal paths orthogonal in
/// implementation while preserving identical NaN and `+inf` error semantics.
fn check_log_q_ratio(log_q: f64) -> Result<(), McmcError> {
    if log_q.is_nan() {
        cold_path();
        return Err(McmcError::NanLogQRatio);
    }
    if log_q == f64::INFINITY {
        cold_path();
        return Err(McmcError::InfiniteLogQRatio);
    }
    Ok(())
}

/// Check that a delayed commit produced the state scored before acceptance.
///
/// The checked delayed path recomputes the committed state's log-probability so
/// proposal authors can catch plan/commit mismatches without leaving the chain's
/// cached log-probability stale.
fn check_committed_log_prob(scored: f64, committed: f64) -> Result<(), McmcError> {
    if committed.is_nan() {
        cold_path();
        return Err(McmcError::NanCommittedLogProb);
    }
    if committed == f64::INFINITY {
        cold_path();
        return Err(McmcError::InfiniteCommittedLogProb);
    }
    if committed.partial_cmp(&scored) != Some(Ordering::Equal) {
        cold_path();
        return Err(McmcError::InconsistentDelayedCommitLogProb);
    }
    Ok(())
}

/// Telemetry for a single Metropolis-Hastings step.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[must_use]
pub struct Step<I> {
    /// Step outcome encoded as a single invariant-bearing value.
    outcome: StepOutcome,
    /// Proposal-specific metadata for the concrete proposal or no-proposal outcome.
    info: Option<I>,
    /// Cached log-probability before the step.
    log_prob_before: f64,
    /// Cached log-probability after the step, when it changed.
    log_prob_after: Option<f64>,
    /// Metropolis-Hastings log acceptance ratio, when one was evaluated.
    log_alpha: Option<f64>,
}

impl<I> Step<I> {
    /// Build telemetry for a no-proposal self-loop.
    ///
    /// This keeps the [`Step`] field-level contract synchronized whenever an
    /// in-place or delayed proposal produces no concrete transition.
    pub(crate) const fn no_proposal(info: Option<I>, log_prob_before: f64) -> Self {
        Self {
            outcome: StepOutcome::NoProposal,
            info,
            log_prob_before,
            log_prob_after: None,
            log_alpha: None,
        }
    }

    /// Build telemetry for an accepted concrete proposal.
    ///
    /// This records the accepted outcome together with the post-step
    /// log-probability used by the chain cache.
    pub(crate) const fn accepted_proposal(
        info: I,
        log_prob_before: f64,
        log_prob_after: f64,
        log_alpha: f64,
    ) -> Self {
        Self {
            outcome: StepOutcome::Accepted,
            info: Some(info),
            log_prob_before,
            log_prob_after: Some(log_prob_after),
            log_alpha: Some(log_alpha),
        }
    }

    /// Build telemetry for a concrete proposal rejected by the M-H draw.
    ///
    /// Rejected proposals leave the cached log-probability unchanged while
    /// preserving the evaluated log-acceptance ratio for diagnostics.
    pub(crate) const fn rejected_proposal(info: I, log_prob_before: f64, log_alpha: f64) -> Self {
        Self {
            outcome: StepOutcome::RejectedProposal,
            info: Some(info),
            log_prob_before,
            log_prob_after: None,
            log_alpha: Some(log_alpha),
        }
    }

    /// Outcome of the completed step.
    ///
    /// The outcome and optional telemetry fields are constructed together, so
    /// callers cannot create contradictory combinations such as an accepted
    /// step without a post-step log-probability.
    pub const fn outcome(&self) -> StepOutcome {
        self.outcome
    }

    /// Proposal-specific metadata for the concrete proposal or no-proposal outcome.
    #[must_use]
    pub const fn info(&self) -> Option<&I> {
        self.info.as_ref()
    }

    /// Cached log-probability before the step.
    #[must_use]
    pub const fn log_prob_before(&self) -> f64 {
        self.log_prob_before
    }

    /// Cached log-probability after the step, when it changed.
    ///
    /// This is `Some` exactly when [`Self::outcome`] is
    /// [`StepOutcome::Accepted`].
    #[must_use]
    pub const fn log_prob_after(&self) -> Option<f64> {
        self.log_prob_after
    }

    /// Metropolis-Hastings log acceptance ratio, when one was evaluated.
    ///
    /// This is `Some` for accepted and rejected concrete proposals and `None`
    /// when no proposal was available.
    #[must_use]
    pub const fn log_alpha(&self) -> Option<f64> {
        self.log_alpha
    }

    /// Why a step was rejected, or `None` when it was accepted.
    ///
    /// This helper distinguishes proposal absence from a concrete proposal
    /// rejected by the Metropolis-Hastings accept/reject draw.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct Flat;
    /// impl Target<()> for Flat {
    ///     fn log_prob(&self, _: &()) -> f64 { 0.0 }
    /// }
    ///
    /// struct NoMove;
    /// impl DelayedProposal<()> for NoMove {
    ///     type Plan = ();
    ///     type Info = ();
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _: &(),
    ///         _: &mut R,
    ///     ) -> Result<Option<()>, Self::Error> {
    ///         Ok(None)
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<()>>(
    ///         &self,
    ///         _: &(),
    ///         _: &(),
    ///         _: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         unreachable!("no plan should not be scored")
    ///     }
    ///
    ///     fn info(&self, _: &()) {}
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _: &mut (),
    ///         _: (),
    ///         _: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         unreachable!("no plan should not be committed")
    ///     }
    /// }
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut proposal = NoMove;
    /// let mut chain = Chain::new((), &Flat).map_err(DelayedStepError::Mcmc)?;
    /// let step = chain.step_delayed(&Flat, &mut proposal, &mut rng)?;
    ///
    /// assert_eq!(step.outcome(), StepOutcome::NoProposal);
    /// assert_eq!(step.rejection_reason(), Some(StepRejectionReason::NoProposal));
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    #[must_use]
    pub const fn rejection_reason(&self) -> Option<StepRejectionReason> {
        match self.outcome {
            StepOutcome::Accepted => None,
            StepOutcome::RejectedProposal => Some(StepRejectionReason::RejectedProposal),
            StepOutcome::NoProposal => Some(StepRejectionReason::NoProposal),
        }
    }
}

/// Outcome of a completed Metropolis-Hastings step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum StepOutcome {
    /// A concrete proposal was accepted and retained.
    Accepted,
    /// A concrete proposal was produced and then rejected by the
    /// Metropolis-Hastings acceptance draw.
    RejectedProposal,
    /// No concrete proposal was available, so the step was an ordinary
    /// self-loop without a Metropolis-Hastings acceptance draw.
    NoProposal,
}

impl StepOutcome {
    /// Whether this outcome accepted and retained a concrete proposal.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::delayed::StepOutcome;
    ///
    /// assert!(StepOutcome::Accepted.is_accepted());
    /// assert!(!StepOutcome::RejectedProposal.is_accepted());
    /// assert!(!StepOutcome::NoProposal.is_accepted());
    /// ```
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        match self {
            Self::Accepted => true,
            Self::RejectedProposal | Self::NoProposal => false,
        }
    }

    /// Whether this outcome includes a concrete proposal.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::delayed::StepOutcome;
    ///
    /// assert!(StepOutcome::Accepted.has_proposal());
    /// assert!(StepOutcome::RejectedProposal.has_proposal());
    /// assert!(!StepOutcome::NoProposal.has_proposal());
    /// ```
    #[must_use]
    pub const fn has_proposal(self) -> bool {
        match self {
            Self::Accepted | Self::RejectedProposal => true,
            Self::NoProposal => false,
        }
    }
}

/// Reason a completed step was counted as a rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum StepRejectionReason {
    /// No concrete proposal was available, so the step was an ordinary
    /// self-loop without a Metropolis-Hastings acceptance draw.
    NoProposal,
    /// A concrete proposal was produced and then rejected by the
    /// Metropolis-Hastings acceptance draw.
    RejectedProposal,
}

/// Telemetry for a delayed-commit Metropolis-Hastings step.
pub type DelayedStep<I> = Step<I>;

/// Portable checkpoint data for a [`Chain`].
///
/// A checkpoint stores the chain state and counters, but deliberately does not
/// store the cached log-probability. Restore checkpoints with
/// [`Chain::from_checkpoint`] so the cache is recomputed from the target that
/// will be used for resumed sampling.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct ChainCheckpoint<S> {
    /// Current state.
    state: S,
    /// Number of accepted moves.
    accepted: usize,
    /// Number of rejected moves.
    rejected: usize,
}

impl<S> ChainCheckpoint<S> {
    /// Create a checkpoint from owned state and counters.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new(1.0_f64, 2, 3);
    ///
    /// assert_eq!(*checkpoint.state(), 1.0);
    /// assert_eq!(checkpoint.accepted(), 2);
    /// assert_eq!(checkpoint.rejected(), 3);
    /// assert_eq!(checkpoint.total_steps(), 5);
    /// ```
    pub const fn new(state: S, accepted: usize, rejected: usize) -> Self {
        Self {
            state,
            accepted,
            rejected,
        }
    }

    /// Shared reference to the checkpointed state.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new("state", 0, 0);
    /// assert_eq!(checkpoint.state(), &"state");
    /// ```
    #[must_use]
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Number of accepted moves in the checkpoint.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new((), 7, 11);
    /// assert_eq!(checkpoint.accepted(), 7);
    /// ```
    #[must_use]
    pub const fn accepted(&self) -> usize {
        self.accepted
    }

    /// Number of rejected moves in the checkpoint.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new((), 7, 11);
    /// assert_eq!(checkpoint.rejected(), 11);
    /// ```
    #[must_use]
    pub const fn rejected(&self) -> usize {
        self.rejected
    }

    /// Total number of counted steps in the checkpoint (`accepted + rejected`).
    ///
    /// The sum saturates at [`usize::MAX`] on overflow rather than wrapping or
    /// panicking.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new((), 7, 11);
    /// assert_eq!(checkpoint.total_steps(), 18);
    /// ```
    #[must_use]
    pub const fn total_steps(&self) -> usize {
        self.accepted.saturating_add(self.rejected)
    }

    /// Consume the checkpoint into its raw parts.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// let checkpoint = ChainCheckpoint::new("state", 7, 11);
    /// assert_eq!(checkpoint.into_parts(), ("state", 7, 11));
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (S, usize, usize) {
        (self.state, self.accepted, self.rejected)
    }
}

/// Errors from a delayed-commit Metropolis-Hastings step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelayedStepError<E> {
    /// The delayed proposal produced invalid MCMC numerics.
    Mcmc(McmcError),
    /// Planning a delayed proposal failed.
    Plan(E),
    /// Evaluating the proposed state's log-probability failed.
    ProposedLogProb(E),
    /// Evaluating the proposal log q-ratio failed.
    LogQRatio(E),
    /// Committing an accepted proposal failed.
    Commit(E),
}

impl<E: fmt::Display> fmt::Display for DelayedStepError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mcmc(err) => write!(f, "{err}"),
            Self::Plan(err) => write!(f, "delayed proposal planning failed: {err}"),
            Self::ProposedLogProb(err) => {
                write!(
                    f,
                    "delayed proposal log-probability evaluation failed: {err}"
                )
            }
            Self::LogQRatio(err) => {
                write!(f, "delayed proposal log q-ratio evaluation failed: {err}")
            }
            Self::Commit(err) => write!(f, "delayed proposal commit failed: {err}"),
        }
    }
}

impl<E> From<McmcError> for DelayedStepError<E> {
    fn from(err: McmcError) -> Self {
        Self::Mcmc(err)
    }
}

impl<E: Error + 'static> Error for DelayedStepError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Mcmc(err) => Some(err),
            Self::Plan(err)
            | Self::ProposedLogProb(err)
            | Self::LogQRatio(err)
            | Self::Commit(err) => Some(err),
        }
    }
}

/// Select how an in-place transition reports its completed outcome.
///
/// The policy keeps one transition algorithm for structured single-step calls
/// and telemetry-free bulk sampling. Implementations are statically dispatched,
/// so the bulk path does not construct proposal metadata or a [`Step`].
trait MutTelemetryMode<S, P: ProposalMut<S> + ?Sized> {
    /// Proposal data retained until the acceptance decision completes.
    type Captured;
    /// Value returned to the caller after the transition completes.
    type Output;

    /// Complete a step for which no concrete proposal was available.
    fn no_proposal(proposal: &mut P, log_prob_before: f64) -> Self::Output;

    /// Capture data from a validated concrete proposal before possible rollback.
    fn capture(proposal: &P, state: &S, token: &P::Undo) -> Self::Captured;

    /// Complete an accepted concrete proposal.
    fn accepted(
        captured: Self::Captured,
        log_prob_before: f64,
        log_prob_after: f64,
        log_alpha: f64,
    ) -> Self::Output;

    /// Complete a rejected concrete proposal after rollback.
    fn rejected(captured: Self::Captured, log_prob_before: f64, log_alpha: f64) -> Self::Output;
}

/// In-place transition policy that returns full structured telemetry.
struct CaptureMutTelemetry;

impl<S, P: ProposalMut<S> + ?Sized> MutTelemetryMode<S, P> for CaptureMutTelemetry {
    type Captured = P::Info;
    type Output = Step<P::Info>;

    fn no_proposal(proposal: &mut P, log_prob_before: f64) -> Self::Output {
        Step::no_proposal(proposal.no_proposal_info(), log_prob_before)
    }

    fn capture(proposal: &P, state: &S, token: &P::Undo) -> Self::Captured {
        proposal.info(state, token)
    }

    fn accepted(
        info: Self::Captured,
        log_prob_before: f64,
        log_prob_after: f64,
        log_alpha: f64,
    ) -> Self::Output {
        Step::accepted_proposal(info, log_prob_before, log_prob_after, log_alpha)
    }

    fn rejected(info: Self::Captured, log_prob_before: f64, log_alpha: f64) -> Self::Output {
        Step::rejected_proposal(info, log_prob_before, log_alpha)
    }
}

/// In-place transition policy that omits all telemetry work.
struct DiscardMutTelemetry;

impl<S, P: ProposalMut<S> + ?Sized> MutTelemetryMode<S, P> for DiscardMutTelemetry {
    type Captured = ();
    type Output = ();

    fn no_proposal(_proposal: &mut P, _log_prob_before: f64) {}

    fn capture(_proposal: &P, _state: &S, _token: &P::Undo) {}

    fn accepted((): Self::Captured, _log_prob_before: f64, _log_prob_after: f64, _log_alpha: f64) {}

    fn rejected((): Self::Captured, _log_prob_before: f64, _log_alpha: f64) {}
}

/// A single MCMC chain.
#[derive(Debug)]
#[must_use]
pub struct Chain<S> {
    /// Current state.
    pub(crate) state: S,
    /// Current log-probability of the state.
    log_prob: f64,
    /// Number of accepted moves.
    accepted: usize,
    /// Number of rejected moves.
    rejected: usize,
}

#[cfg(feature = "serde")]
impl<S: Serialize> Serialize for Chain<S> {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        self.checkpoint().serialize(serializer)
    }
}

impl<S> Chain<S> {
    /// Create a new chain from an initial state.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert_eq!(chain.accepted(), 0);
    /// assert_relative_eq!(chain.log_prob(), -0.5, epsilon = 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanInitialLogProb`] if the target's log-probability
    /// for the initial state is NaN, or [`McmcError::InfiniteInitialLogProb`]
    /// if it is +∞.
    pub fn new<T: Target<S>>(initial: S, target: &T) -> Result<Self, McmcError> {
        let log_prob = target.log_prob(&initial);
        if log_prob.is_nan() {
            cold_path();
            return Err(McmcError::NanInitialLogProb);
        }
        if log_prob == f64::INFINITY {
            cold_path();
            return Err(McmcError::InfiniteInitialLogProb);
        }
        Ok(Self {
            state: initial,
            log_prob,
            accepted: 0,
            rejected: 0,
        })
    }

    /// Restore a chain from checkpoint data and a target distribution.
    ///
    /// The checkpoint does not contain a cached log-probability. This method
    /// recomputes the cache from `target` and preserves the checkpointed
    /// counters.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanCheckpointLogProb`] or
    /// [`McmcError::InfiniteCheckpointLogProb`] if the target's
    /// log-probability for the checkpoint state is NaN or +∞.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// struct Normal;
    /// impl Target<f64> for Normal {
    ///     fn log_prob(&self, state: &f64) -> f64 { -0.5 * state * state }
    /// }
    ///
    /// let checkpoint = ChainCheckpoint::new(2.0, 7, 11);
    /// let chain = Chain::from_checkpoint(checkpoint, &Normal)?;
    ///
    /// assert_eq!(*chain.state(), 2.0);
    /// assert_relative_eq!(chain.log_prob(), -2.0, epsilon = 1e-12);
    /// assert_eq!(chain.total_steps(), 18);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub fn from_checkpoint<T: Target<S>>(
        checkpoint: ChainCheckpoint<S>,
        target: &T,
    ) -> Result<Self, McmcError> {
        let (state, accepted, rejected) = checkpoint.into_parts();
        let log_prob = target.log_prob(&state);
        if log_prob.is_nan() {
            cold_path();
            return Err(McmcError::NanCheckpointLogProb);
        }
        if log_prob == f64::INFINITY {
            cold_path();
            return Err(McmcError::InfiniteCheckpointLogProb);
        }
        Ok(Self {
            state,
            log_prob,
            accepted,
            rejected,
        })
    }

    /// Refresh the cached log-probability for the current state.
    ///
    /// This is used when a chain is paired with a sampler target, so resumed or
    /// transferred chains do not continue sampling from a stale cache.
    pub(crate) fn refresh_current_log_prob<T: Target<S>>(
        &mut self,
        target: &T,
    ) -> Result<(), McmcError> {
        let log_prob = target.log_prob(&self.state);
        if log_prob.is_nan() {
            cold_path();
            return Err(McmcError::NanCurrentLogProb);
        }
        if log_prob == f64::INFINITY {
            cold_path();
            return Err(McmcError::InfiniteCurrentLogProb);
        }
        self.log_prob = log_prob;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) const fn set_cached_log_prob_for_testing(&mut self, log_prob: f64) {
        self.log_prob = log_prob;
    }

    /// Perform a single Metropolis–Hastings step with a by-value proposal.
    ///
    /// The proposal returns a new state by value.  For state spaces where
    /// constructing a whole proposed state is expensive, use
    /// [`step_mut`](Self::step_mut).
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// chain.step(&T, &P, &mut rng)?;
    /// assert_eq!(chain.total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`] or
    /// [`McmcError::InfiniteProposedLogProb`] if the target's log-probability
    /// for the proposed state is NaN or +∞, [`McmcError::NanLogQRatio`] or
    /// [`McmcError::InfiniteLogQRatio`] if the proposal's log q-ratio is
    /// NaN or +∞.
    pub fn step<T: Target<S>, P: Proposal<S> + ?Sized, R: Rng + ?Sized>(
        &mut self,
        target: &T,
        proposal: &P,
        rng: &mut R,
    ) -> Result<(), McmcError> {
        let proposed = proposal.propose(&self.state, rng);
        let log_prob_new = target.log_prob(&proposed);
        check_proposed_log_prob(log_prob_new)?;

        let log_q = proposal.log_q_ratio(&self.state, &proposed);
        check_log_q_ratio(log_q)?;

        let log_alpha = log_prob_new - self.log_prob + log_q;

        let accept = accept_log_alpha(log_alpha, rng);

        if accept {
            self.state = proposed;
            self.log_prob = log_prob_new;
            self.accepted = self.accepted.saturating_add(1);
        } else {
            self.rejected = self.rejected.saturating_add(1);
        }
        Ok(())
    }

    /// Perform a single Metropolis–Hastings step (in-place with rollback).
    ///
    /// Unlike [`step`](Self::step), this method avoids constructing a whole
    /// proposed state. The proposal mutates the state in place and returns an
    /// undo token; on rejection (or NaN error) the state is rolled back
    /// automatically.
    ///
    /// Returns structured [`Step`] telemetry that distinguishes acceptance,
    /// Metropolis-Hastings rejection, and proposal absence while preserving
    /// proposal-specific metadata.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::in_place::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl ProposalMut<S> for P {
    /// #     type Undo = f64;
    /// #     type Info = f64;
    /// #     fn propose_mut<R: Rng + ?Sized>(&mut self, s: &mut S, r: &mut R) -> Option<f64> {
    /// #         let old = s.0; s.0 += r.random_range(-1.0..1.0); Some(old)
    /// #     }
    /// #     fn info(&self, s: &S, _: &f64) -> f64 { s.0 }
    /// #     fn undo(&mut self, s: &mut S, old: f64) { s.0 = old; }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// let mut proposal = P;
    /// let step = chain.step_mut(&T, &mut proposal, &mut rng)?;
    /// assert!(step.outcome().has_proposal());
    /// assert_eq!(chain.total_steps(), 1);
    /// # Ok::<(), McmcError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanProposedLogProb`],
    /// [`McmcError::InfiniteProposedLogProb`], [`McmcError::NanLogQRatio`],
    /// or [`McmcError::InfiniteLogQRatio`] after rolling back the state.
    pub fn step_mut<T: Target<S>, P: ProposalMut<S> + ?Sized, R: Rng + ?Sized>(
        &mut self,
        target: &T,
        proposal: &mut P,
        rng: &mut R,
    ) -> Result<Step<P::Info>, McmcError> {
        self.step_mut_with_mode::<CaptureMutTelemetry, T, P, R>(target, proposal, rng)
    }

    /// Perform an in-place step without constructing proposal telemetry.
    ///
    /// Bulk sampler methods use this path when their return type does not expose
    /// per-step metadata. The transition mechanics are shared with [`step_mut`](Self::step_mut).
    pub(crate) fn step_mut_without_telemetry<
        T: Target<S>,
        P: ProposalMut<S> + ?Sized,
        R: Rng + ?Sized,
    >(
        &mut self,
        target: &T,
        proposal: &mut P,
        rng: &mut R,
    ) -> Result<(), McmcError> {
        self.step_mut_with_mode::<DiscardMutTelemetry, T, P, R>(target, proposal, rng)
    }

    /// Execute one in-place transition with a statically selected telemetry policy.
    fn step_mut_with_mode<
        M: MutTelemetryMode<S, P>,
        T: Target<S>,
        P: ProposalMut<S> + ?Sized,
        R: Rng + ?Sized,
    >(
        &mut self,
        target: &T,
        proposal: &mut P,
        rng: &mut R,
    ) -> Result<M::Output, McmcError> {
        let log_prob_before = self.log_prob;
        let Some(token) = proposal.propose_mut(&mut self.state, rng) else {
            let output = M::no_proposal(proposal, log_prob_before);
            self.rejected = self.rejected.saturating_add(1);
            return Ok(output);
        };

        let log_prob_new = target.log_prob(&self.state);
        if let Err(err) = check_proposed_log_prob(log_prob_new) {
            proposal.undo(&mut self.state, token);
            return Err(err);
        }

        let log_q = proposal.log_q_ratio(&self.state, &token);
        if let Err(err) = check_log_q_ratio(log_q) {
            proposal.undo(&mut self.state, token);
            return Err(err);
        }

        let log_alpha = log_prob_new - self.log_prob + log_q;
        let captured = M::capture(proposal, &self.state, &token);

        let accept = accept_log_alpha(log_alpha, rng);

        if accept {
            self.log_prob = log_prob_new;
            self.accepted = self.accepted.saturating_add(1);
            Ok(M::accepted(
                captured,
                log_prob_before,
                log_prob_new,
                log_alpha,
            ))
        } else {
            proposal.undo(&mut self.state, token);
            self.rejected = self.rejected.saturating_add(1);
            Ok(M::rejected(captured, log_prob_before, log_alpha))
        }
    }

    /// Perform a single delayed-commit Metropolis-Hastings step.
    ///
    /// The proposal first returns a concrete move plan without mutating the
    /// state.  `Chain` evaluates the Metropolis-Hastings accept/reject decision
    /// for that exact transition and calls [`DelayedProposal::commit`] only
    /// after acceptance.
    ///
    /// A delayed plan should already identify the local site or handle needed
    /// to apply the move.  If no valid site exists, the proposal should return
    /// `Ok(None)` from [`DelayedProposal::propose_plan`], which `Chain` records
    /// as a rejection without calling scoring or commit hooks.
    ///
    /// `commit` must be failure-atomic: if it returns an error, it must leave
    /// `state` exactly as it was before the commit attempt.  `Chain` keeps a
    /// cached log-probability and cannot restore an arbitrary partially
    /// applied domain-specific move on its own.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct TargetLine;
    /// impl Target<i32> for TargetLine {
    ///     fn log_prob(&self, state: &i32) -> f64 {
    ///         -f64::from(state.abs())
    ///     }
    /// }
    ///
    /// struct MoveRight;
    /// impl DelayedProposal<i32> for MoveRight {
    ///     type Plan = i32;
    ///     type Info = i32;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _state: &i32,
    ///         _rng: &mut R,
    ///     ) -> Result<Option<i32>, Self::Error> {
    ///         Ok(Some(1))
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<i32>>(
    ///         &self,
    ///         state: &i32,
    ///         plan: &i32,
    ///         target: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         Ok(target.log_prob(&(*state + *plan)))
    ///     }
    ///
    ///     fn info(&self, plan: &i32) -> i32 {
    ///         *plan
    ///     }
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         state: &mut i32,
    ///         plan: i32,
    ///         _rng: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         *state += plan;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let target = TargetLine;
    /// let mut proposal = MoveRight;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(-1, &target)?;
    ///
    /// let step = chain.step_delayed(&target, &mut proposal, &mut rng)?;
    /// assert_eq!(step.outcome(), StepOutcome::Accepted);
    /// assert_eq!(*chain.state(), 0);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelayedStepError::Plan`] if planning fails,
    /// [`DelayedStepError::ProposedLogProb`] if proposed log-probability
    /// evaluation fails, [`DelayedStepError::LogQRatio`] if proposal-ratio
    /// evaluation fails, [`DelayedStepError::Mcmc`] on invalid
    /// log-probability or log q-ratio values, and [`DelayedStepError::Commit`]
    /// if applying an accepted move fails.
    pub fn step_delayed<T: Target<S>, P: DelayedProposal<S> + ?Sized, R: Rng + ?Sized>(
        &mut self,
        target: &T,
        proposal: &mut P,
        rng: &mut R,
    ) -> Result<DelayedStep<P::Info>, DelayedStepError<P::Error>> {
        let log_prob_before = self.log_prob;
        let Some(plan) = proposal
            .propose_plan(&self.state, rng)
            .map_err(DelayedStepError::Plan)?
        else {
            let info = proposal.no_plan_info();
            self.rejected = self.rejected.saturating_add(1);
            return Ok(Step::no_proposal(info, log_prob_before));
        };

        let log_prob_new = proposal
            .proposed_log_prob(&self.state, &plan, target)
            .map_err(DelayedStepError::ProposedLogProb)?;
        check_proposed_log_prob(log_prob_new).map_err(DelayedStepError::Mcmc)?;

        let log_q = proposal
            .log_q_ratio(&self.state, &plan)
            .map_err(DelayedStepError::LogQRatio)?;
        check_log_q_ratio(log_q).map_err(DelayedStepError::Mcmc)?;

        let log_alpha = log_prob_new - self.log_prob + log_q;
        let accept = accept_log_alpha(log_alpha, rng);
        let info = proposal.info(&plan);

        if accept {
            proposal
                .commit(&mut self.state, plan, rng)
                .map_err(DelayedStepError::Commit)?;
            self.log_prob = log_prob_new;
            self.accepted = self.accepted.saturating_add(1);
            Ok(Step::accepted_proposal(
                info,
                log_prob_before,
                log_prob_new,
                log_alpha,
            ))
        } else {
            self.rejected = self.rejected.saturating_add(1);
            Ok(Step::rejected_proposal(info, log_prob_before, log_alpha))
        }
    }

    /// Perform a delayed-commit step and verify the committed state afterward.
    ///
    /// This variant is intended for proposal development and invariant-heavy
    /// state spaces.  It follows the same Metropolis-Hastings decision as
    /// [`step_delayed`](Self::step_delayed), then recomputes the target
    /// log-probability after an accepted commit.  If the committed state's
    /// log-probability is invalid or differs from the value used for the
    /// acceptance decision, the original state is restored and an
    /// [`McmcError`] is returned through [`DelayedStepError::Mcmc`].
    ///
    /// The method requires `S: Clone` so it can restore the prior state when a
    /// proposal violates the delayed-commit contract.
    ///
    /// If [`DelayedProposal::commit`] returns an error, the chain state and
    /// cached log-probability are restored before the error is returned.  The
    /// proposal remains responsible for its own internal state.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// struct TargetLine;
    /// impl Target<i32> for TargetLine {
    ///     fn log_prob(&self, state: &i32) -> f64 {
    ///         -f64::from(state.abs())
    ///     }
    /// }
    ///
    /// struct MoveRight;
    /// impl DelayedProposal<i32> for MoveRight {
    ///     type Plan = i32;
    ///     type Info = i32;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _state: &i32,
    ///         _rng: &mut R,
    ///     ) -> Result<Option<i32>, Self::Error> {
    ///         Ok(Some(1))
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<i32>>(
    ///         &self,
    ///         state: &i32,
    ///         plan: &i32,
    ///         target: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         Ok(target.log_prob(&(*state + *plan)))
    ///     }
    ///
    ///     fn info(&self, plan: &i32) -> i32 {
    ///         *plan
    ///     }
    ///
    ///     fn commit<R: Rng + ?Sized>(
    ///         &mut self,
    ///         state: &mut i32,
    ///         plan: i32,
    ///         _rng: &mut R,
    ///     ) -> Result<(), Self::Error> {
    ///         *state += plan;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let target = TargetLine;
    /// let mut proposal = MoveRight;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(-1, &target)?;
    ///
    /// let step = chain.step_delayed_checked(&target, &mut proposal, &mut rng)?;
    /// assert_eq!(step.outcome(), StepOutcome::Accepted);
    /// assert_eq!(*chain.state(), 0);
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`step_delayed`](Self::step_delayed), plus
    /// [`McmcError::NanCommittedLogProb`],
    /// [`McmcError::InfiniteCommittedLogProb`], or
    /// [`McmcError::InconsistentDelayedCommitLogProb`] when post-commit
    /// validation fails.
    pub fn step_delayed_checked<T, P, R>(
        &mut self,
        target: &T,
        proposal: &mut P,
        rng: &mut R,
    ) -> Result<DelayedStep<P::Info>, DelayedStepError<P::Error>>
    where
        S: Clone,
        T: Target<S>,
        P: DelayedProposal<S> + ?Sized,
        R: Rng + ?Sized,
    {
        let log_prob_before = self.log_prob;
        let Some(plan) = proposal
            .propose_plan(&self.state, rng)
            .map_err(DelayedStepError::Plan)?
        else {
            let info = proposal.no_plan_info();
            self.rejected = self.rejected.saturating_add(1);
            return Ok(Step::no_proposal(info, log_prob_before));
        };

        let log_prob_new = proposal
            .proposed_log_prob(&self.state, &plan, target)
            .map_err(DelayedStepError::ProposedLogProb)?;
        check_proposed_log_prob(log_prob_new).map_err(DelayedStepError::Mcmc)?;

        let log_q = proposal
            .log_q_ratio(&self.state, &plan)
            .map_err(DelayedStepError::LogQRatio)?;
        check_log_q_ratio(log_q).map_err(DelayedStepError::Mcmc)?;

        let log_alpha = log_prob_new - self.log_prob + log_q;
        let accept = accept_log_alpha(log_alpha, rng);
        let info = proposal.info(&plan);

        if accept {
            let state_before_commit = self.state.clone();
            if let Err(err) = proposal.commit(&mut self.state, plan, rng) {
                self.state = state_before_commit;
                return Err(DelayedStepError::Commit(err));
            }

            let committed_log_prob = target.log_prob(&self.state);
            if let Err(err) = check_committed_log_prob(log_prob_new, committed_log_prob) {
                self.state = state_before_commit;
                return Err(DelayedStepError::Mcmc(err));
            }

            self.log_prob = committed_log_prob;
            self.accepted = self.accepted.saturating_add(1);
            Ok(Step::accepted_proposal(
                info,
                log_prob_before,
                committed_log_prob,
                log_alpha,
            ))
        } else {
            self.rejected = self.rejected.saturating_add(1);
            Ok(Step::rejected_proposal(info, log_prob_before, log_alpha))
        }
    }

    /// Shared reference to the current state.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert_relative_eq!(*chain.state(), 1.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Replace the current state and recompute the cached log-probability.
    ///
    /// This is the safe way to externally mutate the chain state.  The
    /// cached `log_prob` is always kept in sync.
    ///
    /// # Errors
    ///
    /// Returns [`McmcError::NanReplacementLogProb`] or
    /// [`McmcError::InfiniteReplacementLogProb`] if the target's
    /// log-probability for `new_state` is NaN or +∞ (the chain is unchanged on
    /// error).
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let mut chain = Chain::new(0.0_f64, &T)?;
    /// chain.replace_state(2.0, &T)?;
    /// assert_relative_eq!(*chain.state(), 2.0);
    /// assert_relative_eq!(chain.log_prob(), -2.0, epsilon = 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub fn replace_state<T: Target<S>>(
        &mut self,
        new_state: S,
        target: &T,
    ) -> Result<(), McmcError> {
        let lp = target.log_prob(&new_state);
        if lp.is_nan() {
            cold_path();
            return Err(McmcError::NanReplacementLogProb);
        }
        if lp == f64::INFINITY {
            cold_path();
            return Err(McmcError::InfiniteReplacementLogProb);
        }
        self.state = new_state;
        self.log_prob = lp;
        Ok(())
    }

    /// Consume the chain and return the state.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x } }
    /// let chain = Chain::new(3.0_f64, &T)?;
    /// let state = chain.into_state();
    /// assert_relative_eq!(state, 3.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub fn into_state(self) -> S {
        self.state
    }

    /// Borrow checkpoint data for serialization without cloning the state.
    ///
    /// Use [`into_checkpoint`](Self::into_checkpoint) when an owned checkpoint
    /// is needed.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<String> for T { fn log_prob(&self, _: &String) -> f64 { 0.0 } }
    /// let chain = Chain::new(String::from("state"), &T)?;
    /// let checkpoint = chain.checkpoint();
    ///
    /// assert_eq!(checkpoint.state().as_str(), "state");
    /// assert_eq!(checkpoint.total_steps(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn checkpoint(&self) -> ChainCheckpoint<&S> {
        ChainCheckpoint::new(&self.state, self.accepted, self.rejected)
    }

    /// Consume the chain into an owned checkpoint.
    ///
    /// Restore the returned checkpoint with [`from_checkpoint`](Self::from_checkpoint)
    /// and the target that will be used for resumed sampling.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<String> for T { fn log_prob(&self, _: &String) -> f64 { 0.0 } }
    /// let chain = Chain::new(String::from("state"), &T)?;
    /// let checkpoint = chain.into_checkpoint();
    ///
    /// assert_eq!(checkpoint.into_parts(), (String::from("state"), 0, 0));
    /// # Ok::<(), McmcError>(())
    /// ```
    pub fn into_checkpoint(self) -> ChainCheckpoint<S> {
        ChainCheckpoint::new(self.state, self.accepted, self.rejected)
    }

    /// Current log-probability of the chain state.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T {
    /// #     fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x }
    /// # }
    /// let chain = Chain::new(1.0_f64, &T)?;
    /// assert_relative_eq!(chain.log_prob(), -0.5, epsilon = 1e-12);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// Number of accepted moves.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, _: &f64) -> f64 { 0.0 } }
    /// let chain = Chain::new(0.0_f64, &T)?;
    /// assert_eq!(chain.accepted(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn accepted(&self) -> usize {
        self.accepted
    }

    /// Number of rejected moves.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::*;
    ///
    /// # struct T;
    /// # impl Target<f64> for T { fn log_prob(&self, _: &f64) -> f64 { 0.0 } }
    /// let chain = Chain::new(0.0_f64, &T)?;
    /// assert_eq!(chain.rejected(), 0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn rejected(&self) -> usize {
        self.rejected
    }

    /// Total number of steps taken (`accepted + rejected`).
    ///
    /// The sum saturates at [`usize::MAX`] on overflow rather than wrapping or
    /// panicking.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// for _ in 0..100 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert_eq!(chain.total_steps(), 100);
    /// assert_eq!(chain.total_steps(), chain.accepted() + chain.rejected());
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    pub const fn total_steps(&self) -> usize {
        self.accepted.saturating_add(self.rejected)
    }

    /// Acceptance rate of the chain.
    ///
    /// Returns `accepted / (accepted + rejected)`, or `0.0` when no steps have
    /// been counted.  The ratio is computed in floating point so very large
    /// counters preserve the acceptance fraction instead of saturating the
    /// denominator first.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(S(0.0), &T)?;
    /// assert_eq!(chain.acceptance_rate(), 0.0); // no steps yet
    ///
    /// for _ in 0..1000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert!(chain.acceptance_rate() > 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "large acceptance counts are intentionally converted for ratio calculation"
    )]
    pub fn acceptance_rate(&self) -> f64 {
        let accepted = self.accepted as f64;
        let total = accepted + self.rejected as f64;
        if total == 0.0 { 0.0 } else { accepted / total }
    }

    /// Reset acceptance and rejection counters to zero.
    ///
    /// Useful after burn-in to measure the acceptance rate of the
    /// production phase only.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::*;
    /// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # #[derive(Clone)] struct S(f64);
    /// # struct T;
    /// # impl Target<S> for T { fn log_prob(&self, s: &S) -> f64 { -0.5 * s.0 * s.0 } }
    /// # struct P;
    /// # impl Proposal<S> for P {
    /// #     fn propose<R: Rng + ?Sized>(&self, c: &S, r: &mut R) -> S {
    /// #         S(c.0 + r.random_range(-1.0..1.0))
    /// #     }
    /// # }
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new(S(0.0), &T)?;
    ///
    /// // Burn-in
    /// for _ in 0..1000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    ///
    /// chain.reset_counters();
    /// assert_eq!(chain.total_steps(), 0);
    ///
    /// // Production — acceptance rate reflects only this phase
    /// for _ in 0..5000 {
    ///     chain.step(&T, &P, &mut rng)?;
    /// }
    /// assert!(chain.acceptance_rate() > 0.0);
    /// # Ok::<(), McmcError>(())
    /// ```
    pub const fn reset_counters(&mut self) {
        self.accepted = 0;
        self.rejected = 0;
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;
    use std::assert_matches;

    use approx::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::AdditiveTarget;

    // --- Test fixtures ---

    #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
    #[derive(Clone, Debug, PartialEq)]
    struct Scalar(f64);

    /// Target: standard normal log-density, log p(x) = -x²/2
    struct Normal;
    impl Target<Scalar> for Normal {
        fn log_prob(&self, state: &Scalar) -> f64 {
            -0.5 * state.0 * state.0
        }
    }

    /// Symmetric random-walk proposal: x' = x + U(-width, width)
    struct RandomWalk {
        width: f64,
    }
    impl Proposal<Scalar> for RandomWalk {
        fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
            let delta: f64 = rng.random_range(-self.width..self.width);
            Scalar(current.0 + delta)
        }
    }

    /// Deterministic proposal that always returns a fixed value.
    struct FixedProposal(f64);
    impl Proposal<Scalar> for FixedProposal {
        fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(self.0)
        }
    }

    // --- Chain::new ---

    #[test]
    fn new_initial_state() {
        let chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert_eq!(chain.state, Scalar(1.0));
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn new_initial_log_prob() {
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        assert_relative_eq!(chain.log_prob(), 0.0, epsilon = 1e-12);

        let chain2 = Chain::new(Scalar(1.0), &Normal).unwrap();
        assert_relative_eq!(chain2.log_prob(), -0.5, epsilon = 1e-12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_checkpoint_resumes_sampling() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        chain.step(&Normal, &FixedProposal(0.0), &mut rng).unwrap();

        let checkpoint = serde_json::to_string(&chain).unwrap();
        let checkpoint: ChainCheckpoint<Scalar> = serde_json::from_str(&checkpoint).unwrap();
        let mut restored = Chain::from_checkpoint(checkpoint, &Normal).unwrap();

        assert_eq!(restored.state(), &Scalar(0.0));
        assert_relative_eq!(
            restored.log_prob(),
            Normal.log_prob(restored.state()),
            epsilon = 1e-12
        );
        assert_eq!(restored.accepted(), 1);
        assert_eq!(restored.rejected(), 0);
        assert_eq!(restored.total_steps(), 1);

        restored
            .step(&Normal, &FixedProposal(100.0), &mut rng)
            .unwrap();

        assert_eq!(restored.total_steps(), 2);
        assert_relative_eq!(
            restored.log_prob(),
            Normal.log_prob(restored.state()),
            epsilon = 1e-12
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_checkpoint_does_not_trust_cached_log_prob() {
        let checkpoint = r#"{"state":2.0,"log_prob":1000.0,"accepted":3,"rejected":4}"#;
        let checkpoint: ChainCheckpoint<Scalar> = serde_json::from_str(checkpoint).unwrap();
        let restored = Chain::from_checkpoint(checkpoint, &Normal).unwrap();

        assert_eq!(restored.state(), &Scalar(2.0));
        assert_relative_eq!(restored.log_prob(), -2.0, epsilon = 1e-12);
        assert_eq!(restored.accepted(), 3);
        assert_eq!(restored.rejected(), 4);
    }

    #[test]
    fn from_checkpoint_rejects_invalid_target_log_prob() {
        struct NanTarget;
        impl Target<Scalar> for NanTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::NAN
            }
        }

        let checkpoint = ChainCheckpoint::new(Scalar(0.0), 1, 2);
        let result = Chain::from_checkpoint(checkpoint, &NanTarget);

        assert_matches!(result, Err(McmcError::NanCheckpointLogProb));
    }

    #[test]
    fn from_checkpoint_rejects_infinite_target_log_prob() {
        struct InfTarget;
        impl Target<Scalar> for InfTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::INFINITY
            }
        }

        let checkpoint = ChainCheckpoint::new(Scalar(0.0), 1, 2);
        let result = Chain::from_checkpoint(checkpoint, &InfTarget);

        assert_matches!(result, Err(McmcError::InfiniteCheckpointLogProb));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_allows_nonserializable_state() {
        struct NonSerializableState(f64);

        struct NonSerializableTarget;
        impl Target<NonSerializableState> for NonSerializableTarget {
            fn log_prob(&self, state: &NonSerializableState) -> f64 {
                -0.5 * state.0 * state.0
            }
        }

        let chain = Chain::new(NonSerializableState(1.0), &NonSerializableTarget).unwrap();

        assert_relative_eq!(chain.log_prob(), -0.5, epsilon = 1e-12);
        assert_eq!(chain.total_steps(), 0);
    }

    // --- acceptance_rate ---

    #[test]
    fn acceptance_rate_zero_steps() {
        let chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        assert_relative_eq!(chain.acceptance_rate(), 0.0);
    }

    #[test]
    fn accept_log_alpha_rejects_nan() {
        let mut rng = StdRng::seed_from_u64(42);

        assert!(
            !accept_log_alpha(f64::NAN, &mut rng),
            "NaN log acceptance ratio should be an explicit rejection"
        );
    }

    #[test]
    fn accept_extreme_tail_log_space() {
        let log_alpha = -800.0;

        assert!(
            accept_from_log_uniform(log_alpha, -801.0),
            "Should accept when log(u) is below an extreme log acceptance ratio"
        );
        assert!(
            !accept_from_log_uniform(log_alpha, -799.0),
            "Should reject when log(u) is above an extreme log acceptance ratio"
        );
    }

    // --- MH acceptance logic ---

    #[test]
    fn step_accepts_uphill() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(Scalar(2.0), &Normal).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();

        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should accept move to higher probability"
        );
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn step_rejects_downhill() {
        // From x=0.0 (log_prob=0) to x=100.0 (log_prob=-5000): virtually always reject
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = FixedProposal(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();

        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should reject move to much lower probability"
        );
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    struct FlatBool;
    impl Target<bool> for FlatBool {
        fn log_prob(&self, _: &bool) -> f64 {
            0.0
        }
    }

    struct FavorTrue {
        true_log_weight: f64,
    }
    impl Target<bool> for FavorTrue {
        fn log_prob(&self, state: &bool) -> f64 {
            if *state { self.true_log_weight } else { 0.0 }
        }
    }

    struct FlipBool;
    impl Proposal<bool> for FlipBool {
        fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
            !*current
        }
    }

    #[test]
    fn unbiased_two_state_baseline_samples_uniformly() {
        let mut chain = Chain::new(false, &FlatBool).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let samples = 1_000_u32;
        let mut true_count = 0_u32;
        for _ in 0..samples {
            chain.step(&FlatBool, &FlipBool, &mut rng).unwrap();
            true_count += u32::from(*chain.state());
        }

        assert_eq!(true_count, samples / 2);
        assert_eq!(chain.accepted(), usize::try_from(samples).unwrap());
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn additive_bias_changes_observed_stationary_distribution() {
        let target = AdditiveTarget::new(
            FlatBool,
            FavorTrue {
                true_log_weight: 3.0_f64.ln(),
            },
        );
        let mut chain = Chain::new(false, &target).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..5_000 {
            chain.step(&target, &FlipBool, &mut rng).unwrap();
        }

        let samples = 50_000_u32;
        let mut true_count = 0_u32;
        for _ in 0..samples {
            chain.step(&target, &FlipBool, &mut rng).unwrap();
            true_count += u32::from(*chain.state());
        }

        let true_fraction = f64::from(true_count) / f64::from(samples);
        assert_relative_eq!(true_fraction, 0.75, epsilon = 0.03);
    }

    struct DelayedFlip {
        log_q: f64,
    }
    impl DelayedProposal<bool> for DelayedFlip {
        type Plan = bool;
        type Info = bool;
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            state: &bool,
            _: &mut R,
        ) -> Result<Option<bool>, Self::Error> {
            Ok(Some(!*state))
        }

        fn proposed_log_prob<T: Target<bool>>(
            &self,
            _: &bool,
            plan: &bool,
            target: &T,
        ) -> Result<f64, Self::Error> {
            Ok(target.log_prob(plan))
        }

        fn log_q_ratio(&self, _: &bool, _: &bool) -> Result<f64, Self::Error> {
            Ok(self.log_q)
        }

        fn info(&self, plan: &bool) -> bool {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut bool,
            plan: bool,
            _: &mut R,
        ) -> Result<(), Self::Error> {
            *state = plan;
            Ok(())
        }
    }

    #[test]
    fn delayed_log_alpha_adds_target_bias_and_proposal_ratio_with_correct_sign() {
        let target = AdditiveTarget::new(
            FlatBool,
            FavorTrue {
                true_log_weight: 3.0_f64.ln(),
            },
        );
        let mut proposal = DelayedFlip {
            log_q: -6.0_f64.ln(),
        };
        let mut chain = Chain::new(false, &target).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&target, &mut proposal, &mut rng)
            .unwrap();

        assert!(step.outcome().has_proposal());
        assert_relative_eq!(step.log_alpha().unwrap(), -2.0_f64.ln(), epsilon = 1e-12);
        assert_eq!(step.info(), Some(&true));
    }

    #[test]
    fn delayed_additive_target_commits_accepted_bias_move() {
        let target = AdditiveTarget::new(
            FlatBool,
            FavorTrue {
                true_log_weight: 3.0_f64.ln(),
            },
        );
        let mut proposal = DelayedFlip { log_q: 0.0 };
        let mut chain = Chain::new(false, &target).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&target, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_relative_eq!(step.log_alpha().unwrap(), 3.0_f64.ln(), epsilon = 1e-12);
        assert_eq!(step.info(), Some(&true));
        assert!(*chain.state());
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    // --- Error handling ---

    #[test]
    fn new_rejects_nan_initial_log_prob() {
        struct NanTarget;
        impl Target<Scalar> for NanTarget {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let result = Chain::new(Scalar(0.0), &NanTarget);
        assert_matches!(result, Err(McmcError::NanInitialLogProb));
    }

    #[test]
    fn step_rejects_nan_proposal() {
        struct NanAtOrigin;
        impl Target<Scalar> for NanAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0 == 0.0 {
                    f64::NAN
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &NanAtOrigin).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&NanAtOrigin, &proposal, &mut rng);
        assert_matches!(result, Err(McmcError::NanProposedLogProb));
    }

    #[test]
    fn step_rejects_nan_log_q_ratio() {
        struct NanProposal;
        impl Proposal<Scalar> for NanProposal {
            fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
                Scalar(0.0)
            }
            fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&Normal, &NanProposal, &mut rng);
        assert_matches!(result, Err(McmcError::NanLogQRatio));
    }

    #[test]
    fn new_rejects_inf_initial_log_prob() {
        struct InfTarget;
        impl Target<Scalar> for InfTarget {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let result = Chain::new(Scalar(0.0), &InfTarget);
        assert_matches!(result, Err(McmcError::InfiniteInitialLogProb));
    }

    #[test]
    fn step_rejects_inf_proposal() {
        struct InfAtOrigin;
        impl Target<Scalar> for InfAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0 == 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &InfAtOrigin).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&InfAtOrigin, &proposal, &mut rng);
        assert_matches!(result, Err(McmcError::InfiniteProposedLogProb));
    }

    #[test]
    fn step_rejects_inf_log_q_ratio() {
        struct InfQProposal;
        impl Proposal<Scalar> for InfQProposal {
            fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
                Scalar(0.0)
            }
            fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step(&Normal, &InfQProposal, &mut rng);
        assert_matches!(result, Err(McmcError::InfiniteLogQRatio));
    }

    #[test]
    fn step_mut_undoes_inf_log_q() {
        struct InfQMutProposal;
        impl ProposalMut<MutScalar> for InfQMutProposal {
            type Undo = f64;
            type Info = ();
            fn propose_mut<R: Rng + ?Sized>(
                &mut self,
                state: &mut MutScalar,
                _rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 = 0.0;
                Some(old)
            }
            fn info(&self, _state: &MutScalar, _old: &f64) {}
            fn undo(&mut self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
            fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = InfQMutProposal;
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&Normal, &mut proposal, &mut rng);
        assert_matches!(result, Err(McmcError::InfiniteLogQRatio));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after +inf log_q_ratio"
        );
    }

    #[test]
    fn step_mut_undoes_inf_log_prob() {
        struct InfAtOrigin;
        impl Target<MutScalar> for InfAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &InfAtOrigin).unwrap();
        let mut proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&InfAtOrigin, &mut proposal, &mut rng);
        assert_matches!(result, Err(McmcError::InfiniteProposedLogProb));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after +inf log_prob"
        );
    }

    // --- Seeded determinism ---

    #[test]
    fn step_deterministic() {
        let proposal = RandomWalk { width: 1.0 };
        let steps = 100;

        let mut chain1 = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng1 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain1.step(&Normal, &proposal, &mut rng1).unwrap();
        }

        let mut chain2 = Chain::new(Scalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            chain2.step(&Normal, &proposal, &mut rng2).unwrap();
        }

        assert_eq!(
            chain1.state, chain2.state,
            "Same seed should produce identical final state"
        );
        assert_eq!(chain1.accepted(), chain2.accepted());
        assert_eq!(chain1.rejected(), chain2.rejected());
    }

    // --- Statistical sanity ---

    #[test]
    fn step_samples_near_normal_mode() {
        let proposal = RandomWalk { width: 1.0 };
        let mut chain = Chain::new(Scalar(5.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Burn-in
        for _ in 0..1_000 {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }

        // Collect samples
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
            sum += chain.state.0;
        }
        let mean = sum / f64::from(n);

        assert_relative_eq!(mean, 0.0, epsilon = 0.1);

        let rate = chain.acceptance_rate();
        assert!(
            (0.1..0.95).contains(&rate),
            "Acceptance rate {rate} should be in a reasonable range"
        );
    }

    // --- log_q_ratio default ---

    #[test]
    fn symmetric_proposal_zero_log_q() {
        let proposal = RandomWalk { width: 1.0 };
        let ratio = proposal.log_q_ratio(&Scalar(0.0), &Scalar(1.0));
        assert_relative_eq!(ratio, 0.0);
    }

    // =====================================================================
    // ProposalMut / step_mut tests
    // =====================================================================

    // --- Test fixtures for step_mut ---

    /// Non-Clone state for testing `ProposalMut`.
    #[derive(Debug, PartialEq)]
    struct MutScalar(f64);

    impl Target<MutScalar> for Normal {
        fn log_prob(&self, state: &MutScalar) -> f64 {
            -0.5 * state.0 * state.0
        }
    }

    /// Deterministic in-place proposal: set state to a fixed value.
    struct FixedMutProposal(f64);
    impl ProposalMut<MutScalar> for FixedMutProposal {
        type Undo = f64; // store old value
        type Info = f64;
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<f64> {
            let old = state.0;
            state.0 = self.0;
            Some(old)
        }
        fn info(&self, state: &MutScalar, _old: &f64) -> f64 {
            state.0
        }
        fn undo(&mut self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
    }

    /// Proposal that always returns None (no valid move).
    struct NoMoveProposal;
    impl ProposalMut<MutScalar> for NoMoveProposal {
        type Undo = ();
        type Info = ();
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            _state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<()> {
            None
        }
        fn info(&self, _state: &MutScalar, _token: &()) {}
        fn undo(&mut self, _state: &mut MutScalar, _token: ()) {}
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MutMoveFamily {
        Add,
    }

    struct FamilyNoMoveProposal {
        last_family: Option<MutMoveFamily>,
    }

    impl ProposalMut<MutScalar> for FamilyNoMoveProposal {
        type Undo = ();
        type Info = MutMoveFamily;

        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            _state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<()> {
            self.last_family = Some(MutMoveFamily::Add);
            None
        }

        fn info(&self, _state: &MutScalar, _token: &()) -> Self::Info {
            unreachable!("no proposal should not produce concrete proposal info")
        }

        fn no_proposal_info(&mut self) -> Option<Self::Info> {
            self.last_family.take()
        }

        fn undo(&mut self, _state: &mut MutScalar, _token: ()) {}
    }

    /// Proposal that probes a mutation, rolls it back internally, and returns None.
    struct ProbeThenNoMoveProposal;
    impl ProposalMut<MutScalar> for ProbeThenNoMoveProposal {
        type Undo = ();
        type Info = ();
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<()> {
            state.0 += 1.0;
            state.0 -= 1.0;
            None
        }
        fn info(&self, _state: &MutScalar, _token: &()) {}
        fn undo(&mut self, _state: &mut MutScalar, _token: ()) {}
    }

    // --- step_mut acceptance ---

    #[test]
    fn step_mut_accepts_uphill() {
        // From x=2.0 (log_prob=-2) to x=0.0 (log_prob=0): always accept
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_eq!(step.info(), Some(&0.0));
        assert_eq!(step.rejection_reason(), None);
        assert_relative_eq!(step.log_prob_before(), -2.0, epsilon = 1e-12);
        assert_eq!(step.log_prob_after(), Some(0.0));
        assert_eq!(step.log_alpha(), Some(2.0));
        assert_eq!(chain.state, MutScalar(0.0));
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn step_mut_rejects_downhill() {
        // From x=0.0 (log_prob=0) to x=100.0 (log_prob=-5000): virtually always reject
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut proposal = FixedMutProposal(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::RejectedProposal);
        assert_eq!(step.info(), Some(&100.0));
        assert_eq!(
            step.rejection_reason(),
            Some(StepRejectionReason::RejectedProposal)
        );
        assert_eq!(step.log_prob_before(), 0.0);
        assert_eq!(step.log_prob_after(), None);
        assert_eq!(step.log_alpha(), Some(-5000.0));
        // State should be rolled back to original
        assert_eq!(
            chain.state,
            MutScalar(0.0),
            "State should be rolled back after rejection"
        );
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    // --- step_mut None proposal ---

    #[test]
    fn step_mut_none_rejects() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = NoMoveProposal;
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::NoProposal);
        assert_eq!(step.info(), None);
        assert_eq!(
            step.rejection_reason(),
            Some(StepRejectionReason::NoProposal)
        );
        assert_eq!(step.log_prob_before(), log_prob);
        assert_eq!(step.log_prob_after(), None);
        assert_eq!(step.log_alpha(), None);
        assert_eq!(chain.state, MutScalar(1.0), "State should be unchanged");
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn step_mut_none_can_report_family_info() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = FamilyNoMoveProposal { last_family: None };
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::NoProposal);
        assert_eq!(step.info(), Some(&MutMoveFamily::Add));
        assert_eq!(proposal.last_family, None);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn step_mut_none_allows_probe() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = ProbeThenNoMoveProposal;
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::NoProposal);
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "ProposalMut::propose_mut(None) must leave state unchanged"
        );
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    // --- step_mut NaN rollback ---

    #[test]
    fn step_mut_undoes_nan_log_prob() {
        struct NanAtOrigin;
        impl Target<MutScalar> for NanAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::NAN
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &NanAtOrigin).unwrap();
        let mut proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&NanAtOrigin, &mut proposal, &mut rng);
        assert_matches!(result, Err(McmcError::NanProposedLogProb));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after NaN log_prob"
        );
    }

    #[test]
    fn step_mut_undoes_nan_log_q() {
        struct NanQProposal;
        impl ProposalMut<MutScalar> for NanQProposal {
            type Undo = f64;
            type Info = ();
            fn propose_mut<R: Rng + ?Sized>(
                &mut self,
                state: &mut MutScalar,
                _rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 = 0.0;
                Some(old)
            }
            fn info(&self, _state: &MutScalar, _old: &f64) {}
            fn undo(&mut self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
            fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = NanQProposal;
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_mut(&Normal, &mut proposal, &mut rng);
        assert_matches!(result, Err(McmcError::NanLogQRatio));
        assert_eq!(
            chain.state,
            MutScalar(1.0),
            "State should be rolled back after NaN log_q_ratio"
        );
    }

    // --- step_mut seeded determinism ---

    #[test]
    fn step_mut_deterministic() {
        /// Random-walk `ProposalMut` for `MutScalar`.
        struct MutRandomWalk {
            width: f64,
        }
        impl ProposalMut<MutScalar> for MutRandomWalk {
            type Undo = f64;
            type Info = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &mut self,
                state: &mut MutScalar,
                rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                let delta: f64 = rng.random_range(-self.width..self.width);
                state.0 += delta;
                Some(old)
            }
            fn info(&self, state: &MutScalar, _old: &f64) -> f64 {
                state.0
            }
            fn undo(&mut self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
        }

        let mut proposal = MutRandomWalk { width: 1.0 };
        let steps = 100;

        let mut chain1 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng1 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            let _ = chain1.step_mut(&Normal, &mut proposal, &mut rng1).unwrap();
        }

        let mut chain2 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut rng2 = StdRng::seed_from_u64(12345);
        for _ in 0..steps {
            let _ = chain2.step_mut(&Normal, &mut proposal, &mut rng2).unwrap();
        }

        assert_eq!(
            chain1.state, chain2.state,
            "Same seed should produce identical final state"
        );
        assert_eq!(chain1.accepted(), chain2.accepted());
        assert_eq!(chain1.rejected(), chain2.rejected());
    }

    // --- step_mut statistical sanity ---

    #[test]
    fn step_mut_samples_near_normal_mode() {
        struct MutRandomWalk {
            width: f64,
        }
        impl ProposalMut<MutScalar> for MutRandomWalk {
            type Undo = f64;
            type Info = f64;
            fn propose_mut<R: Rng + ?Sized>(
                &mut self,
                state: &mut MutScalar,
                rng: &mut R,
            ) -> Option<f64> {
                let old = state.0;
                state.0 += rng.random_range(-self.width..self.width);
                Some(old)
            }
            fn info(&self, state: &MutScalar, _old: &f64) -> f64 {
                state.0
            }
            fn undo(&mut self, state: &mut MutScalar, old: f64) {
                state.0 = old;
            }
        }

        let mut proposal = MutRandomWalk { width: 1.0 };
        let mut chain = Chain::new(MutScalar(5.0), &Normal).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Burn-in
        for _ in 0..1_000 {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        }

        // Collect samples
        let n = 10_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let _ = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
            sum += chain.state.0;
        }
        let mean = sum / f64::from(n);

        assert_relative_eq!(mean, 0.0, epsilon = 0.1);

        let rate = chain.acceptance_rate();
        assert!(
            (0.1..0.95).contains(&rate),
            "Acceptance rate {rate} should be in a reasonable range"
        );
    }

    // --- step_mut non-Clone state ---

    #[test]
    fn step_mut_non_clone_state() {
        // MutScalar intentionally does not derive Clone.
        // This test verifies the ProposalMut path compiles and works.
        let mut chain = Chain::new(MutScalar(5.0), &Normal).unwrap();
        let mut proposal = FixedMutProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        // Should accept (moving to mode)
        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_eq!(chain.state, MutScalar(0.0));
    }

    // --- ProposalMut log_q_ratio default ---

    #[test]
    fn symmetric_proposal_mut_zero_log_q() {
        let proposal = FixedMutProposal(0.0);
        let ratio = proposal.log_q_ratio(&MutScalar(1.0), &2.0);
        assert_relative_eq!(ratio, 0.0);
    }

    // =====================================================================
    // DelayedProposal / step_delayed tests
    // =====================================================================

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DelayedFixtureError {
        Plan,
        Score,
        Ratio,
        Commit,
    }

    impl fmt::Display for DelayedFixtureError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl Error for DelayedFixtureError {}

    struct FixedDelayedProposal {
        proposed: f64,
        log_q: Result<f64, DelayedFixtureError>,
        plan_error: Option<DelayedFixtureError>,
        score_error: Option<DelayedFixtureError>,
        commit_error: Option<DelayedFixtureError>,
    }

    impl FixedDelayedProposal {
        const fn new(proposed: f64) -> Self {
            Self {
                proposed,
                log_q: Ok(0.0),
                plan_error: None,
                score_error: None,
                commit_error: None,
            }
        }
    }

    impl DelayedProposal<MutScalar> for FixedDelayedProposal {
        type Plan = f64;
        type Info = f64;
        type Error = DelayedFixtureError;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &MutScalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            if let Some(err) = self.plan_error {
                Err(err)
            } else {
                Ok(Some(self.proposed))
            }
        }

        fn proposed_log_prob<T: Target<MutScalar>>(
            &self,
            _state: &MutScalar,
            plan: &f64,
            target: &T,
        ) -> Result<f64, Self::Error> {
            self.score_error
                .map_or_else(|| Ok(target.log_prob(&MutScalar(*plan))), Err)
        }

        fn log_q_ratio(&self, _state: &MutScalar, _plan: &f64) -> Result<f64, Self::Error> {
            self.log_q
        }

        fn info(&self, plan: &f64) -> f64 {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            if let Some(err) = self.commit_error {
                Err(err)
            } else {
                state.0 = plan;
                Ok(())
            }
        }
    }

    struct NoDelayedProposal;

    impl DelayedProposal<MutScalar> for NoDelayedProposal {
        type Plan = f64;
        type Info = f64;
        type Error = DelayedFixtureError;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &MutScalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            Ok(None)
        }

        fn proposed_log_prob<T: Target<MutScalar>>(
            &self,
            _state: &MutScalar,
            _plan: &f64,
            _target: &T,
        ) -> Result<f64, Self::Error> {
            unreachable!("no plan should not be scored")
        }

        fn info(&self, plan: &f64) -> f64 {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            _state: &mut MutScalar,
            _plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            unreachable!("no plan should not be committed")
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MoveFamily {
        Add,
    }

    struct FamilyNoPlanProposal {
        last_family: Option<MoveFamily>,
    }

    impl DelayedProposal<MutScalar> for FamilyNoPlanProposal {
        type Plan = f64;
        type Info = MoveFamily;
        type Error = DelayedFixtureError;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &MutScalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            self.last_family = Some(MoveFamily::Add);
            Ok(None)
        }

        fn no_plan_info(&mut self) -> Option<Self::Info> {
            self.last_family.take()
        }

        fn proposed_log_prob<T: Target<MutScalar>>(
            &self,
            _state: &MutScalar,
            _plan: &f64,
            _target: &T,
        ) -> Result<f64, Self::Error> {
            unreachable!("no plan should not be scored")
        }

        fn info(&self, _plan: &f64) -> Self::Info {
            unreachable!("no plan should not produce plan info")
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            _state: &mut MutScalar,
            _plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            unreachable!("no plan should not be committed")
        }
    }

    struct CheckedCommitProposal {
        scored: f64,
        committed: f64,
    }

    impl DelayedProposal<Scalar> for CheckedCommitProposal {
        type Plan = f64;
        type Info = f64;
        type Error = DelayedFixtureError;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &Scalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            Ok(Some(self.scored))
        }

        fn proposed_log_prob<T: Target<Scalar>>(
            &self,
            _state: &Scalar,
            plan: &f64,
            target: &T,
        ) -> Result<f64, Self::Error> {
            Ok(target.log_prob(&Scalar(*plan)))
        }

        fn info(&self, plan: &f64) -> f64 {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut Scalar,
            _plan: f64,
            _rng: &mut R,
        ) -> Result<(), Self::Error> {
            state.0 = self.committed;
            Ok(())
        }
    }

    #[test]
    fn delayed_accepts_uphill() {
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = FixedDelayedProposal::new(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&Normal, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert!(step.outcome().is_accepted());
        assert!(step.outcome().has_proposal());
        assert!(step.rejection_reason().is_none());
        assert_eq!(step.info(), Some(&0.0));
        assert_relative_eq!(step.log_prob_before(), -2.0, epsilon = 1e-12);
        assert_eq!(step.log_prob_after(), Some(0.0));
        assert_eq!(step.log_alpha(), Some(2.0));
        assert_eq!(chain.state, MutScalar(0.0));
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_rejects_downhill() {
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut proposal = FixedDelayedProposal::new(100.0);
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&Normal, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::RejectedProposal);
        assert!(!step.outcome().is_accepted());
        assert!(step.outcome().has_proposal());
        assert_eq!(
            step.rejection_reason(),
            Some(StepRejectionReason::RejectedProposal)
        );
        assert_eq!(step.info(), Some(&100.0));
        assert_relative_eq!(step.log_prob_before(), 0.0);
        assert_eq!(step.log_prob_after(), None);
        assert_eq!(step.log_alpha(), Some(-5000.0));
        assert_eq!(chain.state, MutScalar(0.0));
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn delayed_no_plan_rejects() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = NoDelayedProposal;
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&Normal, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::NoProposal);
        assert!(!step.outcome().is_accepted());
        assert!(!step.outcome().has_proposal());
        assert_eq!(
            step.rejection_reason(),
            Some(StepRejectionReason::NoProposal)
        );
        assert_eq!(step.info(), None);
        assert_relative_eq!(step.log_prob_before(), log_prob);
        assert_eq!(step.log_prob_after(), None);
        assert_eq!(step.log_alpha(), None);
        assert_eq!(chain.state, MutScalar(1.0));
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn delayed_no_plan_can_report_family_info() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = FamilyNoPlanProposal { last_family: None };
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed(&Normal, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::NoProposal);
        assert!(!step.outcome().is_accepted());
        assert!(!step.outcome().has_proposal());
        assert_eq!(
            step.rejection_reason(),
            Some(StepRejectionReason::NoProposal)
        );
        assert_eq!(step.info(), Some(&MoveFamily::Add));
        assert_eq!(proposal.last_family, None);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn delayed_rejects_nan_log_prob() {
        struct NanAtOrigin;
        impl Target<MutScalar> for NanAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::NAN
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }

        let mut chain = Chain::new(MutScalar(1.0), &NanAtOrigin).unwrap();
        let mut proposal = FixedDelayedProposal::new(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&NanAtOrigin, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(McmcError::NanProposedLogProb))
        );
        assert_eq!(chain.state, MutScalar(1.0));
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_rejects_inf_log_prob() {
        struct InfAtOrigin;
        impl Target<MutScalar> for InfAtOrigin {
            fn log_prob(&self, state: &MutScalar) -> f64 {
                if state.0 == 0.0 {
                    f64::INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }

        let mut chain = Chain::new(MutScalar(1.0), &InfAtOrigin).unwrap();
        let mut proposal = FixedDelayedProposal::new(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&InfAtOrigin, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(McmcError::InfiniteProposedLogProb))
        );
        assert_eq!(chain.state, MutScalar(1.0));
    }

    #[test]
    fn delayed_rejects_bad_log_q() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = FixedDelayedProposal {
            log_q: Ok(f64::NAN),
            ..FixedDelayedProposal::new(0.0)
        };
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);

        assert_matches!(result, Err(DelayedStepError::Mcmc(McmcError::NanLogQRatio)));
        assert_eq!(chain.state, MutScalar(1.0));

        proposal.log_q = Ok(f64::INFINITY);
        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);
        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(McmcError::InfiniteLogQRatio))
        );
        assert_eq!(chain.state, MutScalar(1.0));
    }

    #[test]
    fn delayed_maps_errors() {
        let mut chain = Chain::new(MutScalar(1.0), &Normal).unwrap();
        let mut proposal = FixedDelayedProposal {
            plan_error: Some(DelayedFixtureError::Plan),
            ..FixedDelayedProposal::new(0.0)
        };
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);
        assert_matches!(
            result,
            Err(DelayedStepError::Plan(DelayedFixtureError::Plan))
        );

        proposal.plan_error = None;
        proposal.score_error = Some(DelayedFixtureError::Score);
        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);
        assert_matches!(
            result,
            Err(DelayedStepError::ProposedLogProb(
                DelayedFixtureError::Score
            ))
        );

        proposal.score_error = None;
        proposal.log_q = Err(DelayedFixtureError::Ratio);
        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);
        assert_matches!(
            result,
            Err(DelayedStepError::LogQRatio(DelayedFixtureError::Ratio))
        );
    }

    #[test]
    fn delayed_error_messages_name_stage() {
        assert_eq!(
            DelayedStepError::Plan(DelayedFixtureError::Plan).to_string(),
            "delayed proposal planning failed: Plan"
        );
        assert_eq!(
            DelayedStepError::ProposedLogProb(DelayedFixtureError::Score).to_string(),
            "delayed proposal log-probability evaluation failed: Score"
        );
        assert_eq!(
            DelayedStepError::LogQRatio(DelayedFixtureError::Ratio).to_string(),
            "delayed proposal log q-ratio evaluation failed: Ratio"
        );
        assert_eq!(
            DelayedStepError::Commit(DelayedFixtureError::Commit).to_string(),
            "delayed proposal commit failed: Commit"
        );
        assert_eq!(
            DelayedStepError::<DelayedFixtureError>::Mcmc(McmcError::NanLogQRatio).to_string(),
            "proposal returned NaN log q-ratio"
        );
    }

    #[test]
    fn delayed_error_sources() {
        let mcmc: DelayedStepError<DelayedFixtureError> = McmcError::NanLogQRatio.into();
        assert_matches!(mcmc, DelayedStepError::Mcmc(McmcError::NanLogQRatio));
        assert_eq!(
            mcmc.source().map(ToString::to_string),
            Some("proposal returned NaN log q-ratio".to_owned())
        );

        let cases = [
            (DelayedStepError::Plan(DelayedFixtureError::Plan), "Plan"),
            (
                DelayedStepError::ProposedLogProb(DelayedFixtureError::Score),
                "Score",
            ),
            (
                DelayedStepError::LogQRatio(DelayedFixtureError::Ratio),
                "Ratio",
            ),
            (
                DelayedStepError::Commit(DelayedFixtureError::Commit),
                "Commit",
            ),
        ];

        for (err, source) in cases {
            assert_eq!(
                err.source().map(ToString::to_string),
                Some(source.to_owned())
            );
        }
    }

    #[test]
    fn delayed_commit_error_is_atomic() {
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = FixedDelayedProposal {
            commit_error: Some(DelayedFixtureError::Commit),
            ..FixedDelayedProposal::new(0.0)
        };
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Commit(DelayedFixtureError::Commit))
        );
        assert_eq!(chain.state, MutScalar(2.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_commit_error_may_restore_after_mutating() {
        struct RestoringCommitError;

        impl DelayedProposal<MutScalar> for RestoringCommitError {
            type Plan = f64;
            type Info = f64;
            type Error = DelayedFixtureError;

            fn propose_plan<R: Rng + ?Sized>(
                &mut self,
                _state: &MutScalar,
                _rng: &mut R,
            ) -> Result<Option<f64>, Self::Error> {
                Ok(Some(0.0))
            }

            fn proposed_log_prob<T: Target<MutScalar>>(
                &self,
                _state: &MutScalar,
                plan: &f64,
                target: &T,
            ) -> Result<f64, Self::Error> {
                Ok(target.log_prob(&MutScalar(*plan)))
            }

            fn info(&self, plan: &f64) -> f64 {
                *plan
            }

            fn commit<R: Rng + ?Sized>(
                &mut self,
                state: &mut MutScalar,
                plan: f64,
                _rng: &mut R,
            ) -> Result<(), Self::Error> {
                let old = state.0;
                state.0 = plan;
                state.0 = old;
                Err(DelayedFixtureError::Commit)
            }
        }

        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = RestoringCommitError;
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed(&Normal, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Commit(DelayedFixtureError::Commit))
        );
        assert_eq!(chain.state, MutScalar(2.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_checked_accepts_consistent_commit() {
        struct CheckedMove;
        impl DelayedProposal<Scalar> for CheckedMove {
            type Plan = f64;
            type Info = f64;
            type Error = DelayedFixtureError;

            fn propose_plan<R: Rng + ?Sized>(
                &mut self,
                _state: &Scalar,
                _rng: &mut R,
            ) -> Result<Option<f64>, Self::Error> {
                Ok(Some(0.0))
            }

            fn proposed_log_prob<T: Target<Scalar>>(
                &self,
                _state: &Scalar,
                plan: &f64,
                target: &T,
            ) -> Result<f64, Self::Error> {
                Ok(target.log_prob(&Scalar(*plan)))
            }

            fn info(&self, plan: &f64) -> f64 {
                *plan
            }

            fn commit<R: Rng + ?Sized>(
                &mut self,
                state: &mut Scalar,
                plan: f64,
                _rng: &mut R,
            ) -> Result<(), Self::Error> {
                state.0 = plan;
                Ok(())
            }
        }

        let mut chain = Chain::new(Scalar(2.0), &Normal).unwrap();
        let mut proposal = CheckedMove;
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_delayed_checked(&Normal, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_eq!(chain.state, Scalar(0.0));
        assert_relative_eq!(chain.log_prob(), 0.0);
        assert_eq!(chain.accepted(), 1);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_checked_restores_after_mismatched_commit() {
        struct MismatchedCommit;
        impl DelayedProposal<Scalar> for MismatchedCommit {
            type Plan = f64;
            type Info = f64;
            type Error = DelayedFixtureError;

            fn propose_plan<R: Rng + ?Sized>(
                &mut self,
                _state: &Scalar,
                _rng: &mut R,
            ) -> Result<Option<f64>, Self::Error> {
                Ok(Some(0.0))
            }

            fn proposed_log_prob<T: Target<Scalar>>(
                &self,
                _state: &Scalar,
                plan: &f64,
                target: &T,
            ) -> Result<f64, Self::Error> {
                Ok(target.log_prob(&Scalar(*plan)))
            }

            fn info(&self, plan: &f64) -> f64 {
                *plan
            }

            fn commit<R: Rng + ?Sized>(
                &mut self,
                state: &mut Scalar,
                _plan: f64,
                _rng: &mut R,
            ) -> Result<(), Self::Error> {
                state.0 = 2.0;
                Ok(())
            }
        }

        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = MismatchedCommit;
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed_checked(&Normal, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(
                McmcError::InconsistentDelayedCommitLogProb
            ))
        );
        assert_eq!(chain.state, Scalar(1.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_checked_restores_after_nan_committed_log_prob() {
        struct NanAtTwo;
        impl Target<Scalar> for NanAtTwo {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0.to_bits() == 2.0_f64.to_bits() {
                    f64::NAN
                } else {
                    Normal.log_prob(state)
                }
            }
        }

        let mut chain = Chain::new(Scalar(1.0), &NanAtTwo).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = CheckedCommitProposal {
            scored: 0.0,
            committed: 2.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed_checked(&NanAtTwo, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(McmcError::NanCommittedLogProb))
        );
        assert_eq!(chain.state, Scalar(1.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_checked_restores_after_infinite_committed_log_prob() {
        struct InfiniteAtTwo;
        impl Target<Scalar> for InfiniteAtTwo {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0.to_bits() == 2.0_f64.to_bits() {
                    f64::INFINITY
                } else {
                    Normal.log_prob(state)
                }
            }
        }

        let mut chain = Chain::new(Scalar(1.0), &InfiniteAtTwo).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = CheckedCommitProposal {
            scored: 0.0,
            committed: 2.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed_checked(&InfiniteAtTwo, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Mcmc(McmcError::InfiniteCommittedLogProb))
        );
        assert_eq!(chain.state, Scalar(1.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    #[test]
    fn delayed_checked_restores_after_mutating_commit_error() {
        struct MutatingCommitError;
        impl DelayedProposal<Scalar> for MutatingCommitError {
            type Plan = f64;
            type Info = f64;
            type Error = DelayedFixtureError;

            fn propose_plan<R: Rng + ?Sized>(
                &mut self,
                _state: &Scalar,
                _rng: &mut R,
            ) -> Result<Option<f64>, Self::Error> {
                Ok(Some(0.0))
            }

            fn proposed_log_prob<T: Target<Scalar>>(
                &self,
                _state: &Scalar,
                plan: &f64,
                target: &T,
            ) -> Result<f64, Self::Error> {
                Ok(target.log_prob(&Scalar(*plan)))
            }

            fn info(&self, plan: &f64) -> f64 {
                *plan
            }

            fn commit<R: Rng + ?Sized>(
                &mut self,
                state: &mut Scalar,
                plan: f64,
                _rng: &mut R,
            ) -> Result<(), Self::Error> {
                state.0 = plan;
                Err(DelayedFixtureError::Commit)
            }
        }

        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let log_prob = chain.log_prob();
        let mut proposal = MutatingCommitError;
        let mut rng = StdRng::seed_from_u64(42);

        let result = chain.step_delayed_checked(&Normal, &mut proposal, &mut rng);

        assert_matches!(
            result,
            Err(DelayedStepError::Commit(DelayedFixtureError::Commit))
        );
        assert_eq!(chain.state, Scalar(1.0));
        assert_relative_eq!(chain.log_prob(), log_prob);
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
    }

    // --- state accessors ---

    #[test]
    fn replace_state_updates_log_prob() {
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        chain.replace_state(Scalar(2.0), &Normal).unwrap();
        assert_eq!(chain.state, Scalar(2.0));
        assert_relative_eq!(chain.log_prob(), -2.0, epsilon = 1e-12);
    }

    #[test]
    fn replace_state_rejects_nan() {
        struct NanTarget;
        impl Target<Scalar> for NanTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::NAN
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let result = chain.replace_state(Scalar(0.0), &NanTarget);
        assert_matches!(result, Err(McmcError::NanReplacementLogProb));
        // State should be unchanged on error
        assert_eq!(chain.state, Scalar(1.0));
    }

    #[test]
    fn replace_state_rejects_inf() {
        struct InfTarget;
        impl Target<Scalar> for InfTarget {
            fn log_prob(&self, _: &Scalar) -> f64 {
                f64::INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(1.0), &Normal).unwrap();
        let result = chain.replace_state(Scalar(0.0), &InfTarget);
        assert_matches!(result, Err(McmcError::InfiniteReplacementLogProb));
        assert_eq!(chain.state, Scalar(1.0));
    }

    #[test]
    fn into_state_returns_state() {
        let chain = Chain::new(Scalar(3.0), &Normal).unwrap();
        let state = chain.into_state();
        assert_eq!(state, Scalar(3.0));
    }

    // --- reset_counters and total_steps ---

    #[test]
    fn reset_counters_zeros_counts() {
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = RandomWalk { width: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }
        assert!(chain.total_steps() > 0);

        chain.reset_counters();
        assert_eq!(chain.accepted(), 0);
        assert_eq!(chain.rejected(), 0);
        assert_eq!(chain.total_steps(), 0);
        assert_relative_eq!(chain.acceptance_rate(), 0.0);
    }

    #[test]
    fn total_steps_matches_counts() {
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = RandomWalk { width: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        let steps = 50;
        for _ in 0..steps {
            chain.step(&Normal, &proposal, &mut rng).unwrap();
        }
        assert_eq!(chain.total_steps(), steps);
        assert_eq!(chain.total_steps(), chain.accepted() + chain.rejected());
    }

    #[test]
    fn checkpoint_total_steps_saturates_at_usize_max() {
        let checkpoint = ChainCheckpoint::new((), usize::MAX, 1);

        assert_eq!(checkpoint.total_steps(), usize::MAX);
    }

    #[test]
    fn chain_total_steps_saturates_after_checkpoint_restore() {
        let checkpoint = ChainCheckpoint::new(Scalar(0.0), usize::MAX, 1);
        let chain = Chain::from_checkpoint(checkpoint, &Normal).unwrap();

        assert_eq!(chain.total_steps(), usize::MAX);
        assert_relative_eq!(chain.acceptance_rate(), 1.0);
    }

    // --- Asymmetric proposal tests ---

    /// Deterministic proposal with a configurable log q-ratio.
    struct AsymmetricCloneProposal {
        target_value: f64,
        log_q: f64,
    }
    impl Proposal<Scalar> for AsymmetricCloneProposal {
        fn propose<R: Rng + ?Sized>(&self, _current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(self.target_value)
        }
        fn log_q_ratio(&self, _current: &Scalar, _proposed: &Scalar) -> f64 {
            self.log_q
        }
    }

    /// In-place version of `AsymmetricCloneProposal`.
    struct AsymmetricMutProposal {
        target_value: f64,
        log_q: f64,
    }
    impl ProposalMut<MutScalar> for AsymmetricMutProposal {
        type Undo = f64;
        type Info = f64;
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<f64> {
            let old = state.0;
            state.0 = self.target_value;
            Some(old)
        }
        fn info(&self, state: &MutScalar, _old: &f64) -> f64 {
            state.0
        }
        fn undo(&mut self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
        fn log_q_ratio(&self, _state: &MutScalar, _token: &f64) -> f64 {
            self.log_q
        }
    }

    /// In-place proposal whose proposal ratio depends on both old and new state.
    struct StateDependentMutProposal {
        target_value: f64,
    }
    impl ProposalMut<MutScalar> for StateDependentMutProposal {
        type Undo = f64;
        type Info = f64;
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            state: &mut MutScalar,
            _rng: &mut R,
        ) -> Option<f64> {
            let old = state.0;
            state.0 = self.target_value;
            Some(old)
        }
        fn info(&self, state: &MutScalar, _old: &f64) -> f64 {
            state.0
        }
        fn undo(&mut self, state: &mut MutScalar, old: f64) {
            state.0 = old;
        }
        fn log_q_ratio(&self, state: &MutScalar, old: &f64) -> f64 {
            old - state.0
        }
    }

    #[test]
    fn positive_log_q_accepts() {
        // From x=0 (log_prob=0) to x=1 (log_prob=-0.5).
        // Without asymmetry: log_alpha = -0.5, might reject.
        // With large positive log_q: log_alpha = -0.5 + 100 = 99.5, always accept.
        let mut chain = Chain::new(Scalar(0.0), &Normal).unwrap();
        let proposal = AsymmetricCloneProposal {
            target_value: 1.0,
            log_q: 100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();
        assert_eq!(
            chain.state,
            Scalar(1.0),
            "Large positive log_q should force acceptance"
        );
    }

    #[test]
    fn negative_log_q_promotes_rejection() {
        // From x=2 (log_prob=-2) to x=0 (log_prob=0).
        // Without asymmetry: log_alpha = 2.0, always accept.
        // With large negative log_q: log_alpha = 2 - 100 = -98, always reject.
        let mut chain = Chain::new(Scalar(2.0), &Normal).unwrap();
        let proposal = AsymmetricCloneProposal {
            target_value: 0.0,
            log_q: -100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&Normal, &proposal, &mut rng).unwrap();
        assert_eq!(
            chain.state,
            Scalar(2.0),
            "Large negative log_q should force rejection"
        );
    }

    #[test]
    fn step_mut_respects_log_q() {
        // Acceptance via step_mut with positive log_q
        let mut chain = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut proposal = AsymmetricMutProposal {
            target_value: 1.0,
            log_q: 100.0,
        };
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();
        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_eq!(chain.state, MutScalar(1.0));

        // Rejection via step_mut with negative log_q
        let mut chain2 = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal2 = AsymmetricMutProposal {
            target_value: 0.0,
            log_q: -100.0,
        };
        let mut rng2 = StdRng::seed_from_u64(42);

        let step2 = chain2.step_mut(&Normal, &mut proposal2, &mut rng2).unwrap();
        assert_eq!(step2.outcome(), StepOutcome::RejectedProposal);
        assert_eq!(chain2.state, MutScalar(2.0));
    }

    #[test]
    fn step_mut_log_q_sees_old_and_new() {
        let mut chain = Chain::new(MutScalar(2.0), &Normal).unwrap();
        let mut proposal = StateDependentMutProposal { target_value: 0.0 };
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain.step_mut(&Normal, &mut proposal, &mut rng).unwrap();

        assert_eq!(step.outcome(), StepOutcome::Accepted);
        assert_eq!(chain.state, MutScalar(0.0));

        let mut chain2 = Chain::new(MutScalar(0.0), &Normal).unwrap();
        let mut proposal2 = StateDependentMutProposal { target_value: 2.0 };
        let mut rng2 = StdRng::seed_from_u64(42);

        let step2 = chain2.step_mut(&Normal, &mut proposal2, &mut rng2).unwrap();

        assert_eq!(step2.outcome(), StepOutcome::RejectedProposal);
        assert_eq!(chain2.state, MutScalar(0.0));
    }

    // --- Edge case: -inf log-probability ---

    #[test]
    fn step_rejects_neg_inf() {
        struct NegInfAt(f64);
        impl Target<Scalar> for NegInfAt {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if (state.0 - self.0).abs() < f64::EPSILON {
                    f64::NEG_INFINITY
                } else {
                    -0.5 * state.0 * state.0
                }
            }
        }
        let target = NegInfAt(0.0);
        let mut chain = Chain::new(Scalar(1.0), &target).unwrap();
        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&target, &proposal, &mut rng).unwrap();
        // exp(-inf - (-0.5)) = exp(-inf) = 0 → always reject
        assert_eq!(
            chain.state,
            Scalar(1.0),
            "Should reject move to -inf log_prob"
        );
    }

    #[test]
    fn step_escapes_neg_inf() {
        struct FiniteAtOrigin;
        impl Target<Scalar> for FiniteAtOrigin {
            fn log_prob(&self, state: &Scalar) -> f64 {
                if state.0.abs() < f64::EPSILON {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
        let target = FiniteAtOrigin;
        let mut chain = Chain::new(Scalar(1.0), &target).unwrap();
        assert!(chain.log_prob().is_infinite() && chain.log_prob().is_sign_negative());

        let proposal = FixedProposal(0.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&target, &proposal, &mut rng).unwrap();
        // log_alpha = 0 - (-inf) = inf → always accept
        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should accept escape from -inf log_prob"
        );
    }

    #[test]
    fn step_rejects_both_neg_inf() {
        struct AlwaysNegInf;
        impl Target<Scalar> for AlwaysNegInf {
            fn log_prob(&self, _state: &Scalar) -> f64 {
                f64::NEG_INFINITY
            }
        }
        let mut chain = Chain::new(Scalar(0.0), &AlwaysNegInf).unwrap();
        let proposal = FixedProposal(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        chain.step(&AlwaysNegInf, &proposal, &mut rng).unwrap();
        // log_alpha = (-inf) - (-inf) = NaN → NaN comparisons false → reject
        assert_eq!(
            chain.state,
            Scalar(0.0),
            "Should reject when both states have -inf log_prob"
        );
        assert_eq!(chain.rejected(), 1);
    }

    #[test]
    fn step_mut_rejects_both_neg_inf() {
        struct AlwaysNegInf;
        impl Target<MutScalar> for AlwaysNegInf {
            fn log_prob(&self, _state: &MutScalar) -> f64 {
                f64::NEG_INFINITY
            }
        }

        let mut chain = Chain::new(MutScalar(0.0), &AlwaysNegInf).unwrap();
        let mut proposal = FixedMutProposal(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        let step = chain
            .step_mut(&AlwaysNegInf, &mut proposal, &mut rng)
            .unwrap();

        assert_eq!(step.outcome(), StepOutcome::RejectedProposal);
        assert_eq!(
            chain.state,
            MutScalar(0.0),
            "State should be rolled back after NaN log acceptance ratio"
        );
        assert_eq!(chain.rejected(), 1);
    }
}
