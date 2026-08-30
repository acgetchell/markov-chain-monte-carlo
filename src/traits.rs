//! Core traits for target distributions and proposal distributions.

use core::{fmt, num::NonZeroUsize};
use std::error::Error;

use rand::Rng;

/// One endpoint's inputs to a weighted discrete proposal ratio.
///
/// Grouping the selected move-family weight, the endpoint's total family
/// weight, and the concrete-site count prevents forward and reverse positional
/// arguments from being interleaved accidentally.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DiscreteProposalEndpoint {
    selected_weight: f64,
    total_weight: f64,
    site_count: usize,
}

impl DiscreteProposalEndpoint {
    /// Describe one endpoint of a weighted discrete proposal.
    pub const fn new(selected_weight: f64, total_weight: f64, site_count: usize) -> Self {
        Self {
            selected_weight,
            total_weight,
            site_count,
        }
    }

    /// Selected move-family weight at this endpoint.
    #[must_use]
    pub const fn selected_weight(self) -> f64 {
        self.selected_weight
    }

    /// Sum of all move-family weights at this endpoint.
    #[must_use]
    pub const fn total_weight(self) -> f64 {
        self.total_weight
    }

    /// Number of concrete sites in the selected family at this endpoint.
    #[must_use]
    pub const fn site_count(self) -> usize {
        self.site_count
    }
}

/// Hastings correction for a discrete proposal with weighted move families and
/// uniformly sampled concrete sites.
///
/// This helper covers proposal kernels that first choose a move family, then
/// choose uniformly from that family's concrete valid-site set.  It computes:
///
/// ```text
/// log(q(current | proposed) / q(proposed | current))
/// = log(reverse_weight) - log(reverse_weight_sum)
/// - log(forward_weight) + log(forward_weight_sum)
/// + log(forward_site_count) - log(reverse_site_count)
/// ```
///
/// Use [`from_counts`](Self::from_counts) when the forward and reverse
/// move-family selection probabilities are equal and cancel.
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::prelude::by_value::{
///     DiscreteProposalRatio, DiscreteProposalRatioError,
/// };
///
/// let log_q_ratio = DiscreteProposalRatio::from_counts(3, 1)?.log_q_ratio();
///
/// assert!((log_q_ratio - 3.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), DiscreteProposalRatioError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DiscreteProposalRatio {
    forward_weight: f64,
    forward_weight_sum: f64,
    reverse_weight: f64,
    reverse_weight_sum: f64,
    forward_site_count: NonZeroUsize,
    reverse_site_count: usize,
}

impl DiscreteProposalRatio {
    /// Create a ratio from named forward and reverse endpoint descriptions.
    ///
    /// The forward count and weight must be positive for a successful plan.
    /// Both endpoint weight sums must be positive and finite, and a family
    /// weight cannot exceed its endpoint sum. A zero reverse count or weight is
    /// allowed and yields `f64::NEG_INFINITY` from
    /// [`log_q_ratio`](Self::log_q_ratio).
    ///
    /// # Errors
    ///
    /// Returns the endpoint-specific [`DiscreteProposalRatioError`] variant
    /// when a weight, total weight, or successful forward site count is
    /// invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalEndpoint, DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let forward = DiscreteProposalEndpoint::new(1.0, 4.0, 6);
    /// let reverse = DiscreteProposalEndpoint::new(3.0, 4.0, 2);
    /// let ratio = DiscreteProposalRatio::from_endpoints(forward, reverse)?;
    ///
    /// assert_eq!(ratio.forward_weight(), 1.0);
    /// assert_eq!(ratio.reverse_site_count(), 2);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    pub fn from_endpoints(
        forward: DiscreteProposalEndpoint,
        reverse: DiscreteProposalEndpoint,
    ) -> Result<Self, DiscreteProposalRatioError> {
        let forward_weight = forward.selected_weight;
        let forward_weight_sum = forward.total_weight;
        let forward_site_count = forward.site_count;
        let reverse_weight = reverse.selected_weight;
        let reverse_weight_sum = reverse.total_weight;
        let reverse_site_count = reverse.site_count;

        if !forward_weight_sum.is_finite() || forward_weight_sum <= 0.0 {
            return Err(DiscreteProposalRatioError::InvalidForwardWeightSum {
                weight_sum: forward_weight_sum,
            });
        }
        if !reverse_weight_sum.is_finite() || reverse_weight_sum <= 0.0 {
            return Err(DiscreteProposalRatioError::InvalidReverseWeightSum {
                weight_sum: reverse_weight_sum,
            });
        }
        if !forward_weight.is_finite()
            || forward_weight <= 0.0
            || forward_weight > forward_weight_sum
        {
            return Err(DiscreteProposalRatioError::InvalidForwardWeight {
                weight: forward_weight,
            });
        }
        if !reverse_weight.is_finite()
            || reverse_weight < 0.0
            || reverse_weight > reverse_weight_sum
        {
            return Err(DiscreteProposalRatioError::InvalidReverseWeight {
                weight: reverse_weight,
            });
        }
        let Some(forward_site_count) = NonZeroUsize::new(forward_site_count) else {
            return Err(DiscreteProposalRatioError::ZeroForwardSiteCount);
        };

        Ok(Self {
            forward_weight,
            forward_weight_sum,
            reverse_weight,
            reverse_weight_sum,
            forward_site_count,
            reverse_site_count,
        })
    }

    /// Create a ratio for equal move-family selection probabilities.
    ///
    /// This is the common CDT-style site-count correction:
    ///
    /// ```text
    /// log_q_ratio = log(forward_site_count) - log(reverse_site_count)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DiscreteProposalRatioError::ZeroForwardSiteCount`] when a
    /// successful proposal reports no valid forward sites.  A zero reverse-site
    /// count is valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    /// let log_q_ratio = ratio.log_q_ratio();
    ///
    /// assert!((log_q_ratio - 2.0_f64.ln()).abs() < 1e-12);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    pub fn from_counts(
        forward_site_count: usize,
        reverse_site_count: usize,
    ) -> Result<Self, DiscreteProposalRatioError> {
        Self::from_endpoints(
            DiscreteProposalEndpoint::new(1.0, 1.0, forward_site_count),
            DiscreteProposalEndpoint::new(1.0, 1.0, reverse_site_count),
        )
    }

    /// Forward move-family weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalEndpoint, DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let ratio = DiscreteProposalRatio::from_endpoints(
    ///     DiscreteProposalEndpoint::new(1.0, 4.0, 6),
    ///     DiscreteProposalEndpoint::new(3.0, 4.0, 2),
    /// )?;
    ///
    /// assert_eq!(ratio.forward_weight(), 1.0);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn forward_weight(self) -> f64 {
        self.forward_weight
    }

    /// Sum of all move-family weights at the forward endpoint.
    #[must_use]
    pub const fn forward_weight_sum(self) -> f64 {
        self.forward_weight_sum
    }

    /// Reverse move-family weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalEndpoint, DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let ratio = DiscreteProposalRatio::from_endpoints(
    ///     DiscreteProposalEndpoint::new(1.0, 4.0, 6),
    ///     DiscreteProposalEndpoint::new(3.0, 4.0, 2),
    /// )?;
    ///
    /// assert_eq!(ratio.reverse_weight(), 3.0);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn reverse_weight(self) -> f64 {
        self.reverse_weight
    }

    /// Sum of all move-family weights at the reverse endpoint.
    #[must_use]
    pub const fn reverse_weight_sum(self) -> f64 {
        self.reverse_weight_sum
    }

    /// Number of concrete sites sampled by the forward proposal family.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    ///
    /// assert_eq!(ratio.forward_site_count(), 4);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn forward_site_count(self) -> usize {
        self.forward_site_count.get()
    }

    /// Number of concrete sites sampled by the reverse proposal family.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    ///
    /// assert_eq!(ratio.reverse_site_count(), 2);
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn reverse_site_count(self) -> usize {
        self.reverse_site_count
    }

    /// Compute `log(q(current | proposed) / q(proposed | current))`.
    ///
    /// Construction validates the forward proposal probability and move-family
    /// weights, so this method only performs the log-space arithmetic.  A zero
    /// reverse-site count or zero reverse move-family weight produces
    /// `f64::NEG_INFINITY`.
    ///
    /// # Examples
    ///
    /// Zero reverse-site counts produce `-inf`, representing a transition that
    /// cannot be accepted because no reverse proposal exists.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::by_value::{
    ///     DiscreteProposalRatio, DiscreteProposalRatioError,
    /// };
    ///
    /// let log_q_ratio = DiscreteProposalRatio::from_counts(3, 0)?.log_q_ratio();
    ///
    /// assert!(log_q_ratio.is_infinite());
    /// assert!(log_q_ratio.is_sign_negative());
    /// # Ok::<(), DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub fn log_q_ratio(self) -> f64 {
        let Some(reverse_site_count) = NonZeroUsize::new(self.reverse_site_count) else {
            return f64::NEG_INFINITY;
        };
        if self.reverse_weight == 0.0 {
            return f64::NEG_INFINITY;
        }

        self.reverse_weight.ln() - self.reverse_weight_sum.ln() - self.forward_weight.ln()
            + self.forward_weight_sum.ln()
            + count_ln(self.forward_site_count)
            - count_ln(reverse_site_count)
    }
}

/// Errors from constructing a [`DiscreteProposalRatio`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DiscreteProposalRatioError {
    /// The forward move-family weight is not positive and finite or exceeds its sum.
    #[non_exhaustive]
    InvalidForwardWeight {
        /// Invalid forward move-family weight.
        weight: f64,
    },
    /// The forward endpoint move-family weight sum is not positive and finite.
    #[non_exhaustive]
    InvalidForwardWeightSum {
        /// Invalid forward endpoint weight sum.
        weight_sum: f64,
    },
    /// The reverse move-family weight is negative, non-finite, or exceeds its sum.
    #[non_exhaustive]
    InvalidReverseWeight {
        /// Invalid reverse move-family weight.
        weight: f64,
    },
    /// The reverse endpoint move-family weight sum is not positive and finite.
    #[non_exhaustive]
    InvalidReverseWeightSum {
        /// Invalid reverse endpoint weight sum.
        weight_sum: f64,
    },
    /// A successful forward proposal reported zero valid forward sites.
    ZeroForwardSiteCount,
}

impl fmt::Display for DiscreteProposalRatioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidForwardWeight { weight } => write!(
                f,
                "invalid forward move-family weight {weight}: expected a positive finite value no greater than its endpoint weight sum"
            ),
            Self::InvalidForwardWeightSum { weight_sum } => write!(
                f,
                "invalid forward move-family weight sum {weight_sum}: expected a positive finite value"
            ),
            Self::InvalidReverseWeight { weight } => write!(
                f,
                "invalid reverse move-family weight {weight}: expected a nonnegative finite value no greater than its endpoint weight sum"
            ),
            Self::InvalidReverseWeightSum { weight_sum } => write!(
                f,
                "invalid reverse move-family weight sum {weight_sum}: expected a positive finite value"
            ),
            Self::ZeroForwardSiteCount => {
                f.write_str("invalid forward site count 0 for a successful proposal")
            }
        }
    }
}

impl Error for DiscreteProposalRatioError {}

/// Convert a valid-site count into a logarithm for proposal-ratio arithmetic.
///
/// The [`NonZeroUsize`] argument records the constructor invariant for counts
/// that must be positive before they enter log-space arithmetic.
#[expect(
    clippy::cast_precision_loss,
    reason = "valid-site counts intentionally cross into log-space f64 proposal arithmetic"
)]
fn count_ln(count: NonZeroUsize) -> f64 {
    (count.get() as f64).ln()
}

/// Additive composition of two target log-weight components.
///
/// `AdditiveTarget` is a small adapter for targets whose log weight is a sum
/// of independent model terms, such as an energy-based model term, learned
/// regularizer, physics action, umbrella-sampling bias, or softened
/// constraint.  Each component implements [`Target`] using this crate's
/// ordinary sign convention: return a log probability/log weight, or return
/// negative energy/action when the model is written as `exp(-S)`.
///
/// For action terms, this means:
///
/// ```text
/// log pi(state) = -S_model(state) - S_bias(state)
/// ```
///
/// so Metropolis-Hastings uses:
///
/// ```text
/// log pi(y) - log pi(x) = -(Delta S_model + Delta S_bias)
/// ```
///
/// Proposal-ratio corrections remain separate in [`Proposal::log_q_ratio`],
/// [`ProposalMut::log_q_ratio`], or [`DelayedProposal::log_q_ratio`].
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::prelude::{AdditiveTarget, Target};
///
/// struct ModelAction;
/// impl Target<i32> for ModelAction {
///     fn log_prob(&self, state: &i32) -> f64 {
///         -0.5 * f64::from(*state * *state)
///     }
/// }
///
/// struct BiasTowardTwo;
/// impl Target<i32> for BiasTowardTwo {
///     fn log_prob(&self, state: &i32) -> f64 {
///         let distance = f64::from(*state - 2);
///         -distance * distance
///     }
/// }
///
/// let target = AdditiveTarget::new(ModelAction, BiasTowardTwo);
///
/// assert_eq!(target.log_prob(&2), -2.0);
/// assert_eq!(target.log_prob(&0), -4.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct AdditiveTarget<A, B> {
    primary: A,
    additive: B,
}

impl<A, B> AdditiveTarget<A, B> {
    /// Create a target whose log weight is `primary + additive`.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{AdditiveTarget, Target};
    ///
    /// struct Flat;
    /// impl Target<()> for Flat {
    ///     fn log_prob(&self, _: &()) -> f64 { 0.0 }
    /// }
    ///
    /// struct Offset;
    /// impl Target<()> for Offset {
    ///     fn log_prob(&self, _: &()) -> f64 { -2.0 }
    /// }
    ///
    /// let target = AdditiveTarget::new(Flat, Offset);
    ///
    /// assert_eq!(target.log_prob(&()), -2.0);
    /// ```
    pub const fn new(primary: A, additive: B) -> Self {
        Self { primary, additive }
    }

    /// Primary target component.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{AdditiveTarget, Target};
    ///
    /// struct Model;
    /// impl Target<i32> for Model {
    ///     fn log_prob(&self, state: &i32) -> f64 { -f64::from(*state) }
    /// }
    ///
    /// struct Bias;
    /// impl Target<i32> for Bias {
    ///     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
    /// }
    ///
    /// let target = AdditiveTarget::new(Model, Bias);
    ///
    /// assert_eq!(target.primary().log_prob(&3), -3.0);
    /// ```
    #[must_use]
    pub const fn primary(&self) -> &A {
        &self.primary
    }

    /// Additive target component, such as a bias or auxiliary log weight.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{AdditiveTarget, Target};
    ///
    /// struct Model;
    /// impl Target<i32> for Model {
    ///     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
    /// }
    ///
    /// struct Bias;
    /// impl Target<i32> for Bias {
    ///     fn log_prob(&self, state: &i32) -> f64 {
    ///         -f64::from((*state - 1).abs())
    ///     }
    /// }
    ///
    /// let target = AdditiveTarget::new(Model, Bias);
    ///
    /// assert_eq!(target.additive().log_prob(&3), -2.0);
    /// ```
    #[must_use]
    pub const fn additive(&self) -> &B {
        &self.additive
    }

    /// Consume the adapter into its component targets.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{AdditiveTarget, Target};
    ///
    /// struct Model;
    /// impl Target<()> for Model {
    ///     fn log_prob(&self, _: &()) -> f64 { -1.0 }
    /// }
    ///
    /// struct Bias;
    /// impl Target<()> for Bias {
    ///     fn log_prob(&self, _: &()) -> f64 { -2.0 }
    /// }
    ///
    /// let target = AdditiveTarget::new(Model, Bias);
    /// let (model, bias) = target.into_parts();
    ///
    /// assert_eq!(model.log_prob(&()), -1.0);
    /// assert_eq!(bias.log_prob(&()), -2.0);
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (A, B) {
        (self.primary, self.additive)
    }
}

impl<S, A, B> Target<S> for AdditiveTarget<A, B>
where
    A: Target<S>,
    B: Target<S>,
{
    fn log_prob(&self, state: &S) -> f64 {
        self.primary.log_prob(state) + self.additive.log_prob(state)
    }
}

/// Target distribution.
///
/// `log_prob` returns a value proportional to the natural logarithm of the
/// target probability mass or density at `state`.  It may be an unnormalized
/// log-density or negative energy/action; additive constants do not affect
/// Metropolis-Hastings acceptance probabilities.
///
/// This is not a logit or arbitrary score: differences between two returned
/// values must be log probability ratios for the chain to target the intended
/// distribution.  Return `f64::NEG_INFINITY` for impossible states.
///
/// A target evaluation must be observational: for a fixed behavior-relevant
/// state and target configuration, repeated calls return the same log weight
/// and do not change later transition probabilities. Interior mutability is
/// permitted only when callers synchronize changes between transitions; each
/// transition re-scores its current state before evaluating acceptance.
pub trait Target<S> {
    /// Compute log-probability (or negative energy/action).
    fn log_prob(&self, state: &S) -> f64;
}

/// Proposal distribution for generating new states by value.
///
/// This trait returns a proposed state by value.  Implementations for small
/// state spaces often clone and modify the current state internally, but the
/// trait itself does not require `S: Clone`.  For state spaces where allocating
/// a whole proposed state is expensive (e.g., triangulations, large graphs),
/// see [`ProposalMut`] which mutates in place and supports cheap rollback.
///
/// `Proposal`, [`ProposalMut`], and [`DelayedProposal`] intentionally remain
/// separate strategy contracts. They differ in when state mutation occurs,
/// where rollback evidence lives, and which failure/telemetry capabilities are
/// meaningful; normalizing them into one outcome would obscure those
/// transition guarantees.
pub trait Proposal<S> {
    /// Propose a new state from the current one.
    fn propose<R: Rng + ?Sized>(&self, current: &S, rng: &mut R) -> S;

    /// Log proposal ratio:
    /// log(q(current | proposed) / q(proposed | current))
    ///
    /// Defaults to 0 for symmetric proposals.
    fn log_q_ratio(&self, _current: &S, _proposed: &S) -> f64 {
        0.0
    }
}

impl<S, P: Proposal<S> + ?Sized> Proposal<S> for &P {
    fn propose<R: Rng + ?Sized>(&self, current: &S, rng: &mut R) -> S {
        (**self).propose(current, rng)
    }

    fn log_q_ratio(&self, current: &S, proposed: &S) -> f64 {
        (**self).log_q_ratio(current, proposed)
    }
}

impl<S, P: Proposal<S> + ?Sized> Proposal<S> for &mut P {
    fn propose<R: Rng + ?Sized>(&self, current: &S, rng: &mut R) -> S {
        (**self).propose(current, rng)
    }

    fn log_q_ratio(&self, current: &S, proposed: &S) -> f64 {
        (**self).log_q_ratio(current, proposed)
    }
}

/// In-place proposal distribution with rollback.
///
/// Unlike [`Proposal`], which clones the state for each proposal,
/// `ProposalMut` mutates the state in place and returns an undo token
/// that can reverse the mutation on rejection.  This is the natural
/// model for combinatorial state spaces (e.g., triangulations, graphs)
/// where moves are invertible and cloning is expensive.
///
/// This strategy is intentionally distinct from by-value [`Proposal`] and
/// accept-before-mutation [`DelayedProposal`]: its undo token is part of the
/// transition invariant, not a generic proposal outcome.
///
/// # Associated Types
///
/// * [`Undo`](ProposalMut::Undo) — a small token that captures
///   exactly what is needed to reverse a move.
/// * [`Info`](ProposalMut::Info) — user-facing metadata returned with the
///   completed step.
pub trait ProposalMut<S> {
    /// Token that records how to reverse a proposed move.
    type Undo;
    /// User-facing metadata returned in structured step telemetry.
    ///
    /// Bulk sampler methods whose return type does not include a [`crate::Step`]
    /// do not construct this metadata.
    type Info;

    /// Mutate `state` in place, returning `Some(undo_token)` on success
    /// or `None` if no valid move could be found.
    ///
    /// Returning `None` must leave `state` exactly as it was on entry.  If a
    /// proposal mutates state before discovering that the move is invalid, it
    /// must undo those changes before returning `None`.  Once `Some(token)` is
    /// returned, [`undo`](ProposalMut::undo) must be able to restore the exact
    /// prior state and any proposal-internal transition state changed by the
    /// attempt. Returning `None` must likewise leave transition-relevant
    /// proposal state unchanged; telemetry-only scratch may remain for
    /// [`no_proposal_info`](Self::no_proposal_info) to consume. If this method
    /// unwinds before returning a token, it must likewise leave target and
    /// transition-relevant proposal state unchanged because the chain has no
    /// rollback evidence yet.
    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut S, rng: &mut R) -> Option<Self::Undo>;

    /// Produce telemetry metadata for a concrete in-place proposal.
    ///
    /// `state` is the already-mutated proposed state and `token` is the undo
    /// token returned by [`propose_mut`](Self::propose_mut). This hook runs
    /// after the proposed log-probability and proposal ratio have been
    /// validated, but before a rejected move is undone.
    ///
    /// This hook is observational: it must not change target state or proposal
    /// state that affects future transitions. Bulk sampler methods do not call
    /// it when they discard per-step telemetry.
    fn info(&self, state: &S, token: &Self::Undo) -> Self::Info;

    /// Produce telemetry metadata when no concrete proposal was available.
    ///
    /// The default returns no metadata. Stateful proposals may record a move
    /// family or search outcome during [`propose_mut`](Self::propose_mut) and
    /// return it here without an external side channel. Any mutation performed
    /// here must be limited to consuming telemetry-only scratch storage and
    /// must not affect future transitions. Bulk sampler methods do not call
    /// this hook when they discard per-step telemetry. Scratch retained across
    /// [`propose_mut`](Self::propose_mut) calls must therefore remain bounded
    /// even when this hook is never invoked, for example by using a fixed-size
    /// or single-overwritten slot.
    fn no_proposal_info(&mut self) -> Option<Self::Info> {
        None
    }

    /// Reverse a previously applied move using its undo token.
    ///
    /// Implementations must also restore proposal-internal transition state
    /// associated with the attempted move. Telemetry-only scratch storage need
    /// not be restored because it cannot affect future transitions. This
    /// method must not panic: the chain also invokes it from a drop guard while
    /// unwinding callbacks that run after a token has been produced.
    fn undo(&mut self, state: &mut S, token: Self::Undo);

    /// Log proposal ratio for the in-place move.
    ///
    /// `state` is the already-mutated proposed state.  Implementations that
    /// need the previous state to compute an asymmetric proposal ratio should
    /// store that information in [`Undo`](ProposalMut::Undo).  This keeps the
    /// normal in-place path allocation-free while still allowing exact
    /// forward/reverse proposal accounting.
    ///
    /// Defaults to 0 for symmetric proposals.
    fn log_q_ratio(&self, _state: &S, _token: &Self::Undo) -> f64 {
        0.0
    }
}

impl<S, P: ProposalMut<S> + ?Sized> ProposalMut<S> for &mut P {
    type Undo = P::Undo;
    type Info = P::Info;

    fn propose_mut<R: Rng + ?Sized>(&mut self, state: &mut S, rng: &mut R) -> Option<Self::Undo> {
        (**self).propose_mut(state, rng)
    }

    fn info(&self, state: &S, token: &Self::Undo) -> Self::Info {
        (**self).info(state, token)
    }

    fn no_proposal_info(&mut self) -> Option<Self::Info> {
        (**self).no_proposal_info()
    }

    fn undo(&mut self, state: &mut S, token: Self::Undo) {
        (**self).undo(state, token);
    }

    fn log_q_ratio(&self, state: &S, token: &Self::Undo) -> f64 {
        (**self).log_q_ratio(state, token)
    }
}

/// Proposal distribution for accept-before-mutation workflows.
///
/// `DelayedProposal` separates planning, Metropolis-Hastings evaluation, and
/// mutation:
///
/// 1. [`propose_plan`](Self::propose_plan) chooses a concrete transition
///    descriptor without mutating the state.
/// 2. [`proposed_log_prob`](Self::proposed_log_prob) evaluates the proposed
///    state's log-probability from that descriptor.
/// 3. [`crate::Chain::step_delayed`] performs the accept/reject draw.
/// 4. [`commit`](Self::commit) mutates the state only after acceptance.
///
/// The plan should identify the actual proposed transition, not just a move
/// class.  For example, a triangulation proposal should choose the move kind
/// and the concrete facet, vertex, or handle needed to apply it.  If no valid
/// site exists, return `Ok(None)` from [`propose_plan`](Self::propose_plan)
/// instead of accepting first and discovering that absence during
/// [`commit`](Self::commit).
///
/// When different move kinds or concrete sites have different proposal
/// probabilities, the plan and [`log_q_ratio`](Self::log_q_ratio) must encode
/// the full Hastings correction for the successful transition.  This is common
/// in local combinatorial kernels: a proposal might choose an add/delete move
/// kind, then choose uniformly among valid sites for that kind, while the
/// reverse state has a different number of valid sites.  Failed planning,
/// including bounded searches that find no valid site and return `Ok(None)`,
/// contributes self-loop probability but does not correct asymmetry among
/// successful forward and reverse moves.
///
/// This is useful for combinatorial state spaces where the log-probability
/// delta is cheap to compute from a concrete move descriptor.  If `commit`
/// returns an error, it must be failure-atomic: either the accepted move is
/// applied completely, or `state` is restored before returning `Err`.
///
/// Implement [`no_plan_info`](Self::no_plan_info) when planning may select a
/// move family before discovering that no concrete site exists.  [`crate::Chain`]
/// stores that metadata through [`crate::DelayedStep::info`] even though the step is
/// counted as a rejection with no proposal.
///
/// This strategy is intentionally distinct from [`Proposal`] and
/// [`ProposalMut`]. Its plan is scored before mutation and its typed errors are
/// stage-specific, so collapsing it into a normalized proposal outcome would
/// weaken the accept-before-mutation contract.
pub trait DelayedProposal<S> {
    /// Concrete move descriptor produced before the Metropolis-Hastings decision.
    type Plan;
    /// User-facing metadata returned in delayed-step telemetry.
    type Info;
    /// Proposal-specific error type.
    type Error;

    /// Propose a concrete move descriptor without mutating `state`.
    ///
    /// Return `Ok(None)` when no valid move can be proposed from the current
    /// state.  That is counted as a rejection by [`crate::Chain::step_delayed`].
    /// Ordinary proposal absence, such as failing to find a valid local site,
    /// should use this path rather than a later commit error.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when planning fails for proposal-specific reasons.
    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &S,
        rng: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error>;

    /// Produce telemetry metadata for an `Ok(None)` planning result.
    ///
    /// The default returns no metadata.  Implementations that choose a move
    /// family before discovering that no valid concrete site exists can store
    /// that choice during [`propose_plan`](Self::propose_plan) and return it
    /// here.  The value is attached to the resulting [`crate::DelayedStep`]
    /// with [`crate::StepOutcome::NoProposal`].
    ///
    /// This hook is telemetry-only: it must not affect transition mechanics,
    /// and bulk sampler methods may skip it. Any scratch state retained for this
    /// hook must remain bounded when callers do not request per-step telemetry.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use markov_chain_monte_carlo::prelude::delayed::*;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// enum MoveFamily {
    ///     Add,
    /// }
    ///
    /// struct Flat;
    /// impl Target<()> for Flat {
    ///     fn log_prob(&self, _: &()) -> f64 { 0.0 }
    /// }
    ///
    /// struct NoAddSite {
    ///     last_family: Option<MoveFamily>,
    /// }
    ///
    /// impl DelayedProposal<()> for NoAddSite {
    ///     type Plan = ();
    ///     type Info = MoveFamily;
    ///     type Error = Infallible;
    ///
    ///     fn propose_plan<R: Rng + ?Sized>(
    ///         &mut self,
    ///         _: &(),
    ///         _: &mut R,
    ///     ) -> Result<Option<()>, Self::Error> {
    ///         self.last_family = Some(MoveFamily::Add);
    ///         Ok(None)
    ///     }
    ///
    ///     fn no_plan_info(&mut self) -> Option<Self::Info> {
    ///         self.last_family.take()
    ///     }
    ///
    ///     fn proposed_log_prob<T: Target<()> + ?Sized>(
    ///         &self,
    ///         _: &(),
    ///         _: &(),
    ///         _: &T,
    ///     ) -> Result<f64, Self::Error> {
    ///         unreachable!("no plan should not be scored")
    ///     }
    ///
    ///     fn info(&self, _: &()) -> MoveFamily {
    ///         MoveFamily::Add
    ///     }
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
    /// let mut proposal = NoAddSite { last_family: None };
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let mut chain = Chain::new((), &Flat).map_err(DelayedStepError::Mcmc)?;
    ///
    /// let step = chain.step_delayed(&Flat, &mut proposal, &mut rng)?;
    ///
    /// assert_eq!(step.outcome(), StepOutcome::NoProposal);
    /// assert_eq!(step.info(), Some(&MoveFamily::Add));
    /// # Ok::<(), DelayedStepError<Infallible>>(())
    /// ```
    fn no_plan_info(&mut self) -> Option<Self::Info> {
        None
    }

    /// Compute the concrete proposed state's log-probability without mutating
    /// `state`.
    ///
    /// The returned value has the same numerical contract as
    /// [`Target::log_prob`]: finite values and `f64::NEG_INFINITY` are valid,
    /// while `NaN` and `+∞` are rejected by [`crate::Chain::step_delayed`].
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the proposal cannot evaluate the plan.
    fn proposed_log_prob<T: Target<S> + ?Sized>(
        &self,
        state: &S,
        plan: &Self::Plan,
        target: &T,
    ) -> Result<f64, Self::Error>;

    /// Log proposal ratio:
    /// log(q(current | proposed) / q(proposed | current)).
    ///
    /// This ratio must correspond to the same concrete transition represented
    /// by [`Plan`](DelayedProposal::Plan).  Include proposal asymmetries such
    /// as move-kind weights, site multiplicities, or reverse-move counts when
    /// they affect the forward/reverse transition probabilities.
    /// For example, a uniformly chosen local site usually contributes
    /// `ln(valid_forward_sites / valid_reverse_sites)` after equal move-kind
    /// weights cancel. [`DiscreteProposalRatio`] computes this common
    /// correction for weighted move families with uniformly sampled sites.
    ///
    /// Defaults to 0 for symmetric proposals.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the ratio cannot be evaluated.
    fn log_q_ratio(&self, _state: &S, _plan: &Self::Plan) -> Result<f64, Self::Error> {
        Ok(0.0)
    }

    /// Produce telemetry metadata for `plan`.
    ///
    /// This hook is observational and must not affect transition mechanics.
    /// Bulk sampler methods whose return type contains no [`crate::DelayedStep`]
    /// may skip it.
    fn info(&self, plan: &Self::Plan) -> Self::Info;

    /// Apply an accepted concrete move to `state`.
    ///
    /// This method is called only after the Metropolis-Hastings decision has
    /// accepted `plan`.  On error, implementations must restore `state` before
    /// returning so the chain's state and cached log-probability remain
    /// synchronized.  [`crate::Chain`] cannot repair a partially applied
    /// failed commit without an implementation-provided rollback token, so
    /// failure atomicity is part of this trait's correctness contract.
    /// The same guarantee applies if `commit` unwinds. Checked delayed stepping
    /// restores clone-isolated target state during unwinding, but cannot restore
    /// proposal-internal state; unchecked delayed stepping has no snapshot.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when a concrete accepted move cannot be committed
    /// because of an exceptional proposal, backend, or invariant failure.
    /// Absence of a valid site is ordinary proposal absence and should normally
    /// be reported by [`propose_plan`](DelayedProposal::propose_plan) as
    /// `Ok(None)`.
    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut S,
        plan: Self::Plan,
        rng: &mut R,
    ) -> Result<(), Self::Error>;
}

impl<S, P: DelayedProposal<S> + ?Sized> DelayedProposal<S> for &mut P {
    type Plan = P::Plan;
    type Info = P::Info;
    type Error = P::Error;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &S,
        rng: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error> {
        (**self).propose_plan(state, rng)
    }

    fn no_plan_info(&mut self) -> Option<Self::Info> {
        (**self).no_plan_info()
    }

    fn proposed_log_prob<T: Target<S> + ?Sized>(
        &self,
        state: &S,
        plan: &Self::Plan,
        target: &T,
    ) -> Result<f64, Self::Error> {
        (**self).proposed_log_prob(state, plan, target)
    }

    fn log_q_ratio(&self, state: &S, plan: &Self::Plan) -> Result<f64, Self::Error> {
        (**self).log_q_ratio(state, plan)
    }

    fn info(&self, plan: &Self::Plan) -> Self::Info {
        (**self).info(plan)
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut S,
        plan: Self::Plan,
        rng: &mut R,
    ) -> Result<(), Self::Error> {
        (**self).commit(state, plan, rng)
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use approx::assert_relative_eq;
    use rand::rng;

    use super::*;

    // --- Fixtures ---

    #[derive(Clone, Debug)]
    struct Scalar(f64);

    struct WeightedTarget(f64);
    impl Target<Scalar> for WeightedTarget {
        fn log_prob(&self, state: &Scalar) -> f64 {
            self.0 * state.0
        }
    }

    struct SymmetricProposal;
    impl Proposal<Scalar> for SymmetricProposal {
        fn propose<R: Rng + ?Sized>(&self, current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(-current.0)
        }
        // log_q_ratio intentionally not overridden — uses default
    }

    struct SymmetricMutProposal;
    impl ProposalMut<Scalar> for SymmetricMutProposal {
        type Undo = f64;
        type Info = f64;
        fn propose_mut<R: Rng + ?Sized>(
            &mut self,
            state: &mut Scalar,
            _rng: &mut R,
        ) -> Option<f64> {
            let old = state.0;
            state.0 = -state.0;
            Some(old)
        }
        fn info(&self, state: &Scalar, _old: &f64) -> f64 {
            state.0
        }
        fn undo(&mut self, state: &mut Scalar, old: f64) {
            state.0 = old;
        }
        // log_q_ratio intentionally not overridden — uses default
    }

    // --- Default log_q_ratio tests ---

    #[test]
    fn additive_target_exposes_components_and_parts() {
        let target = AdditiveTarget::new(WeightedTarget(2.0), WeightedTarget(-0.5));

        assert_relative_eq!(target.log_prob(&Scalar(4.0)), 6.0);
        assert_relative_eq!(target.primary().log_prob(&Scalar(4.0)), 8.0);
        assert_relative_eq!(target.additive().log_prob(&Scalar(4.0)), -2.0);

        let (primary, additive) = target.into_parts();

        assert_relative_eq!(primary.log_prob(&Scalar(3.0)), 6.0);
        assert_relative_eq!(additive.log_prob(&Scalar(3.0)), -1.5);
    }

    #[test]
    fn proposal_default_log_q_zero() {
        let p = SymmetricProposal;
        let ratio = p.log_q_ratio(&Scalar(0.0), &Scalar(1.0));
        assert_relative_eq!(ratio, 0.0);
    }

    #[test]
    fn proposal_ref_forwards() {
        let proposal = SymmetricProposal;
        let shared = &proposal;
        let proposed = shared.propose(&Scalar(2.0), &mut rng());

        assert_relative_eq!(proposed.0, -2.0);
        assert_relative_eq!(shared.log_q_ratio(&Scalar(2.0), &proposed), 0.0);
    }

    #[test]
    fn owned_mut_proposal_works() {
        let mut proposal = SymmetricMutProposal;
        let mut state = Scalar(2.0);
        let token = proposal.propose_mut(&mut state, &mut rng()).unwrap();

        assert_relative_eq!(state.0, -2.0);
        assert_relative_eq!(proposal.info(&state, &token), -2.0);
        assert_relative_eq!(proposal.log_q_ratio(&state, &token), 0.0);

        proposal.undo(&mut state, token);
        assert_relative_eq!(state.0, 2.0);
    }

    #[test]
    fn mut_ref_mut_proposal_forwards() {
        let mut proposal = SymmetricMutProposal;
        let shared = &mut proposal;
        let mut state = Scalar(2.0);
        let token = shared.propose_mut(&mut state, &mut rng()).unwrap();

        assert_relative_eq!(state.0, -2.0);
        assert_relative_eq!(shared.log_q_ratio(&state, &token), 0.0);

        shared.undo(&mut state, token);
        assert_relative_eq!(state.0, 2.0);
    }

    #[test]
    fn proposal_mut_default_log_q_zero() {
        let p = SymmetricMutProposal;
        let ratio = p.log_q_ratio(&Scalar(1.0), &0.0_f64);
        assert_relative_eq!(ratio, 0.0);
    }

    #[test]
    fn mut_ref_proposal_forwards() {
        let mut proposal = SymmetricProposal;
        let shared = &mut proposal;
        let proposed = shared.propose(&Scalar(2.0), &mut rng());

        assert_relative_eq!(proposed.0, -2.0);
        assert_relative_eq!(shared.log_q_ratio(&Scalar(2.0), &proposed), 0.0);
    }

    struct SymmetricDelayedProposal;
    impl DelayedProposal<Scalar> for SymmetricDelayedProposal {
        type Plan = f64;
        type Info = f64;
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            state: &Scalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            Ok(Some(-state.0))
        }

        fn proposed_log_prob<T: Target<Scalar> + ?Sized>(
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

    #[test]
    fn delayed_default_log_q_zero() {
        let p = SymmetricDelayedProposal;
        let ratio = p.log_q_ratio(&Scalar(0.0), &1.0).unwrap();
        assert_relative_eq!(ratio, 0.0);
    }

    #[test]
    fn discrete_proposal_ratio_accounts_for_site_counts() {
        let ratio = DiscreteProposalRatio::from_counts(3, 1)
            .unwrap()
            .log_q_ratio();

        assert_relative_eq!(ratio, 3.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn discrete_proposal_ratio_accounts_for_move_weights() {
        let ratio = DiscreteProposalRatio::from_endpoints(
            DiscreteProposalEndpoint::new(1.0, 4.0, 6),
            DiscreteProposalEndpoint::new(3.0, 4.0, 2),
        )
        .unwrap();

        assert_eq!(ratio.forward_weight().to_bits(), 1.0_f64.to_bits());
        assert_eq!(ratio.forward_weight_sum().to_bits(), 4.0_f64.to_bits());
        assert_eq!(ratio.reverse_weight().to_bits(), 3.0_f64.to_bits());
        assert_eq!(ratio.reverse_weight_sum().to_bits(), 4.0_f64.to_bits());
        assert_eq!(ratio.forward_site_count(), 6);
        assert_eq!(ratio.reverse_site_count(), 2);

        assert_relative_eq!(
            ratio.log_q_ratio(),
            3.0_f64.ln() - 4.0_f64.ln() - 1.0_f64.ln() + 4.0_f64.ln() + 6.0_f64.ln() - 2.0_f64.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn discrete_proposal_ratio_accounts_for_state_dependent_weight_sums() {
        let ratio = DiscreteProposalRatio::from_endpoints(
            DiscreteProposalEndpoint::new(1.0, 4.0, 1),
            DiscreteProposalEndpoint::new(1.0, 8.0, 1),
        )
        .unwrap()
        .log_q_ratio();

        assert_relative_eq!(ratio, 0.5_f64.ln(), epsilon = f64::EPSILON);
    }

    #[test]
    fn discrete_proposal_ratio_allows_missing_reverse_sites() {
        let ratio = DiscreteProposalRatio::from_counts(3, 0)
            .unwrap()
            .log_q_ratio();

        assert!(ratio.is_infinite());
        assert!(ratio.is_sign_negative());
    }

    #[test]
    fn discrete_proposal_ratio_allows_zero_reverse_weight() {
        let ratio = DiscreteProposalRatio::from_endpoints(
            DiscreteProposalEndpoint::new(1.0, 1.0, 3),
            DiscreteProposalEndpoint::new(0.0, 1.0, 1),
        )
        .unwrap()
        .log_q_ratio();

        assert!(ratio.is_infinite());
        assert!(ratio.is_sign_negative());
    }

    #[test]
    fn discrete_proposal_ratio_rejects_impossible_successful_forward_plan_at_construction() {
        let err = DiscreteProposalRatio::from_counts(0, 1).unwrap_err();

        assert_eq!(err, DiscreteProposalRatioError::ZeroForwardSiteCount);
        assert_eq!(
            err.to_string(),
            "invalid forward site count 0 for a successful proposal"
        );
    }

    #[test]
    fn discrete_proposal_ratio_rejects_invalid_forward_weights() {
        for weight in [0.0, -1.0, 2.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
            let err = DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(weight, 1.0, 1),
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
            )
            .unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid forward move-family weight {weight}: expected a positive finite value no greater than its endpoint weight sum"
                )
            );

            match err {
                DiscreteProposalRatioError::InvalidForwardWeight { weight: observed } => {
                    if weight.is_nan() {
                        assert!(observed.is_nan());
                    } else {
                        assert_eq!(observed.to_bits(), weight.to_bits());
                    }
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }
    }

    #[test]
    fn discrete_proposal_ratio_rejects_invalid_reverse_weights() {
        for weight in [-1.0, 2.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
            let err = DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
                DiscreteProposalEndpoint::new(weight, 1.0, 1),
            )
            .unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid reverse move-family weight {weight}: expected a nonnegative finite value no greater than its endpoint weight sum"
                )
            );

            match err {
                DiscreteProposalRatioError::InvalidReverseWeight { weight: observed } => {
                    if weight.is_nan() {
                        assert!(observed.is_nan());
                    } else {
                        assert_eq!(observed.to_bits(), weight.to_bits());
                    }
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }
    }

    #[test]
    fn discrete_proposal_ratio_rejects_invalid_endpoint_weight_sums() {
        for weight_sum in [0.0, -1.0, f64::NEG_INFINITY, f64::INFINITY] {
            let forward = DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(1.0, weight_sum, 1),
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
            )
            .unwrap_err();
            assert_eq!(
                forward,
                DiscreteProposalRatioError::InvalidForwardWeightSum { weight_sum }
            );
            assert_eq!(
                forward.to_string(),
                format!(
                    "invalid forward move-family weight sum {weight_sum}: expected a positive finite value"
                )
            );

            let reverse = DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
                DiscreteProposalEndpoint::new(1.0, weight_sum, 1),
            )
            .unwrap_err();
            assert_eq!(
                reverse,
                DiscreteProposalRatioError::InvalidReverseWeightSum { weight_sum }
            );
            assert_eq!(
                reverse.to_string(),
                format!(
                    "invalid reverse move-family weight sum {weight_sum}: expected a positive finite value"
                )
            );
        }

        assert!(matches!(
            DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(1.0, f64::NAN, 1),
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
            ),
            Err(DiscreteProposalRatioError::InvalidForwardWeightSum { weight_sum })
                if weight_sum.is_nan()
        ));
        assert!(matches!(
            DiscreteProposalRatio::from_endpoints(
                DiscreteProposalEndpoint::new(1.0, 1.0, 1),
                DiscreteProposalEndpoint::new(1.0, f64::NAN, 1),
            ),
            Err(DiscreteProposalRatioError::InvalidReverseWeightSum { weight_sum })
                if weight_sum.is_nan()
        ));
    }

    #[test]
    fn delayed_mut_ref_forwards() {
        struct ZeroTarget;
        impl Target<Scalar> for ZeroTarget {
            fn log_prob(&self, state: &Scalar) -> f64 {
                -state.0.abs()
            }
        }

        let mut proposal = SymmetricDelayedProposal;
        let shared = &mut proposal;
        let state = Scalar(1.0);
        let plan = shared.propose_plan(&state, &mut rng()).unwrap().unwrap();

        assert_relative_eq!(plan, -1.0);
        assert_relative_eq!(
            shared
                .proposed_log_prob(&state, &plan, &ZeroTarget)
                .unwrap(),
            -1.0
        );
        assert_relative_eq!(shared.log_q_ratio(&state, &plan).unwrap(), 0.0);
        assert_relative_eq!(shared.info(&plan), -1.0);

        let mut committed = Scalar(1.0);
        shared.commit(&mut committed, plan, &mut rng()).unwrap();
        assert_relative_eq!(committed.0, -1.0);
    }
}
