//! Core traits for target distributions and proposal distributions.

use rand::Rng;

/// Hastings correction for a discrete proposal with weighted move families and
/// uniformly sampled concrete sites.
///
/// This helper covers proposal kernels that first choose a move family, then
/// choose uniformly from that family's concrete valid-site set.  It computes:
///
/// ```text
/// log(q(current | proposed) / q(proposed | current))
/// = log(reverse_weight) - log(forward_weight)
/// + log(forward_site_count) - log(reverse_site_count)
/// ```
///
/// Use [`from_counts`](Self::from_counts) when forward and reverse move-family
/// weights are equal and cancel.
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::DiscreteProposalRatio;
///
/// let log_q_ratio = DiscreteProposalRatio::from_counts(3, 1)?.log_q_ratio();
///
/// assert!((log_q_ratio - 3.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DiscreteProposalRatio {
    forward_weight: f64,
    reverse_weight: f64,
    forward_site_count: usize,
    reverse_site_count: usize,
}

impl DiscreteProposalRatio {
    /// Create a ratio from move-family weights and valid concrete-site counts.
    ///
    /// `forward_weight` is the probability or relative weight of selecting the
    /// move family that proposed the current plan.  `reverse_weight` is the
    /// probability or relative weight of selecting that plan's inverse move
    /// family from the proposed state.
    ///
    /// The forward count and weight must be positive for a successful plan.  A
    /// zero reverse count or weight is allowed and yields `f64::NEG_INFINITY`
    /// from [`log_q_ratio`](Self::log_q_ratio), meaning the forward transition
    /// should never be accepted under the Metropolis-Hastings correction.
    ///
    /// # Errors
    ///
    /// Returns [`DiscreteProposalRatioError::InvalidForwardWeight`] when the
    /// forward move-family weight is not positive and finite.
    ///
    /// Returns [`DiscreteProposalRatioError::InvalidReverseWeight`] when the
    /// reverse move-family weight is negative, `NaN`, or infinite.  A zero
    /// reverse weight is valid.
    ///
    /// Returns [`DiscreteProposalRatioError::ZeroForwardSiteCount`] when a
    /// successful proposal reports no valid forward sites.  A zero reverse-site
    /// count is valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::new(0.25, 6, 0.75, 2)?;
    ///
    /// assert_eq!(ratio.forward_weight(), 0.25);
    /// assert_eq!(ratio.forward_site_count(), 6);
    /// assert_eq!(ratio.reverse_weight(), 0.75);
    /// assert_eq!(ratio.reverse_site_count(), 2);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    pub fn new(
        forward_weight: f64,
        forward_site_count: usize,
        reverse_weight: f64,
        reverse_site_count: usize,
    ) -> Result<Self, DiscreteProposalRatioError> {
        if !forward_weight.is_finite() || forward_weight <= 0.0 {
            return Err(DiscreteProposalRatioError::InvalidForwardWeight {
                weight: forward_weight,
            });
        }
        if !reverse_weight.is_finite() || reverse_weight < 0.0 {
            return Err(DiscreteProposalRatioError::InvalidReverseWeight {
                weight: reverse_weight,
            });
        }
        if forward_site_count == 0 {
            return Err(DiscreteProposalRatioError::ZeroForwardSiteCount);
        }

        Ok(Self {
            forward_weight,
            reverse_weight,
            forward_site_count,
            reverse_site_count,
        })
    }

    /// Create a ratio for equal move-family weights.
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
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    /// let log_q_ratio = ratio.log_q_ratio();
    ///
    /// assert!((log_q_ratio - 2.0_f64.ln()).abs() < 1e-12);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    pub fn from_counts(
        forward_site_count: usize,
        reverse_site_count: usize,
    ) -> Result<Self, DiscreteProposalRatioError> {
        Self::new(1.0, forward_site_count, 1.0, reverse_site_count)
    }

    /// Forward move-family weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::new(0.25, 6, 0.75, 2)?;
    ///
    /// assert_eq!(ratio.forward_weight(), 0.25);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn forward_weight(self) -> f64 {
        self.forward_weight
    }

    /// Reverse move-family weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::new(0.25, 6, 0.75, 2)?;
    ///
    /// assert_eq!(ratio.reverse_weight(), 0.75);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn reverse_weight(self) -> f64 {
        self.reverse_weight
    }

    /// Number of concrete sites sampled by the forward proposal family.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    ///
    /// assert_eq!(ratio.forward_site_count(), 4);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub const fn forward_site_count(self) -> usize {
        self.forward_site_count
    }

    /// Number of concrete sites sampled by the reverse proposal family.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let ratio = DiscreteProposalRatio::from_counts(4, 2)?;
    ///
    /// assert_eq!(ratio.reverse_site_count(), 2);
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
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
    /// use markov_chain_monte_carlo::DiscreteProposalRatio;
    ///
    /// let log_q_ratio = DiscreteProposalRatio::from_counts(3, 0)?.log_q_ratio();
    ///
    /// assert!(log_q_ratio.is_infinite());
    /// assert!(log_q_ratio.is_sign_negative());
    /// # Ok::<(), markov_chain_monte_carlo::DiscreteProposalRatioError>(())
    /// ```
    #[must_use]
    pub fn log_q_ratio(self) -> f64 {
        if self.reverse_weight == 0.0 || self.reverse_site_count == 0 {
            return f64::NEG_INFINITY;
        }

        self.reverse_weight.ln() - self.forward_weight.ln() + count_ln(self.forward_site_count)
            - count_ln(self.reverse_site_count)
    }
}

/// Errors from constructing a [`DiscreteProposalRatio`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DiscreteProposalRatioError {
    /// The forward move-family weight is not positive and finite.
    #[non_exhaustive]
    InvalidForwardWeight {
        /// Invalid forward move-family weight.
        weight: f64,
    },
    /// The reverse move-family weight is negative, `NaN`, or infinite.
    #[non_exhaustive]
    InvalidReverseWeight {
        /// Invalid reverse move-family weight.
        weight: f64,
    },
    /// A successful forward proposal reported zero valid forward sites.
    ZeroForwardSiteCount,
}

impl core::fmt::Display for DiscreteProposalRatioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidForwardWeight { weight } => write!(
                f,
                "invalid forward move-family weight {weight}: expected a positive finite value"
            ),
            Self::InvalidReverseWeight { weight } => write!(
                f,
                "invalid reverse move-family weight {weight}: expected a nonnegative finite value"
            ),
            Self::ZeroForwardSiteCount => {
                f.write_str("invalid forward site count 0 for a successful proposal")
            }
        }
    }
}

impl std::error::Error for DiscreteProposalRatioError {}

/// Convert a valid-site count into a logarithm for proposal-ratio arithmetic.
#[expect(
    clippy::cast_precision_loss,
    reason = "valid-site counts intentionally cross into log-space f64 proposal arithmetic"
)]
fn count_ln(count: usize) -> f64 {
    (count as f64).ln()
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
/// # Associated Types
///
/// * [`Undo`](ProposalMut::Undo) — a small token that captures
///   exactly what is needed to reverse a move.
pub trait ProposalMut<S> {
    /// Token that records how to reverse a proposed move.
    type Undo;

    /// Mutate `state` in place, returning `Some(undo_token)` on success
    /// or `None` if no valid move could be found.
    ///
    /// Returning `None` must leave `state` exactly as it was on entry.  If a
    /// proposal mutates state before discovering that the move is invalid, it
    /// must undo those changes before returning `None`.  Once `Some(token)` is
    /// returned, [`undo`](ProposalMut::undo) must be able to restore the exact
    /// prior state.
    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut S, rng: &mut R) -> Option<Self::Undo>;

    /// Reverse a previously applied move using its undo token.
    fn undo(&self, state: &mut S, token: Self::Undo);

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

impl<S, P: ProposalMut<S> + ?Sized> ProposalMut<S> for &P {
    type Undo = P::Undo;

    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut S, rng: &mut R) -> Option<Self::Undo> {
        (**self).propose_mut(state, rng)
    }

    fn undo(&self, state: &mut S, token: Self::Undo) {
        (**self).undo(state, token);
    }

    fn log_q_ratio(&self, state: &S, token: &Self::Undo) -> f64 {
        (**self).log_q_ratio(state, token)
    }
}

impl<S, P: ProposalMut<S> + ?Sized> ProposalMut<S> for &mut P {
    type Undo = P::Undo;

    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut S, rng: &mut R) -> Option<Self::Undo> {
        (**self).propose_mut(state, rng)
    }

    fn undo(&self, state: &mut S, token: Self::Undo) {
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
    fn proposed_log_prob<T: Target<S>>(
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
    fn info(&self, plan: &Self::Plan) -> Self::Info;

    /// Apply an accepted concrete move to `state`.
    ///
    /// This method is called only after the Metropolis-Hastings decision has
    /// accepted `plan`.  On error, implementations must restore `state` before
    /// returning so the chain's state and cached log-probability remain
    /// synchronized.  [`crate::Chain`] cannot repair a partially applied
    /// failed commit without an implementation-provided rollback token, so
    /// failure atomicity is part of this trait's correctness contract.
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

    fn proposed_log_prob<T: Target<S>>(
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

    struct SymmetricProposal;
    impl Proposal<Scalar> for SymmetricProposal {
        fn propose<R: Rng + ?Sized>(&self, current: &Scalar, _rng: &mut R) -> Scalar {
            Scalar(current.0 + 1.0)
        }
        // log_q_ratio intentionally not overridden — uses default
    }

    struct SymmetricMutProposal;
    impl ProposalMut<Scalar> for SymmetricMutProposal {
        type Undo = f64;
        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut Scalar, _rng: &mut R) -> Option<f64> {
            let old = state.0;
            state.0 += 1.0;
            Some(old)
        }
        fn undo(&self, state: &mut Scalar, old: f64) {
            state.0 = old;
        }
        // log_q_ratio intentionally not overridden — uses default
    }

    // --- Default log_q_ratio tests ---

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

        assert_relative_eq!(proposed.0, 3.0);
        assert_relative_eq!(shared.log_q_ratio(&Scalar(2.0), &proposed), 0.0);
    }

    #[test]
    fn shared_mut_proposal_forwards() {
        let proposal = SymmetricMutProposal;
        let shared = &proposal;
        let mut state = Scalar(2.0);
        let token = shared.propose_mut(&mut state, &mut rng()).unwrap();

        assert_relative_eq!(state.0, 3.0);
        assert_relative_eq!(shared.log_q_ratio(&state, &token), 0.0);

        shared.undo(&mut state, token);
        assert_relative_eq!(state.0, 2.0);
    }

    #[test]
    fn mut_ref_mut_proposal_forwards() {
        let mut proposal = SymmetricMutProposal;
        let shared = &mut proposal;
        let mut state = Scalar(2.0);
        let token = shared.propose_mut(&mut state, &mut rng()).unwrap();

        assert_relative_eq!(state.0, 3.0);
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

        assert_relative_eq!(proposed.0, 3.0);
        assert_relative_eq!(shared.log_q_ratio(&Scalar(2.0), &proposed), 0.0);
    }

    struct SymmetricDelayedProposal;
    impl DelayedProposal<Scalar> for SymmetricDelayedProposal {
        type Plan = f64;
        type Info = f64;
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            _state: &Scalar,
            _rng: &mut R,
        ) -> Result<Option<f64>, Self::Error> {
            Ok(Some(1.0))
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
        let ratio = DiscreteProposalRatio::new(0.25, 6, 0.75, 2).unwrap();

        assert_eq!(ratio.forward_weight().to_bits(), 0.25_f64.to_bits());
        assert_eq!(ratio.reverse_weight().to_bits(), 0.75_f64.to_bits());
        assert_eq!(ratio.forward_site_count(), 6);
        assert_eq!(ratio.reverse_site_count(), 2);

        assert_relative_eq!(
            ratio.log_q_ratio(),
            0.75_f64.ln() - 0.25_f64.ln() + 6.0_f64.ln() - 2.0_f64.ln(),
            epsilon = 1e-12
        );
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
        let ratio = DiscreteProposalRatio::new(1.0, 3, 0.0, 1)
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
        for weight in [0.0, -1.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
            let err = DiscreteProposalRatio::new(weight, 1, 1.0, 1).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid forward move-family weight {weight}: expected a positive finite value"
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
        for weight in [-1.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
            let err = DiscreteProposalRatio::new(1.0, 1, weight, 1).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid reverse move-family weight {weight}: expected a nonnegative finite value"
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
    fn delayed_mut_ref_forwards() {
        struct ZeroTarget;
        impl Target<Scalar> for ZeroTarget {
            fn log_prob(&self, state: &Scalar) -> f64 {
                -state.0.abs()
            }
        }

        let mut proposal = SymmetricDelayedProposal;
        let shared = &mut proposal;
        let state = Scalar(0.0);
        let plan = shared.propose_plan(&state, &mut rng()).unwrap().unwrap();

        assert_relative_eq!(plan, 1.0);
        assert_relative_eq!(
            shared
                .proposed_log_prob(&state, &plan, &ZeroTarget)
                .unwrap(),
            -1.0
        );
        assert_relative_eq!(shared.log_q_ratio(&state, &plan).unwrap(), 0.0);
        assert_relative_eq!(shared.info(&plan), 1.0);

        let mut committed = Scalar(0.0);
        shared.commit(&mut committed, plan, &mut rng()).unwrap();
        assert_relative_eq!(committed.0, 1.0);
    }
}
