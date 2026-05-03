//! Core traits for target distributions and proposal distributions.

use rand::Rng;

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
/// 1. [`propose_plan`](Self::propose_plan) chooses a move descriptor without
///    mutating the state.
/// 2. [`proposed_log_prob`](Self::proposed_log_prob) evaluates the proposed
///    state's log-probability from that descriptor.
/// 3. [`crate::Chain::step_delayed`] performs the accept/reject draw.
/// 4. [`commit`](Self::commit) mutates the state only after acceptance.
///
/// This is useful for combinatorial state spaces where the log-probability
/// delta is cheap to compute from a move descriptor, but applying the move may
/// require searching for a valid local site.  If `commit` returns an error, it
/// must be failure-atomic: either the accepted move is applied completely, or
/// `state` is restored before returning `Err`.
pub trait DelayedProposal<S> {
    /// Move descriptor produced before the Metropolis-Hastings decision.
    type Plan;
    /// User-facing metadata returned in delayed-step telemetry.
    type Info;
    /// Proposal-specific error type.
    type Error;

    /// Propose a move descriptor without mutating `state`.
    ///
    /// Return `Ok(None)` when no valid move can be proposed from the current
    /// state.  That is counted as a rejection by [`crate::Chain::step_delayed`].
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when planning fails for proposal-specific reasons.
    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &S,
        rng: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error>;

    /// Compute the proposed state's log-probability without mutating `state`.
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

    /// Apply an accepted move to `state`.
    ///
    /// This method is called only after the Metropolis-Hastings decision has
    /// accepted `plan`.  On error, implementations must restore `state` before
    /// returning so the chain's state and cached log-probability remain
    /// synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the accepted move cannot be committed.
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

    use super::*;
    use rand::rng;

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
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default Proposal::log_q_ratio should be 0.0"
        );
    }

    #[test]
    fn proposal_ref_forwards() {
        let proposal = SymmetricProposal;
        let shared = &proposal;
        let proposed = shared.propose(&Scalar(2.0), &mut rng());

        assert!((proposed.0 - 3.0).abs() < f64::EPSILON);
        assert!(shared.log_q_ratio(&Scalar(2.0), &proposed).abs() < f64::EPSILON);
    }

    #[test]
    fn shared_mut_proposal_forwards() {
        let proposal = SymmetricMutProposal;
        let shared = &proposal;
        let mut state = Scalar(2.0);
        let token = shared.propose_mut(&mut state, &mut rng()).unwrap();

        assert!((state.0 - 3.0).abs() < f64::EPSILON);
        assert!(shared.log_q_ratio(&state, &token).abs() < f64::EPSILON);

        shared.undo(&mut state, token);
        assert!((state.0 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mut_ref_mut_proposal_forwards() {
        let mut proposal = SymmetricMutProposal;
        let shared = &mut proposal;
        let mut state = Scalar(2.0);
        let token = shared.propose_mut(&mut state, &mut rng()).unwrap();

        assert!((state.0 - 3.0).abs() < f64::EPSILON);
        assert!(shared.log_q_ratio(&state, &token).abs() < f64::EPSILON);

        shared.undo(&mut state, token);
        assert!((state.0 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn proposal_mut_default_log_q_zero() {
        let p = SymmetricMutProposal;
        let ratio = p.log_q_ratio(&Scalar(1.0), &0.0_f64);
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default ProposalMut::log_q_ratio should be 0.0"
        );
    }

    #[test]
    fn mut_ref_proposal_forwards() {
        let mut proposal = SymmetricProposal;
        let shared = &mut proposal;
        let proposed = shared.propose(&Scalar(2.0), &mut rng());

        assert!((proposed.0 - 3.0).abs() < f64::EPSILON);
        assert!(shared.log_q_ratio(&Scalar(2.0), &proposed).abs() < f64::EPSILON);
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
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default DelayedProposal::log_q_ratio should be 0.0"
        );
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

        assert!((plan - 1.0).abs() < f64::EPSILON);
        assert!(
            (shared
                .proposed_log_prob(&state, &plan, &ZeroTarget)
                .unwrap()
                + 1.0)
                .abs()
                < f64::EPSILON
        );
        assert!(shared.log_q_ratio(&state, &plan).unwrap().abs() < f64::EPSILON);
        assert!((shared.info(&plan) - 1.0).abs() < f64::EPSILON);

        let mut committed = Scalar(0.0);
        shared.commit(&mut committed, plan, &mut rng()).unwrap();
        assert!((committed.0 - 1.0).abs() < f64::EPSILON);
    }
}
