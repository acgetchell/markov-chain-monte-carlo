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

#[cfg(test)]
mod tests {
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
    fn proposal_default_log_q_ratio_is_zero() {
        let p = SymmetricProposal;
        let ratio = p.log_q_ratio(&Scalar(0.0), &Scalar(1.0));
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default Proposal::log_q_ratio should be 0.0"
        );
    }

    #[test]
    fn proposal_mut_default_log_q_ratio_is_zero() {
        let p = SymmetricMutProposal;
        let ratio = p.log_q_ratio(&Scalar(1.0), &0.0_f64);
        assert!(
            ratio.abs() < f64::EPSILON,
            "Default ProposalMut::log_q_ratio should be 0.0"
        );
    }
}
