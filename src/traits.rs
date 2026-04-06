//! Core traits for target distributions and proposal distributions.

use rand::Rng;

/// Target distribution
pub trait Target<S> {
    /// Compute log-probability (or negative energy/action).
    fn log_prob(&self, state: &S) -> f64;
}

/// Proposal distribution for generating new states.
///
/// This trait uses a clone-based model: [`propose`](Proposal::propose) returns
/// a new state by value.  For state spaces where cloning is expensive (e.g.,
/// triangulations, large graphs), see [`ProposalMut`] which mutates in place
/// and supports cheap rollback.
pub trait Proposal<S: Clone> {
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
