//! Regression evidence for proposal-author validation patterns.

use core::convert::Infallible;

use approx::assert_relative_eq;
use markov_chain_monte_carlo::prelude::testing::{
    DelayedProposal, DetailedBalanceConfig, DetailedBalanceError, DiscreteProposalRatio,
    DiscreteProposalRatioError, Target, verify_detailed_balance_delayed,
};
use rand::{Rng, RngExt, SeedableRng, TryRng, rngs::StdRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Left,
    Middle,
    Right,
}

impl State {
    const fn valid_sites(self) -> usize {
        match self {
            Self::Middle => 2,
            Self::Left | Self::Right => 1,
        }
    }

    const fn search_success(self) -> f64 {
        match self {
            Self::Middle => 1.0,
            Self::Left | Self::Right => 0.75,
        }
    }
}

struct Flat;

impl Target<State> for Flat {
    fn log_prob(&self, _: &State) -> f64 {
        0.0
    }
}

struct BoundedSearch {
    include_success_factor: bool,
}

impl DelayedProposal<State> for BoundedSearch {
    type Plan = State;
    type Info = State;
    type Error = DiscreteProposalRatioError;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &State,
        rng: &mut R,
    ) -> Result<Option<State>, Self::Error> {
        for _ in 0..2 {
            let site = rng.random_bool(0.5);
            let destination = match (*state, site) {
                (State::Left, false) | (State::Right, true) => Some(State::Middle),
                (State::Middle, false) => Some(State::Left),
                (State::Middle, true) => Some(State::Right),
                (State::Left, true) | (State::Right, false) => None,
            };
            if destination.is_some() {
                return Ok(destination);
            }
        }
        Ok(None)
    }

    fn proposed_log_prob<T: Target<State> + ?Sized>(
        &self,
        _: &State,
        plan: &State,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(plan))
    }

    fn log_q_ratio(&self, state: &State, plan: &State) -> Result<f64, Self::Error> {
        let count_ratio =
            DiscreteProposalRatio::from_counts(state.valid_sites(), plan.valid_sites())?
                .log_q_ratio();
        let success_ratio = if self.include_success_factor {
            (plan.search_success() / state.search_success()).ln()
        } else {
            0.0
        };
        Ok(count_ratio + success_ratio)
    }

    fn info(&self, plan: &State) -> State {
        *plan
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut State,
        plan: State,
        _: &mut R,
    ) -> Result<(), Self::Error> {
        *state = plan;
        Ok(())
    }
}

/// Replay one of the four equally likely two-draw paths, including exhaustion.
struct CandidatePairRng {
    words: std::array::IntoIter<u64, 2>,
}

impl CandidatePairRng {
    fn next_word(&mut self) -> u64 {
        self.words.next().expect("search makes at most two draws")
    }
}

impl TryRng for CandidatePairRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(if self.next_word() == 0 { 0 } else { u32::MAX })
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next_word())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_word().to_le_bytes()[..chunk.len()]);
        }
        Ok(())
    }
}

#[test]
fn bounded_search_probabilities_include_exhaustion() {
    let mut proposal = BoundedSearch {
        include_success_factor: true,
    };
    for (state, expected) in [
        (State::Left, [0, 3, 0, 1]),
        (State::Middle, [2, 0, 2, 0]),
        (State::Right, [0, 3, 0, 1]),
    ] {
        let mut counts = [0; 4];
        for first in [0, u64::MAX] {
            for second in [0, u64::MAX] {
                let mut rng = CandidatePairRng {
                    words: [first, second].into_iter(),
                };
                let outcome = proposal.propose_plan(&state, &mut rng).unwrap();
                let bin = match outcome {
                    Some(State::Left) => 0,
                    Some(State::Middle) => 1,
                    Some(State::Right) => 2,
                    None => 3,
                };
                counts[bin] += 1;
            }
        }
        assert_eq!(counts, expected, "state {state:?}");
    }
}

#[test]
fn bounded_search_requires_success_probability_correction() {
    // Exhaustive paths above give q(Middle|Left)=3/4 and q(Left|Middle)=1/2.
    // On a flat target, the corrected accepted flows are both 1/2; omitting
    // the success factor instead gives 3/8 forward and 1/2 reverse.
    let config = DetailedBalanceConfig::new(16_384, 0.08, 64).unwrap();
    for include_success_factor in [true, false] {
        let mut proposal = BoundedSearch {
            include_success_factor,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let result = verify_detailed_balance_delayed(
            &State::Left,
            &State::Middle,
            &Flat,
            &mut proposal,
            &mut rng,
            config,
            (|plan| *plan == State::Middle, |plan| *plan == State::Left),
        );
        let report = if include_success_factor {
            result.expect("complete proposal ratio must balance")
        } else {
            let Err(DetailedBalanceError::Violation { report, .. }) = result else {
                panic!("count-only ratio must fail detailed balance: {result:?}");
            };
            report
        };
        assert_relative_eq!(report.forward_log_proposal.exp(), 0.75, epsilon = 0.02);
        assert_relative_eq!(report.reverse_log_proposal.exp(), 0.5, epsilon = 0.02);
        assert_relative_eq!(report.reverse_log_transition.exp(), 0.5, epsilon = 0.02);
        let expected_forward = if include_success_factor { 0.5 } else { 0.375 };
        assert_relative_eq!(
            report.forward_log_transition.exp(),
            expected_forward,
            epsilon = 0.02
        );
        let expected_residual = if include_success_factor {
            0.0
        } else {
            0.75_f64.ln()
        };
        assert_relative_eq!(
            report.log_balance_residual,
            expected_residual,
            epsilon = 0.08
        );
    }
}
