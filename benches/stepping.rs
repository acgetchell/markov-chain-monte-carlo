//! Criterion benchmarks for core Metropolis-Hastings stepping paths.

use core::convert::Infallible;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::DelayedProposal;
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{Chain, Sampler, Target};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

const SEED: u64 = 42;
const BULK_STEPS: usize = 100;
const SPIN_COUNT: usize = 256;

#[derive(Clone, Copy)]
struct Scalar(f64);

struct Normal;

impl Target<Scalar> for Normal {
    fn log_prob(&self, state: &Scalar) -> f64 {
        -0.5 * state.0 * state.0
    }
}

struct FlatTarget;

impl<T> Target<T> for FlatTarget {
    fn log_prob(&self, _state: &T) -> f64 {
        0.0
    }
}

struct RandomWalk {
    width: f64,
}

impl Proposal<Scalar> for RandomWalk {
    fn propose<R: Rng + ?Sized>(&self, current: &Scalar, rng: &mut R) -> Scalar {
        let delta = rng.random_range(-self.width..self.width);
        Scalar(current.0 + delta)
    }
}

struct SpinChain {
    spins: Vec<i8>,
}

struct Alignment {
    beta: f64,
}

impl Target<SpinChain> for Alignment {
    fn log_prob(&self, state: &SpinChain) -> f64 {
        self.beta
            * state
                .spins
                .windows(2)
                .map(|pair| f64::from(pair[0]) * f64::from(pair[1]))
                .sum::<f64>()
    }
}

struct SpinFlip;

impl ProposalMut<SpinChain> for SpinFlip {
    type Undo = usize;

    fn propose_mut<R: Rng + ?Sized>(&self, state: &mut SpinChain, rng: &mut R) -> Option<usize> {
        if state.spins.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }

    fn undo(&self, state: &mut SpinChain, idx: usize) {
        state.spins[idx] *= -1;
    }
}

struct DelayedWalk {
    delta: f64,
}

impl DelayedProposal<Scalar> for DelayedWalk {
    type Plan = f64;
    type Info = ();
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        _state: &Scalar,
        _rng: &mut R,
    ) -> Result<Option<f64>, Self::Error> {
        Ok(Some(self.delta))
    }

    fn proposed_log_prob<T: Target<Scalar>>(
        &self,
        state: &Scalar,
        plan: &f64,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(&Scalar(state.0 + *plan)))
    }

    fn info(&self, _plan: &f64) {}

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut Scalar,
        plan: f64,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        state.0 += plan;
        Ok(())
    }
}

struct NoDelayedPlan;

impl DelayedProposal<Scalar> for NoDelayedPlan {
    type Plan = f64;
    type Info = ();
    type Error = Infallible;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        _state: &Scalar,
        _rng: &mut R,
    ) -> Result<Option<f64>, Self::Error> {
        Ok(None)
    }

    fn proposed_log_prob<T: Target<Scalar>>(
        &self,
        _state: &Scalar,
        _plan: &f64,
        _target: &T,
    ) -> Result<f64, Self::Error> {
        unreachable!("no-plan proposals should not be scored")
    }

    fn info(&self, _plan: &f64) {}

    fn commit<R: Rng + ?Sized>(
        &mut self,
        _state: &mut Scalar,
        _plan: f64,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        unreachable!("no-plan proposals should not be committed")
    }
}

/// Build a scalar chain with a valid cached log-probability for scalar benches.
fn scalar_chain(target: &impl Target<Scalar>) -> Chain<Scalar> {
    Chain::new(Scalar(0.0), target).expect("valid scalar benchmark state")
}

/// Build a non-`Clone` spin-chain state used to exercise in-place rollback.
fn spin_chain(target: &Alignment) -> Chain<SpinChain> {
    let state = SpinChain {
        spins: vec![1; SPIN_COUNT],
    };
    Chain::new(state, target).expect("valid spin benchmark state")
}

/// Register single-step chain benchmarks for by-value, in-place, and rollback paths.
fn bench_chain_steps(c: &mut Criterion) {
    let target = Normal;
    let flat = FlatTarget;
    let proposal = RandomWalk { width: 1.0 };
    let spin_target = Alignment { beta: 8.0 };
    let spin_proposal = SpinFlip;

    c.bench_function("chain/step_by_value", |b| {
        let mut chain = scalar_chain(&target);
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            chain.step(&target, &proposal, &mut rng).unwrap();
            black_box(chain.state().0);
        });
    });

    c.bench_function("chain/step_mut_accept", |b| {
        let mut chain = spin_chain(&Alignment { beta: 0.0 });
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            black_box(chain.step_mut(&flat, &spin_proposal, &mut rng).unwrap());
            black_box(chain.state().spins[0]);
        });
    });

    c.bench_function("chain/step_mut_reject_rollback", |b| {
        let mut chain = spin_chain(&spin_target);
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            black_box(
                chain
                    .step_mut(&spin_target, &spin_proposal, &mut rng)
                    .unwrap(),
            );
            black_box(chain.state().spins[0]);
        });
    });
}

/// Register delayed-step benchmarks for accepted, rejected, and no-plan paths.
fn bench_delayed_steps(c: &mut Criterion) {
    let normal = Normal;
    let flat = FlatTarget;

    c.bench_function("chain/step_delayed_accept_commit", |b| {
        let mut chain = scalar_chain(&flat);
        let mut proposal = DelayedWalk { delta: 1.0 };
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let step = chain.step_delayed(&flat, &mut proposal, &mut rng).unwrap();
            black_box(step.accepted);
            black_box(chain.state().0);
        });
    });

    c.bench_function("chain/step_delayed_reject_plan", |b| {
        let mut chain = scalar_chain(&normal);
        let mut proposal = DelayedWalk { delta: 100.0 };
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let step = chain
                .step_delayed(&normal, &mut proposal, &mut rng)
                .unwrap();
            black_box(step.accepted);
            black_box(chain.state().0);
        });
    });

    c.bench_function("chain/step_delayed_no_plan", |b| {
        let mut chain = scalar_chain(&normal);
        let mut proposal = NoDelayedPlan;
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let step = chain
                .step_delayed(&normal, &mut proposal, &mut rng)
                .unwrap();
            black_box(step.proposed);
            black_box(chain.rejected());
        });
    });
}

/// Register bulk sampler benchmarks to watch wrapper overhead across workflows.
fn bench_sampler_runs(c: &mut Criterion) {
    let target = Normal;
    let proposal = RandomWalk { width: 1.0 };
    let flat = FlatTarget;
    let spin_proposal = SpinFlip;

    c.bench_function("sampler/run_by_value_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(chain, mut rng)| {
                let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng);
                sampler.run(black_box(BULK_STEPS)).unwrap();
                black_box(sampler.chain_ref().state().0);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sampler/run_mut_100", |b| {
        b.iter_batched(
            || {
                (
                    spin_chain(&Alignment { beta: 0.0 }),
                    StdRng::seed_from_u64(SEED),
                )
            },
            |(chain, mut rng)| {
                let mut sampler = Sampler::new(chain, &flat, &spin_proposal, &mut rng);
                sampler.run_mut(black_box(BULK_STEPS)).unwrap();
                black_box(sampler.chain_ref().state().spins[0]);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("sampler/run_delayed_100", |b| {
        b.iter_batched(
            || {
                (
                    scalar_chain(&flat),
                    DelayedWalk { delta: 1.0 },
                    StdRng::seed_from_u64(SEED),
                )
            },
            |(chain, mut delayed, mut rng)| {
                let mut sampler = Sampler::new(chain, &flat, &mut delayed, &mut rng);
                sampler.run_delayed(black_box(BULK_STEPS)).unwrap();
                black_box(sampler.chain_ref().state().0);
            },
            BatchSize::SmallInput,
        );
    });
}

/// Register observing benchmarks to compare collection and online accumulation.
fn bench_observing(c: &mut Criterion) {
    let target = Normal;
    let proposal = RandomWalk { width: 1.0 };

    c.bench_function("observing/run_observing_buffer_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(chain, mut rng)| {
                let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng);
                let mut square = |state: &Scalar| state.0 * state.0;
                let observations = sampler
                    .run_observing(black_box(BULK_STEPS), &mut square)
                    .unwrap();
                black_box(observations.as_slice());
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("observing/manual_online_sum_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(mut chain, mut rng)| {
                let mut sum = 0.0;
                for _ in 0..black_box(BULK_STEPS) {
                    chain.step(&target, &proposal, &mut rng).unwrap();
                    sum += chain.state().0 * chain.state().0;
                }
                black_box(sum);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_chain_steps,
    bench_delayed_steps,
    bench_sampler_runs,
    bench_observing
);
criterion_main!(benches);
