//! Criterion benchmarks for core Metropolis-Hastings stepping paths.

use core::{convert::Infallible, fmt};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use markov_chain_monte_carlo::StepOutcome;
use markov_chain_monte_carlo::prelude::by_value::Proposal;
use markov_chain_monte_carlo::prelude::delayed::DelayedProposal;
use markov_chain_monte_carlo::prelude::in_place::ProposalMut;
use markov_chain_monte_carlo::prelude::{BinningAnalysis, Chain, OnlineStats, Sampler, Target};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

const SEED: u64 = 42;
const BULK_STEPS: usize = 100;
const SPIN_COUNT: usize = 256;
// A single boundary-spin flip has log acceptance ratio -2 * beta. This value
// is below every representable `ln(Open01)` draw, making rollback deterministic.
const REJECTION_BETA: f64 = 512.0;

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
    type Info = usize;

    fn propose_mut<R: Rng + ?Sized>(
        &mut self,
        state: &mut SpinChain,
        rng: &mut R,
    ) -> Option<usize> {
        if state.spins.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..state.spins.len());
        state.spins[idx] *= -1;
        Some(idx)
    }

    fn info(&self, _state: &SpinChain, idx: &usize) -> usize {
        *idx
    }

    fn undo(&mut self, state: &mut SpinChain, idx: usize) {
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

/// Convert a fallible benchmark operation into its value or panic with context.
trait OrAbort {
    /// Successful value produced by the operation.
    type Output;

    /// Return the successful value or panic with the named benchmark context.
    ///
    /// # Panics
    ///
    /// Panics when the benchmark operation returns an error.
    fn or_abort(self, context: &str) -> Self::Output;
}

impl<T, E: fmt::Display> OrAbort for Result<T, E> {
    type Output = T;

    fn or_abort(self, context: &str) -> Self::Output {
        match self {
            Ok(value) => value,
            Err(err) => panic!("{context}: {err}"),
        }
    }
}

/// Build a scalar chain with a valid cached log-probability for scalar benches.
fn scalar_chain(target: &impl Target<Scalar>) -> Chain<Scalar> {
    Chain::new(Scalar(0.0), target).or_abort("valid scalar benchmark state")
}

/// Build a non-`Clone` spin-chain state used to exercise in-place rollback.
fn spin_chain(target: &Alignment) -> Chain<SpinChain> {
    let state = SpinChain {
        spins: vec![1; SPIN_COUNT],
    };
    Chain::new(state, target).or_abort("valid spin benchmark state")
}

/// Register single-step chain benchmarks for by-value, in-place, and rollback paths.
fn bench_chain_steps(c: &mut Criterion) {
    let target = Normal;
    let flat = FlatTarget;
    let proposal = RandomWalk { width: 1.0 };
    let spin_target = Alignment {
        beta: REJECTION_BETA,
    };

    {
        let mut chain = spin_chain(&Alignment { beta: 0.0 });
        let mut spin_proposal = SpinFlip;
        let mut rng = StdRng::seed_from_u64(SEED);
        let outcome = chain
            .step_mut(&flat, &mut spin_proposal, &mut rng)
            .or_abort("verify in-place acceptance benchmark")
            .outcome();
        assert_eq!(outcome, StepOutcome::Accepted);
    }

    {
        let mut chain = spin_chain(&spin_target);
        let mut spin_proposal = SpinFlip;
        let mut rng = StdRng::seed_from_u64(SEED);
        let outcome = chain
            .step_mut(&spin_target, &mut spin_proposal, &mut rng)
            .or_abort("verify in-place rollback benchmark")
            .outcome();
        assert_eq!(outcome, StepOutcome::RejectedProposal);
        assert!(chain.state().spins.iter().all(|&spin| spin == 1));
    }

    // These names retain their v0.4.0 steady-state lifecycle contracts. Keep
    // fixture construction outside `b.iter`, or rename a changed workload.
    c.bench_function("chain/step_by_value", |b| {
        let mut chain = scalar_chain(&target);
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let _ = chain
                .step(&target, &proposal, &mut rng)
                .or_abort("chain step by value");
            black_box(chain.state().0);
        });
    });

    c.bench_function("chain/step_mut_accept", |b| {
        let mut chain = spin_chain(&Alignment { beta: 0.0 });
        let mut spin_proposal = SpinFlip;
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let _ = black_box(
                chain
                    .step_mut(&flat, &mut spin_proposal, &mut rng)
                    .or_abort("in-place chain step on flat target"),
            );
            black_box(chain.state().spins[0]);
        });
    });

    c.bench_function("chain/step_mut_reject_rollback", |b| {
        let mut chain = spin_chain(&spin_target);
        let mut spin_proposal = SpinFlip;
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let _ = black_box(
                chain
                    .step_mut(&spin_target, &mut spin_proposal, &mut rng)
                    .or_abort("in-place chain step with rollback"),
            );
            black_box(chain.state().spins[0]);
        });
    });
}

/// Register delayed-step benchmarks for accepted, rejected, and no-plan paths.
fn bench_delayed_steps(c: &mut Criterion) {
    let normal = Normal;
    let flat = FlatTarget;

    {
        let mut chain = scalar_chain(&flat);
        let mut proposal = DelayedWalk { delta: 1.0 };
        let mut rng = StdRng::seed_from_u64(SEED);
        let outcome = chain
            .step_delayed(&flat, &mut proposal, &mut rng)
            .or_abort("verify delayed acceptance benchmark")
            .outcome();
        assert_eq!(outcome, StepOutcome::Accepted);
    }

    {
        let mut chain = scalar_chain(&normal);
        let mut proposal = DelayedWalk { delta: 100.0 };
        let mut rng = StdRng::seed_from_u64(SEED);
        let outcome = chain
            .step_delayed(&normal, &mut proposal, &mut rng)
            .or_abort("verify delayed rejection benchmark")
            .outcome();
        assert_eq!(outcome, StepOutcome::RejectedProposal);
    }

    {
        let mut chain = scalar_chain(&normal);
        let mut proposal = NoDelayedPlan;
        let mut rng = StdRng::seed_from_u64(SEED);
        let outcome = chain
            .step_delayed(&normal, &mut proposal, &mut rng)
            .or_abort("verify delayed no-plan benchmark")
            .outcome();
        assert_eq!(outcome, StepOutcome::NoProposal);
    }

    // These names retain their v0.4.0 steady-state lifecycle contracts.
    c.bench_function("chain/step_delayed_accept_commit", |b| {
        let mut chain = scalar_chain(&flat);
        let mut proposal = DelayedWalk { delta: 1.0 };
        let mut rng = StdRng::seed_from_u64(SEED);

        b.iter(|| {
            let step = chain
                .step_delayed(&flat, &mut proposal, &mut rng)
                .or_abort("delayed accepted chain step");
            let _ = black_box(step.outcome());
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
                .or_abort("delayed rejected chain step");
            let _ = black_box(step.outcome());
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
                .or_abort("delayed no-plan chain step");
            let _ = black_box(step.outcome());
            black_box(chain.rejected());
        });
    });
}

/// Register bulk sampler benchmarks to watch wrapper overhead across workflows.
fn bench_sampler_runs(c: &mut Criterion) {
    let target = Normal;
    let proposal = RandomWalk { width: 1.0 };
    let flat = FlatTarget;

    // Sampler construction is setup; each timed iteration advances the same
    // sampler by 100 steps, matching the v0.4.0 throughput contract.
    c.bench_function("sampler/run_by_value_100", |b| {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut sampler = Sampler::new(scalar_chain(&target), &target, &proposal, &mut rng)
            .or_abort("sampler by-value setup");

        b.iter(|| {
            sampler
                .run(black_box(BULK_STEPS))
                .or_abort("sampler by-value bulk run");
            black_box(sampler.chain_ref().state().0);
        });
    });

    c.bench_function("sampler/run_mut_100", |b| {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut sampler = Sampler::new(
            spin_chain(&Alignment { beta: 0.0 }),
            &flat,
            SpinFlip,
            &mut rng,
        )
        .or_abort("sampler in-place setup");

        b.iter(|| {
            sampler
                .run_mut(black_box(BULK_STEPS))
                .or_abort("sampler in-place bulk run");
            black_box(sampler.chain_ref().state().spins[0]);
        });
    });

    c.bench_function("sampler/run_delayed_100", |b| {
        let mut delayed = DelayedWalk { delta: 1.0 };
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut sampler = Sampler::new(scalar_chain(&flat), &flat, &mut delayed, &mut rng)
            .or_abort("sampler delayed setup");

        b.iter(|| {
            sampler
                .run_delayed(black_box(BULK_STEPS))
                .or_abort("sampler delayed bulk run");
            black_box(sampler.chain_ref().state().0);
        });
    });
}

/// Register observing benchmarks to compare collection and online accumulation.
fn bench_observing(c: &mut Criterion) {
    let target = Normal;
    let proposal = RandomWalk { width: 1.0 };

    // The buffered workflow retains its steady-state sampler contract. The
    // returned Vec allocation, use, and destruction are part of the workload.
    c.bench_function("observing/run_observing_buffer_100", |b| {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut sampler = Sampler::new(scalar_chain(&target), &target, &proposal, &mut rng)
            .or_abort("observing buffer sampler setup");
        let mut square = |state: &Scalar| state.0 * state.0;

        b.iter(|| {
            let observations = sampler
                .run_observing(black_box(BULK_STEPS), &mut square)
                .or_abort("observing buffer run");
            black_box(observations.as_slice());
        });
    });

    // These three legacy comparison workloads use fresh fixed-seed batches.
    // Criterion excludes tuple creation but includes each 100-step workflow.
    c.bench_function("observing/manual_online_sum_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(mut chain, mut rng)| {
                let mut sum = 0.0;
                for _ in 0..black_box(BULK_STEPS) {
                    let _ = chain
                        .step(&target, &proposal, &mut rng)
                        .or_abort("manual observing chain step");
                    let sample = chain.state().0;
                    sum = sample.mul_add(sample, sum);
                }
                black_box(sum);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("observing/run_observing_into_online_stats_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(chain, mut rng)| {
                let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)
                    .or_abort("online stats sampler setup");
                let mut square = |state: &Scalar| state.0 * state.0;
                let mut stats = OnlineStats::new();
                sampler
                    .run_observing_into(black_box(BULK_STEPS), &mut square, &mut stats)
                    .or_abort("online stats observing run");
                black_box(stats.count());
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("observing/run_observing_into_binning_100", |b| {
        b.iter_batched(
            || (scalar_chain(&target), StdRng::seed_from_u64(SEED)),
            |(chain, mut rng)| {
                let mut sampler = Sampler::new(chain, &target, &proposal, &mut rng)
                    .or_abort("binning sampler setup");
                let mut square = |state: &Scalar| state.0 * state.0;
                let mut bins = BinningAnalysis::new();
                sampler
                    .run_observing_into(black_box(BULK_STEPS), &mut square, &mut bins)
                    .or_abort("binning observing run");
                black_box(bins.standard_error());
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
