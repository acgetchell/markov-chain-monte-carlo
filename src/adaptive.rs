//! Bounded scalar proposal tuning during an explicit warmup phase.

use core::{fmt, ops::RangeInclusive};
use std::error::Error;

use rand::Rng;

use crate::{
    DelayedProposal, DelayedStepError, McmcError, Proposal, ProposalMut, Sampler, StepOutcome,
    Target, numerics::count_as_f64,
};

/// A proposal whose step width can be changed between completed transitions.
///
/// Implement alongside any of [`Proposal`], [`ProposalMut`], or [`DelayedProposal`]
/// to use the sampler's adaptive warmup methods. Increasing the scale should
/// generally decrease acceptance. Each fixed scale must define a valid proposal
/// with the corresponding Hastings correction. Do not change scale while a
/// transition is being generated, scored, committed, or undone.
pub trait TunableProposal {
    /// Set the proposal's scale for subsequent transitions.
    ///
    /// Adaptive warmup supplies a positive finite value within the bounds passed
    /// to [`AdaptiveScale::new`]. Implementations must support every value in
    /// those bounds and update all scale-dependent sampling and density terms.
    /// This operation must not change the target or chain state, and must not
    /// panic for supported scales. Setting the same scale again must leave
    /// transition-relevant proposal state unchanged, so warmup chunk boundaries
    /// do not change the kernel or consume randomness.
    fn set_scale(&mut self, scale: f64);
}

impl<P: TunableProposal + ?Sized> TunableProposal for &mut P {
    fn set_scale(&mut self, scale: f64) {
        (**self).set_scale(scale);
    }
}

/// Validated state for acceptance-rate tuning of one positive proposal scale.
///
/// After completed warmup transition `n` (starting at one), use the bounded
/// Robbins–Monro update:
///
/// ```text
/// log_scale += n^(-0.6) * (accepted - target_acceptance)
/// log_scale = clamp(log_scale, ln(min_scale), ln(max_scale))
/// ```
///
/// `accepted` is one for an accepted proposal and zero for rejection or no
/// proposal. Errors do not advance adaptation. The scale is also clamped after
/// exponentiation to preserve the supplied bounds despite floating-point
/// rounding. At `usize::MAX` completed steps, further updates leave it frozen.
///
/// Use with [`Sampler::warm_up`], [`Sampler::warm_up_mut`], or
/// [`Sampler::warm_up_delayed`], then collect production samples with the ordinary
/// sampler methods, which never update this controller. Discard warmup draws;
/// freezing the proposal does not establish stationarity or adequate mixing.
/// This is scalar tuning, not covariance adaptation or a guarantee of attaining
/// the requested acceptance rate. Choose bounds appropriate to the proposal.
///
/// Reuse this value across warmup chunks to preserve the learning schedule,
/// independently of chain counter resets. Chain checkpoints do not contain this
/// controller or the proposal's tuned parameters. Same-build continuation also
/// requires preserving the proposal and RNG state.
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::prelude::by_value::*;
/// use markov_chain_monte_carlo::{AdaptiveScale, AdaptiveScaleError, TunableProposal};
/// use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
///
/// struct Normal;
/// impl Target<f64> for Normal {
///     fn log_prob(&self, x: &f64) -> f64 { -0.5 * x * x }
/// }
/// struct Walk { width: f64 }
/// impl Proposal<f64> for Walk {
///     fn propose<R: Rng + ?Sized>(&self, x: &f64, rng: &mut R) -> f64 {
///         x + self.width * rng.random_range(-1.0..1.0)
///     }
/// }
/// impl TunableProposal for Walk {
///     fn set_scale(&mut self, scale: f64) { self.width = scale; }
/// }
/// let mut tuning = AdaptiveScale::new(0.1, 0.44, 0.001..=100.0)?;
/// let mut rng = StdRng::seed_from_u64(42);
/// # let run = (|| -> Result<(), McmcError> {
/// let chain = Chain::new(0.0, &Normal)?;
/// let mut sampler = Sampler::new(chain, &Normal, Walk { width: 0.1 }, &mut rng)?;
/// sampler.warm_up(2_000, &mut tuning)?;
/// sampler.reset_counters();
/// sampler.run(1_000)?; // Fixed width throughout production.
/// assert_eq!(tuning.completed_steps(), 2_000);
/// # Ok(())
/// # })();
/// # assert!(run.is_ok());
/// # Ok::<(), AdaptiveScaleError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct AdaptiveScale {
    scale: f64,
    log_scale: f64,
    target_acceptance: f64,
    min_scale: f64,
    max_scale: f64,
    log_min: f64,
    log_max: f64,
    completed_steps: usize,
}

impl AdaptiveScale {
    /// Create a tuner with an initial scale, acceptance target, and inclusive bounds.
    ///
    /// The target must be finite and strictly between zero and one. Bounds must
    /// be positive and finite, with minimum no greater than maximum; the initial
    /// scale must lie within them. Equal bounds hold the scale fixed.
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveScaleError`] for an invalid target, bounds, or initial
    /// scale, checked in that order. Construction has no sampling side effects.
    pub fn new(
        initial_scale: f64,
        target_acceptance: f64,
        bounds: RangeInclusive<f64>,
    ) -> Result<Self, AdaptiveScaleError> {
        if !target_acceptance.is_finite() || target_acceptance <= 0.0 || target_acceptance >= 1.0 {
            return Err(AdaptiveScaleError::InvalidTargetAcceptance {
                value: target_acceptance,
            });
        }
        let (min_scale, max_scale) = bounds.into_inner();
        if !min_scale.is_finite()
            || !max_scale.is_finite()
            || min_scale <= 0.0
            || min_scale > max_scale
        {
            return Err(AdaptiveScaleError::InvalidBounds {
                min_scale,
                max_scale,
            });
        }
        if !initial_scale.is_finite() || !(min_scale..=max_scale).contains(&initial_scale) {
            return Err(AdaptiveScaleError::InvalidInitialScale {
                value: initial_scale,
                min_scale,
                max_scale,
            });
        }
        Ok(Self {
            scale: initial_scale,
            log_scale: initial_scale.ln(),
            target_acceptance,
            min_scale,
            max_scale,
            log_min: min_scale.ln(),
            log_max: max_scale.ln(),
            completed_steps: 0,
        })
    }

    /// Current positive finite proposal scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Desired fraction of completed transitions that accept a proposal.
    #[must_use]
    pub const fn target_acceptance(&self) -> f64 {
        self.target_acceptance
    }

    /// Inclusive bounds on the proposal scale.
    #[must_use]
    pub const fn bounds(&self) -> RangeInclusive<f64> {
        self.min_scale..=self.max_scale
    }

    /// Number of completed adaptive transitions, saturating at `usize::MAX`.
    #[must_use]
    pub const fn completed_steps(&self) -> usize {
        self.completed_steps
    }

    /// Advance the bounded schedule after one completed transition.
    fn update(&mut self, accepted: bool) {
        let Some(next) = self.completed_steps.checked_add(1) else {
            return;
        };
        self.completed_steps = next;
        // Rounding large counts only rounds the learning rate; no integer
        // accounting or transition selection depends on the floating value.
        let gain = count_as_f64(next).powf(-0.6);
        self.log_scale = (self.log_scale + gain * (f64::from(accepted) - self.target_acceptance))
            .clamp(self.log_min, self.log_max);
        self.scale = self.log_scale.exp().clamp(self.min_scale, self.max_scale);
    }
}

/// Invalid scalar-adaptation configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveScaleError {
    /// The target is not finite and strictly between zero and one.
    InvalidTargetAcceptance {
        /// Supplied target acceptance rate.
        value: f64,
    },
    /// Bounds are not positive, finite, and ordered.
    InvalidBounds {
        /// Supplied lower bound.
        min_scale: f64,
        /// Supplied upper bound.
        max_scale: f64,
    },
    /// The initial scale is not finite or is outside the bounds.
    InvalidInitialScale {
        /// Supplied initial scale.
        value: f64,
        /// Validated lower bound.
        min_scale: f64,
        /// Validated upper bound.
        max_scale: f64,
    },
}

impl fmt::Display for AdaptiveScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTargetAcceptance { value } => {
                write!(
                    f,
                    "target acceptance must be finite and in (0, 1), got {value}"
                )
            }
            Self::InvalidBounds {
                min_scale,
                max_scale,
            } => {
                write!(
                    f,
                    "scale bounds must be positive, finite, and ordered, got {min_scale}..={max_scale}"
                )
            }
            Self::InvalidInitialScale {
                value,
                min_scale,
                max_scale,
            } => {
                write!(
                    f,
                    "initial scale must be finite and in {min_scale}..={max_scale}, got {value}"
                )
            }
        }
    }
}

impl Error for AdaptiveScaleError {}

impl<S, T: ?Sized, P: TunableProposal, R: ?Sized> Sampler<'_, S, T, P, R> {
    /// Apply scale changes only between successfully completed transitions.
    fn warm_up_with<E>(
        &mut self,
        steps: usize,
        tuning: &mut AdaptiveScale,
        mut step: impl FnMut(&mut Self) -> Result<bool, E>,
    ) -> Result<(), E> {
        #[cfg(feature = "tracing")]
        let _span = tracing::debug_span!(
            target: "markov_chain_monte_carlo",
            "warm_up",
            steps,
            start_step = self.chain_ref().total_steps()
        )
        .entered();
        if steps == 0 {
            return Ok(());
        }
        self.proposal_mut().set_scale(tuning.scale());
        for _ in 0..steps {
            let accepted = step(self)?;
            tuning.update(accepted);
            self.proposal_mut().set_scale(tuning.scale());
        }
        Ok(())
    }
}

impl<S, T: Target<S> + ?Sized, P: Proposal<S> + TunableProposal, R: Rng + ?Sized>
    Sampler<'_, S, T, P, R>
{
    /// Tune proposal scale over `steps` by-value warmup transitions.
    ///
    /// Applies the tuner's scale before the first transition and updates it
    /// after each completed transition. A zero-step call changes nothing.
    /// Retains chain state and counters, but no samples. Reuse the tuner across
    /// chunks; call [`reset_counters`](Self::reset_counters) explicitly when
    /// starting production. Ordinary sampling methods leave the final scale fixed.
    /// See [`AdaptiveScale`] for a complete example and scientific limitations.
    ///
    /// # Errors
    ///
    /// Returns the first [`McmcError`] from [`step`](Self::step). Completed
    /// transitions and tuning updates remain applied. A failed transition does
    /// not update tuning; the proposal retains the last scale applied by warmup.
    pub fn warm_up(&mut self, steps: usize, tuning: &mut AdaptiveScale) -> Result<(), McmcError> {
        self.warm_up_with(steps, tuning, |sampler| {
            sampler.step().map(|step| step.outcome().is_accepted())
        })
    }
}

impl<S, T: Target<S> + ?Sized, P: ProposalMut<S> + TunableProposal, R: Rng + ?Sized>
    Sampler<'_, S, T, P, R>
{
    /// Tune proposal scale over `steps` in-place warmup transitions.
    ///
    /// Follows [`warm_up`](Self::warm_up)'s chunk, zero-step, counter, and freeze
    /// semantics. No-proposal self-loops count as non-acceptances. Scale updates
    /// happen after rollback or acceptance, preserving the transition's kernel.
    /// Proposal metadata hooks are not called because warmup retains no telemetry.
    ///
    /// # Errors
    ///
    /// Returns the first [`McmcError`] from [`step_mut`](Self::step_mut).
    /// Failed transitions do not update tuning; completed work remains applied.
    pub fn warm_up_mut(
        &mut self,
        steps: usize,
        tuning: &mut AdaptiveScale,
    ) -> Result<(), McmcError> {
        self.warm_up_with(steps, tuning, |sampler| {
            sampler.step_mut_outcome().map(StepOutcome::is_accepted)
        })
    }
}

impl<S, T: Target<S> + ?Sized, P: DelayedProposal<S> + TunableProposal, R: Rng + ?Sized>
    Sampler<'_, S, T, P, R>
{
    /// Tune proposal scale over `steps` delayed-commit warmup transitions.
    ///
    /// Follows [`warm_up`](Self::warm_up)'s chunk, zero-step, counter, and freeze
    /// semantics. No-plan self-loops count as non-acceptances. Scale updates occur
    /// only after a completed transition, including successful commit if accepted.
    /// Proposal metadata hooks are not called because warmup retains no telemetry.
    ///
    /// # Errors
    ///
    /// Returns the first [`DelayedStepError`] from [`step_delayed`](Self::step_delayed).
    /// Failed transitions do not update tuning; completed work remains applied.
    pub fn warm_up_delayed(
        &mut self,
        steps: usize,
        tuning: &mut AdaptiveScale,
    ) -> Result<(), DelayedStepError<P::Error>> {
        self.warm_up_with(steps, tuning, |sampler| {
            sampler.step_delayed_outcome().map(StepOutcome::is_accepted)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AdaptiveScale;

    #[test]
    fn saturated_schedule_freezes_without_overflow() {
        let mut tuning = AdaptiveScale::new(1.0, 0.5, 0.1..=10.0).unwrap();
        tuning.completed_steps = usize::MAX;
        let before = tuning.clone();
        tuning.update(true);
        assert_eq!(tuning, before);
    }
}
