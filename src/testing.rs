//! Testing and validation helpers.

use core::convert::Infallible;
use std::error::Error;
use std::fmt;

use rand::Rng;

use crate::{DelayedProposal, Proposal, ProposalMut, Target};

/// Configuration for empirical detailed-balance verification.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DetailedBalanceConfig {
    /// Number of proposal samples drawn in each direction.
    pub samples: usize,
    /// Absolute tolerance for the log detailed-balance residual.
    pub tolerance: f64,
    /// Minimum hit count required in each direction.
    pub min_hits: usize,
}

impl Default for DetailedBalanceConfig {
    fn default() -> Self {
        Self {
            samples: 10_000,
            tolerance: 0.1,
            min_hits: 30,
        }
    }
}

/// Direction of an empirical detailed-balance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailedBalanceDirection {
    /// The transition from the current state to the proposed state.
    Forward,
    /// The transition from the proposed state back to the current state.
    Reverse,
}

impl fmt::Display for DetailedBalanceDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "forward"),
            Self::Reverse => write!(f, "reverse"),
        }
    }
}

/// State role in an empirical detailed-balance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailedBalanceState {
    /// The current state supplied to the check.
    Current,
    /// The proposed state supplied to the check.
    Proposed,
}

impl fmt::Display for DetailedBalanceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Current => write!(f, "current"),
            Self::Proposed => write!(f, "proposed"),
        }
    }
}

/// Empirical detailed-balance verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DetailedBalanceReport {
    /// Number of proposal samples drawn in each direction.
    pub samples: usize,
    /// Number of times the forward transition was proposed.
    pub forward_hits: usize,
    /// Number of times the reverse transition was proposed.
    pub reverse_hits: usize,
    /// Estimated `log(q(proposed | current))`.
    pub forward_log_proposal: f64,
    /// Estimated `log(q(current | proposed))`.
    pub reverse_log_proposal: f64,
    /// Estimated log transition probability after MH acceptance, forward.
    pub forward_log_transition: f64,
    /// Estimated log transition probability after MH acceptance, reverse.
    pub reverse_log_transition: f64,
    /// Estimated `log(forward flow) - log(reverse flow)`.
    pub log_balance_residual: f64,
    /// Approximate standard error of `log_balance_residual`.
    pub log_balance_standard_error: f64,
}

impl DetailedBalanceReport {
    /// Return true when the absolute residual is within `tolerance`.
    ///
    /// ```
    /// use markov_chain_monte_carlo::DetailedBalanceReport;
    ///
    /// let report = DetailedBalanceReport {
    ///     samples: 128,
    ///     forward_hits: 128,
    ///     reverse_hits: 128,
    ///     forward_log_proposal: 0.0,
    ///     reverse_log_proposal: 0.0,
    ///     forward_log_transition: 0.0,
    ///     reverse_log_transition: 0.0,
    ///     log_balance_residual: 1e-3,
    ///     log_balance_standard_error: 0.01,
    /// };
    ///
    /// assert!(report.is_within_tolerance(0.01));
    /// assert!(!report.is_within_tolerance(1e-4));
    /// ```
    #[must_use]
    pub fn is_within_tolerance(&self, tolerance: f64) -> bool {
        self.log_balance_residual.is_finite()
            && tolerance.is_finite()
            && tolerance >= 0.0
            && self.log_balance_residual.abs() <= tolerance
    }

    /// Approximate normal z-score for the observed log residual.
    ///
    /// ```
    /// use markov_chain_monte_carlo::DetailedBalanceReport;
    ///
    /// let report = DetailedBalanceReport {
    ///     samples: 128,
    ///     forward_hits: 128,
    ///     reverse_hits: 128,
    ///     forward_log_proposal: 0.0,
    ///     reverse_log_proposal: 0.0,
    ///     forward_log_transition: 0.0,
    ///     reverse_log_transition: 0.0,
    ///     log_balance_residual: 0.02,
    ///     log_balance_standard_error: 0.01,
    /// };
    ///
    /// assert_eq!(report.z_score(), Some(2.0));
    /// ```
    #[must_use]
    pub fn z_score(&self) -> Option<f64> {
        if self.log_balance_standard_error.is_finite() && self.log_balance_standard_error > 0.0 {
            Some(self.log_balance_residual / self.log_balance_standard_error)
        } else {
            None
        }
    }
}

/// Error returned by empirical detailed-balance verification.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DetailedBalanceError<E = Infallible> {
    /// `samples` must be greater than zero.
    InvalidSamples {
        /// Invalid sample count.
        samples: usize,
    },
    /// `tolerance` must be finite and nonnegative.
    InvalidTolerance {
        /// Invalid tolerance.
        tolerance: f64,
    },
    /// `min_hits` must be greater than zero and no larger than `samples`.
    InvalidMinHits {
        /// Invalid minimum hit count.
        min_hits: usize,
        /// Number of proposal samples drawn in each direction.
        samples: usize,
    },
    /// Target log-probability was `NaN` or positive infinity for one endpoint.
    InvalidTargetLogProb {
        /// Endpoint whose log-probability was invalid.
        state: DetailedBalanceState,
        /// Observed target log-probability.
        log_prob: f64,
    },
    /// Proposal log q-ratio was `NaN` or positive infinity for one direction.
    InvalidLogQRatio {
        /// Direction whose log q-ratio was invalid.
        direction: DetailedBalanceDirection,
        /// Observed log q-ratio.
        log_q_ratio: f64,
    },
    /// Metropolis-Hastings log acceptance ratio was `NaN`.
    InvalidLogAcceptanceRatio {
        /// Direction whose log acceptance ratio was invalid.
        direction: DetailedBalanceDirection,
        /// Observed log acceptance ratio.
        log_acceptance_ratio: f64,
    },
    /// Delayed proposal planning failed.
    Plan {
        /// Direction whose planning failed.
        direction: DetailedBalanceDirection,
        /// Underlying proposal error.
        source: E,
    },
    /// Delayed proposal log-probability evaluation failed.
    ProposedLogProb {
        /// Direction whose proposed log-probability evaluation failed.
        direction: DetailedBalanceDirection,
        /// Underlying proposal error.
        source: E,
    },
    /// Delayed proposal log q-ratio evaluation failed.
    LogQRatio {
        /// Direction whose proposal ratio evaluation failed.
        direction: DetailedBalanceDirection,
        /// Underlying proposal error.
        source: E,
    },
    /// Too few exact proposal hits were observed in one direction.
    InsufficientHits {
        /// Direction with too few hits.
        direction: DetailedBalanceDirection,
        /// Observed hit count.
        hits: usize,
        /// Required hit count.
        min_hits: usize,
    },
    /// Estimated detailed-balance residual exceeded the configured tolerance.
    Violation {
        /// Estimated `log(forward flow) - log(reverse flow)`.
        residual: f64,
        /// Configured absolute tolerance.
        tolerance: f64,
        /// Detailed empirical report.
        report: DetailedBalanceReport,
    },
}

impl<E: fmt::Display> fmt::Display for DetailedBalanceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSamples { samples } => {
                write!(f, "invalid sample count {samples}: expected at least 1")
            }
            Self::InvalidTolerance { tolerance } => write!(
                f,
                "invalid detailed-balance tolerance {tolerance}: expected a finite nonnegative value"
            ),
            Self::InvalidMinHits { min_hits, samples } => write!(
                f,
                "invalid minimum hit count {min_hits}: expected 1..={samples}"
            ),
            Self::InvalidTargetLogProb { state, log_prob } => write!(
                f,
                "{state} target log-probability is {log_prob}: expected finite or -infinity"
            ),
            Self::InvalidLogQRatio {
                direction,
                log_q_ratio,
            } => write!(
                f,
                "{direction} proposal log q-ratio is {log_q_ratio}: expected finite or -infinity"
            ),
            Self::InvalidLogAcceptanceRatio {
                direction,
                log_acceptance_ratio,
            } => write!(
                f,
                "{direction} log acceptance ratio is {log_acceptance_ratio}: expected a non-NaN value"
            ),
            Self::Plan { direction, source } => {
                write!(f, "{direction} delayed proposal planning failed: {source}")
            }
            Self::ProposedLogProb { direction, source } => write!(
                f,
                "{direction} delayed proposal log-probability evaluation failed: {source}"
            ),
            Self::LogQRatio { direction, source } => {
                write!(
                    f,
                    "{direction} delayed proposal ratio evaluation failed: {source}"
                )
            }
            Self::InsufficientHits {
                direction,
                hits,
                min_hits,
            } => write!(
                f,
                "insufficient {direction} proposal hits: observed {hits}, expected at least {min_hits}"
            ),
            Self::Violation {
                residual,
                tolerance,
                ..
            } => write!(
                f,
                "detailed-balance residual {residual} exceeds tolerance {tolerance}"
            ),
        }
    }
}

impl<E> Error for DetailedBalanceError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Plan { source, .. }
            | Self::ProposedLogProb { source, .. }
            | Self::LogQRatio { source, .. } => Some(source),
            Self::InvalidSamples { .. }
            | Self::InvalidTolerance { .. }
            | Self::InvalidMinHits { .. }
            | Self::InvalidTargetLogProb { .. }
            | Self::InvalidLogQRatio { .. }
            | Self::InvalidLogAcceptanceRatio { .. }
            | Self::InsufficientHits { .. }
            | Self::Violation { .. } => None,
        }
    }
}

/// One failed detailed-balance check in a batch.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DetailedBalanceFailure<E = Infallible> {
    /// Zero-based index of the checked transition.
    pub index: usize,
    /// Error reported for this transition.
    pub error: DetailedBalanceError<E>,
}

/// Batch detailed-balance report that preserves every failure.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DetailedBalanceBatchReport<E = Infallible> {
    /// Successful transition reports.
    pub reports: Vec<DetailedBalanceReport>,
    /// Failed transition checks, including all violations.
    pub failures: Vec<DetailedBalanceFailure<E>>,
}

impl<E> DetailedBalanceBatchReport<E> {
    /// Return true when every transition passed.
    ///
    /// ```
    /// use markov_chain_monte_carlo::DetailedBalanceBatchReport;
    ///
    /// let batch: DetailedBalanceBatchReport = DetailedBalanceBatchReport {
    ///     reports: Vec::new(),
    ///     failures: Vec::new(),
    /// };
    ///
    /// assert!(batch.is_success());
    /// ```
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.failures.is_empty()
    }
}

/// One delayed-commit transition to check in a batch.
#[must_use]
pub struct DetailedBalanceDelayedTransition<'a, S, Plan> {
    /// Current endpoint for the transition.
    pub current: &'a S,
    /// Proposed endpoint for the transition.
    pub proposed: &'a S,
    /// Predicate identifying sampled forward plans from `current` to `proposed`.
    pub forward_plan_matches: &'a dyn Fn(&Plan) -> bool,
    /// Predicate identifying sampled reverse plans from `proposed` to `current`.
    pub reverse_plan_matches: &'a dyn Fn(&Plan) -> bool,
}

/// Empirically verify detailed balance for one discrete by-value transition.
///
/// This helper samples `proposal` from `current` and `proposed`, estimates the
/// off-diagonal Metropolis-Hastings transition probabilities in both directions,
/// and checks that the log transition flows agree within `config.tolerance`.
///
/// Because it compares states with [`PartialEq`], this is intended for
/// discrete, quantized, or otherwise exactly comparable state spaces.  For
/// continuous proposals, exact hits are usually too rare; use a coarsened state
/// representation or a domain-specific diagnostic instead.
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct Flip;
/// impl Proposal<bool> for Flip {
///     fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
///         !current
///     }
/// }
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let report = verify_detailed_balance(
///     &false,
///     &true,
///     &Flat,
///     &Flip,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
/// )?;
///
/// assert_eq!(report.forward_hits, 128);
/// assert!(report.is_within_tolerance(1e-12));
/// # Ok::<(), DetailedBalanceError>(())
/// ```
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid, target
/// log-probabilities or proposal ratios are `NaN`/`+infinity`, too few exact
/// hits are observed in either direction, or the estimated detailed-balance
/// residual exceeds the configured tolerance.
pub fn verify_detailed_balance<S, T, P, R>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceReport, DetailedBalanceError>
where
    S: PartialEq,
    T: Target<S>,
    P: Proposal<S> + ?Sized,
    R: Rng + ?Sized,
{
    validate_config(config)?;
    verify_detailed_balance_unchecked(current, proposed, target, proposal, rng, config)
}

/// Verify many by-value transitions and return every transition violation.
///
/// This is useful for checking a small grid, graph edge list, or representative
/// set of local moves without stopping on the first failed transition.
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct Flip;
/// impl Proposal<bool> for Flip {
///     fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
///         !current
///     }
/// }
///
/// let pairs = [(false, true), (true, false)];
/// let mut rng = StdRng::seed_from_u64(42);
/// let batch = verify_detailed_balance_many(
///     pairs.iter().map(|(current, proposed)| (current, proposed)),
///     &Flat,
///     &Flip,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
/// )?;
///
/// assert!(batch.is_success());
/// assert_eq!(batch.reports.len(), 2);
/// # Ok::<(), DetailedBalanceError>(())
/// ```
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid.  Per-transition
/// failures are collected in [`DetailedBalanceBatchReport::failures`].
pub fn verify_detailed_balance_many<'a, S, T, P, R, I>(
    pairs: I,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceBatchReport, DetailedBalanceError>
where
    S: PartialEq + 'a,
    T: Target<S>,
    P: Proposal<S> + ?Sized,
    R: Rng + ?Sized,
    I: IntoIterator<Item = (&'a S, &'a S)>,
{
    let mut batch = DetailedBalanceBatchReport {
        reports: Vec::new(),
        failures: Vec::new(),
    };

    validate_config(config)?;

    for (index, (current, proposed)) in pairs.into_iter().enumerate() {
        match verify_detailed_balance_unchecked(current, proposed, target, proposal, rng, config) {
            Ok(report) => batch.reports.push(report),
            Err(error) => batch.failures.push(DetailedBalanceFailure { index, error }),
        }
    }

    Ok(batch)
}

/// Empirically verify detailed balance for one in-place transition.
///
/// This helper clones each endpoint before calling [`ProposalMut::propose_mut`],
/// so it is intended for test code and representative transitions rather than
/// production sampling.  It estimates the full Metropolis-Hastings transition
/// probability by combining exact endpoint hits with each hit's undo-token-based
/// log q-ratio.
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct Flip;
/// impl ProposalMut<bool> for Flip {
///     type Undo = bool;
///
///     fn propose_mut<R: Rng + ?Sized>(&self, state: &mut bool, _: &mut R) -> Option<bool> {
///         let old = *state;
///         *state = !*state;
///         Some(old)
///     }
///
///     fn undo(&self, state: &mut bool, token: bool) {
///         *state = token;
///     }
/// }
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let report = verify_detailed_balance_mut(
///     &false,
///     &true,
///     &Flat,
///     &Flip,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
/// )?;
///
/// assert!(report.is_within_tolerance(1e-12));
/// # Ok::<(), DetailedBalanceError>(())
/// ```
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid, target
/// log-probabilities or proposal ratios are `NaN`/`+infinity`, too few exact
/// hits are observed in either direction, or the estimated detailed-balance
/// residual exceeds the configured tolerance.
pub fn verify_detailed_balance_mut<S, T, P, R>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceReport, DetailedBalanceError>
where
    S: Clone + PartialEq,
    T: Target<S>,
    P: ProposalMut<S> + ?Sized,
    R: Rng + ?Sized,
{
    validate_config(config)?;
    verify_detailed_balance_mut_unchecked(current, proposed, target, proposal, rng, config)
}

/// Verify many in-place transitions and return every transition violation.
///
/// ```
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct Flip;
/// impl ProposalMut<bool> for Flip {
///     type Undo = bool;
///
///     fn propose_mut<R: Rng + ?Sized>(&self, state: &mut bool, _: &mut R) -> Option<bool> {
///         let old = *state;
///         *state = !*state;
///         Some(old)
///     }
///
///     fn undo(&self, state: &mut bool, token: bool) {
///         *state = token;
///     }
/// }
///
/// let pairs = [(false, true), (true, false)];
/// let mut rng = StdRng::seed_from_u64(42);
/// let batch = verify_detailed_balance_mut_many(
///     pairs.iter().map(|(current, proposed)| (current, proposed)),
///     &Flat,
///     &Flip,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
/// )?;
///
/// assert!(batch.is_success());
/// assert_eq!(batch.reports.len(), 2);
/// # Ok::<(), DetailedBalanceError>(())
/// ```
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid.  Per-transition
/// failures are collected in [`DetailedBalanceBatchReport::failures`].
pub fn verify_detailed_balance_mut_many<'a, S, T, P, R, I>(
    pairs: I,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceBatchReport, DetailedBalanceError>
where
    S: Clone + PartialEq + 'a,
    T: Target<S>,
    P: ProposalMut<S> + ?Sized,
    R: Rng + ?Sized,
    I: IntoIterator<Item = (&'a S, &'a S)>,
{
    let mut batch = DetailedBalanceBatchReport {
        reports: Vec::new(),
        failures: Vec::new(),
    };

    validate_config(config)?;

    for (index, (current, proposed)) in pairs.into_iter().enumerate() {
        match verify_detailed_balance_mut_unchecked(
            current, proposed, target, proposal, rng, config,
        ) {
            Ok(report) => batch.reports.push(report),
            Err(error) => batch.failures.push(DetailedBalanceFailure { index, error }),
        }
    }

    Ok(batch)
}

/// Empirically verify detailed balance for one delayed-commit transition.
///
/// `plan_matches.0` identifies sampled forward plans from `current` to
/// `proposed`; `plan_matches.1` identifies sampled reverse plans from
/// `proposed` to `current`.  This keeps the delayed helper plan-based, matching
/// the [`DelayedProposal`] contract where plans are the concrete transition
/// descriptors used for scoring and proposal ratios.
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid, delayed
/// proposal planning/scoring fails, target log-probabilities or proposal ratios
/// are `NaN`/`+infinity`, too few matching plans are observed in either
/// direction, or the estimated detailed-balance residual exceeds the configured
/// tolerance.
///
/// ```
/// use core::convert::Infallible;
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct FlipPlan;
/// impl DelayedProposal<bool> for FlipPlan {
///     type Plan = bool;
///     type Info = bool;
///     type Error = Infallible;
///
///     fn propose_plan<R: Rng + ?Sized>(
///         &mut self,
///         state: &bool,
///         _: &mut R,
///     ) -> Result<Option<bool>, Infallible> {
///         Ok(Some(!*state))
///     }
///
///     fn proposed_log_prob<T: Target<bool>>(
///         &self,
///         _: &bool,
///         plan: &bool,
///         target: &T,
///     ) -> Result<f64, Infallible> {
///         Ok(target.log_prob(plan))
///     }
///
///     fn info(&self, plan: &bool) -> bool { *plan }
///
///     fn commit<R: Rng + ?Sized>(
///         &mut self,
///         state: &mut bool,
///         plan: bool,
///         _: &mut R,
///     ) -> Result<(), Infallible> {
///         *state = plan;
///         Ok(())
///     }
/// }
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let mut proposal = FlipPlan;
/// let report = verify_detailed_balance_delayed(
///     &false,
///     &true,
///     &Flat,
///     &mut proposal,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
///     (|plan| *plan, |plan| !*plan),
/// )?;
///
/// assert!(report.is_within_tolerance(1e-12));
/// # Ok::<(), DetailedBalanceError<Infallible>>(())
/// ```
pub fn verify_detailed_balance_delayed<S, T, P, R>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &mut P,
    rng: &mut R,
    config: DetailedBalanceConfig,
    plan_matches: (impl Fn(&P::Plan) -> bool, impl Fn(&P::Plan) -> bool),
) -> Result<DetailedBalanceReport, DetailedBalanceError<P::Error>>
where
    T: Target<S>,
    P: DelayedProposal<S> + ?Sized,
    R: Rng + ?Sized,
{
    validate_config(config)?;
    let (forward_matches, reverse_matches) = plan_matches;
    verify_detailed_balance_delayed_unchecked(
        current,
        proposed,
        target,
        proposal,
        rng,
        config,
        (&forward_matches, &reverse_matches),
    )
}

/// Verify many delayed-commit transitions and return every transition violation.
///
/// Each [`DetailedBalanceDelayedTransition`] supplies endpoint states and the
/// forward/reverse plan predicates for that concrete transition.
///
/// ```
/// use core::convert::Infallible;
/// use markov_chain_monte_carlo::prelude::testing::*;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// struct Flat;
/// impl Target<bool> for Flat {
///     fn log_prob(&self, _: &bool) -> f64 { 0.0 }
/// }
///
/// struct FlipPlan;
/// impl DelayedProposal<bool> for FlipPlan {
///     type Plan = bool;
///     type Info = bool;
///     type Error = Infallible;
///
///     fn propose_plan<R: Rng + ?Sized>(
///         &mut self,
///         state: &bool,
///         _: &mut R,
///     ) -> Result<Option<bool>, Infallible> {
///         Ok(Some(!*state))
///     }
///
///     fn proposed_log_prob<T: Target<bool>>(
///         &self,
///         _: &bool,
///         plan: &bool,
///         target: &T,
///     ) -> Result<f64, Infallible> {
///         Ok(target.log_prob(plan))
///     }
///
///     fn info(&self, plan: &bool) -> bool { *plan }
///
///     fn commit<R: Rng + ?Sized>(
///         &mut self,
///         state: &mut bool,
///         plan: bool,
///         _: &mut R,
///     ) -> Result<(), Infallible> {
///         *state = plan;
///         Ok(())
///     }
/// }
///
/// let forward = |plan: &bool| *plan;
/// let reverse = |plan: &bool| !*plan;
/// let transitions = [DetailedBalanceDelayedTransition {
///     current: &false,
///     proposed: &true,
///     forward_plan_matches: &forward,
///     reverse_plan_matches: &reverse,
/// }];
/// let mut proposal = FlipPlan;
/// let mut rng = StdRng::seed_from_u64(42);
/// let batch = verify_detailed_balance_delayed_many(
///     transitions,
///     &Flat,
///     &mut proposal,
///     &mut rng,
///     DetailedBalanceConfig {
///         samples: 128,
///         tolerance: 1e-12,
///         min_hits: 1,
///     },
/// )?;
///
/// assert!(batch.is_success());
/// assert_eq!(batch.reports.len(), 1);
/// # Ok::<(), DetailedBalanceError<Infallible>>(())
/// ```
///
/// # Errors
///
/// Returns [`DetailedBalanceError`] if the configuration is invalid.  Per-transition
/// failures are collected in [`DetailedBalanceBatchReport::failures`].
pub fn verify_detailed_balance_delayed_many<'a, S, T, P, R, I>(
    transitions: I,
    target: &T,
    proposal: &mut P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceBatchReport<P::Error>, DetailedBalanceError<P::Error>>
where
    S: 'a,
    T: Target<S>,
    P: DelayedProposal<S> + ?Sized,
    P::Plan: 'a,
    R: Rng + ?Sized,
    I: IntoIterator<Item = DetailedBalanceDelayedTransition<'a, S, P::Plan>>,
{
    let mut batch = DetailedBalanceBatchReport {
        reports: Vec::new(),
        failures: Vec::new(),
    };

    validate_config(config)?;

    for (index, transition) in transitions.into_iter().enumerate() {
        match verify_detailed_balance_delayed_unchecked(
            transition.current,
            transition.proposed,
            target,
            proposal,
            rng,
            config,
            (
                transition.forward_plan_matches,
                transition.reverse_plan_matches,
            ),
        ) {
            Ok(report) => batch.reports.push(report),
            Err(error) => batch.failures.push(DetailedBalanceFailure { index, error }),
        }
    }

    Ok(batch)
}

/// Run a by-value detailed-balance check after the public caller has already
/// validated the sampling configuration.
fn verify_detailed_balance_unchecked<S, T, P, R>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceReport, DetailedBalanceError>
where
    S: PartialEq,
    T: Target<S>,
    P: Proposal<S> + ?Sized,
    R: Rng + ?Sized,
{
    let endpoint_log_probs = endpoint_log_probs(current, proposed, target)?;
    let forward_log_q_ratio = proposal.log_q_ratio(current, proposed);
    check_log_q_ratio(DetailedBalanceDirection::Forward, forward_log_q_ratio)?;
    let reverse_log_q_ratio = proposal.log_q_ratio(proposed, current);
    check_log_q_ratio(DetailedBalanceDirection::Reverse, reverse_log_q_ratio)?;

    let forward_acceptance = acceptance_probability(
        DetailedBalanceDirection::Forward,
        endpoint_log_probs.proposed - endpoint_log_probs.current + forward_log_q_ratio,
    )?;
    let reverse_acceptance = acceptance_probability(
        DetailedBalanceDirection::Reverse,
        endpoint_log_probs.current - endpoint_log_probs.proposed + reverse_log_q_ratio,
    )?;

    let forward = estimate_by_value_transition(
        current,
        proposed,
        proposal,
        rng,
        config.samples,
        forward_acceptance,
    );
    let reverse = estimate_by_value_transition(
        proposed,
        current,
        proposal,
        rng,
        config.samples,
        reverse_acceptance,
    );

    finish_report(endpoint_log_probs, forward, reverse, config)
}

/// Run an in-place detailed-balance check after configuration validation,
/// estimating each direction from cloned endpoint states.
fn verify_detailed_balance_mut_unchecked<S, T, P, R>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &P,
    rng: &mut R,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceReport, DetailedBalanceError>
where
    S: Clone + PartialEq,
    T: Target<S>,
    P: ProposalMut<S> + ?Sized,
    R: Rng + ?Sized,
{
    let endpoint_log_probs = endpoint_log_probs(current, proposed, target)?;
    let forward = estimate_mut_transition(
        current,
        proposed,
        target,
        proposal,
        rng,
        TransitionRequest {
            samples: config.samples,
            from_log_prob: endpoint_log_probs.current,
            direction: DetailedBalanceDirection::Forward,
        },
    )?;
    let reverse = estimate_mut_transition(
        proposed,
        current,
        target,
        proposal,
        rng,
        TransitionRequest {
            samples: config.samples,
            from_log_prob: endpoint_log_probs.proposed,
            direction: DetailedBalanceDirection::Reverse,
        },
    )?;

    finish_report(endpoint_log_probs, forward, reverse, config)
}

/// Run a delayed-proposal detailed-balance check after configuration validation,
/// using caller-supplied predicates to identify matching concrete plans.
fn verify_detailed_balance_delayed_unchecked<S, T, P, R, F, G>(
    current: &S,
    proposed: &S,
    target: &T,
    proposal: &mut P,
    rng: &mut R,
    config: DetailedBalanceConfig,
    plan_matches: (&F, &G),
) -> Result<DetailedBalanceReport, DetailedBalanceError<P::Error>>
where
    T: Target<S>,
    P: DelayedProposal<S> + ?Sized,
    R: Rng + ?Sized,
    F: Fn(&P::Plan) -> bool + ?Sized,
    G: Fn(&P::Plan) -> bool + ?Sized,
{
    let endpoint_log_probs = endpoint_log_probs(current, proposed, target)?;
    let forward = estimate_delayed_transition(
        current,
        target,
        proposal,
        rng,
        TransitionRequest {
            samples: config.samples,
            from_log_prob: endpoint_log_probs.current,
            direction: DetailedBalanceDirection::Forward,
        },
        plan_matches.0,
    )?;
    let reverse = estimate_delayed_transition(
        proposed,
        target,
        proposal,
        rng,
        TransitionRequest {
            samples: config.samples,
            from_log_prob: endpoint_log_probs.proposed,
            direction: DetailedBalanceDirection::Reverse,
        },
        plan_matches.1,
    )?;

    finish_report(endpoint_log_probs, forward, reverse, config)
}

/// Validate detailed-balance configuration before sampling.
fn validate_config<E>(config: DetailedBalanceConfig) -> Result<(), DetailedBalanceError<E>> {
    if config.samples == 0 {
        return Err(DetailedBalanceError::InvalidSamples {
            samples: config.samples,
        });
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(DetailedBalanceError::InvalidTolerance {
            tolerance: config.tolerance,
        });
    }
    if config.min_hits == 0 || config.min_hits > config.samples {
        return Err(DetailedBalanceError::InvalidMinHits {
            min_hits: config.min_hits,
            samples: config.samples,
        });
    }
    Ok(())
}

/// Cached endpoint target log-probabilities for the two states in a check.
#[derive(Debug, Clone, Copy)]
struct EndpointLogProbs {
    current: f64,
    proposed: f64,
}

/// Evaluate and validate both endpoint target log-probabilities before sampling
/// proposal transitions.
fn endpoint_log_probs<S, T, E>(
    current: &S,
    proposed: &S,
    target: &T,
) -> Result<EndpointLogProbs, DetailedBalanceError<E>>
where
    T: Target<S>,
{
    let current_log_prob = target.log_prob(current);
    check_log_prob(DetailedBalanceState::Current, current_log_prob)?;
    let proposed_log_prob = target.log_prob(proposed);
    check_log_prob(DetailedBalanceState::Proposed, proposed_log_prob)?;

    Ok(EndpointLogProbs {
        current: current_log_prob,
        proposed: proposed_log_prob,
    })
}

/// Check that an endpoint has a valid target log-probability.
const fn check_log_prob<E>(
    state: DetailedBalanceState,
    log_prob: f64,
) -> Result<(), DetailedBalanceError<E>> {
    if log_prob.is_nan() || log_prob == f64::INFINITY {
        Err(DetailedBalanceError::InvalidTargetLogProb { state, log_prob })
    } else {
        Ok(())
    }
}

/// Check that a proposal log q-ratio is valid.
const fn check_log_q_ratio<E>(
    direction: DetailedBalanceDirection,
    log_q_ratio: f64,
) -> Result<(), DetailedBalanceError<E>> {
    if log_q_ratio.is_nan() || log_q_ratio == f64::INFINITY {
        Err(DetailedBalanceError::InvalidLogQRatio {
            direction,
            log_q_ratio,
        })
    } else {
        Ok(())
    }
}

/// Check that empirical sampling produced enough exact hits.
const fn check_hits<E>(
    direction: DetailedBalanceDirection,
    hits: usize,
    min_hits: usize,
) -> Result<(), DetailedBalanceError<E>> {
    if hits >= min_hits {
        Ok(())
    } else {
        Err(DetailedBalanceError::InsufficientHits {
            direction,
            hits,
            min_hits,
        })
    }
}

/// Empirical hit and acceptance-weight totals for one transition direction.
#[derive(Debug, Clone, Copy)]
struct TransitionEstimate {
    hits: usize,
    weight_sum: f64,
    weight_square_sum: f64,
}

/// Shared sampling metadata for one transition-estimation pass.
#[derive(Debug, Clone, Copy)]
struct TransitionRequest {
    samples: usize,
    from_log_prob: f64,
    direction: DetailedBalanceDirection,
}

impl TransitionEstimate {
    /// Start an empty empirical transition estimate.
    const fn empty() -> Self {
        Self {
            hits: 0,
            weight_sum: 0.0,
            weight_square_sum: 0.0,
        }
    }

    /// Add one matching proposal hit with its Metropolis-Hastings acceptance
    /// probability.
    fn push(&mut self, acceptance_probability: f64) {
        self.hits += 1;
        self.weight_sum += acceptance_probability;
        self.weight_square_sum += acceptance_probability * acceptance_probability;
    }

    /// Estimate the log proposal probability from exact endpoint hits.
    fn log_proposal_probability(self, samples: usize) -> f64 {
        log_empirical_probability(self.hits, samples)
    }

    /// Estimate the log accepted transition probability from accumulated
    /// acceptance weights.
    fn log_transition_probability(self, samples: usize) -> f64 {
        log_empirical_weight(self.weight_sum, samples)
    }

    /// Approximate the standard error of the accepted log transition estimate
    /// using the observed acceptance-weight variance.
    fn log_standard_error(self, samples: usize) -> f64 {
        let mean = empirical_mean(self.weight_sum, samples);
        if mean <= 0.0 {
            return f64::INFINITY;
        }

        let second_moment = empirical_mean(self.weight_square_sum, samples);
        let variance = mean.mul_add(-mean, second_moment).max(0.0);
        let mean_standard_error = (variance / usize_to_f64(samples)).sqrt();
        mean_standard_error / mean
    }
}

/// Sample a by-value proposal repeatedly and accumulate hits that exactly match
/// the requested endpoint.
fn estimate_by_value_transition<S, P, R>(
    from: &S,
    to: &S,
    proposal: &P,
    rng: &mut R,
    samples: usize,
    acceptance_probability: f64,
) -> TransitionEstimate
where
    S: PartialEq,
    P: Proposal<S> + ?Sized,
    R: Rng + ?Sized,
{
    let mut estimate = TransitionEstimate::empty();
    for _ in 0..samples {
        if proposal.propose(from, rng).eq(to) {
            estimate.push(acceptance_probability);
        }
    }
    estimate
}

/// Sample an in-place proposal from cloned endpoints and score each exact hit
/// with the proposal's undo-token log ratio.
fn estimate_mut_transition<S, T, P, R>(
    from: &S,
    to: &S,
    target: &T,
    proposal: &P,
    rng: &mut R,
    request: TransitionRequest,
) -> Result<TransitionEstimate, DetailedBalanceError>
where
    S: Clone + PartialEq,
    T: Target<S>,
    P: ProposalMut<S> + ?Sized,
    R: Rng + ?Sized,
{
    let mut estimate = TransitionEstimate::empty();
    for _ in 0..request.samples {
        let mut candidate = from.clone();
        let Some(token) = proposal.propose_mut(&mut candidate, rng) else {
            continue;
        };
        if candidate.eq(to) {
            let proposed_log_prob = target.log_prob(&candidate);
            check_log_prob(DetailedBalanceState::Proposed, proposed_log_prob)?;
            let log_q_ratio = proposal.log_q_ratio(&candidate, &token);
            check_log_q_ratio(request.direction, log_q_ratio)?;
            estimate.push(acceptance_probability(
                request.direction,
                proposed_log_prob - request.from_log_prob + log_q_ratio,
            )?);
        }
    }
    Ok(estimate)
}

/// Sample delayed proposal plans and score the plans accepted by the caller's
/// transition predicate.
fn estimate_delayed_transition<S, T, P, R, F>(
    from: &S,
    target: &T,
    proposal: &mut P,
    rng: &mut R,
    request: TransitionRequest,
    plan_matches: &F,
) -> Result<TransitionEstimate, DetailedBalanceError<P::Error>>
where
    T: Target<S>,
    P: DelayedProposal<S> + ?Sized,
    R: Rng + ?Sized,
    F: Fn(&P::Plan) -> bool + ?Sized,
{
    let mut estimate = TransitionEstimate::empty();
    for _ in 0..request.samples {
        let Some(plan) =
            proposal
                .propose_plan(from, rng)
                .map_err(|source| DetailedBalanceError::Plan {
                    direction: request.direction,
                    source,
                })?
        else {
            continue;
        };
        if plan_matches(&plan) {
            let proposed_log_prob =
                proposal
                    .proposed_log_prob(from, &plan, target)
                    .map_err(|source| DetailedBalanceError::ProposedLogProb {
                        direction: request.direction,
                        source,
                    })?;
            check_log_prob(DetailedBalanceState::Proposed, proposed_log_prob)?;
            let log_q_ratio = proposal.log_q_ratio(from, &plan).map_err(|source| {
                DetailedBalanceError::LogQRatio {
                    direction: request.direction,
                    source,
                }
            })?;
            check_log_q_ratio(request.direction, log_q_ratio)?;
            estimate.push(acceptance_probability(
                request.direction,
                proposed_log_prob - request.from_log_prob + log_q_ratio,
            )?);
        }
    }
    Ok(estimate)
}

/// Combine forward and reverse empirical estimates into the public report and
/// turn insufficient hits or excessive residuals into typed errors.
fn finish_report<E>(
    endpoint_log_probs: EndpointLogProbs,
    forward: TransitionEstimate,
    reverse: TransitionEstimate,
    config: DetailedBalanceConfig,
) -> Result<DetailedBalanceReport, DetailedBalanceError<E>> {
    check_hits(
        DetailedBalanceDirection::Forward,
        forward.hits,
        config.min_hits,
    )?;
    check_hits(
        DetailedBalanceDirection::Reverse,
        reverse.hits,
        config.min_hits,
    )?;

    let forward_log_transition = forward.log_transition_probability(config.samples);
    let reverse_log_transition = reverse.log_transition_probability(config.samples);
    let forward_log_flow = endpoint_log_probs.current + forward_log_transition;
    let reverse_log_flow = endpoint_log_probs.proposed + reverse_log_transition;
    let log_balance_residual = log_residual(forward_log_flow, reverse_log_flow);
    let log_balance_standard_error = forward
        .log_standard_error(config.samples)
        .hypot(reverse.log_standard_error(config.samples));

    let report = DetailedBalanceReport {
        samples: config.samples,
        forward_hits: forward.hits,
        reverse_hits: reverse.hits,
        forward_log_proposal: forward.log_proposal_probability(config.samples),
        reverse_log_proposal: reverse.log_proposal_probability(config.samples),
        forward_log_transition,
        reverse_log_transition,
        log_balance_residual,
        log_balance_standard_error,
    };

    if !report.is_within_tolerance(config.tolerance) {
        return Err(DetailedBalanceError::Violation {
            residual: log_balance_residual,
            tolerance: config.tolerance,
            report,
        });
    }

    Ok(report)
}

/// Compute the log-flow difference while treating two impossible flows as
/// exactly balanced.
fn log_residual(forward_log_flow: f64, reverse_log_flow: f64) -> f64 {
    if forward_log_flow == f64::NEG_INFINITY && reverse_log_flow == f64::NEG_INFINITY {
        0.0
    } else {
        forward_log_flow - reverse_log_flow
    }
}

/// Convert a log Metropolis-Hastings ratio into an acceptance probability,
/// rejecting undefined ratios before they contaminate empirical estimates.
fn acceptance_probability<E>(
    direction: DetailedBalanceDirection,
    log_acceptance_ratio: f64,
) -> Result<f64, DetailedBalanceError<E>> {
    if log_acceptance_ratio.is_nan() {
        Err(DetailedBalanceError::InvalidLogAcceptanceRatio {
            direction,
            log_acceptance_ratio,
        })
    } else {
        Ok(log_acceptance_ratio.min(0.0).exp())
    }
}

/// Convert an exact hit count into an empirical log-proposal probability.
fn log_empirical_probability(hits: usize, samples: usize) -> f64 {
    log_empirical_weight(usize_to_f64(hits), samples)
}

/// Convert an accumulated transition weight into an empirical log probability.
fn log_empirical_weight(weight_sum: f64, samples: usize) -> f64 {
    empirical_mean(weight_sum, samples).ln()
}

/// Divide an accumulated empirical weight by the number of proposal samples.
fn empirical_mean(weight_sum: f64, samples: usize) -> f64 {
    weight_sum / usize_to_f64(samples)
}

/// Convert bounded sample counts into floating-point values for empirical
/// probability estimates.
#[expect(
    clippy::cast_precision_loss,
    reason = "empirical probabilities intentionally convert bounded sample counts to f64"
)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    struct TwoStateTarget;
    impl Target<bool> for TwoStateTarget {
        fn log_prob(&self, state: &bool) -> f64 {
            if *state { -2.0 } else { 0.0 }
        }
    }

    struct ImpossibleTarget;
    impl Target<bool> for ImpossibleTarget {
        fn log_prob(&self, _: &bool) -> f64 {
            f64::NEG_INFINITY
        }
    }

    struct Flip;
    impl Proposal<bool> for Flip {
        fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
            !current
        }
    }

    struct BadLogQ;
    impl Proposal<bool> for BadLogQ {
        fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
            !current
        }

        fn log_q_ratio(&self, _: &bool, _: &bool) -> f64 {
            1.0
        }
    }

    struct Stuck;
    impl Proposal<bool> for Stuck {
        fn propose<R: Rng + ?Sized>(&self, current: &bool, _: &mut R) -> bool {
            *current
        }
    }

    struct FlipMut;
    impl ProposalMut<bool> for FlipMut {
        type Undo = bool;

        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut bool, _: &mut R) -> Option<bool> {
            let old = *state;
            *state = !*state;
            Some(old)
        }

        fn undo(&self, state: &mut bool, token: bool) {
            *state = token;
        }
    }

    struct BadLogQMut;
    impl ProposalMut<bool> for BadLogQMut {
        type Undo = bool;

        fn propose_mut<R: Rng + ?Sized>(&self, state: &mut bool, _: &mut R) -> Option<bool> {
            let old = *state;
            *state = !*state;
            Some(old)
        }

        fn undo(&self, state: &mut bool, token: bool) {
            *state = token;
        }

        fn log_q_ratio(&self, _: &bool, _: &bool) -> f64 {
            1.0
        }
    }

    struct DelayedFlip;
    impl DelayedProposal<bool> for DelayedFlip {
        type Plan = bool;
        type Info = bool;
        type Error = Infallible;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            state: &bool,
            _: &mut R,
        ) -> Result<Option<bool>, Infallible> {
            Ok(Some(!*state))
        }

        fn proposed_log_prob<T: Target<bool>>(
            &self,
            _: &bool,
            plan: &bool,
            target: &T,
        ) -> Result<f64, Infallible> {
            Ok(target.log_prob(plan))
        }

        fn info(&self, plan: &bool) -> bool {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut bool,
            plan: bool,
            _: &mut R,
        ) -> Result<(), Infallible> {
            *state = plan;
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DelayedFailure {
        Plan,
        ProposedLogProb,
        LogQRatio,
    }

    impl fmt::Display for DelayedFailure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Plan => write!(f, "plan failed"),
                Self::ProposedLogProb => write!(f, "proposed log-probability failed"),
                Self::LogQRatio => write!(f, "log q-ratio failed"),
            }
        }
    }

    impl Error for DelayedFailure {}

    struct FailingDelayed {
        failure: DelayedFailure,
    }

    impl DelayedProposal<bool> for FailingDelayed {
        type Plan = bool;
        type Info = bool;
        type Error = DelayedFailure;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            state: &bool,
            _: &mut R,
        ) -> Result<Option<bool>, DelayedFailure> {
            if self.failure == DelayedFailure::Plan {
                Err(DelayedFailure::Plan)
            } else {
                Ok(Some(!*state))
            }
        }

        fn proposed_log_prob<T: Target<bool>>(
            &self,
            _: &bool,
            plan: &bool,
            target: &T,
        ) -> Result<f64, DelayedFailure> {
            if self.failure == DelayedFailure::ProposedLogProb {
                Err(DelayedFailure::ProposedLogProb)
            } else {
                Ok(target.log_prob(plan))
            }
        }

        fn log_q_ratio(&self, _: &bool, _: &bool) -> Result<f64, DelayedFailure> {
            if self.failure == DelayedFailure::LogQRatio {
                Err(DelayedFailure::LogQRatio)
            } else {
                Ok(0.0)
            }
        }

        fn info(&self, plan: &bool) -> bool {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut bool,
            plan: bool,
            _: &mut R,
        ) -> Result<(), DelayedFailure> {
            *state = plan;
            Ok(())
        }
    }

    struct ReverseFailingDelayed {
        failure: DelayedFailure,
    }

    impl DelayedProposal<bool> for ReverseFailingDelayed {
        type Plan = bool;
        type Info = bool;
        type Error = DelayedFailure;

        fn propose_plan<R: Rng + ?Sized>(
            &mut self,
            state: &bool,
            _: &mut R,
        ) -> Result<Option<bool>, DelayedFailure> {
            if *state && self.failure == DelayedFailure::Plan {
                Err(DelayedFailure::Plan)
            } else {
                Ok(Some(!*state))
            }
        }

        fn proposed_log_prob<T: Target<bool>>(
            &self,
            state: &bool,
            plan: &bool,
            target: &T,
        ) -> Result<f64, DelayedFailure> {
            if *state && self.failure == DelayedFailure::ProposedLogProb {
                Err(DelayedFailure::ProposedLogProb)
            } else {
                Ok(target.log_prob(plan))
            }
        }

        fn log_q_ratio(&self, state: &bool, _: &bool) -> Result<f64, DelayedFailure> {
            if *state && self.failure == DelayedFailure::LogQRatio {
                Err(DelayedFailure::LogQRatio)
            } else {
                Ok(0.0)
            }
        }

        fn info(&self, plan: &bool) -> bool {
            *plan
        }

        fn commit<R: Rng + ?Sized>(
            &mut self,
            state: &mut bool,
            plan: bool,
            _: &mut R,
        ) -> Result<(), DelayedFailure> {
            *state = plan;
            Ok(())
        }
    }

    /// Return a fast deterministic configuration for exact two-state tests.
    fn small_config() -> DetailedBalanceConfig {
        DetailedBalanceConfig {
            samples: 128,
            tolerance: 1e-12,
            min_hits: 1,
        }
    }

    #[test]
    fn verifies_symmetric_two_state_transition() {
        let mut rng = StdRng::seed_from_u64(42);
        let report = verify_detailed_balance(
            &false,
            &true,
            &TwoStateTarget,
            &Flip,
            &mut rng,
            small_config(),
        )
        .unwrap();

        assert_eq!(report.forward_hits, 128);
        assert_eq!(report.reverse_hits, 128);
        assert!(report.is_within_tolerance(1e-12));
        assert!(report.z_score().is_some_and(|score| score.abs() < 1e-6));
    }

    #[test]
    fn verifies_mutable_two_state_transition() {
        let mut rng = StdRng::seed_from_u64(42);
        let report = verify_detailed_balance_mut(
            &false,
            &true,
            &TwoStateTarget,
            &FlipMut,
            &mut rng,
            small_config(),
        )
        .unwrap();

        assert_eq!(report.forward_hits, 128);
        assert_eq!(report.reverse_hits, 128);
        assert!(report.is_within_tolerance(1e-12));
    }

    #[test]
    fn verifies_delayed_two_state_transition() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = DelayedFlip;
        let report = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap();

        assert_eq!(report.forward_hits, 128);
        assert_eq!(report.reverse_hits, 128);
        assert!(report.is_within_tolerance(1e-12));
    }

    #[test]
    fn rejects_bad_log_q_ratio() {
        let mut rng = StdRng::seed_from_u64(42);
        let err = verify_detailed_balance(
            &false,
            &true,
            &TwoStateTarget,
            &BadLogQ,
            &mut rng,
            small_config(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::Violation {
                residual,
                tolerance: 1e-12,
                ..
            } if (residual - 1.0).abs() < 1e-12
        ));
    }

    #[test]
    fn mutable_rejects_bad_log_q_ratio() {
        let mut rng = StdRng::seed_from_u64(42);
        let err = verify_detailed_balance_mut(
            &false,
            &true,
            &TwoStateTarget,
            &BadLogQMut,
            &mut rng,
            small_config(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::Violation {
                residual,
                tolerance: 1e-12,
                ..
            } if (residual - 1.0).abs() < 1e-12
        ));
    }

    #[test]
    fn delayed_reports_plan_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = FailingDelayed {
            failure: DelayedFailure::Plan,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::Plan {
                direction: DetailedBalanceDirection::Forward,
                source: DelayedFailure::Plan,
            }
        ));
    }

    #[test]
    fn delayed_reports_proposed_log_prob_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = FailingDelayed {
            failure: DelayedFailure::ProposedLogProb,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::ProposedLogProb {
                direction: DetailedBalanceDirection::Forward,
                source: DelayedFailure::ProposedLogProb,
            }
        ));
    }

    #[test]
    fn delayed_reports_log_q_ratio_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = FailingDelayed {
            failure: DelayedFailure::LogQRatio,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::LogQRatio {
                direction: DetailedBalanceDirection::Forward,
                source: DelayedFailure::LogQRatio,
            }
        ));
    }

    #[test]
    fn delayed_reports_reverse_plan_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = ReverseFailingDelayed {
            failure: DelayedFailure::Plan,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::Plan {
                direction: DetailedBalanceDirection::Reverse,
                source: DelayedFailure::Plan,
            }
        ));
    }

    #[test]
    fn delayed_reports_reverse_proposed_log_prob_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = ReverseFailingDelayed {
            failure: DelayedFailure::ProposedLogProb,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::ProposedLogProb {
                direction: DetailedBalanceDirection::Reverse,
                source: DelayedFailure::ProposedLogProb,
            }
        ));
    }

    #[test]
    fn delayed_reports_reverse_log_q_ratio_errors() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = ReverseFailingDelayed {
            failure: DelayedFailure::LogQRatio,
        };
        let err = verify_detailed_balance_delayed(
            &false,
            &true,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
            (|plan| *plan, |plan| !*plan),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::LogQRatio {
                direction: DetailedBalanceDirection::Reverse,
                source: DelayedFailure::LogQRatio,
            }
        ));
    }

    #[test]
    fn reports_insufficient_hits() {
        let mut rng = StdRng::seed_from_u64(42);
        let err = verify_detailed_balance(
            &false,
            &true,
            &TwoStateTarget,
            &Stuck,
            &mut rng,
            small_config(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InsufficientHits {
                direction: DetailedBalanceDirection::Forward,
                hits: 0,
                min_hits: 1,
            }
        ));
    }

    #[test]
    fn reports_invalid_log_acceptance_ratio() {
        let mut rng = StdRng::seed_from_u64(42);
        let err = verify_detailed_balance(
            &false,
            &true,
            &ImpossibleTarget,
            &Flip,
            &mut rng,
            small_config(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidLogAcceptanceRatio {
                direction: DetailedBalanceDirection::Forward,
                log_acceptance_ratio,
            } if log_acceptance_ratio.is_nan()
        ));
    }

    #[test]
    fn batch_reports_all_failures() {
        let mut rng = StdRng::seed_from_u64(42);
        let pairs = [(false, true), (true, false)];
        let batch = verify_detailed_balance_many(
            pairs.iter().map(|(current, proposed)| (current, proposed)),
            &TwoStateTarget,
            &BadLogQ,
            &mut rng,
            small_config(),
        )
        .unwrap();

        assert!(!batch.is_success());
        assert_eq!(batch.failures.len(), 2);
        assert_eq!(batch.failures[0].index, 0);
        assert_eq!(batch.failures[1].index, 1);
    }

    #[test]
    fn delayed_batch_reports_successes() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut proposal = DelayedFlip;
        let forward = |plan: &bool| *plan;
        let reverse = |plan: &bool| !*plan;
        let transitions = [DetailedBalanceDelayedTransition {
            current: &false,
            proposed: &true,
            forward_plan_matches: &forward,
            reverse_plan_matches: &reverse,
        }];

        let batch = verify_detailed_balance_delayed_many(
            transitions,
            &TwoStateTarget,
            &mut proposal,
            &mut rng,
            small_config(),
        )
        .unwrap();

        assert!(batch.is_success());
        assert_eq!(batch.reports.len(), 1);
    }

    #[test]
    fn batch_rejects_invalid_config_without_fake_transition_index() {
        let mut rng = StdRng::seed_from_u64(42);
        let err = verify_detailed_balance_many(
            core::iter::empty::<(&bool, &bool)>(),
            &TwoStateTarget,
            &Flip,
            &mut rng,
            DetailedBalanceConfig {
                samples: 0,
                tolerance: 1e-12,
                min_hits: 1,
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidSamples { samples: 0 }
        ));
    }

    #[test]
    fn rejects_invalid_config() {
        let err: DetailedBalanceError = validate_config(DetailedBalanceConfig {
            samples: 0,
            tolerance: 0.0,
            min_hits: 1,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidSamples { samples: 0 }
        ));

        let err = validate_config::<Infallible>(DetailedBalanceConfig {
            samples: 1,
            tolerance: -1.0,
            min_hits: 1,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidTolerance { tolerance }
                if tolerance.to_bits() == (-1.0_f64).to_bits()
        ));

        let err = validate_config::<Infallible>(DetailedBalanceConfig {
            samples: 1,
            tolerance: f64::NAN,
            min_hits: 1,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidTolerance { tolerance } if tolerance.is_nan()
        ));

        let err = validate_config::<Infallible>(DetailedBalanceConfig {
            samples: 1,
            tolerance: f64::INFINITY,
            min_hits: 1,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidTolerance { tolerance }
                if tolerance.is_infinite() && tolerance.is_sign_positive()
        ));

        let err = validate_config::<Infallible>(DetailedBalanceConfig {
            samples: 1,
            tolerance: 0.0,
            min_hits: 0,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidMinHits {
                min_hits: 0,
                samples: 1,
            }
        ));

        let err = validate_config::<Infallible>(DetailedBalanceConfig {
            samples: 1,
            tolerance: 0.0,
            min_hits: 2,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            DetailedBalanceError::InvalidMinHits {
                min_hits: 2,
                samples: 1,
            }
        ));
    }
}
