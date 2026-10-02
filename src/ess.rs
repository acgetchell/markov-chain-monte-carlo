//! Multi-chain effective sample sizes and scalar Monte Carlo standard errors.
//!
//! These estimators follow the split-chain algorithm in `ArviZ` 0.22.0, with
//! explicit unavailable results in place of constant-data sentinels. Sampling
//! assumptions and numerical conventions are documented on [`EssEstimate`].

use std::{error::Error, fmt, time::Duration};

use statrs::function::beta::beta_reg;

use crate::{
    SplitRhatError,
    convergence::{RankedRhatInput, SplitRhatInput},
    numerics::{compensated_sum, count_as_f64},
};

/// Invalid input or an unavailable multi-chain ESS/MCSE estimate.
///
/// [`Self::Input`] and [`Self::InvalidProbability`] reject the request. Other
/// variants describe valid observations whose requested diagnostic is unavailable.
/// None is evidence of mixing; retain the reason alongside observable/chain IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MonteCarloError {
    /// Shared chain-count, shape, finite-draw, or supported-count rejection.
    Input(SplitRhatError),
    /// Quantile probabilities must be finite and strictly between zero and one.
    InvalidProbability,
    /// All retained observations are exactly equal, including signed-zero ties.
    ConstantSamples,
    /// Every retained half is constant, although half locations can differ.
    NoWithinChainVariation,
    /// A quantile's retained indicators are all zero or all one.
    DegenerateIndicator,
    /// Variation in a half was lost during scaled moment calculation.
    #[non_exhaustive]
    UnresolvedVariance {
        /// Position of the original chain in the supplied slice.
        chain_index: usize,
        /// Zero for the first half, one for the last half.
        half: usize,
    },
    /// A numerical result or beta-quantile calculation is unresolved/nonfinite.
    NumericalFailure,
    /// Quantile uncertainty bounds coincide after order-statistic selection.
    ///
    /// This can occur for tied/discrete draws or insufficient resolution. A
    /// collapsed interval does not establish zero Monte Carlo uncertainty.
    CollapsedQuantileInterval,
}

impl fmt::Display for MonteCarloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => error.fmt(f),
            Self::InvalidProbability => {
                f.write_str("quantile probability must be finite and in (0, 1)")
            }
            Self::ConstantSamples => {
                f.write_str("ESS is unavailable for constant retained samples")
            }
            Self::NoWithinChainVariation => {
                f.write_str("ESS is unavailable without variation in any split half")
            }
            Self::DegenerateIndicator => {
                f.write_str("quantile ESS is unavailable for constant retained indicators")
            }
            Self::UnresolvedVariance { chain_index, half } => write!(
                f,
                "ESS variance is unresolved in chain {chain_index}, half {half}"
            ),
            Self::NumericalFailure => {
                f.write_str("ESS or MCSE arithmetic is numerically unresolved")
            }
            Self::CollapsedQuantileInterval => {
                f.write_str("quantile MCSE interval collapsed; uncertainty is unavailable")
            }
        }
    }
}

impl Error for MonteCarloError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SplitRhatError> for MonteCarloError {
    fn from(error: SplitRhatError) -> Self {
        Self::Input(error)
    }
}

/// Quantity whose split-chain effective sample size is being estimated.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum EssEstimator {
    /// Raw-scale scalar mean; requires finite population variance for interpretation.
    Mean,
    /// Pooled rank-normalized split draws; this is the bulk ESS estimator.
    Bulk,
    /// Indicators at the pooled type-7 quantile of all original draws.
    ///
    /// The probability is checked by [`EssEstimate::estimate`], and must be in
    /// `(0, 1)`. This is an indicator ESS, not ESS of a filtered tail subset.
    Quantile(f64),
}

/// Positive finite multi-chain ESS and its estimator/sample metadata.
///
/// Supply at least two comparable, independently initialized post-warmup chains
/// of equal length, with at least four finite draws each. Keep recording cadence,
/// units, observable, and warmup policy consistent; preserve repeated rejected
/// states. Scalar slices cannot verify these sampling assumptions. Every original
/// draw is checked, including odd middle draws omitted from split moments.
///
/// Each chain contributes first/last `floor(N/2)` halves. Mean ESS uses raw
/// retained values; bulk ESS pools and rank-normalizes those values with the
/// same average-tie/normal-score conventions as [`crate::RankNormalizedSplitRhat`].
/// Quantile cutoffs use all original draws before splitting, with linear type-7
/// interpolation and `x <= cutoff` indicators. Endpoints are unsupported.
///
/// Biased within-half autocovariances (divisor `n`, at every lag) are combined
/// with unbiased half variances and between-half mean variation. Geyer's initial
/// positive/monotone sequence, the extra final positive even lag, and finite-lag
/// boundary follow [ArviZ 0.22.0](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py).
/// The estimated time is bounded below by `1/log10(S)`, so ESS can exceed `S`
/// but is capped at `S * log10(S)`. [`Self::is_regularized`] reports this cap.
/// This regularization differs from the existing single-chain mean ESS contract.
///
/// Constant individual halves are allowed, as required for sparse quantile
/// indicators. A constant retained pool or zero variation in every half remains
/// unavailable. Scaled centering and compensated sums avoid overflow; unresolved
/// variation and nonrepresentable final values return typed errors. There is no
/// absolute near-constant tolerance or constant-data passing sentinel: unlike
/// `ArviZ`'s raw ESS span-below-`1e-15` shortcut, changes of units preserve
/// computable variation here. Extreme-range results can intentionally differ.
///
/// Autocovariances are computed on demand through the reference stopping rule.
/// With `K` inspected lags, cost is `O(S * K)` time and `O(S)` space, worst-case
/// `K = floor(N/2)`. Counts are bounded by the shared `2^50` rank/count contract.
/// Estimates require sufficiently long, stationary, reversible traces and
/// summable correlations. A finite output does not prove convergence, adequate
/// information, target correctness, or existence of population moments.
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::prelude::{EssEstimate, EssEstimator, MonteCarloError};
/// let a = [0., 1., 3., 2., 1., 0., 2., 3.];
/// let b = [2., 0., 1., 3., 0., 2., 3., 1.];
/// let ess = EssEstimate::estimate(&[&a, &b], EssEstimator::Bulk)?;
/// assert_eq!(ess.sample_count(), 16);
/// assert_eq!(ess.samples_per_split_chain(), 4);
/// assert!(ess.value() > 0.0);
/// # Ok::<(), MonteCarloError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct EssEstimate {
    value: f64,
    estimator: EssEstimator,
    counts: SampleCounts,
    regularized: bool,
}

impl EssEstimate {
    /// Estimate a selected quantity without mutating or retaining input buffers.
    ///
    /// # Errors
    ///
    /// Quantile probability is checked first. [`MonteCarloError::Input`] then
    /// preserves the ordered checks on [`crate::RankNormalizedSplitRhat::estimate`].
    /// Constant pools, degenerate indicators, absent within-half variation,
    /// unresolved variances, and final arithmetic failures remain unavailable as
    /// documented on [`MonteCarloError`]. No missing estimate becomes a sentinel.
    pub fn estimate(chains: &[&[f64]], estimator: EssEstimator) -> Result<Self, MonteCarloError> {
        check_estimator(estimator)?;
        let input = SplitRhatInput::parse(chains)?.ranked()?;
        estimate_input(&input, estimator)
    }

    /// Positive finite effective sample count.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Estimator identity, including the probability for a quantile ESS.
    #[must_use]
    pub const fn estimator(self) -> EssEstimator {
        self.estimator
    }

    /// Total retained draws used by split autocovariances and ESS/S.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.counts.retained
    }

    /// Total original draws, including omitted odd middle draws.
    #[must_use]
    pub const fn original_sample_count(self) -> usize {
        self.counts.chains * self.counts.length
    }

    /// Original chain count, before splitting.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.counts.chains
    }

    /// Original draws per chain.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.counts.length
    }

    /// Retained draws per split half.
    #[must_use]
    pub const fn samples_per_split_chain(self) -> usize {
        self.counts.length / 2
    }

    /// ESS/S, using the actual retained count; may exceed one.
    #[must_use]
    pub fn relative(self) -> f64 {
        self.value / count_as_f64(self.sample_count())
    }

    /// Whether the reference antithetic time bound changed the estimate.
    #[must_use]
    pub const fn is_regularized(self) -> bool {
        self.regularized
    }

    /// ESS per measured workload second, with explicit timing availability.
    ///
    /// # Errors
    ///
    /// Rejects missing timing or original chain/count mismatches. The timing
    /// constructor rejects zero duration. Callers must measure the same workload;
    /// matching counts cannot prove trace identity. Parallel runs use actual wall
    /// time, not summed chain durations, and prefixes need their own timing.
    pub fn per_second(
        self,
        timing: Option<&DiagnosticTiming>,
    ) -> Result<f64, DiagnosticTimingError> {
        Ok(self.value / self.counts.elapsed(timing)?.as_secs_f64())
    }
}

/// Both 0.05/0.95 quantile ESS components and their available minimum.
///
/// Input failures reject the report. Valid draws preserve each component's
/// success or typed unavailability independently. The minimum is available only
/// when both succeed; it never discards a degenerate tail component.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct TailEss {
    lower: Result<EssEstimate, MonteCarloError>,
    upper: Result<EssEstimate, MonteCarloError>,
    counts: SampleCounts,
}

impl TailEss {
    /// Estimate both tail components with the input conventions of [`EssEstimate`].
    ///
    /// # Errors
    ///
    /// Returns [`MonteCarloError::Input`] for invalid shape/values/counts. Component
    /// failures remain in [`Self::lower`] and [`Self::upper`] on a successful report.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::{TailEss, MonteCarloError};
    /// let counts = [0., 1., 0., 1., 0., 1., 0., 1.];
    /// let tail = TailEss::estimate(&[&counts, &counts])?;
    /// assert!(tail.lower().is_ok());
    /// assert_eq!(tail.upper(), Err(MonteCarloError::DegenerateIndicator));
    /// assert_eq!(tail.value(), None);
    /// # Ok::<(), MonteCarloError>(())
    /// ```
    pub fn estimate(chains: &[&[f64]]) -> Result<Self, MonteCarloError> {
        let input = SplitRhatInput::parse(chains)?.ranked()?;
        let (lower_cutoff, upper_cutoff) = {
            let sorted = sorted_original(&input);
            (quantile(&sorted, 0.05), quantile(&sorted, 0.95))
        };
        Ok(Self {
            lower: estimate_quantile(&input, 0.05, lower_cutoff),
            upper: estimate_quantile(&input, 0.95, upper_cutoff),
            counts: SampleCounts::from_input(&input),
        })
    }

    /// The 0.05 indicator ESS or its unavailable reason.
    ///
    /// # Errors
    ///
    /// Preserves the component errors documented on [`EssEstimate::estimate`].
    pub const fn lower(self) -> Result<EssEstimate, MonteCarloError> {
        self.lower
    }

    /// The 0.95 indicator ESS or its unavailable reason.
    ///
    /// # Errors
    ///
    /// Preserves the component errors documented on [`EssEstimate::estimate`].
    pub const fn upper(self) -> Result<EssEstimate, MonteCarloError> {
        self.upper
    }

    /// Borrow the smaller successful component; absent if either failed.
    #[must_use]
    pub fn minimum(&self) -> Option<&EssEstimate> {
        let lower = self.lower.as_ref().ok()?;
        let upper = self.upper.as_ref().ok()?;
        Some(if lower.value <= upper.value {
            lower
        } else {
            upper
        })
    }

    /// Minimum effective sample count, available only with both components.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        self.minimum().map(|ess| ess.value)
    }

    /// Total retained draws, including when components are unavailable.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.counts.retained
    }

    /// Total original draws, including omitted odd middle draws.
    #[must_use]
    pub const fn original_sample_count(self) -> usize {
        self.counts.chains * self.counts.length
    }

    /// Original chain count, including when components are unavailable.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.counts.chains
    }

    /// Original draws per chain, including any odd middle draw.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.counts.length
    }

    /// Retained draws per split half, including when components are unavailable.
    #[must_use]
    pub const fn samples_per_split_chain(self) -> usize {
        self.counts.length / 2
    }

    /// Tail ESS/S when both components are available.
    #[must_use]
    pub fn relative(&self) -> Option<f64> {
        self.minimum().map(|ess| ess.relative())
    }

    /// Available tail ESS per measured second, under [`EssEstimate::per_second`]'s contract.
    ///
    /// # Errors
    ///
    /// Rejects missing/mismatched timing even if a component is unavailable.
    pub fn per_second(
        &self,
        timing: Option<&DiagnosticTiming>,
    ) -> Result<Option<f64>, DiagnosticTimingError> {
        let seconds = self.counts.elapsed(timing)?.as_secs_f64();
        Ok(self.value().map(|value| value / seconds))
    }
}

/// Mean Monte Carlo standard error in the observable's original units.
///
/// Computes pooled sample standard deviation of **all original draws** (variance
/// divisor `T-1`), divided by the square root of raw-scale split mean ESS,
/// matching `ArviZ`'s `mcse(mean)`.
/// An odd middle draw therefore contributes to the variance but not ESS moments;
/// [`Self::effective_sample_size`] exposes both original and retained counts.
/// Finite population mean/variance and the sampling assumptions on [`EssEstimate`]
/// are necessary for interpretation. Finite output for heavy tails does not
/// establish existence of those moments. This is distinct from posterior spread
/// and from [`crate::BinningAnalysis`]'s blocked mean-error estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MeanMcse {
    value: f64,
    ess: EssEstimate,
}

impl MeanMcse {
    /// Estimate original-unit mean error from borrowed comparable chains.
    ///
    /// # Errors
    ///
    /// Preserves raw mean ESS errors. [`MonteCarloError::NumericalFailure`] also
    /// rejects a nonpositive or nonfinite final error; no zero uncertainty is invented.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::{MeanMcse, MonteCarloError};
    /// let a = [0., 1., 3., 2., 1., 0., 2., 3.];
    /// let b = [2., 0., 1., 3., 0., 2., 3., 1.];
    /// let error = MeanMcse::estimate(&[&a, &b])?;
    /// assert!(error.value() > 0.0);
    /// assert_eq!(error.effective_sample_size().original_sample_count(), 16);
    /// # Ok::<(), MonteCarloError>(())
    /// ```
    pub fn estimate(chains: &[&[f64]]) -> Result<Self, MonteCarloError> {
        let input = SplitRhatInput::parse(chains)?.ranked()?;
        let ess = estimate_input(&input, EssEstimator::Mean)?;
        let mut normalized: Vec<_> = chains
            .iter()
            .flat_map(|chain| chain.iter().copied())
            .collect();
        let scaled = ScaledSamples::new(&normalized)?;
        let anchor = normalized[0];
        for value in &mut normalized {
            *value = scaled.difference(*value, anchor) / scaled.scale;
        }
        let mean = compensated_sum(normalized.iter().copied()) / count_as_f64(normalized.len());
        let variance = compensated_sum(normalized.iter().map(|&x| (x - mean).powi(2)))
            / count_as_f64(normalized.len() - 1);
        let factor = if scaled.halve { 2.0 } else { 1.0 };
        let value = scaled.scale * (factor * (variance / ess.value).sqrt());
        positive(value)?;
        Ok(Self { value, ess })
    }

    /// Positive finite mean MCSE in original observable units.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Borrow the raw-scale mean ESS and its sample metadata.
    pub const fn effective_sample_size(&self) -> &EssEstimate {
        &self.ess
    }
}

/// Quantile MCSE using the beta/order-statistic method of Vehtari et al. (2021).
///
/// Uses indicator ESS at `p` and beta probabilities `0.1586553`/`0.8413447`
/// with shapes `ESS*p+1` and `ESS*(1-p)+1`. Selects original pooled order
/// statistics at `max(floor(T*a_low),1)` and `min(ceil(T*a_high),T)` (one based),
/// where `T` includes all original draws, including odd middle draws, and
/// `a_low`/`a_high` are the beta quantiles at those probabilities. Returns half
/// their separation, following `ArviZ` 0.22.0's conventions.
/// [`Self::interval`] exposes the selected bounds; they are
/// an approximate central uncertainty interval, not posterior spread or a
/// simultaneous coverage guarantee. [`Self::quantile`] uses linear type 7.
///
/// Tied/count observations can collapse the bounds despite varying indicators;
/// that condition is unavailable, never zero MCSE. Discrete quantiles lack the
/// usual continuous-density interpretation; callers must assess that limitation.
/// Beta inversion uses bounded bisection of statrs' regularized beta function,
/// with a checked finite CDF residual rather than unbounded Newton iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct QuantileMcse {
    value: f64,
    quantile: f64,
    interval: [f64; 2],
    ess: EssEstimate,
}

impl QuantileMcse {
    /// Estimate quantile uncertainty for a finite probability in `(0, 1)`.
    ///
    /// # Errors
    ///
    /// Preserves quantile ESS/input errors. Unresolved beta inversion or final
    /// arithmetic returns [`MonteCarloError::NumericalFailure`]; coinciding bounds
    /// return [`MonteCarloError::CollapsedQuantileInterval`].
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::{QuantileMcse, MonteCarloError};
    /// let a = [0., 1., 3., 2., 1., 0., 2., 3.];
    /// let b = [2., 0., 1., 3., 0., 2., 3., 1.];
    /// let median_error = QuantileMcse::estimate(&[&a, &b], 0.5)?;
    /// assert_eq!(median_error.quantile(), 1.5);
    /// assert!(median_error.value() > 0.0);
    /// # Ok::<(), MonteCarloError>(())
    /// ```
    pub fn estimate(chains: &[&[f64]], probability: f64) -> Result<Self, MonteCarloError> {
        check_estimator(EssEstimator::Quantile(probability))?;
        let input = SplitRhatInput::parse(chains)?.ranked()?;
        let sorted = sorted_original(&input);
        let ess = estimate_quantile(&input, probability, quantile(&sorted, probability))?;
        let a = ess.value.mul_add(probability, 1.0);
        let b = ess.value.mul_add(1.0 - probability, 1.0);
        let lower_probability = beta_quantile(a, b, 0.158_655_3)?;
        let upper_probability = beta_quantile(a, b, 0.841_344_7)?;
        let size = count_as_f64(sorted.len());
        let lower = sorted[order_index((lower_probability * size).floor().max(1.0) - 1.0)];
        let upper = sorted[order_index((upper_probability * size).ceil().min(size) - 1.0)];
        if lower >= upper {
            return Err(MonteCarloError::CollapsedQuantileInterval);
        }
        let difference = upper - lower;
        let value = if difference.is_finite() {
            difference * 0.5
        } else {
            lower.mul_add(-0.5, upper * 0.5)
        };
        positive(value)?;
        Ok(Self {
            value,
            quantile: quantile(&sorted, probability),
            interval: [lower, upper],
            ess,
        })
    }

    /// Positive finite quantile MCSE in original observable units.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Pooled original-draw type-7 quantile at the requested probability.
    #[must_use]
    pub const fn quantile(self) -> f64 {
        self.quantile
    }

    /// Selected lower/upper original-unit uncertainty bounds.
    #[must_use]
    pub const fn interval(self) -> [f64; 2] {
        self.interval
    }

    /// Borrow the indicator ESS, probability identity, and retained/original counts.
    pub const fn effective_sample_size(&self) -> &EssEstimate {
        &self.ess
    }
}

/// Timing request rejected before calculating an ESS rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticTimingError {
    /// No timing was supplied for the analyzed workload.
    MissingTiming,
    /// The measured clock could not resolve a positive duration.
    ZeroDuration,
    /// A timing declaration needs positive chain and original draw counts.
    InvalidCounts,
    /// Timing and estimator declare different original workloads.
    #[non_exhaustive]
    CountMismatch {
        /// Estimator's original chain count.
        expected_chains: usize,
        /// Estimator's original length per chain.
        expected_samples_per_chain: usize,
        /// Timing declaration's original chain count.
        actual_chains: usize,
        /// Timing declaration's original length per chain.
        actual_samples_per_chain: usize,
    },
}

impl fmt::Display for DiagnosticTimingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTiming => f.write_str("ESS rate requires timing of the analyzed workload"),
            Self::ZeroDuration => f.write_str("ESS rate requires a nonzero duration"),
            Self::InvalidCounts => f.write_str("timed chain and draw counts must be positive"),
            Self::CountMismatch {
                expected_chains,
                expected_samples_per_chain,
                actual_chains,
                actual_samples_per_chain,
            } => write!(
                f,
                "timing covers {actual_chains} chains of {actual_samples_per_chain} draws; estimator uses {expected_chains} chains of {expected_samples_per_chain} draws"
            ),
        }
    }
}

impl Error for DiagnosticTimingError {}

/// Positive measured duration with declared original production chain/draw counts.
///
/// This type checks declared counts, not trace identity or timing provenance.
/// Measure the analyzed production workload, excluding unrelated warmup/analysis.
/// Parallel workloads use wall time, not summed chain times; record timing scope
/// in exported evidence. Do not reuse whole-run time for an untimed prefix.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use markov_chain_monte_carlo::{DiagnosticTiming, EssEstimate, EssEstimator};
/// let a = [0., 1., 3., 2., 1., 0., 2., 3.];
/// let b = [2., 0., 1., 3., 0., 2., 3., 1.];
/// let ess = EssEstimate::estimate(&[&a, &b], EssEstimator::Mean)?;
/// // Illustrative caller-supplied measurement of those two production chains.
/// let measured_wall_time = Duration::from_secs(2);
/// let rate = DiagnosticTiming::try_new(measured_wall_time, 2, 8)
///     .and_then(|timing| ess.per_second(Some(&timing)));
/// assert_eq!(rate, Ok(ess.value() / 2.0));
/// assert!(ess.per_second(None).is_err());
/// # Ok::<(), markov_chain_monte_carlo::MonteCarloError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosticTiming {
    elapsed: Duration,
    chains: usize,
    length: usize,
}

impl DiagnosticTiming {
    /// Accept a measured duration and positive original workload counts.
    ///
    /// # Errors
    ///
    /// Rejects zero duration before zero chain/draw counts.
    pub const fn try_new(
        elapsed: Duration,
        chain_count: usize,
        samples_per_chain: usize,
    ) -> Result<Self, DiagnosticTimingError> {
        if elapsed.is_zero() {
            return Err(DiagnosticTimingError::ZeroDuration);
        }
        if chain_count == 0 || samples_per_chain == 0 {
            return Err(DiagnosticTimingError::InvalidCounts);
        }
        Ok(Self {
            elapsed,
            chains: chain_count,
            length: samples_per_chain,
        })
    }

    /// Declared measured workload wall time.
    #[must_use]
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }

    /// Declared original chain count.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.chains
    }

    /// Declared original production draws per chain.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.length
    }
}

/// Checked original and retained counts shared by ESS and timing comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleCounts {
    chains: usize,
    length: usize,
    retained: usize,
}

impl SampleCounts {
    const fn from_input(input: &RankedRhatInput<'_>) -> Self {
        Self {
            chains: input.chains().len(),
            length: input.samples_per_chain(),
            retained: input.retained_count(),
        }
    }

    fn elapsed(self, timing: Option<&DiagnosticTiming>) -> Result<Duration, DiagnosticTimingError> {
        let timing = timing.ok_or(DiagnosticTimingError::MissingTiming)?;
        if self.chains != timing.chains || self.length != timing.length {
            return Err(DiagnosticTimingError::CountMismatch {
                expected_chains: self.chains,
                expected_samples_per_chain: self.length,
                actual_chains: timing.chains,
                actual_samples_per_chain: timing.length,
            });
        }
        Ok(timing.elapsed)
    }
}

/// Carry accepted probability bounds into the quantile calculation.
fn check_estimator(estimator: EssEstimator) -> Result<(), MonteCarloError> {
    if let EssEstimator::Quantile(p) = estimator
        && (!p.is_finite() || p <= 0.0 || p >= 1.0)
    {
        return Err(MonteCarloError::InvalidProbability);
    }
    Ok(())
}

/// Original-chain/first-half/last-half order, matching shared pooled ranking.
fn retained<'a>(input: &'a RankedRhatInput<'_>) -> impl Iterator<Item = f64> + 'a {
    let n = input.samples_per_chain() / 2;
    input
        .chains()
        .iter()
        .flat_map(move |chain| chain[..n].iter().chain(&chain[chain.len() - n..]))
        .copied()
}

/// Estimate only after the shared parser has established storage/count bounds.
fn estimate_input(
    input: &RankedRhatInput<'_>,
    estimator: EssEstimator,
) -> Result<EssEstimate, MonteCarloError> {
    match estimator {
        EssEstimator::Mean => {
            estimate_values(&retained(input).collect::<Vec<_>>(), input, estimator)
        }
        EssEstimator::Bulk => estimate_values(&input.normal_scores(), input, estimator),
        EssEstimator::Quantile(p) => {
            estimate_quantile(input, p, quantile(&sorted_original(input), p))
        }
    }
}

/// Indicators retain sparse constant halves, but reject an entirely constant pool.
#[expect(
    clippy::float_cmp,
    reason = "indicators are exactly represented zeros/ones"
)]
fn estimate_quantile(
    input: &RankedRhatInput<'_>,
    p: f64,
    cutoff: f64,
) -> Result<EssEstimate, MonteCarloError> {
    let indicators: Vec<_> = retained(input).map(|x| f64::from(x <= cutoff)).collect();
    if indicators.iter().all(|&x| x == indicators[0]) {
        return Err(MonteCarloError::DegenerateIndicator);
    }
    estimate_values(&indicators, input, EssEstimator::Quantile(p))
}

/// Common shifted scaling; physical units are restored only for mean MCSE.
struct ScaledSamples {
    scale: f64,
    halve: bool,
}

impl ScaledSamples {
    /// Choose one difference scale for a nonempty, finite, accepted sample pool.
    ///
    /// Halve every sample when the full range would overflow. Sharing this
    /// choice and scale across halves preserves their relative locations and
    /// variances; mean MCSE restores physical units with the scale and a factor
    /// of two when halving was needed.
    fn new(values: &[f64]) -> Result<Self, MonteCarloError> {
        let origin = values[0];
        let (low, high) = values
            .iter()
            .fold((origin, origin), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        let halve = !(high - low).is_finite();
        let difference = |x: f64| {
            if halve {
                origin.mul_add(-0.5, x * 0.5)
            } else {
                x - origin
            }
        };
        let scale = values
            .iter()
            .map(|&x| difference(x).abs())
            .fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Err(MonteCarloError::ConstantSamples);
        }
        positive(scale)?;
        Ok(Self { scale, halve })
    }

    /// Subtract a local anchor using the pool's common halving convention.
    /// Both values must belong to the accepted pool; callers divide by `scale`.
    fn difference(&self, value: f64, anchor: f64) -> f64 {
        if self.halve {
            anchor.mul_add(-0.5, value * 0.5)
        } else {
            value - anchor
        }
    }
}

/// Direct biased autocovariances on centered halves, with between-half information.
#[expect(
    clippy::float_cmp,
    reason = "detect represented constancy before variance arithmetic"
)]
fn estimate_values(
    values: &[f64],
    input: &RankedRhatInput<'_>,
    estimator: EssEstimator,
) -> Result<EssEstimate, MonteCarloError> {
    let scaling = ScaledSamples::new(values)?;
    let n = input.samples_per_chain() / 2;
    let n_float = count_as_f64(n);
    let mut centered = Vec::with_capacity(values.len());
    let mut means = Vec::with_capacity(2 * input.chains().len());
    let mut variances = Vec::with_capacity(means.capacity());
    for (index, half) in values.chunks_exact(n).enumerate() {
        let anchor = half[0];
        let delta = |x| scaling.difference(x, anchor) / scaling.scale;
        let mean = compensated_sum(half.iter().map(|&x| delta(x))) / n_float;
        let start = centered.len();
        centered.extend(half.iter().map(|&x| delta(x) - mean));
        let variance = compensated_sum(centered[start..].iter().map(|x| x * x)) / n_float;
        if !variance.is_finite() || (variance <= 0.0 && half.iter().any(|&x| x != anchor)) {
            return Err(MonteCarloError::UnresolvedVariance {
                chain_index: index / 2,
                half: index % 2,
            });
        }
        means.push(scaling.difference(anchor, values[0]) / scaling.scale + mean);
        variances.push(variance);
    }
    let m = count_as_f64(means.len());
    let variance_sum = compensated_sum(variances);
    if variance_sum <= 0.0 {
        return Err(MonteCarloError::NoWithinChainVariation);
    }
    let biased_within = variance_sum / m;
    positive(biased_within)?;
    let within = biased_within * n_float / (n_float - 1.0);
    let grand_mean = compensated_sum(means.iter().copied()) / m;
    let between = compensated_sum(means.iter().map(|&x| (x - grand_mean).powi(2))) / (m - 1.0);
    let marginal = biased_within + between;
    positive(marginal)?;
    let rho = |lag| {
        let covariance = compensated_sum(centered.chunks_exact(n).map(|half| {
            compensated_sum(half.iter().zip(&half[lag..]).map(|(a, b)| a * b)) / n_float
        })) / m;
        1.0 - (within - covariance) / marginal
    };
    let mut correlations = vec![0.0; n];
    correlations[0] = 1.0;
    let mut even = 1.0;
    let mut odd = rho(1);
    correlations[1] = odd;
    let mut t = 1;
    while t < n.saturating_sub(3) && even + odd > 0.0 {
        even = rho(t + 1);
        odd = rho(t + 2);
        if even + odd >= 0.0 {
            correlations[t + 1] = even;
            correlations[t + 2] = odd;
        }
        t += 2;
    }
    let last_even = t - 1;
    if even > 0.0 {
        correlations[last_even] = even;
    }
    for index in (1..t.saturating_sub(3)).step_by(2) {
        let previous = correlations[index - 1] + correlations[index];
        if correlations[index + 1] + correlations[index + 2] > previous {
            correlations[index + 1] = previous * 0.5;
            correlations[index + 2] = previous * 0.5;
        }
    }
    let time = compensated_sum(correlations[..last_even].iter().copied()).mul_add(2.0, -1.0)
        + correlations[last_even];
    if !time.is_finite() {
        return Err(MonteCarloError::NumericalFailure);
    }
    let total = count_as_f64(input.retained_count());
    let bound = 1.0 / total.log10();
    let value = total / time.max(bound);
    positive(value)?;
    Ok(EssEstimate {
        value,
        estimator,
        counts: SampleCounts::from_input(input),
        regularized: time < bound,
    })
}

fn sorted_original(input: &RankedRhatInput<'_>) -> Vec<f64> {
    let mut sorted: Vec<_> = input
        .chains()
        .iter()
        .flat_map(|chain| chain.iter().copied())
        .collect();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted
}

/// Bounds/counts are checked before converting an order-statistic position.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite nonnegative indices below checked pooled storage count"
)]
const fn order_index(value: f64) -> usize {
    value as usize
}

/// Type 7 with anchored/convex interpolation to avoid finite-range overflow.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let position = count_as_f64(sorted.len() - 1) * p;
    let index = order_index(position.floor());
    let fraction = position - position.floor();
    let lower = sorted[index];
    let upper = sorted[(index + 1).min(sorted.len() - 1)];
    if lower.is_sign_negative() == upper.is_sign_negative() {
        (upper - lower).mul_add(fraction, lower)
    } else {
        (1.0 - fraction).mul_add(lower, fraction * upper)
    }
}

/// Check external beta inversion rather than accepting a finite-looking sentinel.
fn beta_quantile(a: f64, b: f64, p: f64) -> Result<f64, MonteCarloError> {
    let (mut lower, mut upper) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let middle = lower.midpoint(upper);
        let cdf = beta_reg(a, b, middle);
        if !cdf.is_finite() || !(0.0..=1.0).contains(&cdf) {
            return Err(MonteCarloError::NumericalFailure);
        }
        if cdf < p {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let x = lower.midpoint(upper);
    if !x.is_finite() || x <= 0.0 || x >= 1.0 {
        return Err(MonteCarloError::NumericalFailure);
    }
    let residual = beta_reg(a, b, x) - p;
    if !residual.is_finite() || residual.abs() > 1e-8 {
        return Err(MonteCarloError::NumericalFailure);
    }
    Ok(x)
}

fn positive(value: f64) -> Result<(), MonteCarloError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(MonteCarloError::NumericalFailure)
    }
}
