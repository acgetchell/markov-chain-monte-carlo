//! Streaming statistics for MCMC measurements.

use core::hint::cold_path;

use std::{error::Error, fmt};

use crate::TryAccumulator;

#[expect(
    clippy::cast_precision_loss,
    reason = "sample counts are expected to stay below the exact f64 integer range"
)]
/// Convert a sample count into the `f64` denominator used by online formulas.
///
/// This keeps the precision-loss lint scoped to the one place where the
/// statistics API intentionally crosses from integer counts to floating-point
/// arithmetic.
const fn count_as_f64(count: usize) -> f64 {
    count as f64
}

/// Errors from fallible statistical accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatisticsError {
    /// A measurement sample was NaN.
    NanSample,
    /// A measurement sample was infinite.
    InfiniteSample,
    /// Updating the running mean produced NaN.
    NanMean,
    /// Updating the running mean produced infinity.
    InfiniteMean,
    /// Updating the variance accumulator produced NaN.
    NanVarianceAccumulator,
    /// Updating the variance accumulator produced infinity.
    InfiniteVarianceAccumulator,
}

impl fmt::Display for StatisticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanSample => write!(f, "statistics sample was NaN"),
            Self::InfiniteSample => write!(f, "statistics sample was infinite"),
            Self::NanMean => write!(f, "online mean became NaN while updating statistics"),
            Self::InfiniteMean => {
                write!(f, "online mean became infinite while updating statistics")
            }
            Self::NanVarianceAccumulator => {
                write!(
                    f,
                    "online variance accumulator became NaN while updating statistics"
                )
            }
            Self::InfiniteVarianceAccumulator => {
                write!(
                    f,
                    "online variance accumulator became infinite while updating statistics"
                )
            }
        }
    }
}

impl Error for StatisticsError {}

/// Finite measurement accepted by the public fallible statistics APIs.
///
/// Keeping this proof type private makes raw `f64` parsing a boundary concern:
/// once a sample reaches the online formulas or binning hierarchy, downstream
/// helpers can rely on it not being `NaN` or infinite.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FiniteSample(f64);

impl FiniteSample {
    /// Parse a raw measurement into a finite sample.
    const fn new(sample: f64) -> Result<Self, StatisticsError> {
        if sample.is_nan() {
            cold_path();
            return Err(StatisticsError::NanSample);
        }
        if sample.is_infinite() {
            cold_path();
            return Err(StatisticsError::InfiniteSample);
        }
        Ok(Self(sample))
    }

    /// Return the validated raw measurement for floating-point arithmetic.
    const fn into_inner(self) -> f64 {
        self.0
    }
}

/// Validate the floating-point state produced by one Welford update.
///
/// This separates bad input samples from arithmetic overflow or invalid
/// accumulator state, which makes streaming errors more useful to debug.
const fn check_stats(stats: &OnlineStats) -> Result<(), StatisticsError> {
    if stats.mean.is_nan() {
        cold_path();
        return Err(StatisticsError::NanMean);
    }
    if stats.mean.is_infinite() {
        cold_path();
        return Err(StatisticsError::InfiniteMean);
    }
    if stats.m2.is_nan() {
        cold_path();
        return Err(StatisticsError::NanVarianceAccumulator);
    }
    if stats.m2.is_infinite() {
        cold_path();
        return Err(StatisticsError::InfiniteVarianceAccumulator);
    }
    Ok(())
}

/// Online mean and variance accumulator using Welford's algorithm.
///
/// `OnlineStats` updates in constant memory and is suitable for long
/// production runs where retaining every measurement would be expensive.
/// Measurements enter through [`try_push`](Self::try_push),
/// [`try_extend`](Self::try_extend), or [`try_from_iter`](Self::try_from_iter)
/// so invalid floating-point values are rejected before they can become part of
/// the accumulator state.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
///
/// let mut stats = OnlineStats::new();
/// stats.try_extend([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])?;
///
/// assert_eq!(stats.count(), 8);
/// assert_eq!(stats.mean(), Some(5.0));
/// assert_eq!(stats.population_variance(), Some(4.0));
/// # Ok::<(), StatisticsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OnlineStats {
    count: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    /// Create an empty accumulator.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::OnlineStats;
    ///
    /// let stats = OnlineStats::new();
    /// assert!(stats.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Build an accumulator from finite samples.
    ///
    /// This validates every sample and leaves no partially constructed
    /// accumulator behind on error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// assert_eq!(
    ///     OnlineStats::try_from_iter([1.0, f64::NAN]),
    ///     Err(StatisticsError::NanSample)
    /// );
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.mean(), Some(2.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] for the first non-finite input
    /// sample.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if Welford's update
    /// produces non-finite accumulator state.  No partially constructed
    /// accumulator is returned on error.
    pub fn try_from_iter(iter: impl IntoIterator<Item = f64>) -> Result<Self, StatisticsError> {
        let mut stats = Self::new();
        stats.try_extend(iter)?;
        Ok(stats)
    }

    /// Apply one already validated sample to Welford state.
    ///
    /// Public callers reach this only through [`Self::try_push`], which stages
    /// the mutation and rejects non-finite arithmetic before committing it.
    fn push_sample(&mut self, sample: FiniteSample) {
        self.count += 1;
        let sample = sample.into_inner();
        let delta = sample - self.mean;
        self.mean += delta / count_as_f64(self.count);
        let delta_after = sample - self.mean;
        self.m2 = delta.mul_add(delta_after, self.m2);
    }

    /// Add one finite sample to the accumulator.
    ///
    /// The accumulator is unchanged on error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let mut stats = OnlineStats::new();
    /// assert_eq!(stats.try_push(f64::NAN), Err(StatisticsError::NanSample));
    /// assert!(stats.is_empty());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] if `sample` is non-finite.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if Welford's update
    /// produces non-finite accumulator state.
    pub fn try_push(&mut self, sample: f64) -> Result<(), StatisticsError> {
        let sample = FiniteSample::new(sample)?;
        let mut next = *self;
        next.push_sample(sample);
        check_stats(&next)?;
        *self = next;
        Ok(())
    }

    /// Add finite samples to the accumulator.
    ///
    /// The accumulator retains samples accepted before the first error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let mut stats = OnlineStats::new();
    /// let err = stats.try_extend([1.0, 2.0, f64::NAN, 4.0]);
    ///
    /// assert_eq!(err, Err(StatisticsError::NanSample));
    /// assert_eq!(stats.count(), 2);
    /// assert_eq!(stats.mean(), Some(1.5));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] for the first non-finite input
    /// sample.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if Welford's update
    /// produces non-finite accumulator state.  Samples accepted before the error
    /// remain in the accumulator.
    pub fn try_extend(
        &mut self,
        iter: impl IntoIterator<Item = f64>,
    ) -> Result<(), StatisticsError> {
        for sample in iter {
            self.try_push(sample)?;
        }
        Ok(())
    }

    /// Remove all accumulated samples.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let mut stats = OnlineStats::try_from_iter([1.0, 2.0])?;
    /// stats.clear();
    /// assert_eq!(stats.count(), 0);
    /// # Ok::<(), StatisticsError>(())
    /// ```
    pub const fn clear(&mut self) {
        *self = Self::new();
    }

    /// Number of accumulated samples.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 2.0, 3.0])?;
    /// assert_eq!(stats.count(), 3);
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Whether the accumulator is empty.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let mut stats = OnlineStats::new();
    /// assert!(stats.is_empty());
    /// stats.try_push(1.0)?;
    /// assert!(!stats.is_empty());
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Current sample mean.
    ///
    /// Returns `None` until at least one sample has been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([2.0, 4.0])?;
    /// assert_eq!(stats.mean(), Some(3.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Population variance, using `n` in the denominator.
    ///
    /// Returns `None` until at least one sample has been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.population_variance(), Some(1.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn population_variance(&self) -> Option<f64> {
        (self.count > 0).then_some(self.m2 / count_as_f64(self.count))
    }

    /// Unbiased sample variance, using `n - 1` in the denominator.
    ///
    /// Returns `None` until at least two samples have been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.sample_variance(), Some(2.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn sample_variance(&self) -> Option<f64> {
        (self.count > 1).then(|| self.m2 / count_as_f64(self.count - 1))
    }

    /// Population standard deviation.
    ///
    /// Returns `None` until at least one sample has been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.population_std_dev(), Some(1.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn population_std_dev(&self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }

    /// Unbiased sample standard deviation.
    ///
    /// Returns `None` until at least two samples have been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.sample_std_dev(), Some(2.0_f64.sqrt()));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn sample_std_dev(&self) -> Option<f64> {
        self.sample_variance().map(f64::sqrt)
    }

    /// Naive standard error of the mean, ignoring autocorrelation.
    ///
    /// For autocorrelated MCMC measurements, prefer [`BinningAnalysis`] and
    /// inspect the blocked standard-error estimates.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{OnlineStats, StatisticsError};
    ///
    /// let stats = OnlineStats::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(stats.standard_error(), Some(1.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn standard_error(&self) -> Option<f64> {
        self.sample_variance()
            .map(|variance| (variance / count_as_f64(self.count)).sqrt())
    }
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TryAccumulator<f64> for OnlineStats {
    type Error = StatisticsError;

    fn try_push(&mut self, sample: f64) -> Result<(), Self::Error> {
        Self::try_push(self, sample)
    }
}

/// Standard-error estimate at one binning level.
///
/// A level stores the statistics of completed block means for a fixed block
/// size.  The standard error is the standard deviation of those block means
/// divided by the square root of the number of completed blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BinningEstimate {
    block_size: usize,
    block_count: usize,
    mean: f64,
    sample_variance: Option<f64>,
    standard_error: Option<f64>,
}

impl BinningEstimate {
    /// Number of original samples per block at this level.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert_eq!(bins.estimates().nth(1).map(|estimate| estimate.block_size()), Some(2));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of completed block means included in this estimate.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert_eq!(bins.estimates().next().map(|estimate| estimate.block_count()), Some(4));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Mean of the completed block means.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert_eq!(bins.estimates().next().map(|estimate| estimate.mean()), Some(2.5));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Unbiased variance of the completed block means.
    ///
    /// Returns `None` until at least two completed blocks exist at this level.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert_eq!(
    ///     bins.estimates().nth(2).and_then(|estimate| estimate.sample_variance()),
    ///     None
    /// );
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn sample_variance(&self) -> Option<f64> {
        self.sample_variance
    }

    /// Estimated standard error of the overall mean at this block size.
    ///
    /// Returns `None` until at least two completed blocks exist at this level.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert!(
    ///     bins.estimates()
    ///         .next()
    ///         .and_then(|estimate| estimate.standard_error())
    ///         .is_some()
    /// );
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BinningLevel {
    block_size: usize,
    stats: OnlineStats,
    pending: Option<FiniteSample>,
}

impl BinningLevel {
    /// Create one level for completed block means of `block_size` samples.
    ///
    /// The binning hierarchy grows lazily, so this is called only when a sample
    /// first reaches a previously unused block size.
    const fn new(block_size: usize) -> Self {
        Self {
            block_size,
            stats: OnlineStats::new(),
            pending: None,
        }
    }

    /// Build the public estimate for this level when at least one block exists.
    ///
    /// `pending` is not added separately here because it has already been
    /// included in this level's statistics.  It is only pending for pairing into
    /// the next coarser level.
    fn estimate(&self) -> Option<BinningEstimate> {
        let mean = self.stats.mean()?;
        let sample_variance = self.stats.sample_variance();
        let standard_error =
            sample_variance.map(|variance| (variance / count_as_f64(self.stats.count())).sqrt());

        Some(BinningEstimate {
            block_size: self.block_size,
            block_count: self.stats.count(),
            mean,
            sample_variance,
            standard_error,
        })
    }

    /// Apply one validated per-level update during staged accumulation.
    ///
    /// Returns a coarser block mean only when this level pairs the incoming block
    /// with an existing pending block.
    fn push_block_mean(
        &mut self,
        block_mean: FiniteSample,
    ) -> Result<Option<FiniteSample>, StatisticsError> {
        self.stats.push_sample(block_mean);
        check_stats(&self.stats)?;
        if let Some(previous) = self.pending.take() {
            FiniteSample::new(0.5_f64.mul_add(block_mean.into_inner(), 0.5 * previous.into_inner()))
                .map(Some)
        } else {
            self.pending = Some(block_mean);
            Ok(None)
        }
    }
}

/// Streaming binning analysis for autocorrelation-corrected error estimates.
///
/// Each pushed sample updates a hierarchy of power-of-two block means.  The
/// level with block size 1 matches the naive standard error.  Coarser levels
/// estimate the error after progressively averaging nearby correlated samples;
/// once the estimates plateau, that value is the usual binning estimate for the
/// standard error of an MCMC mean.
///
/// Measurements enter through [`try_push`](Self::try_push),
/// [`try_extend`](Self::try_extend), or [`try_from_iter`](Self::try_from_iter)
/// so invalid floating-point values are rejected before they can become part of
/// the binning hierarchy.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
///
/// let mut bins = BinningAnalysis::new();
/// bins.try_extend((1..=8).map(f64::from))?;
///
/// let estimates: Vec<_> = bins.estimates().collect();
/// assert_eq!(estimates[0].block_size(), 1);
/// assert_eq!(estimates[1].block_size(), 2);
/// assert_eq!(estimates[2].block_size(), 4);
/// assert_eq!(bins.mean(), Some(4.5));
/// # Ok::<(), StatisticsError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BinningAnalysis {
    count: usize,
    levels: Vec<BinningLevel>,
    staged_levels: Vec<(usize, BinningLevel)>,
}

impl BinningAnalysis {
    /// Create an empty binning analysis.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::BinningAnalysis;
    ///
    /// let bins = BinningAnalysis::new();
    /// assert!(bins.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self {
            count: 0,
            levels: Vec::new(),
            staged_levels: Vec::new(),
        }
    }

    /// Build a binning analysis from finite samples.
    ///
    /// This validates every sample and leaves no partially constructed analysis
    /// behind on error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// assert_eq!(
    ///     BinningAnalysis::try_from_iter([1.0, f64::INFINITY]),
    ///     Err(StatisticsError::InfiniteSample)
    /// );
    ///
    /// let bins = BinningAnalysis::try_from_iter([1.0, 2.0, 3.0])?;
    /// assert_eq!(bins.mean(), Some(2.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] for the first non-finite input
    /// sample.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if any binning-level
    /// accumulator update produces non-finite state.  No partially constructed
    /// analysis is returned on error.
    pub fn try_from_iter(iter: impl IntoIterator<Item = f64>) -> Result<Self, StatisticsError> {
        let mut analysis = Self::new();
        analysis.try_extend(iter)?;
        Ok(analysis)
    }

    /// Add one finite measurement.
    ///
    /// The analysis is unchanged on error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let mut bins = BinningAnalysis::new();
    /// assert_eq!(bins.try_push(f64::INFINITY), Err(StatisticsError::InfiniteSample));
    /// assert!(bins.is_empty());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] if `sample` is non-finite.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if any staged
    /// binning-level accumulator update produces non-finite state.
    pub fn try_push(&mut self, sample: f64) -> Result<(), StatisticsError> {
        let sample = FiniteSample::new(sample)?;
        self.stage_push(sample)?;

        self.count += 1;
        for (level_index, level) in self.staged_levels.drain(..) {
            if level_index == self.levels.len() {
                self.levels.push(level);
            } else {
                self.levels[level_index] = level;
            }
        }
        Ok(())
    }

    /// Add finite measurements to the analysis.
    ///
    /// The analysis retains samples accepted before the first error.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let mut bins = BinningAnalysis::new();
    /// let err = bins.try_extend([1.0, 2.0, f64::INFINITY, 4.0]);
    ///
    /// assert_eq!(err, Err(StatisticsError::InfiniteSample));
    /// assert_eq!(bins.count(), 2);
    /// assert_eq!(bins.mean(), Some(1.5));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StatisticsError::NanSample`] or
    /// [`StatisticsError::InfiniteSample`] for the first non-finite input
    /// sample.
    ///
    /// Returns [`StatisticsError::NanMean`], [`StatisticsError::InfiniteMean`],
    /// [`StatisticsError::NanVarianceAccumulator`], or
    /// [`StatisticsError::InfiniteVarianceAccumulator`] if any binning-level
    /// accumulator update produces non-finite state.  Samples accepted before the
    /// error remain in the analysis.
    pub fn try_extend(
        &mut self,
        iter: impl IntoIterator<Item = f64>,
    ) -> Result<(), StatisticsError> {
        for sample in iter {
            self.try_push(sample)?;
        }
        Ok(())
    }

    /// Remove all accumulated samples and bins.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let mut bins = BinningAnalysis::try_from_iter([1.0, 2.0])?;
    /// bins.clear();
    /// assert!(bins.is_empty());
    /// # Ok::<(), StatisticsError>(())
    /// ```
    pub fn clear(&mut self) {
        self.count = 0;
        self.levels.clear();
        self.staged_levels.clear();
    }

    /// Number of original measurements pushed into the analysis.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter([1.0, 2.0, 3.0])?;
    /// assert_eq!(bins.count(), 3);
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Whether the analysis contains no measurements.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let mut bins = BinningAnalysis::new();
    /// assert!(bins.is_empty());
    /// bins.try_push(1.0)?;
    /// assert!(!bins.is_empty());
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Mean of all original measurements.
    ///
    /// Returns `None` until at least one sample has been added.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter([1.0, 2.0, 3.0])?;
    /// assert_eq!(bins.mean(), Some(2.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        self.levels.first().and_then(|level| level.stats.mean())
    }

    /// Naive standard error from unblocked samples.
    ///
    /// This ignores autocorrelation and is equivalent to
    /// [`OnlineStats::standard_error`] over the original measurements.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter([1.0, 3.0])?;
    /// assert_eq!(bins.unblocked_standard_error(), Some(1.0));
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn unblocked_standard_error(&self) -> Option<f64> {
        self.levels
            .first()
            .and_then(|level| level.stats.standard_error())
    }

    /// Coarsest available binning estimate with at least two completed blocks.
    ///
    /// This is a convenient single estimate for streaming use.  For production
    /// analysis, inspect [`estimates`](Self::estimates) to confirm that the
    /// standard error has stabilized across block sizes.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=8).map(f64::from))?;
    /// assert_eq!(
    ///     bins.coarsest_estimate().map(|estimate| estimate.block_size()),
    ///     Some(4)
    /// );
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn coarsest_estimate(&self) -> Option<BinningEstimate> {
        self.estimates()
            .filter(|estimate| estimate.standard_error().is_some())
            .last()
    }

    /// Coarsest available standard error from the binning hierarchy.
    ///
    /// Returns `None` until at least two completed blocks exist at some level.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// assert!(bins.standard_error().is_some());
    /// # Ok::<(), StatisticsError>(())
    /// ```
    #[must_use]
    pub fn standard_error(&self) -> Option<f64> {
        self.coarsest_estimate()
            .and_then(|estimate| estimate.standard_error())
    }

    /// Iterate over estimates from fine to coarse block sizes.
    ///
    /// The iterator includes levels with at least one completed block.  A
    /// level's [`BinningEstimate::standard_error`] is `None` until two
    /// completed blocks are available at that block size.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{BinningAnalysis, StatisticsError};
    ///
    /// let bins = BinningAnalysis::try_from_iter((1..=4).map(f64::from))?;
    /// let block_sizes: Vec<_> = bins.estimates().map(|estimate| estimate.block_size()).collect();
    ///
    /// assert_eq!(block_sizes, vec![1, 2, 4]);
    /// # Ok::<(), StatisticsError>(())
    /// ```
    pub fn estimates(&self) -> impl Iterator<Item = BinningEstimate> + '_ {
        self.levels.iter().filter_map(BinningLevel::estimate)
    }

    /// Stage the exact levels changed by a fallible push.
    ///
    /// A single sample only touches a prefix of the binning hierarchy.  Staging
    /// that prefix keeps `try_push` failure-atomic without cloning untouched
    /// coarser levels.
    fn stage_push(&mut self, sample: FiniteSample) -> Result<(), StatisticsError> {
        self.staged_levels.clear();
        let mut level_index = 0;
        let mut block_mean = sample;

        loop {
            let mut level = self
                .levels
                .get(level_index)
                .cloned()
                .unwrap_or_else(|| self.new_staged_level(level_index));
            let next_block_mean = match level.push_block_mean(block_mean) {
                Ok(next_block_mean) => next_block_mean,
                Err(err) => {
                    self.staged_levels.clear();
                    return Err(err);
                }
            };
            self.staged_levels.push((level_index, level));

            if let Some(mean) = next_block_mean {
                block_mean = mean;
                level_index += 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Create a lazily allocated level without mutating the visible hierarchy.
    ///
    /// This mirrors `ensure_level` for staged updates, deriving the next block
    /// size from either the latest staged level or the existing hierarchy.
    fn new_staged_level(&self, level_index: usize) -> BinningLevel {
        let block_size = if level_index == 0 {
            1
        } else if let Some((_, previous)) = self.staged_levels.last() {
            previous.block_size.saturating_mul(2)
        } else {
            self.levels[level_index - 1].block_size.saturating_mul(2)
        };

        BinningLevel::new(block_size)
    }
}

impl Default for BinningAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl TryAccumulator<f64> for BinningAnalysis {
    type Error = StatisticsError;

    fn try_push(&mut self, sample: f64) -> Result<(), Self::Error> {
        Self::try_push(self, sample)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert_relative_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn statistics_error_messages() {
        assert_eq!(
            StatisticsError::NanSample.to_string(),
            "statistics sample was NaN"
        );
        assert_eq!(
            StatisticsError::InfiniteSample.to_string(),
            "statistics sample was infinite"
        );
        assert_eq!(
            StatisticsError::NanMean.to_string(),
            "online mean became NaN while updating statistics"
        );
        assert_eq!(
            StatisticsError::InfiniteMean.to_string(),
            "online mean became infinite while updating statistics"
        );
        assert_eq!(
            StatisticsError::NanVarianceAccumulator.to_string(),
            "online variance accumulator became NaN while updating statistics"
        );
        assert_eq!(
            StatisticsError::InfiniteVarianceAccumulator.to_string(),
            "online variance accumulator became infinite while updating statistics"
        );
    }

    #[test]
    fn online_stats_reports_empty_state() {
        let stats = OnlineStats::new();

        assert!(stats.is_empty());
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.population_variance(), None);
        assert_eq!(stats.sample_variance(), None);
        assert_eq!(stats.standard_error(), None);
    }

    #[test]
    fn online_stats_matches_known_values() {
        let stats = OnlineStats::try_from_iter([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();

        assert_eq!(stats.count(), 8);
        assert_close(stats.mean().unwrap(), 5.0);
        assert_close(stats.population_variance().unwrap(), 4.0);
        assert_close(stats.sample_variance().unwrap(), 32.0 / 7.0);
        assert_close(stats.population_std_dev().unwrap(), 2.0);
        assert_close(stats.standard_error().unwrap(), (4.0_f64 / 7.0).sqrt());
    }

    #[test]
    fn online_stats_pins_fused_variance_update() {
        let stats = OnlineStats::try_from_iter([1.0, 1.0e12, -1.0e12, 3.5, -2.25, 8.125]).unwrap();

        assert_eq!(stats.count(), 6);
        assert_eq!(stats.mean.to_bits(), 0x3ffb_aaa0_0000_0000);
        assert_eq!(stats.m2.to_bits(), 0x44fa_7843_79d9_9db4);
        assert_eq!(
            stats.population_variance().unwrap().to_bits(),
            0x44d1_a582_513b_be78
        );
        assert_eq!(
            stats.sample_variance().unwrap().to_bits(),
            0x44d5_2d02_c7e1_4af6
        );
    }

    #[test]
    fn online_stats_can_clear_and_reuse() {
        let mut stats = OnlineStats::default();
        stats.try_extend([1.0, 3.0]).unwrap();
        stats.clear();
        stats.try_push(10.0).unwrap();

        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(10.0));
        assert_eq!(stats.sample_variance(), None);
    }

    #[test]
    fn online_stats_try_push_rejects_invalid_samples_atomically() {
        let mut stats = OnlineStats::new();
        stats.try_push(1.0).unwrap();

        assert_eq!(stats.try_push(f64::NAN), Err(StatisticsError::NanSample));
        assert_eq!(
            stats.try_push(f64::NEG_INFINITY),
            Err(StatisticsError::InfiniteSample)
        );
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(1.0));
    }

    #[test]
    fn online_stats_try_push_rejects_nonfinite_accumulator_atomically() {
        let mut stats = OnlineStats::new();
        stats.try_push(f64::MAX).unwrap();

        assert_eq!(
            stats.try_push(-f64::MAX),
            Err(StatisticsError::InfiniteMean)
        );
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(f64::MAX));
    }

    #[test]
    fn online_stats_try_push_rejects_infinite_variance_accumulator_atomically() {
        let mut stats = OnlineStats::new();
        stats.try_push(f64::MAX).unwrap();

        assert_eq!(
            stats.try_push(0.0),
            Err(StatisticsError::InfiniteVarianceAccumulator)
        );
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(f64::MAX));
        assert_eq!(stats.sample_variance(), None);
    }

    #[test]
    fn online_stats_try_push_rejects_stale_nan_mean_atomically() {
        let mut stats = OnlineStats {
            count: 1,
            mean: f64::NAN,
            m2: 0.0,
        };

        assert_eq!(stats.try_push(1.0), Err(StatisticsError::NanMean));
        assert_eq!(stats.count, 1);
        assert!(stats.mean.is_nan());
        assert_close(stats.m2, 0.0);
    }

    #[test]
    fn online_stats_try_push_rejects_stale_nan_variance_accumulator_atomically() {
        let mut stats = OnlineStats {
            count: 1,
            mean: 1.0,
            m2: f64::NAN,
        };

        assert_eq!(
            stats.try_push(2.0),
            Err(StatisticsError::NanVarianceAccumulator)
        );
        assert_eq!(stats.count, 1);
        assert_close(stats.mean, 1.0);
        assert!(stats.m2.is_nan());
    }

    #[test]
    fn online_stats_try_extend_keeps_prior_successes() {
        let mut stats = OnlineStats::new();

        assert_eq!(
            stats.try_extend([1.0, 2.0, f64::NAN, 4.0]),
            Err(StatisticsError::NanSample)
        );
        assert_eq!(stats.count(), 2);
        assert_eq!(stats.mean(), Some(1.5));
    }

    #[test]
    fn online_stats_try_from_iter_validates_all_samples() {
        let stats = OnlineStats::try_from_iter([1.0, 3.0]).unwrap();
        assert_eq!(stats.count(), 2);
        assert_eq!(stats.mean(), Some(2.0));

        assert_eq!(
            OnlineStats::try_from_iter([1.0, f64::NAN, 3.0]),
            Err(StatisticsError::NanSample)
        );
    }

    #[test]
    fn binning_analysis_builds_power_of_two_levels() {
        let bins = BinningAnalysis::try_from_iter((1..=8).map(f64::from)).unwrap();
        let estimates: Vec<_> = bins.estimates().collect();

        assert_eq!(bins.count(), 8);
        assert_eq!(bins.mean(), Some(4.5));
        assert_eq!(estimates.len(), 4);
        assert_eq!(estimates[0].block_size(), 1);
        assert_eq!(estimates[0].block_count(), 8);
        assert_eq!(estimates[1].block_size(), 2);
        assert_eq!(estimates[1].block_count(), 4);
        assert_eq!(estimates[2].block_size(), 4);
        assert_eq!(estimates[2].block_count(), 2);
        assert_eq!(estimates[3].block_size(), 8);
        assert_eq!(estimates[3].block_count(), 1);
    }

    #[test]
    fn binning_analysis_reports_blocked_errors() {
        let bins = BinningAnalysis::try_from_iter((1..=8).map(f64::from)).unwrap();
        let estimates: Vec<_> = bins.estimates().collect();

        assert_close(estimates[0].sample_variance().unwrap(), 6.0);
        assert_close(
            estimates[0].standard_error().unwrap(),
            (6.0_f64 / 8.0).sqrt(),
        );
        assert_close(estimates[1].sample_variance().unwrap(), 20.0 / 3.0);
        assert_close(
            estimates[1].standard_error().unwrap(),
            (20.0_f64 / 12.0).sqrt(),
        );
        assert_close(estimates[2].sample_variance().unwrap(), 8.0);
        assert_close(estimates[2].standard_error().unwrap(), 2.0);
        assert_eq!(estimates[3].sample_variance(), None);
        assert_eq!(estimates[3].standard_error(), None);

        let coarsest = bins.coarsest_estimate().unwrap();
        assert_eq!(coarsest.block_size(), 4);
        assert_eq!(bins.standard_error(), Some(2.0));
    }

    #[test]
    fn binning_analysis_handles_partial_tail_blocks() {
        let bins = BinningAnalysis::try_from_iter((1..=5).map(f64::from)).unwrap();
        let estimates: Vec<_> = bins.estimates().collect();

        assert_eq!(bins.count(), 5);
        assert_eq!(estimates.len(), 3);
        assert_eq!(estimates[0].block_size(), 1);
        assert_eq!(estimates[0].block_count(), 5);
        assert_eq!(estimates[1].block_size(), 2);
        assert_eq!(estimates[1].block_count(), 2);
        assert_eq!(estimates[2].block_size(), 4);
        assert_eq!(estimates[2].block_count(), 1);
        assert_close(estimates[1].mean(), 2.5);
    }

    #[test]
    fn binning_analysis_can_clear_and_reuse() {
        let mut bins = BinningAnalysis::default();
        bins.try_extend([1.0, 2.0, 3.0, 4.0]).unwrap();
        bins.clear();
        bins.try_extend([10.0, 14.0]).unwrap();

        assert_eq!(bins.count(), 2);
        assert_eq!(bins.mean(), Some(12.0));
        assert_eq!(bins.estimates().count(), 2);
    }

    #[test]
    fn binning_analysis_try_push_rejects_invalid_samples_atomically() {
        let mut bins = BinningAnalysis::new();
        bins.try_push(1.0).unwrap();

        assert_eq!(bins.try_push(f64::NAN), Err(StatisticsError::NanSample));
        assert_eq!(
            bins.try_push(f64::INFINITY),
            Err(StatisticsError::InfiniteSample)
        );
        assert_eq!(bins.count(), 1);
        assert_eq!(bins.mean(), Some(1.0));
    }

    #[test]
    fn binning_analysis_try_push_rejects_nonfinite_accumulator_atomically() {
        let mut bins = BinningAnalysis::new();
        bins.try_push(f64::MAX).unwrap();

        assert_eq!(bins.try_push(-f64::MAX), Err(StatisticsError::InfiniteMean));
        assert_eq!(bins.count(), 1);
        assert_eq!(bins.mean(), Some(f64::MAX));
        assert_eq!(bins.estimates().next().unwrap().block_count(), 1);
        assert!(bins.staged_levels.is_empty());
    }

    #[test]
    fn binning_analysis_try_push_matches_try_from_iter_hierarchy() {
        let samples = (1..=9).map(f64::from);
        let mut fallible = BinningAnalysis::new();

        for sample in samples.clone() {
            fallible.try_push(sample).unwrap();
        }
        let from_iter = BinningAnalysis::try_from_iter(samples).unwrap();

        assert_eq!(fallible, from_iter);

        let estimates: Vec<_> = fallible.estimates().collect();
        assert_eq!(
            estimates
                .iter()
                .map(BinningEstimate::block_size)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert_eq!(
            estimates
                .iter()
                .map(BinningEstimate::block_count)
                .collect::<Vec<_>>(),
            vec![9, 4, 2, 1]
        );
    }

    #[test]
    fn binning_analysis_try_extend_keeps_prior_successes() {
        let mut bins = BinningAnalysis::new();

        assert_eq!(
            bins.try_extend([1.0, 2.0, f64::NAN, 4.0]),
            Err(StatisticsError::NanSample)
        );
        assert_eq!(bins.count(), 2);
        assert_eq!(bins.mean(), Some(1.5));
    }

    #[test]
    fn binning_analysis_try_from_iter_validates_all_samples() {
        let bins = BinningAnalysis::try_from_iter([1.0, 2.0, 3.0, 4.0]).unwrap();
        let estimates: Vec<_> = bins.estimates().collect();

        assert_eq!(bins.count(), 4);
        assert_eq!(bins.mean(), Some(2.5));
        assert_eq!(estimates.len(), 3);
        assert_eq!(estimates[0].block_count(), 4);
        assert_eq!(estimates[1].block_count(), 2);
        assert_eq!(estimates[2].block_count(), 1);

        assert_eq!(
            BinningAnalysis::try_from_iter([1.0, f64::INFINITY, 3.0]),
            Err(StatisticsError::InfiniteSample)
        );
    }
}
