//! Classical and rank-normalized split R-hat for comparable, finite scalar chains.
//!
//! [`SplitRhat`] compares within-half and between-half variation, while
//! [`RankNormalizedSplitRhat`] applies the same ratio to pooled normal scores.
//! [`SplitRhatError`] keeps invalid and unresolved inputs explicit. These types are
//! re-exported at the crate root and in [`crate::prelude`].

use std::{error::Error, fmt};

use crate::numerics::{compensated_sum, count_as_f64, inverse_normal_lower_tail};

/// Invalid or numerically unresolved input to either split R-hat estimator.
///
/// Every `chain_index` is a zero-based position in the supplied slice, not a
/// [`crate::ChainId`]. Keep the identifiers alongside those slices when reporting
/// failures for a [`crate::Trace`]. A `half` of zero selects the first half and
/// one selects the last half; sample indices refer to the original unsplit chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplitRhatError {
    /// At least two original chains are required, before splitting.
    #[non_exhaustive]
    InsufficientChains {
        /// Number of original chains.
        count: usize,
    },
    /// Every original chain needs at least four samples.
    #[non_exhaustive]
    InsufficientSamples {
        /// Index of the short chain in the input slice.
        chain_index: usize,
        /// Number of samples in that chain.
        count: usize,
    },
    /// Chains must have equal lengths; no implicit trimming is performed.
    #[non_exhaustive]
    UnequalLengths {
        /// Index of the chain with a different length.
        chain_index: usize,
        /// Length of the first chain.
        expected: usize,
        /// Length of this chain.
        actual: usize,
    },
    /// A sample is NaN or infinite, including an otherwise unused middle draw.
    #[non_exhaustive]
    NonFiniteSample {
        /// Index of the original chain.
        chain_index: usize,
        /// Index within that original chain.
        sample_index: usize,
    },
    /// A split chain has no variation; a stuck chain must not imply convergence.
    #[non_exhaustive]
    ConstantSplitChain {
        /// Index of the original chain.
        chain_index: usize,
        /// Zero for the first half, one for the second.
        half: usize,
    },
    /// A nonconstant half's variance is not positive and finite after scaling.
    ///
    /// Unlike [`Self::ConstantSplitChain`], the original samples do vary.
    /// Extreme scale differences can make that variation numerically unresolved.
    #[non_exhaustive]
    UnresolvedVariance {
        /// Index of the original chain containing the unresolved half.
        chain_index: usize,
        /// Zero for the first half, one for the second.
        half: usize,
    },
    /// The final R-hat estimate is not representable as a positive finite value.
    ///
    /// Every half passed the individual variance checks before this failure.
    NumericalFailure,
    /// Rank normalization would pool more than `2^50` retained draws, or the
    /// pooled count overflows `usize`. This bound keeps average ranks and the
    /// fractional offsets in the normal-score transform exactly representable.
    ///
    /// Only [`RankNormalizedSplitRhat::estimate`] returns this variant;
    /// [`SplitRhat::estimate`] does not rank observations.
    #[non_exhaustive]
    TooManyRankedSamples {
        /// Number of original chains whose retained samples would be pooled.
        chain_count: usize,
        /// Retained draws per half; each original chain contributes two halves.
        samples_per_split_chain: usize,
        /// Maximum pooled retained count: the smaller of `2^50` and `usize::MAX`.
        max_retained_samples: usize,
    },
}

impl fmt::Display for SplitRhatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientChains { count } => {
                write!(
                    f,
                    "split R-hat needs at least two original chains, got {count}"
                )
            }
            Self::InsufficientSamples { chain_index, count } => {
                write!(
                    f,
                    "chain {chain_index} needs at least four samples, got {count}"
                )
            }
            Self::UnequalLengths {
                chain_index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "chain {chain_index} has {actual} samples; expected {expected}"
                )
            }
            Self::NonFiniteSample {
                chain_index,
                sample_index,
            } => {
                write!(f, "chain {chain_index} sample {sample_index} is not finite")
            }
            Self::ConstantSplitChain { chain_index, half } => {
                write!(f, "chain {chain_index} half {half} is constant")
            }
            Self::UnresolvedVariance { chain_index, half } => write!(
                f,
                "chain {chain_index} half {half} has varying samples but its variance is unresolved after scaling"
            ),
            Self::NumericalFailure => {
                f.write_str("final split R-hat estimate is not positive and finite")
            }
            Self::TooManyRankedSamples {
                chain_count,
                samples_per_split_chain,
                max_retained_samples,
            } => write!(
                f,
                "rank normalization of {chain_count} chains with {samples_per_split_chain} samples per half exceeds the limit of {max_retained_samples} retained draws"
            ),
        }
    }
}

impl Error for SplitRhatError {}

/// Classical split potential scale reduction statistic for one scalar observable.
///
/// Use [`Self::estimate`] to compare chains for location disagreement or drift.
/// The result owns a scalar estimate and count metadata; it does not borrow or
/// retain the input observations. Inspect it with [`Self::value`], and use
/// [`crate::IntegratedAutocorrelationTime::effective_sample_size`] separately
/// for single-chain mean precision. Both diagnostics are available without
/// enabling any Cargo feature.
///
/// Supply at least two independently initialized original chains targeting the
/// same distribution, with the same observable, units, recording interval, and
/// warmup policy. Discard warmup first, preserve time order, and retain rejected
/// steps and no-proposal self-loops. Scalar slices cannot verify those assumptions.
///
/// Each chain is split into first and last halves of length `n = floor(N/2)`;
/// the middle draw of an odd-length chain is omitted. With `m = 2 * chains`,
/// let `W` be the mean unbiased within-half sample variance, and `B = n` times
/// the unbiased sample variance of the half means. Return
/// `sqrt(((n-1)/n * W + B/n) / W)`. Values below one are retained.
/// See the [classical split R-hat definition](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html).
///
/// This raw-moment estimator assumes finite marginal mean and variance. It is
/// **not rank-normalized or folded R-hat** and can miss scale or tail differences.
/// A value near one is not proof of convergence, adequate length, or exploration
/// of all modes. Use longer runs, dispersed starts, ESS, and trace inspection.
/// [`RankNormalizedSplitRhat`] helps detect location disagreement with heavy
/// tails. Neither estimator includes folded diagnostics for scale sensitivity.
///
/// # Examples
///
/// Select the same named observable from each original [`crate::Trace`] chain,
/// then borrow the collected columns for estimation. The rows below are short
/// synthetic production records for demonstrating selection, not evidence of
/// adequate sampling. Real inputs must satisfy the comparability and warmup
/// requirements above; selection preserves insertion order and checks neither.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{
///     ChainId, SplitRhat, Trace, TraceError, TraceRecord, TraceStepOutcome,
/// };
///
/// let chain_ids = [ChainId::new(10), ChainId::new(20)];
/// let mut trace = Trace::new(["energy"])?;
/// for (id, energies) in chain_ids.into_iter().zip([
///     [1.0, 2.0, 3.0, 4.0],
///     [3.0, 4.0, 5.0, 6.0],
/// ]) {
///     for (index, energy) in energies.into_iter().enumerate() {
///         trace.push(TraceRecord::new(
///             id, index + 1, TraceStepOutcome::accepted(), -energy, vec![energy],
///         ))?;
///     }
/// }
/// let columns: Result<Vec<Vec<f64>>, TraceError> = chain_ids.iter().map(|&id| {
///     trace.observable_values(id, "energy").map(|values| values.copied().collect())
/// }).collect();
/// let columns = columns?;
/// let chains: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
/// let result = SplitRhat::estimate(&chains);
/// assert!(matches!(result, Ok(rhat)
///     if rhat.chain_count() == 2 && (rhat.value() - (35.0_f64 / 6.0).sqrt()).abs() < 1e-14));
/// # Ok::<(), TraceError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SplitRhat {
    /// Positive finite result of the classical split variance ratio.
    estimate: f64,
    /// Number of original chains, before constructing twice as many halves.
    chain_count: usize,
    /// Original length, retained to report whether splitting omitted a draw.
    samples_per_chain: usize,
}

impl SplitRhat {
    /// Estimate classical split R-hat from borrowed, equally long scalar chains.
    ///
    /// Supply at least two original chains, each with at least four draws,
    /// using the observable and sampling prerequisites described on [`Self`].
    /// Each chain contributes its first and last `floor(N/2)` draws; an odd
    /// middle draw is omitted from moments but still checked for finiteness.
    /// These minimum counts make the formula defined, not statistically reliable.
    /// The returned summary is independent of the input buffers, which are unchanged.
    ///
    /// Time is `O(chains * samples)` and auxiliary memory is `O(chains)`.
    /// Common shifted scaling and compensated sums avoid overflow; centering
    /// within each half preserves small local variation at large offsets.
    /// Extremely different scales can still underflow within-half variances.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{SplitRhat, SplitRhatError};
    ///
    /// let left = [0.0, 2.0, 0.0, 2.0];
    /// let right = [2.0, 4.0, 2.0, 4.0];
    /// let rhat = SplitRhat::estimate(&[&left, &right])?;
    /// assert!((rhat.value() - (7.0_f64 / 6.0).sqrt()).abs() < 1e-14);
    /// assert_eq!(rhat.chain_count(), 2);
    /// assert_eq!(rhat.samples_per_split_chain(), 2);
    /// # Ok::<(), SplitRhatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - [`SplitRhatError::InsufficientChains`]: fewer than two original chains.
    /// - [`SplitRhatError::InsufficientSamples`]: a chain has fewer than four draws.
    /// - [`SplitRhatError::UnequalLengths`]: a chain differs in length from the first.
    /// - [`SplitRhatError::NonFiniteSample`]: any supplied draw is NaN or infinite,
    ///   including a middle draw omitted by splitting.
    /// - [`SplitRhatError::ConstantSplitChain`]: all draws in a half are exactly
    ///   equal as represented by `f64`, even if other halves vary.
    /// - [`SplitRhatError::UnresolvedVariance`]: a nonconstant half's computed
    ///   variance is nonpositive or nonfinite, including underflow after common
    ///   scaling. The error identifies the original chain and half.
    /// - [`SplitRhatError::NumericalFailure`]: the final estimate is nonpositive
    ///   or nonfinite after all individual halves passed their variance checks.
    ///
    /// Chain count is checked first. Then lengths are checked in input order,
    /// with minimum length before equality for each chain, followed by finiteness
    /// of every original sample. Moment calculation then checks each half for
    /// constancy and numerical failure. Chain indices in errors are input-slice
    /// positions; they do not identify [`crate::ChainId`] values.
    pub fn estimate(chains: &[&[f64]]) -> Result<Self, SplitRhatError> {
        let input = SplitRhatInput::parse(chains)?;
        let sample_count = input.samples_per_chain;
        let half_length = sample_count / 2;
        let halves = input
            .chains
            .iter()
            .flat_map(|chain| [&chain[..half_length], &chain[sample_count - half_length..]]);
        let origin = input.chains[0][0];
        // Use a common factor for every half; separate per-chain scaling would
        // erase between-chain differences and change the diagnostic.
        let (minimum, maximum) = halves
            .clone()
            .flatten()
            .fold((origin, origin), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        let halve = !(maximum - minimum).is_finite();
        let difference = |value: f64, anchor: f64| {
            if halve {
                anchor.mul_add(-0.5, value * 0.5)
            } else {
                value - anchor
            }
        };
        let scale = halves
            .clone()
            .flatten()
            .map(|&value| difference(value, origin).abs())
            .fold(0.0_f64, f64::max);
        let n = count_as_f64(half_length);
        let mut moments = Vec::with_capacity(2 * input.chains.len());
        for (index, half) in halves.enumerate() {
            let anchor = half[0];
            #[expect(
                clippy::float_cmp,
                reason = "constant means exactly equal represented observations, including signed zero"
            )]
            let constant = half.iter().all(|&value| value == anchor);
            if constant {
                return Err(SplitRhatError::ConstantSplitChain {
                    chain_index: index / 2,
                    half: index % 2,
                });
            }
            let delta = |value| difference(value, anchor) / scale;
            let mean_delta = compensated_sum(half.iter().map(|&value| delta(value))) / n;
            let variance = compensated_sum(half.iter().map(|&value| {
                let residual = delta(value) - mean_delta;
                residual * residual
            })) / (n - 1.0);
            if !variance.is_finite() || variance <= 0.0 {
                return Err(SplitRhatError::UnresolvedVariance {
                    chain_index: index / 2,
                    half: index % 2,
                });
            }
            moments.push((difference(anchor, origin) / scale + mean_delta, variance));
        }
        let m = count_as_f64(moments.len());
        let mean = compensated_sum(moments.iter().map(|&(mean, _)| mean)) / m;
        let between_over_n = compensated_sum(
            moments
                .iter()
                .map(|&(half_mean, _)| (half_mean - mean).powi(2)),
        ) / (m - 1.0);
        let within = compensated_sum(moments.into_iter().map(|(_, variance)| variance)) / m;
        // Taking square roots before division avoids overflowing B/W when
        // R-hat itself is still representable.
        let estimate = ((n - 1.0) / n).mul_add(within, between_over_n).sqrt() / within.sqrt();
        if !estimate.is_finite() || estimate <= 0.0 {
            return Err(SplitRhatError::NumericalFailure);
        }
        Ok(Self {
            estimate,
            chain_count: input.chains.len(),
            samples_per_chain: sample_count,
        })
    }

    /// Estimated dimensionless classical split R-hat, without a floor at one.
    ///
    /// Always positive and finite after successful estimation. A value near
    /// one means the estimated variances agree; it does not prove convergence.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.estimate
    }

    /// Number of original chains, before splitting.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.chain_count
    }

    /// Supplied sample count per original chain, including an omitted middle draw.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.samples_per_chain
    }

    /// Used sample count per split chain; half the original length, rounded down.
    #[must_use]
    pub const fn samples_per_split_chain(self) -> usize {
        self.samples_per_chain / 2
    }
}

/// Rank-normalized split R-hat component for one scalar observable.
///
/// Use [`Self::estimate`] with borrowed scalar chains, then read the component
/// with [`Self::value`] and inspect the original and retained sample counts.
///
/// This distinct type identifies the estimator; [`SplitRhat`] remains the
/// classical raw-moment estimator. Supply comparable post-warmup chains under
/// the sampling assumptions documented on [`SplitRhat`]. Callers own warmup,
/// recording cadence, chain independence, and any decision thresholds.
///
/// Split first, pool all retained draws, assign one-based average ranks to ties,
/// then transform each rank `r` to `Phi^-1((r - 3/8) / (S + 1/4))`, where `S`
/// counts retained draws across all chains. Apply the split variance ratio to
/// these normal scores, preserving chain and half identity. Signed zeros tie;
/// integer/count observations represented as `f64` need no jitter. Strictly
/// monotone transformations that preserve represented ties and ordering preserve
/// this diagnostic up to floating-point rounding, including order reversal.
/// Conversion to `f64` can merge distinct integers beyond `2^53`; preserve the
/// observable's relevant distinctions before supplying these scalar slices.
///
/// See Vehtari et al. (2021), [Sections 3.1 and 4.1, equation (14)](https://arxiv.org/html/1903.08008v5#S4.SS1).
/// Rank normalization avoids requiring finite marginal moments of the original
/// distribution, but every supplied draw must still be finite. This component
/// **omits folding and the combined maximum** from Section 4.2. It can miss
/// scale disagreement and does not certify convergence or adequate sampling.
///
/// # Examples
///
/// These short synthetic count traces demonstrate estimation and count metadata,
/// not adequate sampling. Real post-warmup traces must satisfy the comparability
/// assumptions above. With five draws per chain, each middle draw is omitted
/// before ranking and each retained half contains two draws.
///
/// ```
/// use markov_chain_monte_carlo::prelude::{RankNormalizedSplitRhat, SplitRhatError};
///
/// let counts_a = [2.0, 4.0, 3.0, 4.0, 6.0];
/// let counts_b = [3.0, 5.0, 4.0, 5.0, 7.0];
/// let result = RankNormalizedSplitRhat::estimate(&[&counts_a, &counts_b])?;
/// println!("Rank-normalized split R-hat component: {}", result.value());
/// assert_eq!(result.chain_count(), 2);
/// assert_eq!(result.samples_per_chain(), 5);
/// assert_eq!(result.samples_per_split_chain(), 2);
/// # Ok::<(), SplitRhatError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RankNormalizedSplitRhat {
    /// Variance ratio of normal scores, with original input counts restored.
    summary: SplitRhat,
}

impl RankNormalizedSplitRhat {
    /// Estimate the rank-normalized split component from borrowed scalar chains.
    ///
    /// Requires two or more equal-length original chains of at least four draws.
    /// Each contributes its first and last `floor(N/2)` draws, in that order;
    /// an odd middle draw is checked for finiteness but omitted before ranking.
    /// These minima define the formula, not its statistical reliability.
    /// Inputs remain unchanged and the result retains no borrow of them.
    /// Sorting costs `O(S log S)` time and `O(S)` auxiliary memory.
    ///
    /// Normal scores use Wichura's AS 241 inverse-normal approximation, with
    /// lower-tail evaluation and symmetry to avoid upper-tail cancellation.
    /// Independent fixtures check inverse-normal absolute error within `2e-14`
    /// and diagnostic relative error within `5e-13`; these are tested tolerances,
    /// not a correctly-rounded guarantee for every platform.
    ///
    /// # Errors
    ///
    /// Checks original chain count, lengths, then finiteness in the same order
    /// as [`SplitRhat::estimate`]. Next, [`SplitRhatError::TooManyRankedSamples`]
    /// rejects pooled retained counts above `2^50` or overflowing `usize`, before
    /// allocation. Its fields preserve the chain count, retained half length,
    /// and effective limit even when the pooled count cannot fit in `usize`.
    ///
    /// Moment errors from [`SplitRhat::estimate`] retain the original chain and
    /// half indices. Each half is checked for constancy, then unresolved score
    /// variance, in original chain order. Any constant half is rejected, including
    /// when other halves vary; signed zeros compare equal. A varying half with
    /// unresolvable score variance yields [`SplitRhatError::UnresolvedVariance`].
    /// [`SplitRhatError::NumericalFailure`] applies only after all halves pass
    /// these checks and the final ratio is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{RankNormalizedSplitRhat, SplitRhatError};
    ///
    /// let a = [0.0, 1.0, 0.0, 1.0];
    /// let b = [2.0, 3.0, 2.0, 3.0];
    /// let result = RankNormalizedSplitRhat::estimate(&[&a, &b])?;
    /// assert!(result.value() > 1.0);
    /// assert_eq!(result.chain_count(), 2);
    /// assert_eq!(result.samples_per_chain(), 4);
    /// assert_eq!(result.samples_per_split_chain(), 2);
    /// # Ok::<(), SplitRhatError>(())
    /// ```
    pub fn estimate(chains: &[&[f64]]) -> Result<Self, SplitRhatError> {
        let input = SplitRhatInput::parse(chains)?;
        let scores = input.normal_scores()?;
        let retained_per_chain = 2 * (input.samples_per_chain / 2);
        let ranked_chains: Vec<_> = scores.chunks_exact(retained_per_chain).collect();
        let mut summary = SplitRhat::estimate(&ranked_chains)?;
        summary.samples_per_chain = input.samples_per_chain;
        Ok(Self { summary })
    }

    /// Positive finite rank-normalized split component, without a floor at one.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.summary.value()
    }

    /// Number of original chains, before splitting.
    #[must_use]
    pub const fn chain_count(self) -> usize {
        self.summary.chain_count()
    }

    /// Original draws per chain, including any omitted middle draw.
    #[must_use]
    pub const fn samples_per_chain(self) -> usize {
        self.summary.samples_per_chain()
    }

    /// Retained draws per half; each original chain contributes two halves.
    #[must_use]
    pub const fn samples_per_split_chain(self) -> usize {
        self.summary.samples_per_split_chain()
    }
}

/// Borrowed chains with the shape and finite-input preconditions for R-hat moments.
///
/// Checking two or more chains and equal lengths of at least four makes the
/// estimator's indexing, half slicing, and `n - 1`/`m - 1` denominators valid.
/// The shared borrow preserves these facts throughout estimation. It does not
/// establish independence, stationarity, or a common target. Constant halves
/// and numerical failures are classified later during moment calculation.
struct SplitRhatInput<'a> {
    chains: &'a [&'a [f64]],
    samples_per_chain: usize,
}

impl<'a> SplitRhatInput<'a> {
    /// Rank only retained draws, then scatter scores back into original
    /// chain/first-half/last-half order. No raw-value arithmetic is needed.
    #[expect(
        clippy::float_cmp,
        reason = "ties mean exact represented equality, including signed zero"
    )]
    fn normal_scores(&self) -> Result<Vec<f64>, SplitRhatError> {
        let half_length = self.samples_per_chain / 2;
        let max_retained_samples = usize::try_from(1_u64 << 50).unwrap_or(usize::MAX);
        let count = self
            .chains
            .len()
            .checked_mul(2 * half_length)
            .filter(|&count| count <= max_retained_samples)
            .ok_or(SplitRhatError::TooManyRankedSamples {
                chain_count: self.chains.len(),
                samples_per_split_chain: half_length,
                max_retained_samples,
            })?;
        let mut ordered = Vec::with_capacity(count);
        ordered.extend(
            self.chains
                .iter()
                .flat_map(|chain| {
                    chain[..half_length]
                        .iter()
                        .chain(&chain[self.samples_per_chain - half_length..])
                })
                .copied()
                .enumerate(),
        );
        ordered.sort_unstable_by(|(_, left), (_, right)| left.total_cmp(right));
        let mut scores = vec![0.0; count];
        let total = count_as_f64(count);
        let mut start = 0;
        while start < count {
            let mut end = start + 1;
            while end < count && ordered[end].1 == ordered[start].1 {
                end += 1;
            }
            // Positions start..end have one-based average rank (start+1+end)/2.
            let rank = (count_as_f64(start) + count_as_f64(end) + 1.0) * 0.5;
            let lower = rank - 0.375;
            let upper = total - rank + 0.625;
            let score = inverse_normal_lower_tail(lower.min(upper) / (total + 0.25));
            let score = if lower <= upper { score } else { -score };
            for &(index, _) in &ordered[start..end] {
                scores[index] = score;
            }
            start = end;
        }
        Ok(scores)
    }

    /// Check lengths before values, with shortness before unequal length within
    /// each original chain. Check all samples, including omitted middle draws.
    fn parse(chains: &'a [&'a [f64]]) -> Result<Self, SplitRhatError> {
        if chains.len() < 2 {
            return Err(SplitRhatError::InsufficientChains {
                count: chains.len(),
            });
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if chain.len() < 4 {
                return Err(SplitRhatError::InsufficientSamples {
                    chain_index,
                    count: chain.len(),
                });
            }
            if chain.len() != chains[0].len() {
                return Err(SplitRhatError::UnequalLengths {
                    chain_index,
                    expected: chains[0].len(),
                    actual: chain.len(),
                });
            }
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if let Some(sample_index) = chain.iter().position(|value| !value.is_finite()) {
                return Err(SplitRhatError::NonFiniteSample {
                    chain_index,
                    sample_index,
                });
            }
        }
        Ok(Self {
            chains,
            samples_per_chain: chains[0].len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{SplitRhatError, SplitRhatInput};

    #[test]
    fn rank_count_overflow_precedes_sample_access() {
        // Synthetic private count metadata exercises an otherwise infeasible
        // allocation boundary. Empty backing slices deliberately fail if the
        // guard starts reading samples before rejecting the count.
        let chains: [&[f64]; 2] = [&[], &[]];
        let input = SplitRhatInput {
            chains: &chains,
            samples_per_chain: usize::MAX,
        };
        let error = input.normal_scores().unwrap_err();
        let SplitRhatError::TooManyRankedSamples {
            chain_count,
            samples_per_split_chain,
            max_retained_samples,
        } = error
        else {
            panic!("expected a rank-count error, got {error:?}");
        };
        assert_eq!(chain_count, 2);
        assert_eq!(samples_per_split_chain, usize::MAX / 2);
        assert_eq!(
            max_retained_samples as u128,
            (usize::MAX as u128).min(1_u128 << 50)
        );
        let message = error.to_string();
        for detail in [
            "2 chains".to_owned(),
            format!("{} samples per half", usize::MAX / 2),
            format!("{max_retained_samples} retained draws"),
        ] {
            assert!(message.contains(&detail), "missing {detail:?}: {message}");
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rank_count_limit_precedes_sample_access() {
        // Pooling 2 * (2^49 + 2) retained draws fits usize but exceeds 2^50.
        // As above, exercise only the count preflight with synthetic metadata.
        // On 32-bit targets the precision bound is above usize::MAX.
        let chains: [&[f64]; 2] = [&[], &[]];
        let input = SplitRhatInput {
            chains: &chains,
            samples_per_chain: (1_usize << 49) + 2,
        };
        assert_eq!(
            input.normal_scores().unwrap_err(),
            SplitRhatError::TooManyRankedSamples {
                chain_count: 2,
                samples_per_split_chain: (1_usize << 48) + 1,
                max_retained_samples: 1_usize << 50,
            }
        );
    }

    #[test]
    fn pooled_average_ranks_restore_original_chain_and_half_order() {
        let a = [3.0, 2.0, f64::MAX, 1.0, 2.0];
        let b = [4.0, -0.0, -f64::MAX, 0.0, 2.0];
        let chains = [&a[..], &b[..]];
        let scores = SplitRhatInput::parse(&chains)
            .unwrap()
            .normal_scores()
            .unwrap();
        // Retained draws: [3,2,1,2,4,-0,+0,2]. Average pooled ranks:
        // [7,5,3,5,8,1.5,1.5,5]. In particular, signed zeros tie across halves.
        // scipy.special.ndtri((rank - 0.375) / 8.25), SciPy 1.16.2.
        let expected = [
            0.852_495_034_274_693_9,
            0.152_505_974_246_244_24,
            -0.472_789_120_992_267_4,
            0.152_505_974_246_244_24,
            1.434_200_159_686_379_4,
            -1.096_803_562_093_512_8,
            -1.096_803_562_093_512_8,
            0.152_505_974_246_244_24,
        ];
        assert_eq!(scores.len(), expected.len());
        for (score, reference) in scores.iter().zip(expected) {
            assert_abs_diff_eq!(*score, reference, epsilon = 2e-14);
        }
        assert_eq!(scores[5].to_bits(), scores[6].to_bits());
        assert_eq!(scores[1].to_bits(), scores[3].to_bits());
        assert_eq!(scores[1].to_bits(), scores[7].to_bits());
    }
}
