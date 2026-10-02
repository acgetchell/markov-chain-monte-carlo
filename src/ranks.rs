//! Pooled average ranks in original chain and draw order, for consumer plots.

use std::{error::Error, fmt, ops::Range};

use crate::numerics::count_as_f64;

/// Invalid observations or an unsupported count for [`PooledRanks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PooledRankError {
    /// Supply at least one original chain.
    NoChains,
    /// Every supplied chain must contain at least one draw.
    #[non_exhaustive]
    EmptyChain {
        /// Zero-based original chain position in the input slice.
        chain_index: usize,
    },
    /// Every draw must be finite.
    #[non_exhaustive]
    NonFiniteSample {
        /// Zero-based original chain position in the input slice.
        chain_index: usize,
        /// Zero-based original draw position within that chain.
        sample_index: usize,
    },
    /// The pooled count exceeds `min(2^50, usize::MAX)` or overflows `usize`.
    #[non_exhaustive]
    TooManySamples {
        /// Number of draws counted before the rejected addition.
        current_count: usize,
        /// Number of draws in the next chain.
        additional_count: usize,
        /// Largest supported pooled count on this architecture.
        max_samples: usize,
    },
}

impl fmt::Display for PooledRankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoChains => f.write_str("pooled ranks require at least one chain"),
            Self::EmptyChain { chain_index } => {
                write!(f, "chain {chain_index} has no draws to rank")
            }
            Self::NonFiniteSample {
                chain_index,
                sample_index,
            } => {
                write!(f, "chain {chain_index} sample {sample_index} is not finite")
            }
            Self::TooManySamples {
                current_count,
                additional_count,
                max_samples,
            } => {
                write!(
                    f,
                    "pooled rank count {current_count} + {additional_count} exceeds supported maximum {max_samples}"
                )
            }
        }
    }
}

impl Error for PooledRankError {}

/// One-based pooled average ranks for plotting original chains.
///
/// [`Self::from_chains`] ranks **every supplied draw**, including odd middle
/// draws, and preserves the original chain and draw order. Each chain's index
/// is its position in the input slice; retain application identifiers such as
/// [`crate::ChainId`] alongside that slice. The returned ranks own their storage;
/// [`Self::chain`] borrows a view without exposing mutable state.
///
/// Exactly equal represented values, including `-0.0` and `0.0`, receive the
/// average of their occupied one-based ranks. No normal-score transformation,
/// jitter, warmup removal, or chain splitting occurs. Recompute ranks after
/// selecting a prefix; ranks from the complete run do not describe that prefix.
///
/// Unlike split diagnostics, plotting accepts a single chain, unequal lengths,
/// and constant chains. A constant pool has rank `(S+1)/2` everywhere. Discrete
/// ties can give nonuniform pooled histograms even for matching distributions:
/// compare each chain's bin proportions with the pooled bin proportions, rather
/// than treating a continuous-uniform reference as a convergence test.
///
/// Rank-based split R-hat and ESS omit odd middle draws **before** ranking.
/// These plot ranks therefore need not match their internal ranks. An explicit
/// half-chain plot must select its halves first and label them as such.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PooledRanks {
    ranks: Vec<f64>,
    chains: Vec<Range<usize>>,
}

impl PooledRanks {
    /// Rank finite observations without changing them or their identities.
    ///
    /// Uses `O(S log S)` time and `O(S + M)` storage for `S` draws in `M` chains.
    /// The count bound preserves exact integer and half-integer ranks and is
    /// shared with the crate's rank-normalized diagnostics.
    ///
    /// # Errors
    ///
    /// Rejects no chains, an empty chain, an unsupported pooled count, or a
    /// nonfinite draw with [`PooledRankError`]. Shape/count checks precede value
    /// checks and rank allocation. Constant observations remain valid plot data.
    ///
    /// # Examples
    ///
    /// ```
    /// use markov_chain_monte_carlo::{PooledRanks, PooledRankError};
    /// let ranks = PooledRanks::from_chains(&[&[3., 1., 1.], &[2., 3.]])?;
    /// assert_eq!(ranks.chain(0), Some([4.5, 1.5, 1.5].as_slice()));
    /// assert_eq!(ranks.chain(1), Some([3., 4.5].as_slice()));
    /// assert_eq!(ranks.sample_count(), 5);
    /// # Ok::<(), PooledRankError>(())
    /// ```
    pub fn from_chains(chains: &[&[f64]]) -> Result<Self, PooledRankError> {
        if chains.is_empty() {
            return Err(PooledRankError::NoChains);
        }
        let mut count = 0_usize;
        for (chain_index, chain) in chains.iter().enumerate() {
            if chain.is_empty() {
                return Err(PooledRankError::EmptyChain { chain_index });
            }
            count = checked_rank_count(count, chain.len())?;
        }
        for (chain_index, chain) in chains.iter().enumerate() {
            if let Some(sample_index) = chain.iter().position(|value| !value.is_finite()) {
                return Err(PooledRankError::NonFiniteSample {
                    chain_index,
                    sample_index,
                });
            }
        }
        let ranks = rank_transform(
            chains.iter().flat_map(|chain| chain.iter().copied()),
            count,
            |rank| rank,
        );
        let mut start = 0;
        let chains = chains
            .iter()
            .map(|chain| {
                let end = start + chain.len();
                let range = start..end;
                start = end;
                range
            })
            .collect();
        Ok(Self { ranks, chains })
    }

    /// Borrow ranks for an original chain, in its original draw order.
    /// Returns `None` for an index outside `0..self.chain_count()`.
    #[must_use]
    pub fn chain(&self, chain_index: usize) -> Option<&[f64]> {
        self.chains
            .get(chain_index)
            .map(|range| &self.ranks[range.clone()])
    }

    /// Number of original chains; no implicit splitting is performed.
    #[must_use]
    pub const fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Number of ranked observations, including any odd middle draws.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.ranks.len()
    }
}

/// Cap ranks at `2^50` to preserve the fractional offsets used by shared
/// normal-score diagnostics, and at `usize::MAX` for the architecture's storage.
#[expect(
    clippy::redundant_pub_crate,
    reason = "shared rank precision bound is crate-internal"
)]
pub(super) fn max_ranked_samples() -> usize {
    usize::try_from(1_u64 << 50).unwrap_or(usize::MAX)
}

/// Reject count overflow and unsupported rank precision before allocation.
fn checked_rank_count(current: usize, additional: usize) -> Result<usize, PooledRankError> {
    current
        .checked_add(additional)
        .filter(|&count| count <= max_ranked_samples())
        .ok_or_else(|| PooledRankError::TooManySamples {
            current_count: current,
            additional_count: additional,
            max_samples: max_ranked_samples(),
        })
}

/// Shared tie grouping and reconstruction. Callers prove finite values and a
/// matching count within the rank bound before allocating or transforming.
#[expect(
    clippy::float_cmp,
    reason = "ties are exact represented equality, including signed zero"
)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "unchecked rank transform is only for validated internal callers"
)]
pub(super) fn rank_transform(
    values: impl Iterator<Item = f64>,
    count: usize,
    transform: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let mut ordered = Vec::with_capacity(count);
    ordered.extend(values.enumerate());
    ordered.sort_unstable_by(|(_, left), (_, right)| left.total_cmp(right));
    let mut ranks = vec![0.0; count];
    let mut start = 0;
    while start < count {
        let mut end = start + 1;
        while end < count && ordered[end].1 == ordered[start].1 {
            end += 1;
        }
        let rank = (count_as_f64(start) + count_as_f64(end) + 1.0) * 0.5;
        let value = transform(rank);
        for &(index, _) in &ordered[start..end] {
            ranks[index] = value;
        }
        start = end;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::{PooledRankError, checked_rank_count, max_ranked_samples};

    #[test]
    fn rank_counts_reject_precision_and_storage_overflow_before_allocation() {
        let limit = max_ranked_samples();
        assert_eq!(checked_rank_count(limit - 1, 1), Ok(limit));
        assert_eq!(
            checked_rank_count(limit, 1),
            Err(PooledRankError::TooManySamples {
                current_count: limit,
                additional_count: 1,
                max_samples: limit,
            })
        );
        assert_eq!(
            checked_rank_count(usize::MAX, 1),
            Err(PooledRankError::TooManySamples {
                current_count: usize::MAX,
                additional_count: 1,
                max_samples: limit,
            })
        );
        for current in [limit, usize::MAX] {
            let message = checked_rank_count(current, 1).unwrap_err().to_string();
            assert!(message.contains(&format!("{current} + 1")));
            assert!(message.contains(&format!("maximum {limit}")));
        }
    }
}
