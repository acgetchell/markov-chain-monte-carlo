//! Error types for MCMC operations.

use std::error::Error;
use std::fmt;

/// The target scores that disagreed after a checked delayed commit.
///
/// Values of this type are produced only after both scores have passed the
/// crate's log-probability validation, so equality is well-defined even though
/// the underlying representation is `f64`.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct DelayedCommitLogProbMismatch {
    scored: f64,
    committed: f64,
}

impl DelayedCommitLogProbMismatch {
    pub(crate) const fn new(scored: f64, committed: f64) -> Self {
        Self { scored, committed }
    }

    /// Log-probability used for the acceptance decision.
    #[must_use]
    pub const fn scored(self) -> f64 {
        self.scored
    }

    /// Log-probability obtained by re-scoring the committed state.
    #[must_use]
    pub const fn committed(self) -> f64 {
        self.committed
    }
}

impl PartialEq for DelayedCommitLogProbMismatch {
    fn eq(&self, other: &Self) -> bool {
        self.scored == other.scored && self.committed == other.committed
    }
}

impl Eq for DelayedCommitLogProbMismatch {}

/// Errors that can occur during MCMC operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum McmcError {
    /// Target returned NaN log-probability for the initial state.
    NanInitialLogProb,
    /// Target returned NaN log-probability for a proposed state.
    NanProposedLogProb,
    /// Proposal returned NaN log q-ratio.
    NanLogQRatio,
    /// Target returned NaN log-probability for a replacement state.
    NanReplacementLogProb,
    /// Target returned NaN log-probability for a checkpoint state.
    NanCheckpointLogProb,
    /// Target returned NaN log-probability for the current chain state.
    NanCurrentLogProb,
    /// Target returned NaN log-probability after a checked delayed commit.
    ///
    /// This is reported by [`crate::Chain::step_delayed_checked`] or
    /// [`crate::Sampler::step_delayed_checked`] after an accepted delayed
    /// proposal has been committed and re-scored.
    NanCommittedLogProb,
    /// A checked delayed proposal committed a state whose target log-probability
    /// differs from the log-probability used for the acceptance decision.
    ///
    /// This is reported by [`crate::Chain::step_delayed_checked`] or
    /// [`crate::Sampler::step_delayed_checked`] when the delayed plan scores
    /// one target score but [`crate::DelayedProposal::commit`] produces another.
    /// Equal target scores do not establish state or transition identity.
    #[non_exhaustive]
    InconsistentDelayedCommitLogProb {
        /// The score used for acceptance and the score observed after commit.
        mismatch: DelayedCommitLogProbMismatch,
    },
    /// An in-place transition lost the undo token required for guarded access
    /// or explicit rollback.
    ///
    /// This reports an internal transition invariant failure without panicking.
    MissingRollbackToken,
    /// Target returned +∞ log-probability for the initial state.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.
    InfiniteInitialLogProb,
    /// Target returned +∞ log-probability for a proposed state.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.  If accepted, the chain
    /// would become permanently stuck.
    InfiniteProposedLogProb,
    /// Proposal returned +∞ log q-ratio.
    ///
    /// This indicates a degenerate proposal where the forward transition
    /// probability is zero (yet a state was somehow proposed), almost
    /// certainly a bug in the proposal implementation.
    InfiniteLogQRatio,
    /// Target returned +∞ log-probability for a replacement state.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.
    InfiniteReplacementLogProb,
    /// Target returned +∞ log-probability for a checkpoint state.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.
    InfiniteCheckpointLogProb,
    /// Target returned +∞ log-probability for the current chain state.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.
    InfiniteCurrentLogProb,
    /// Target returned +∞ log-probability after a checked delayed commit.
    ///
    /// This is reported by [`crate::Chain::step_delayed_checked`] or
    /// [`crate::Sampler::step_delayed_checked`] after an accepted delayed
    /// proposal has been committed and re-scored.
    ///
    /// This indicates infinite probability, which is invalid for any
    /// proper (normalizable) distribution.
    InfiniteCommittedLogProb,
}

impl fmt::Display for McmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanInitialLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for the initial state"
                )
            }
            Self::NanProposedLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for a proposed state"
                )
            }
            Self::NanLogQRatio => write!(f, "proposal returned NaN log q-ratio"),
            Self::NanReplacementLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for a replacement state"
                )
            }
            Self::NanCheckpointLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for a checkpoint state"
                )
            }
            Self::NanCurrentLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability for the current chain state"
                )
            }
            Self::NanCommittedLogProb => {
                write!(
                    f,
                    "target returned NaN log-probability after a checked delayed commit"
                )
            }
            Self::InconsistentDelayedCommitLogProb { mismatch } => {
                write!(
                    f,
                    "checked delayed commit used log-probability {} for acceptance but committed a state with log-probability {}",
                    mismatch.scored(),
                    mismatch.committed(),
                )
            }
            Self::MissingRollbackToken => {
                write!(f, "in-place rollback guard is missing its undo token")
            }
            Self::InfiniteInitialLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability for the initial state"
                )
            }
            Self::InfiniteProposedLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability for a proposed state"
                )
            }
            Self::InfiniteLogQRatio => write!(f, "proposal returned +inf log q-ratio"),
            Self::InfiniteReplacementLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability for a replacement state"
                )
            }
            Self::InfiniteCheckpointLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability for a checkpoint state"
                )
            }
            Self::InfiniteCurrentLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability for the current chain state"
                )
            }
            Self::InfiniteCommittedLogProb => {
                write!(
                    f,
                    "target returned +inf log-probability after a checked delayed commit"
                )
            }
        }
    }
}

impl Error for McmcError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_name_each_error() {
        let mismatch = DelayedCommitLogProbMismatch::new(-0.5, -2.0);
        let cases = [
            (
                McmcError::NanInitialLogProb,
                "target returned NaN log-probability for the initial state",
            ),
            (
                McmcError::NanProposedLogProb,
                "target returned NaN log-probability for a proposed state",
            ),
            (McmcError::NanLogQRatio, "proposal returned NaN log q-ratio"),
            (
                McmcError::NanReplacementLogProb,
                "target returned NaN log-probability for a replacement state",
            ),
            (
                McmcError::NanCheckpointLogProb,
                "target returned NaN log-probability for a checkpoint state",
            ),
            (
                McmcError::NanCurrentLogProb,
                "target returned NaN log-probability for the current chain state",
            ),
            (
                McmcError::NanCommittedLogProb,
                "target returned NaN log-probability after a checked delayed commit",
            ),
            (
                McmcError::InconsistentDelayedCommitLogProb { mismatch },
                "checked delayed commit used log-probability -0.5 for acceptance but committed a state with log-probability -2",
            ),
            (
                McmcError::MissingRollbackToken,
                "in-place rollback guard is missing its undo token",
            ),
            (
                McmcError::InfiniteInitialLogProb,
                "target returned +inf log-probability for the initial state",
            ),
            (
                McmcError::InfiniteProposedLogProb,
                "target returned +inf log-probability for a proposed state",
            ),
            (
                McmcError::InfiniteLogQRatio,
                "proposal returned +inf log q-ratio",
            ),
            (
                McmcError::InfiniteReplacementLogProb,
                "target returned +inf log-probability for a replacement state",
            ),
            (
                McmcError::InfiniteCheckpointLogProb,
                "target returned +inf log-probability for a checkpoint state",
            ),
            (
                McmcError::InfiniteCurrentLogProb,
                "target returned +inf log-probability for the current chain state",
            ),
            (
                McmcError::InfiniteCommittedLogProb,
                "target returned +inf log-probability after a checked delayed commit",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected, "wrong message for {error:?}");
        }

        assert_eq!(mismatch.scored().to_bits(), (-0.5_f64).to_bits());
        assert_eq!(mismatch.committed().to_bits(), (-2.0_f64).to_bits());
    }

    #[test]
    fn error_is_copy() {
        let err = McmcError::NanInitialLogProb;
        let err2 = err; // copy
        assert_eq!(err, err2);
    }

    #[test]
    fn error_implements_std_error() {
        let err: &dyn Error = &McmcError::NanLogQRatio;
        // source() should be None for leaf errors
        assert!(err.source().is_none());
    }
}
