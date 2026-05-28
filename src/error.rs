//! Error types for MCMC operations.

use std::error::Error;
use std::fmt;

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
    /// one transition but [`crate::DelayedProposal::commit`] applies a
    /// different one.
    InconsistentDelayedCommitLogProb,
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
            Self::InconsistentDelayedCommitLogProb => {
                write!(
                    f,
                    "checked delayed commit produced a state with a different log-probability than the accepted plan"
                )
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
    fn display_nan_initial_log_prob() {
        let msg = McmcError::NanInitialLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability for the initial state"
        );
    }

    #[test]
    fn display_nan_proposed_log_prob() {
        let msg = McmcError::NanProposedLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability for a proposed state"
        );
    }

    #[test]
    fn display_nan_log_q_ratio() {
        let msg = McmcError::NanLogQRatio.to_string();
        assert_eq!(msg, "proposal returned NaN log q-ratio");
    }

    #[test]
    fn display_nan_replacement_log_prob() {
        let msg = McmcError::NanReplacementLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability for a replacement state"
        );
    }

    #[test]
    fn display_nan_checkpoint_log_prob() {
        let msg = McmcError::NanCheckpointLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability for a checkpoint state"
        );
    }

    #[test]
    fn display_nan_current_log_prob() {
        let msg = McmcError::NanCurrentLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability for the current chain state"
        );
    }

    #[test]
    fn display_nan_committed_log_prob() {
        let msg = McmcError::NanCommittedLogProb.to_string();
        assert_eq!(
            msg,
            "target returned NaN log-probability after a checked delayed commit"
        );
    }

    #[test]
    fn display_inconsistent_delayed_commit_log_prob() {
        let msg = McmcError::InconsistentDelayedCommitLogProb.to_string();
        assert_eq!(
            msg,
            "checked delayed commit produced a state with a different log-probability than the accepted plan"
        );
    }

    #[test]
    fn display_infinite_initial_log_prob() {
        let msg = McmcError::InfiniteInitialLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability for the initial state"
        );
    }

    #[test]
    fn display_inf_proposed_log_prob() {
        let msg = McmcError::InfiniteProposedLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability for a proposed state"
        );
    }

    #[test]
    fn display_infinite_log_q_ratio() {
        let msg = McmcError::InfiniteLogQRatio.to_string();
        assert_eq!(msg, "proposal returned +inf log q-ratio");
    }

    #[test]
    fn display_infinite_replacement_log_prob() {
        let msg = McmcError::InfiniteReplacementLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability for a replacement state"
        );
    }

    #[test]
    fn display_infinite_checkpoint_log_prob() {
        let msg = McmcError::InfiniteCheckpointLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability for a checkpoint state"
        );
    }

    #[test]
    fn display_infinite_current_log_prob() {
        let msg = McmcError::InfiniteCurrentLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability for the current chain state"
        );
    }

    #[test]
    fn display_infinite_committed_log_prob() {
        let msg = McmcError::InfiniteCommittedLogProb.to_string();
        assert_eq!(
            msg,
            "target returned +inf log-probability after a checked delayed commit"
        );
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
