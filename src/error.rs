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
