//! Internal scalar arithmetic shared by diagnostics and streaming statistics.
//!
//! Domain modules own input validation and statistical policy. Keeping these
//! primitives here avoids dependencies between otherwise independent estimators.

#![expect(
    clippy::redundant_pub_crate,
    reason = "restrict helper visibility so accidental public re-exports fail to compile"
)]

/// Reduce accumulation error in an ordered sum using Kahan summation.
///
/// ACF, integrated-time, split R-hat, and proposal-bin checks share this
/// arithmetic. Callers supply finite terms scaled so that the sum and its
/// intermediates remain representable. Compensation limits rounding error,
/// including cancellation in signed sums; it does not validate inputs or
/// prevent overflow.
pub(crate) fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut accumulator = CompensatedSum::default();
    for value in values {
        accumulator.add(value);
    }
    accumulator.total()
}

/// Independent Kahan state for one ordered sum of scaled finite terms.
///
/// Incremental access lets the ACF accumulate several lag covariances together
/// without changing the arithmetic order within any individual covariance.
#[derive(Clone, Copy, Default)]
pub(crate) struct CompensatedSum {
    /// Accumulated value, including the most recent adjusted term.
    sum: f64,
    /// Rounding correction carried into the next addition.
    correction: f64,
}

impl CompensatedSum {
    /// Incorporate one term in order, retaining its rounding correction.
    pub(crate) fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }

    /// Read the accumulated value without resetting the rounding correction.
    #[must_use]
    pub(crate) const fn total(&self) -> f64 {
        self.sum
    }
}

/// Convert an observation, lag-pair, chain, or bin count to floating point.
///
/// This is the shared integer-to-float boundary for diagnostics and streaming
/// statistics. It does not check minimum counts or denominator validity.
/// Counts through `2^53` are exact; larger counts may be rounded. Callers that
/// require exact conversion, such as proposal-bin checks, enforce that bound
/// before calling this helper.
#[expect(
    clippy::cast_precision_loss,
    reason = "callers own count bounds; counts beyond 2^53 may be rounded"
)]
pub(crate) const fn count_as_f64(count: usize) -> f64 {
    count as f64
}

#[cfg(test)]
mod tests {
    use super::compensated_sum;

    #[test]
    fn compensated_sum_accepts_iterables_and_preserves_small_terms() {
        // An array is IntoIterator but not Iterator. Two half-ULP terms
        // would both disappear in a naive left-to-right sum starting at 1.
        let half_ulp = f64::EPSILON / 2.0;
        let sum = compensated_sum([1.0, half_ulp, half_ulp, -1.0]);
        assert_eq!(sum.to_bits(), f64::EPSILON.to_bits());
    }
}
