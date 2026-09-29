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

/// Standard-normal quantile for `0 < p <= 0.5`, using Wichura's AS 241
/// rational approximations (1988), DOI: <https://doi.org/10.2307/2347330>.
/// Coefficient tables can also be checked against `CPython` 3.13.7's
/// `Modules/_statisticsmodule.c`. This evaluates the mathematical tables with
/// Horner's rule with fused multiply-add; callers own the domain check and
/// reflect upper-tail ranks before division so that computing `1 - p` never
/// loses tail precision.
pub(crate) fn inverse_normal_lower_tail(p: f64) -> f64 {
    let q = p - 0.5;
    if q >= -0.425 {
        let r = 0.180_625 - q * q;
        let numerator = polynomial(
            r,
            &[
                2_509.080_928_730_122_7,
                33_430.575_583_588_13,
                67_265.770_927_008_7,
                45_921.953_931_549_87,
                13_731.693_765_509_46,
                1_971.590_950_306_551_3,
                133.141_667_891_784_38,
                3.387_132_872_796_366_5,
            ],
        );
        let denominator = polynomial(
            r,
            &[
                5_226.495_278_852_854,
                28_729.085_735_721_943,
                39_307.895_800_092_71,
                21_213.794_301_586_597,
                5_394.196_021_424_751,
                687.187_007_492_057_9,
                42.313_330_701_600_91,
                1.0,
            ],
        );
        return q * numerator / denominator;
    }
    let r = (-p.ln()).sqrt();
    let (r, numerator, denominator) = if r <= 5.0 {
        (
            r - 1.6,
            [
                7.745_450_142_783_414e-4,
                0.022_723_844_989_269_184,
                0.241_780_725_177_450_6,
                1.270_458_252_452_368_4,
                3.647_848_324_763_204_5,
                5.769_497_221_460_691,
                4.630_337_846_156_546,
                1.423_437_110_749_683_5,
            ],
            [
                1.050_750_071_644_416_9e-9,
                5.475_938_084_995_345e-4,
                0.015_198_666_563_616_457,
                0.148_103_976_427_480_08,
                0.689_767_334_985_1,
                1.676_384_830_183_803_8,
                2.053_191_626_637_759,
                1.0,
            ],
        )
    } else {
        (
            r - 5.0,
            [
                2.010_334_399_292_288e-7,
                2.711_555_568_743_487_6e-5,
                0.001_242_660_947_388_078_4,
                0.026_532_189_526_576_123,
                0.296_560_571_828_504_87,
                1.784_826_539_917_291_3,
                5.463_784_911_164_114,
                6.657_904_643_501_104,
            ],
            [
                2.044_263_103_389_939_7e-15,
                1.421_511_758_316_446e-7,
                1.846_318_317_510_054_8e-5,
                7.868_691_311_456_133e-4,
                0.014_875_361_290_850_615,
                0.136_929_880_922_735_8,
                0.599_832_206_555_887_9,
                1.0,
            ],
        )
    };
    -polynomial(r, &numerator) / polynomial(r, &denominator)
}

/// Evaluate coefficients ordered from highest to lowest degree.
fn polynomial(x: f64, coefficients: &[f64; 8]) -> f64 {
    coefficients
        .iter()
        .fold(0.0, |value, &coefficient| value.mul_add(x, coefficient))
}

#[cfg(test)]
mod tests {
    use super::{compensated_sum, inverse_normal_lower_tail};

    #[test]
    fn inverse_normal_matches_independent_scipy_quantiles() {
        // scipy.special.ndtri(p), SciPy 1.16.2 (Cephes, independently of AS 241).
        // Includes both region boundaries, both tails used by the rank-count
        // bound, the center, and its nearby cancellation-sensitive values.
        for (p, expected) in [
            (0.5, 0.0),
            (0.49, -0.025_068_908_258_711_06),
            (0.25, -0.674_489_750_196_081_7),
            (0.075, -1.439_531_470_938_456_3),
            (0.075_000_000_000_000_1, -1.439_531_470_938_455),
            (0.074_999_999_999_999_9, -1.439_531_470_938_456_8),
            (0.01, -2.326_347_874_040_840_8),
            (1e-6, -4.753_424_308_822_899),
            (1.388_794_386_496_402_1e-11, -6.657_904_643_501_103),
            (1.388_794_386_5e-11, -6.657_904_643_500_722),
            (1.388_794_386_49e-11, -6.657_904_643_501_782),
            (1e-12, -7.034_483_825_301_131),
            (5.551_115_123_125_782e-16, -8.014_015_948_775_546),
            (0.499_999_999_999_999_9, -2.782_916_424_671_767e-16),
        ] {
            let actual = inverse_normal_lower_tail(p);
            // Also constrain relative error near the center, where an absolute
            // bound alone would allow returning zero for a nonzero quantile.
            let error = (actual - expected).abs();
            assert!(
                error <= 2e-14 && error <= 2e-14 * expected.abs(),
                "p={p:e}: actual={actual:e}, expected={expected:e}, error={error:e}"
            );
        }
    }

    #[test]
    fn compensated_sum_accepts_iterables_and_preserves_small_terms() {
        // An array is IntoIterator but not Iterator. Two half-ULP terms
        // would both disappear in a naive left-to-right sum starting at 1.
        let half_ulp = f64::EPSILON / 2.0;
        let sum = compensated_sum([1.0, half_ulp, half_ulp, -1.0]);
        assert_eq!(sum.to_bits(), f64::EPSILON.to_bits());
    }
}
