//! Fixed reference distributions for sampler validation and mixing experiments.

use std::f64::consts::LN_2;

use crate::Target;

/// Two-dimensional reference targets with analytical population moments.
///
/// Available with the `benchmarks` Cargo feature. Each variant implements
/// [`Target<[f64; 2]>`] for coordinates `[x, y]` on the whole real plane, using
/// the fixed parameters below. `N(mean, variance)` denotes a normal distribution
/// parameterized by its **variance**, and all auxiliary normal draws are independent.
/// These presets intentionally fix dimension and difficulty so comparisons name
/// the same distribution. They introduce no additional dependencies.
///
/// # Numerical behavior
///
/// Log weights omit state-independent normalizing constants. Any non-finite
/// coordinate returns `NaN`, which the chain reports as an invalid target value.
/// For finite coordinates, a negative log weight outside the representable
/// range becomes negative infinity. Extremely small tail contributions can
/// underflow. As with other `f64` targets, these are floating-point evaluations,
/// not exact real arithmetic.
///
/// # Examples
///
/// ```
/// use markov_chain_monte_carlo::{BenchmarkTarget, Chain, Target};
///
/// let target = BenchmarkTarget::Rosenbrock;
/// let chain = Chain::new([1.0, 1.0], &target)?;
/// assert_eq!(target.log_prob(chain.state()), 0.0);
/// // The mean differs from the mode [1, 1].
/// assert_eq!(target.mean(), [1.0, 1.5]);
/// assert_eq!(target.covariance(), [[0.5, 1.0], [1.0, 2.505]]);
/// # Ok::<(), markov_chain_monte_carlo::McmcError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BenchmarkTarget {
    /// Rosenbrock density proportional to `exp(-(x - 1)^2 - 100(y - x^2)^2)`.
    ///
    /// Equivalently, `X ~ N(1, 1/2)` and `Y | X ~ N(X^2, 1/200)`.
    /// Mean: `[1, 3/2]`; covariance: `[[1/2, 1], [1, 501/200]]`.
    /// The normalization integral of the returned log weight is `pi / 10`.
    /// This is the two-dimensional Rosenbrock family described by
    /// [Pagani, Wiegand, and Nadarajah](https://arxiv.org/abs/1903.09556),
    /// with coefficients `a = 1`, `b = 100`, and location `mu = 1`.
    Rosenbrock,
    /// Two-dimensional marginal of Neal's funnel: `X ~ N(0, 9)` and
    /// `Y | X ~ N(0, exp(X))`.
    ///
    /// Coordinate zero is the log-variance variable; the conditional standard
    /// deviation is `exp(X/2)`. The log weight includes the essential `-X/2`
    /// conditional normalization term. Mean: `[0, 0]`; covariance:
    /// `diag(9, exp(9/2))`. The normalization integral is `6 pi`.
    /// This uses the scale convention in the
    /// [Stan funnel example](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#example-neals-funnel),
    /// retaining one lower-level coordinate from the original ten-dimensional model.
    NealsFunnel,
    /// Equal mixture of two unit-covariance Gaussians centered at `[-5, 0]`
    /// and `[5, 0]`.
    ///
    /// Mean: `[0, 0]`; covariance: `diag(26, 1)`. The returned log weight
    /// includes the mixture weights `1/2`; its normalization integral is `2 pi`.
    /// Log-space evaluation preserves finite weights between widely separated modes.
    GaussianMixture,
    /// Banana-shaped Gaussian shear: `X ~ N(0, 100)` and
    /// `Y = Z + 0.03 (X^2 - 100)`, where `Z ~ N(0, 1)` is independent of `X`.
    ///
    /// Mean: `[0, 0]`; covariance: `diag(100, 19)`. The shear has unit
    /// Jacobian, so no state-dependent normalization is needed. The normalization
    /// integral is `20 pi`. The positive bend convention is fixed by this formula.
    Banana,
}

impl BenchmarkTarget {
    /// Analytical population mean in coordinate order `[x, y]`.
    #[must_use]
    pub const fn mean(&self) -> [f64; 2] {
        match self {
            Self::Rosenbrock => [1.0, 1.5],
            Self::NealsFunnel | Self::GaussianMixture | Self::Banana => [0.0, 0.0],
        }
    }

    /// Analytical population covariance: entry `[i][j]` is `Cov(state[i], state[j])`.
    ///
    /// Zero covariance does not imply independence for the funnel or banana.
    /// The funnel's second-coordinate variance is evaluated as `exp(4.5)` in `f64`.
    #[must_use]
    pub fn covariance(&self) -> [[f64; 2]; 2] {
        match self {
            Self::Rosenbrock => [[0.5, 1.0], [1.0, 2.505]],
            Self::NealsFunnel => [[9.0, 0.0], [0.0, 4.5_f64.exp()]],
            Self::GaussianMixture => [[26.0, 0.0], [0.0, 1.0]],
            Self::Banana => [[100.0, 0.0], [0.0, 19.0]],
        }
    }
}

impl Target<[f64; 2]> for BenchmarkTarget {
    fn log_prob(&self, &[x, y]: &[f64; 2]) -> f64 {
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        match self {
            Self::Rosenbrock => {
                let centered = x - 1.0;
                let residual = 10.0 * (-x).mul_add(x, y);
                (-centered).mul_add(centered, -(residual * residual))
            }
            Self::NealsFunnel => {
                // Compute y^2 exp(-x) / 2 in log space to avoid overflowing
                // either y^2 or exp(-x) when their product is representable.
                // Zero needs its own branch, including in the narrowest neck.
                let conditional_energy = if y == 0.0 {
                    0.0
                } else {
                    (2.0_f64.mul_add(y.abs().ln(), -x) - LN_2).exp()
                };
                let standardized = x / 3.0;
                0.5_f64.mul_add(-x, -0.5 * standardized * standardized) - conditional_energy
            }
            Self::GaussianMixture => {
                // Factored log-sum-exp: the nearer component is dominant and
                // the farther/nearer density ratio is exp(-10 |x|). This also
                // avoids subtracting two negative infinities in the far tails.
                let distance = x.abs() - 5.0;
                (0.5 * y).mul_add(-y, -0.5 * distance * distance) + (-10.0 * x.abs()).exp().ln_1p()
                    - LN_2
            }
            Self::Banana => {
                let standardized = x / 10.0;
                // Scale before squaring so x^2 alone need not be representable.
                let bend = (0.03 * x).mul_add(x, -3.0);
                let residual = y - bend;
                (0.5 * residual).mul_add(-residual, -0.5 * standardized * standardized)
            }
        }
    }
}
