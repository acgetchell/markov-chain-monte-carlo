//! Independent density, moment, and numerical-boundary checks for benchmark targets.

#![cfg(feature = "benchmarks")]
#![expect(
    clippy::suboptimal_flops,
    reason = "Independent oracle formulas deliberately use ordinary unfused arithmetic"
)]

use std::f64::consts::{LN_2, PI, SQRT_2};

use approx::assert_relative_eq;
use markov_chain_monte_carlo::{BenchmarkTarget, Chain, McmcError, Target};

const TARGETS: [BenchmarkTarget; 4] = [
    BenchmarkTarget::Rosenbrock,
    BenchmarkTarget::NealsFunnel,
    BenchmarkTarget::GaussianMixture,
    BenchmarkTarget::Banana,
];

#[test]
fn known_log_weights_and_coordinate_conventions() {
    // Exact points from the defining densities, including the funnel normalizer.
    for (target, point, expected) in [
        (BenchmarkTarget::Rosenbrock, [1.0, 1.0], 0.0),
        (BenchmarkTarget::Rosenbrock, [0.0, 0.0], -1.0),
        (BenchmarkTarget::Rosenbrock, [1.0, 2.0], -100.0),
        (BenchmarkTarget::NealsFunnel, [0.0, 2.0], -2.0),
        (BenchmarkTarget::NealsFunnel, [3.0, 0.0], -2.0),
        (BenchmarkTarget::NealsFunnel, [-3.0, 0.0], 1.0),
        (BenchmarkTarget::GaussianMixture, [0.0, 0.0], -12.5),
        (BenchmarkTarget::Banana, [0.0, -3.0], 0.0),
        (BenchmarkTarget::Banana, [10.0, 0.0], -0.5),
        (BenchmarkTarget::Banana, [0.0, 0.0], -4.5),
    ] {
        assert_relative_eq!(target.log_prob(&point), expected, epsilon = 1e-14);
    }
}

#[test]
fn mixture_agrees_with_direct_density_where_exponentials_are_resolved() {
    for x in [-9.0_f64, -5.0, -0.2, 0.0, 0.2, 5.0, 9.0] {
        for y in [-3.0_f64, 0.0, 2.0] {
            let left = (-((x + 5.0).powi(2) + y * y) / 2.0).exp();
            let right = (-((x - 5.0).powi(2) + y * y) / 2.0).exp();
            assert_relative_eq!(
                BenchmarkTarget::GaussianMixture.log_prob(&[x, y]),
                left.midpoint(right).ln(),
                epsilon = 2e-14
            );
        }
    }
    // A probability-space implementation would underflow here.
    assert_relative_eq!(
        BenchmarkTarget::GaussianMixture.log_prob(&[1000.0, 0.0]),
        -495_012.5 - LN_2,
        epsilon = 1e-10
    );
}

#[test]
fn funnel_preserves_representable_weights_in_extreme_necks_and_tails() {
    let target = BenchmarkTarget::NealsFunnel;
    for x in [-1500.0_f64, 1500.0] {
        assert_relative_eq!(
            target.log_prob(&[x, 0.0]),
            -125_000.0 - x / 2.0,
            epsilon = 1e-10
        );
    }
    // exp(-x) or y^2 individually overflow/underflow, but these points have
    // conditional standardized coordinate exactly one in real arithmetic.
    for x in [-1400.0_f64, -800.0, 800.0, 1400.0] {
        let y = (x / 2.0).exp();
        let expected = -(x / 3.0).powi(2) / 2.0 - x / 2.0 - 0.5;
        assert_relative_eq!(target.log_prob(&[x, y]), expected, epsilon = 2e-11);
    }
    // Even a subnormal coordinate need not disappear when the neck rescales it.
    let y = f64::from_bits(1);
    let x = 2.0 * y.ln();
    assert_relative_eq!(
        target.log_prob(&[x, y]),
        -(x / 3.0).powi(2) / 2.0 - x / 2.0 - 0.5,
        epsilon = 2e-11
    );
}

#[test]
fn nonfinite_coordinates_are_invalid_and_finite_overflow_is_zero_weight() {
    for target in TARGETS {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for point in [[invalid, 0.0], [0.0, invalid]] {
                assert!(target.log_prob(&point).is_nan());
                assert!(matches!(
                    Chain::new(point, &target),
                    Err(McmcError::NanInitialLogProb)
                ));
            }
        }
        for point in [[f64::MAX, f64::MAX], [-f64::MAX, 0.0], [0.0, f64::MAX]] {
            let value = target.log_prob(&point);
            assert!(value.is_infinite() && value.is_sign_negative());
        }
        for x in [-1e150, -1e10, -1.0, 0.0, 1.0, 1e10, 1e150] {
            for y in [-1e150, -1.0, 0.0, 1.0, 1e150] {
                let value = target.log_prob(&[x, y]);
                assert!(value.is_finite() || value == f64::NEG_INFINITY);
            }
        }
    }
}

#[test]
fn large_representable_log_weights_do_not_overflow_prematurely() {
    // Squaring 1.5e154 first would overflow, although half its square is
    // representable. The expected values come from the defining quadratics.
    for (target, point, expected) in [
        (BenchmarkTarget::Rosenbrock, [0.0, 1e153], -1e308),
        (BenchmarkTarget::NealsFunnel, [0.0, 1.5e154], -1.125e308),
        (BenchmarkTarget::GaussianMixture, [5.0, 1.5e154], -1.125e308),
        (BenchmarkTarget::Banana, [0.0, 1.5e154], -1.125e308),
    ] {
        let actual = target.log_prob(&point);
        assert!(actual.is_finite(), "{target:?} at {point:?}: {actual}");
        // The funnel uses ln/exp near exponent 709; allow their accumulated
        // roundoff while distinguishing finite values from negative infinity.
        assert_relative_eq!(actual, expected, epsilon = 0.0, max_relative = 5e-13);
    }
}

#[test]
fn nonlinear_ridges_preserve_their_marginal_gaussian_weights() {
    for x in [-20.0_f64, -3.0, 0.0, 1.0, 5.0, 20.0] {
        assert_relative_eq!(
            BenchmarkTarget::Rosenbrock.log_prob(&[x, x * x]),
            -(x - 1.0).powi(2),
            epsilon = 1e-12
        );
        let ridge = 0.03 * (x * x - 100.0);
        assert_relative_eq!(
            BenchmarkTarget::Banana.log_prob(&[x, ridge]),
            -x * x / 200.0,
            epsilon = 1e-12
        );
    }
}

// Deterministic midpoint quadrature of the *production density*. Independent
// changes of variables resolve curved ridges and the funnel neck. The Jacobian
// and normalization constants are supplied separately, so missing conditional
// normalizers, incorrect coefficients, and wrong moments change the integrals.
// The latent domain [-12, 12]^2 includes the shifted Gaussian contributing to
// E[Y^2] in the funnel (center 3). Its omitted moment tail is below 1e-15.
// The mixture uses [-16, 16] x [-12, 12], at least 11 SD beyond either x center.
fn check_integrals(
    target: BenchmarkTarget,
    log_normalizer: f64,
    transform: impl Fn(f64, f64) -> ([f64; 2], f64),
) {
    let step = 0.075;
    let mut mass = 0.0;
    let mut first = [0.0; 2];
    let mut second = [[0.0; 2]; 2];
    for i in -160..160 {
        for j in -160..160 {
            let (point, log_jacobian) =
                transform((f64::from(i) + 0.5) * step, (f64::from(j) + 0.5) * step);
            let weight =
                (target.log_prob(&point) - log_normalizer + log_jacobian).exp() * step * step;
            mass += weight;
            for a in 0..2 {
                first[a] += weight * point[a];
                for b in 0..2 {
                    second[a][b] += weight * point[a] * point[b];
                }
            }
        }
    }
    // These conservative absolute tolerances allow accumulated f64 roundoff
    // in ~100,000 quadrature cells; no stochastic convergence assumption enters.
    assert_relative_eq!(mass, 1.0, epsilon = 2e-11);
    let mean = target.mean();
    let covariance = target.covariance();
    for a in 0..2 {
        assert_relative_eq!(first[a], mean[a], epsilon = 2e-10);
        for b in 0..2 {
            assert_relative_eq!(
                second[a][b] - first[a] * first[b],
                covariance[a][b],
                epsilon = 2e-9
            );
        }
    }
}

#[test]
fn rosenbrock_density_normalizes_and_reproduces_analytical_moments() {
    check_integrals(BenchmarkTarget::Rosenbrock, (PI / 10.0).ln(), |u, v| {
        let x = 1.0 + u / SQRT_2;
        ([x, x * x + v / 200.0_f64.sqrt()], -20.0_f64.ln())
    });
}

#[test]
fn funnel_density_normalizes_and_reproduces_analytical_moments() {
    check_integrals(BenchmarkTarget::NealsFunnel, (6.0 * PI).ln(), |u, v| {
        let x = 3.0 * u;
        ([x, (x / 2.0).exp() * v], 3.0_f64.ln() + x / 2.0)
    });
}

#[test]
fn banana_density_normalizes_and_reproduces_analytical_moments() {
    check_integrals(BenchmarkTarget::Banana, (20.0 * PI).ln(), |u, v| {
        let x = 10.0 * u;
        ([x, v + 0.03 * (x * x - 100.0)], 10.0_f64.ln())
    });
}

#[test]
fn mixture_density_normalizes_and_reproduces_analytical_moments() {
    check_integrals(BenchmarkTarget::GaussianMixture, (2.0 * PI).ln(), |u, v| {
        ([u * 4.0 / 3.0, v], (4.0_f64 / 3.0).ln())
    });
}
