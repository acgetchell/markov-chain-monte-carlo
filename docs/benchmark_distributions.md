# Reference benchmark distributions

The unreleased `benchmarks` feature provides `BenchmarkTarget` at the crate root. It implements `Target<[f64; 2]>` for four fixed reference distributions and
exposes their analytical population `mean()` and `covariance()`. No additional dependencies are enabled. Coordinates are always `[x, y]`; dimensions and
parameters are fixed so experiments can identify the exact target by its variant. Custom parameters or higher-dimensional extensions belong in a separate
`Target` implementation.

These targets exercise curved ridges, changing conditional scale, separated modes, and nonlinear dependence. They support model checks and mixing
comparisons; they do not supply a tuned proposal or certify convergence. The funnel here is a two-dimensional marginal of the original ten-dimensional
example, so results are not interchangeable with ten-dimensional funnel benchmarks.

## Definitions and ground truth

Here `N(m, v)` denotes a normal distribution with mean `m` and **variance** `v`. Every auxiliary normal draw below is independent. Log weights omit only
state-independent constants. Covariance rows and columns follow `[x, y]`.

| Variant           | Construction                                                | Mean       | Covariance                       |
| ----------------- | ----------------------------------------------------------- | ---------- | -------------------------------- |
| `Rosenbrock`       | `X ~ N(1, 1/2)`; `Y = X^2 + E`, `E ~ N(0, 1/200)`             | `[1, 3/2]` | `[[1/2, 1], [1, 501/200]]`        |
| `NealsFunnel`     | `X ~ N(0, 9)`; `Y = exp(X/2) Z`, `Z ~ N(0, 1)`                | `[0, 0]`   | `[[9, 0], [0, exp(9/2)]]`         |
| `GaussianMixture` | `X = 5 S + Z`, equal `S = -1, +1`; `Z, Y ~ N(0, 1)`          | `[0, 0]`   | `[[26, 0], [0, 1]]`               |
| `Banana`          | `X ~ N(0, 100)`; `Y = Z + 0.03 (X^2 - 100)`, `Z ~ N(0, 1)`   | `[0, 0]`   | `[[100, 0], [0, 19]]`             |

For a normal `X` with mean `m` and variance `v`, `E[X^2] = m^2 + v`, `Var(X^2) = 2 v^2 + 4 m^2 v`, and `Cov(X, X^2) = 2 m v`.
The Rosenbrock moments follow immediately after adding the independent residual variance `1/200`. Its mode `[1, 1]` differs from its mean `[1, 3/2]`.
The density is proportional to `exp(-(x - 1)^2 - 100 (y - x^2)^2)`, using the
[two-dimensional family studied by Pagani, Wiegand, and Nadarajah](https://arxiv.org/abs/1903.09556) with `a = 1`, `b = 100`, and `mu = 1`.

For the funnel, conditional centering gives zero means and cross covariance. The marginal variance of `Y` is `E[exp(X)] = exp(9/2)`, approximately `90.01713`.
Its log weight is `-x^2/18 - x/2 - y^2 exp(-x)/2`; omitting `-x/2` would change the marginal distribution of `X`. The conditional standard deviation is
`exp(X/2)`, following the [Stan funnel convention](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#example-neals-funnel).

For the mixture, the independent sign adds variance `25` to the unit Gaussian variance in `X`. For the banana, centering the quadratic term gives zero
mean, and `Var(Y) = 1 + 0.03^2 (2 * 100^2) = 19`. Its covariance is diagonal even though the coordinates are dependent. The positive bend is part of this
preset's definition; other banana conventions can reverse the bend or choose different scales.

The normalization integrals of `exp(log_prob)` are `pi/10` (Rosenbrock), `6 pi` (funnel), `2 pi` (mixture, including its equal mixture weights), and `20 pi`
(banana). The constructions also describe independent reference sampling if callers supply their own normal random variates.

## Numerical and validation scope

Non-finite coordinates return `NaN`, which produces the chain's invalid-target error. Finite coordinates whose negative log weight overflows return negative
infinity. The mixture uses a factored log-sum-exp; the funnel computes its conditional quadratic energy in log space, handling exact zero separately. This
avoids spurious overflow from separately squaring a coordinate or exponentiating the inverse conditional variance. Floating-point roundoff and underflow of
negligible tail contributions still apply.

`tests/benchmark_distributions.rs` uses exact density points, an independent probability-space mixture calculation in a resolved range, numerical boundary
cases, and deterministic midpoint quadrature of the production densities. Independent transformations and Jacobians resolve the nonlinear shapes. The
integrals check unit mass, both means, both variances, and cross covariance against the formulas above. They do not require a random walk to have mixed.

## Reproduce a mixing experiment

```bash
cargo run --release --features benchmarks --example benchmark_distributions
```

The example uses fixed independent coordinate uniform random-walk increments. It prints each target, seed, starting point, proposal half-widths, discarded
warmup count, retained count, recording interval, lag budget, acceptance rate, marginal moment errors, cross covariance and its error, and scalar mean ESS
with measured ESS/second. Every production transition is retained, including rejections. Unavailable diagnostic estimates retain their error messages.

Timing includes production transitions and recording both coordinates, and excludes warmup, allocation of the trace buffers, printing, and diagnostics. Each
coordinate's rate uses the same full production duration; the two rates must not be added. Variance and cross covariance use empirical sample moments; the ESS
shown is for estimating the coordinate mean, not its variance or covariance. Seeded trajectories are repeatable within the same build and environment; bitwise
stability across Rust, `rand`, platform, or architecture changes is not promised. Wall time varies with hardware, load, and build profile.

A random walk can remain in one mixture component or miss a funnel tail, even when its local ESS looks good. The example intentionally keeps such failures
visible instead of enforcing stochastic pass/fail thresholds. Compare analytical moments, inspect traces, use dispersed independent starts, extend runs and
lag windows, and assess convergence before treating ESS/second as useful sampling efficiency. Use identical target definitions, observables, production
budgets, recording policies, timing scopes, and environments when comparing algorithms. The fixed warmup count is an example policy, not an established
mixing time.
