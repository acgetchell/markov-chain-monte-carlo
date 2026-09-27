# References and citations

This page owns bibliographic records and method provenance. See the [README](README.md) for getting started and
[scientific basis and scope](docs/scientific_basis.md) for assumptions, mathematical contracts, and limitations.

## Contents

- [How to cite this library](#how-to-cite-this-library)
- [Method-to-source index](#method-to-source-index)
- [Background references](#background-references)
- [Related crates](#related-crates)
- [AI-assisted development tools](#ai-assisted-development-tools)

## How to cite this library

If you use this library in your research or project, please cite the Zenodo DOI and the structured metadata in [CITATION.cff](CITATION.cff). This file can be
processed by GitHub and other citation platforms.

- DOI: <https://doi.org/10.5281/zenodo.20033111>
- Citation metadata: [CITATION.cff](CITATION.cff)

## Method-to-source index

Sources for implemented methods are distinguished from general background and comparisons with methods outside this crate's scope.
Independent methods are alphabetical within each group; the bibliography retains its existing citation numbers.

| Group | Method or topic | Sources and role |
| --- | --- | --- |
| Sampling | [Adaptive warmup](docs/scientific_basis.md#adaptive-warmup) | [12](#ref-12), acceptance-based scale adaptation background; the bounded indicator update is specified by this crate |
| Sampling | [Additive target terms](docs/scientific_basis.md#additive-target-terms) | [1](#ref-1), [2](#ref-2), the Metropolis-Hastings contract applied to a combined log weight |
| Sampling | [Metropolis-Hastings acceptance](docs/scientific_basis.md#metropolis-hastings-contract) | [1](#ref-1), [2](#ref-2), original algorithm sources |
| Statistics and diagnostics | [Autocorrelation and integrated time](docs/scientific_basis.md#autocorrelation-estimator-contract) | [8](#ref-8), initial sequence estimators; [15](#ref-15), trace-length caution only |
| Statistics and diagnostics | [Binning](docs/scientific_basis.md#binning-analysis) | [7](#ref-7), blocking for correlated uncertainty |
| Statistics and diagnostics | [Classical split R-hat](docs/scientific_basis.md#classical-split-r-hat) | [13](#ref-13), the implemented variance-ratio definition; [14](#ref-14), limitations and unimplemented improvements |
| Statistics and diagnostics | [ESS and efficiency](docs/scientific_basis.md#ess-and-wall-clock-efficiency) | [8](#ref-8), [13](#ref-13), integrated-time relation; Stan's pooled multi-chain ESS is a different estimator |
| Statistics and diagnostics | [Online statistics](docs/scientific_basis.md#online-statistics) | [6](#ref-6), Welford accumulation |
| Statistics and diagnostics | [Proposal validation](docs/scientific_basis.md#proposal-validation) | [1](#ref-1), [2](#ref-2), transition-flow identity; [11](#ref-11), independent-bin tolerance bound |
| Example models | Neal's funnel | [10](#ref-10), conditional scale convention |
| Example models | Rosenbrock target | [9](#ref-9), normalized conditional-normal construction |
| General background | MCMC and statistical physics | [3](#ref-3), [4](#ref-4), [5](#ref-5), textbooks and surveys |

## Background references

Reference numbers are stable citation identifiers. Each entry has a permanent anchor such as `#ref-6`; append new entries without renumbering existing ones.

1. <a name="ref-1"></a> Metropolis, Nicholas, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller. "Equation of State
   Calculations by Fast Computing Machines." *The Journal of Chemical Physics* 21, no. 6 (1953): 1087-1092. DOI:
   [10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
2. <a name="ref-2"></a> Hastings, W. K. "Monte Carlo Sampling Methods Using Markov Chains and Their Applications." *Biometrika* 57, no. 1 (1970): 97-109. DOI:
   [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
3. <a name="ref-3"></a> Brooks, Steve, Andrew Gelman, Galin Jones, and Xiao-Li Meng, eds. *Handbook of Markov Chain Monte Carlo*. Boca Raton: CRC Press, 2011.
   DOI: [10.1201/b10905](https://doi.org/10.1201/b10905)
4. <a name="ref-4"></a> Robert, Christian P., and George Casella. *Monte Carlo Statistical Methods*. 2nd ed. New York: Springer, 2004. DOI:
   [10.1007/978-1-4757-4145-2](https://doi.org/10.1007/978-1-4757-4145-2)
5. <a name="ref-5"></a> Landau, David P., and Kurt Binder. *A Guide to Monte Carlo Simulations in Statistical Physics*. 4th ed. Cambridge: Cambridge University
   Press, 2014. DOI: [10.1017/CBO9781139696463](https://doi.org/10.1017/CBO9781139696463)
6. <a name="ref-6"></a> Welford, B. P. "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics* 4, no. 3 (1962): 419-420. DOI:
   [10.1080/00401706.1962.10490022](https://doi.org/10.1080/00401706.1962.10490022)
7. <a name="ref-7"></a> Flyvbjerg, H., and H. G. Petersen. "Error Estimates on Averages of Correlated Data." *The Journal of Chemical Physics* 91, no. 1 (1989):
   461-466. DOI: [10.1063/1.457480](https://doi.org/10.1063/1.457480)
8. <a name="ref-8"></a> Geyer, C. J. "Practical Markov Chain Monte Carlo." *Statistical Science* 7 (1992): 473-483.
   [Author's initial sequence estimator documentation](https://www.stat.umn.edu/geyer/mcmc/library/mcmc/html/initseq.html).
9. <a name="ref-9"></a> Pagani, Filippo, Martin Wiegand, and Saralees Nadarajah. "An n-dimensional Rosenbrock Distribution for MCMC Testing." 2020.
   [arXiv:1903.09556](https://arxiv.org/abs/1903.09556). Section 4 supplies the conditional-normal construction and normalization of the two-dimensional family.
10. <a name="ref-10"></a> Stan Development Team. "Efficiency Tuning: Example: Neal's Funnel." *Stan User's Guide*.
    [Funnel definition and scale convention](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#example-neals-funnel).
11. <a name="ref-11"></a> Hoeffding, Wassily. "Probability Inequalities for Sums of Bounded Random Variables." *Journal of the American Statistical Association*
    58, no. 301 (1963): 13-30. DOI: [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830). The bounded independent-sum inequality and
    a union bound supply the simultaneous tolerance used by `verify_proposal_bins`.
12. <a name="ref-12"></a> Andrieu, Christophe, and Johannes Thoms. "A Tutorial on Adaptive MCMC." *Statistics and Computing* 18 (2008): 343-373. DOI:
    [10.1007/s11222-008-9110-y](https://doi.org/10.1007/s11222-008-9110-y). Section 5.1.2 discusses acceptance-based global scale adaptation; this crate uses
    a bounded acceptance-indicator update during finite warmup without covariance learning.
13. <a name="ref-13"></a> Stan Development Team. *Stan Reference Manual*, version 2.29.
    [Section 16.3: Notation for samples, chains, and draws](https://mc-stan.org/docs/2_29/reference-manual/notation-for-samples-chains-and-draws.html)
    specifies classical split R-hat;
    [section 16.4: Effective sample size](https://mc-stan.org/docs/2_29/reference-manual/effective-sample-size.html)
    gives the integrated-time relationship and also describes Stan's distinct multi-chain estimator.
14. <a name="ref-14"></a> Vehtari, Aki, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner.
    "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC." *Bayesian Analysis* 16, no. 2 (2021): 667-718.
    DOI: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221). [arXiv:1903.08008](https://arxiv.org/abs/1903.08008).
    Documents limitations of classical R-hat and improvements not implemented by this crate.
15. <a name="ref-15"></a> Foreman-Mackey, Dan, and contributors. "Autocorrelation Analysis & Convergence." *emcee documentation*.
    [Trace-length discussion](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).
    Context for the example notebook's heuristic short-trace caution; the crate uses Geyer's estimator, not emcee's window selection.

## Related crates

This crate is part of a small Rust ecosystem for geometry, linear algebra, sampling, and simulation:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations)
- [`delaunay`](https://crates.io/crates/delaunay)
- [`la-stack`](https://crates.io/crates/la-stack)

## AI-assisted development tools

- Anthropic. "Claude." <https://www.anthropic.com/claude>.
- CodeRabbit AI, Inc. "CodeRabbit." <https://coderabbit.ai/>.
- OpenAI. "ChatGPT." <https://openai.com/chatgpt>.
- OpenAI. "Codex." <https://openai.com/codex>.

All AI-generated output was reviewed and/or edited by the maintainer. No generated content was used without human oversight.
