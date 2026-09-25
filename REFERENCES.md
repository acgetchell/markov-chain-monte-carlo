# References and citations

## How to cite this library

If you use this library in your research or project, please cite the Zenodo DOI and the structured metadata in [CITATION.cff](CITATION.cff). This file can be
processed by GitHub and other citation platforms.

- DOI: <https://doi.org/10.5281/zenodo.20033111>
- Citation metadata: [CITATION.cff](CITATION.cff)

## Background references

These references provide the standard background for the algorithms and examples currently implemented in this crate.

1. Metropolis, Nicholas, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller. "Equation of State Calculations by Fast Computing
   Machines." *The Journal of Chemical Physics* 21, no. 6 (1953): 1087-1092. DOI: [10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
2. Hastings, W. K. "Monte Carlo Sampling Methods Using Markov Chains and Their Applications." *Biometrika* 57, no. 1 (1970): 97-109. DOI:
   [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
3. Brooks, Steve, Andrew Gelman, Galin Jones, and Xiao-Li Meng, eds. *Handbook of Markov Chain Monte Carlo*. Boca Raton: CRC Press, 2011. DOI:
   [10.1201/b10905](https://doi.org/10.1201/b10905)
4. Robert, Christian P., and George Casella. *Monte Carlo Statistical Methods*. 2nd ed. New York: Springer, 2004. DOI:
   [10.1007/978-1-4757-4145-2](https://doi.org/10.1007/978-1-4757-4145-2)
5. Landau, David P., and Kurt Binder. *A Guide to Monte Carlo Simulations in Statistical Physics*. 4th ed. Cambridge: Cambridge University Press, 2014. DOI:
   [10.1017/CBO9781139696463](https://doi.org/10.1017/CBO9781139696463)
6. Welford, B. P. "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics* 4, no. 3 (1962): 419-420. DOI:
   [10.1080/00401706.1962.10490022](https://doi.org/10.1080/00401706.1962.10490022)
7. Flyvbjerg, H., and H. G. Petersen. "Error Estimates on Averages of Correlated Data." *The Journal of Chemical Physics* 91, no. 1 (1989): 461-466. DOI:
   [10.1063/1.457480](https://doi.org/10.1063/1.457480)
8. Geyer, C. J. "Practical Markov Chain Monte Carlo." *Statistical Science* 7 (1992): 473-483.
   [Author's initial sequence estimator documentation](https://www.stat.umn.edu/geyer/mcmc/library/mcmc/html/initseq.html).
9. Pagani, Filippo, Martin Wiegand, and Saralees Nadarajah. "An n-dimensional Rosenbrock Distribution for MCMC Testing." 2020.
   [arXiv:1903.09556](https://arxiv.org/abs/1903.09556). Section 4 supplies the conditional-normal construction and normalization of the two-dimensional family.
10. Stan Development Team. "Efficiency Tuning: Example: Neal's Funnel." *Stan User's Guide*.
    [Funnel definition and scale convention](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#example-neals-funnel).
11. Hoeffding, Wassily. "Probability Inequalities for Sums of Bounded Random Variables." *Journal of the American Statistical Association* 58, no. 301 (1963):
    13-30. DOI: [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830). The bounded independent-sum inequality and a union bound
    supply the simultaneous tolerance used by `verify_proposal_bins`.
12. Andrieu, Christophe, and Johannes Thoms. "A Tutorial on Adaptive MCMC." *Statistics and Computing* 18 (2008): 343-373. DOI:
    [10.1007/s11222-008-9110-y](https://doi.org/10.1007/s11222-008-9110-y). Section 5.1.2 discusses acceptance-based global scale adaptation; this crate uses
    a bounded acceptance-indicator update during finite warmup without covariance learning.

## Related crates

This crate is part of a small Rust ecosystem for geometry, linear algebra, sampling, and simulation:

- [`causal-triangulations`](https://crates.io/crates/causal-triangulations)
- [`delaunay`](https://crates.io/crates/delaunay)
- [`la-stack`](https://crates.io/crates/la-stack)

## AI-assisted development tools

- CodeRabbit AI, Inc. "CodeRabbit." <https://coderabbit.ai/>.
- Anthropic. "Claude." <https://www.anthropic.com/claude>.
- OpenAI. "ChatGPT." <https://openai.com/chatgpt>.
- OpenAI. "Codex." <https://openai.com/codex>.

All AI-generated output was reviewed and/or edited by the maintainer. No generated content was used without human oversight.
