<!-- rumdl-disable MD041 -->

These fixed-seed stepping workloads describe transition and observation overhead.
They do not establish convergence, mixing, effective sample size, or scientific efficiency.

Research-repo-tools 0.2.5 measures the entire baseline phase before the current phase.
Ratios describe sequential observations on one host. Temperature, clock frequency,
power state and background load can drift between phases; matching host identities
does not control those effects. Criterion's within-phase timing intervals do not
include systematic drift between phases. The runner does not retain an observed
acquisition-order or phase-timestamp receipt.

Ratios are baseline median time divided by current median time. A value above one
means a lower observed current median; attributing that difference to the release
or claiming statistical significance requires stronger experimental evidence.
Retain absolute timings and raw samples, and withhold comparisons when observed
conditions are incompatible. Counterbalanced trials and retained execution receipts
require a published shared-runner capability before adoption here.
