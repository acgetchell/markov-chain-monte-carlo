<!-- rumdl-disable MD041 -->

These fixed-seed stepping workloads describe transition and observation overhead.
They do not establish convergence, mixing, effective sample size, or scientific efficiency.

The campaign acquires two paired trials on one host, baseline first in trial 1
and current first in trial 2. The README shows trial 1 by prior declaration.
The complete report retains both trials, original raw samples, realized phase
order, and collection timestamps. Temperature, clock frequency, power state
and background load can still drift within or between pairs.

Ratios are baseline median time divided by current median time. A value above one
means a lower observed current median. Criterion's intervals describe variation
within a phase. Comparing the paired results can reveal sensitivity to phase
order, but two paired trials provide limited evidence about between-trial
variation and causal release effects.
