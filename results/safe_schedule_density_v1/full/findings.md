# Completed exposed-development diagnostic

Experiment 23A **NO_GO / BLOCKED_INPUT** and frozen, unexecuted 23B are preserved.
All 400 predeclared execution clips / 200 groups / 101 classes were processed.
All percentages below use equal-class, equal-group-within-class, equal-clip weights.
Primary endpoint is B4+C, epsilon0; tolerance 1e-10 is separate.

1. **Density:** full-population exact joint fraction averages 44.61% [95% marginal interval 42.75%, 46.53%].
   Dense-correct conditional P_safe averages 46.71% [95% marginal interval 44.99%, 48.52%].
   These are densities under uniform schedule enumeration, not best-subset existence.
2. **Periodic adequacy:** Periodic passes joint on 47.52% [95% marginal interval 44.06%, 50.99%] of weighted clips;
   accuracy alone is 94.31% [95% marginal interval 92.57%, 96.04%]. CE nondegradation and correctness differ.
3. **Periodic failures:** 188 Dense-correct clips in 139 groups
   (47.28% of whole-cohort weight) fail Periodic joint.
   Of this conditional mass, 98.43% [95% marginal interval 97.80%, 99.00%] has a safe alternative;
   1.57% [95% marginal interval 1.00%, 2.20%] has none. Exact density averages 34.61% [95% marginal interval 32.82%, 36.49%],
   with weighted clip-density median 0.33406593406593404 and quartiles
   0.1824175824175824, 0.4879120879120879 (zeros included).
   Whole-cohort mass of such rescuable Periodic failures: 46.53% [95% marginal interval 43.07%, 50.00%].
4. **Frozen methods:** missed_safe whole-cohort mass is 48.27% [95% marginal interval 45.05%, 51.73%]
   for H and 48.02% [95% marginal interval 44.55%, 51.73%] for H+M. This means method joint
   failure with a passing same-K/C/epsilon alternative; actual comparisons apply
   only to archived executed B4+C. U is offline re-pooling; K8 H/H+M is N/A.
5. **Unknowns:** label/future-informed density cannot establish a usable causal
   pre-decision signal, fresh-population generalization, or complete execution
   cost. The reused recognition head was development-selected and reached its
   historical iteration limit; controller/head training overlap remains. Unknown
   parent/reupload/pretraining relationships and single-group classes limit
   independence and uncertainty. CPU diagnostic time is not inference latency.
   Offline alternatives alone do not justify fitting another controller.

Dense-wrong clips remain in all-population summaries; correction and CE
preservation are separate rows, conditional P_safe is N/A. Dense agreement is
fidelity only. RandomExact averages individual schedule losses/correctness.
The fixed 0.01-nat Periodic-improvement fraction is 18.16% [95% marginal interval 16.16%, 20.16%].
Sensitivity rows for K8, U and epsilon .01/.05 are retained without selecting a
favorable result. No K8>=K4 safety assumption is imposed. All four correctness
transitions, paired contrasts, group/class tables and exact distributions are saved.

Intervals use 2000 paired PCG64 class-stratified group draws, seed20261021. They are marginal/unadjusted and conditional on reused inputs/head. Empty bootstrap conditioning sets are omitted with valid draw counts. Single-group classes have no within-class resampling variance.
New encoder calls=0; fits=0; decodes=0; downloads=0; GPU initializations=0.
