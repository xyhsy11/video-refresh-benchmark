# PARTIAL_PILOT — technical validation only

Experiment 23A **NO_GO / BLOCKED_INPUT** and frozen, unexecuted 23B are preserved.
Only the 20 input-selected technical clips were processed. These are not population conclusions and have no population confidence intervals.
All percentages below use equal-class, equal-group-within-class, equal-clip weights.
Primary endpoint is B4+C, epsilon0; tolerance 1e-10 is separate.

1. **Density:** full-population exact joint fraction averages 45.27%.
   Dense-correct conditional P_safe averages 45.27%.
   These are densities under uniform schedule enumeration, not best-subset existence.
2. **Periodic adequacy:** Periodic passes joint on 40.00% of weighted clips;
   accuracy alone is 100.00%. CE nondegradation and correctness differ.
3. **Periodic failures:** 12 Dense-correct clips in 12 groups
   (60.00% of whole-cohort weight) fail Periodic joint.
   Of this conditional mass, 100.00% has a safe alternative;
   0.00% has none. Exact density averages 39.32%,
   with weighted clip-density median 0.4153846153846154 and quartiles
   0.1978021978021978, 0.5472527472527473 (zeros included).
   Whole-cohort mass of such rescuable Periodic failures: 60.00%.
4. **Frozen methods:** missed_safe whole-cohort mass is 60.00%
   for H and 35.00% for H+M. This means method joint
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
The fixed 0.01-nat Periodic-improvement fraction is 13.20%.
Sensitivity rows for K8, U and epsilon .01/.05 are retained without selecting a
favorable result. No K8>=K4 safety assumption is imposed. All four correctness
transitions, paired contrasts, group/class tables and exact distributions are saved.

No bootstrap population intervals are produced for this technical pilot.
New encoder calls=0; fits=0; decodes=0; downloads=0; GPU initializations=0.
