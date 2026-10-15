# Cached-history information audit

T contains exactly 44 deterministic functions of already-computed periodic cached scores, not 44 independent new observations. This audit is algebra/implementation evidence, not a prediction-performance result.

The history follows only refreshes at candidate ordinals 0,4,8 before the nine decisions. Each vector is added once; repeated outputs at the next two opportunities add no event. No skipped past RGB, future refreshed scores, hypothetical alternative histories, labels or outcomes enter the builder.

For t=1,2,3, only q0 is available. All44 T values are zero, including masks, standard deviation and switch rate. T adds no information.

For t=5,6,7, m=[4*q0+(t-4)*q4]/t, so q0=[t*m-(t-4)*q4]/4. All T values can already be derived from old C (including t and q4). Differences, nonlinear softmax/margin functions and masks make information explicit for the fixed shallow forest; they are not new observations.

For t=9,10,11, m=[4*q0+4*q4+(t-8)*q8]/t, so old C identifies q0+q4 but generally not their individual values or order. Ordered score differences can preserve information lost by that compression. The controlled fixture swaps distinct q0 and q4, holds q8, ages, budgets, motion and all other C observations fixed, and leaves old C unchanged while T changes. It is not a performance example.

Masks are redundant with the candidate block. Centered20-vectors each sum to zero. In the final block, given old C and latest difference L=q8-q4, q4=q8-L and q0=(q0+q4)-q4; the previous difference is then determined. Margin standard deviation is permutation-invariant, while the consecutive top1 switch rate can depend on order. Confidence statistics are nonlinear derived features. Do not count all44 columns as independent new information.

Softmax uses exp(q-max(q))/sum in float64; argmax ties use the lowest class index. Margin standard deviation is population ddof0, with k1 zero. All numeric definitions, order and missing-history conventions were frozen before fitting.

Controlled fixture C discrepancy: 0.0; changed T fields: history_latest_score_difference_00, history_latest_score_difference_01, history_latest_score_difference_02, history_latest_score_difference_03, history_latest_score_difference_04, history_latest_score_difference_05, history_latest_score_difference_06, history_latest_score_difference_07, history_latest_score_difference_08, history_latest_score_difference_09, history_latest_score_difference_10, history_latest_score_difference_11, history_latest_score_difference_12, history_latest_score_difference_13, history_latest_score_difference_14, history_latest_score_difference_15, history_latest_score_difference_16, history_latest_score_difference_17, history_latest_score_difference_18, history_latest_score_difference_19, history_previous_score_difference_00, history_previous_score_difference_01, history_previous_score_difference_02, history_previous_score_difference_03, history_previous_score_difference_04, history_previous_score_difference_05, history_previous_score_difference_06, history_previous_score_difference_07, history_previous_score_difference_08, history_previous_score_difference_09, history_previous_score_difference_10, history_previous_score_difference_11, history_previous_score_difference_12, history_previous_score_difference_13, history_previous_score_difference_14, history_previous_score_difference_15, history_previous_score_difference_16, history_previous_score_difference_17, history_previous_score_difference_18, history_previous_score_difference_19, history_top1_switch_rate.

Real-data weighted-past-mean maximum discrepancies: development=7.1054273576e-15, exploratory=5.3290705182e-15. Cached score repetitions agree exactly. Incremental T matches the independent builder.

Full-clip candidate selection and existing future-reference codec vectors keep this offline. Prefix-only score handling does not validate strict streaming causality. No temporal video backbone is evaluated.
