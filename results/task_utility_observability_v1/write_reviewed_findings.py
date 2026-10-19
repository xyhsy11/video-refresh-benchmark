"""Render reviewed EXPLORATORY interpretation from genuine retained tables.

No fitting, inference or target construction. Default prints findings; --write
writes reports only within this new study. The research decision is explicitly
reviewed, not selected by an automatic significance test.
"""
from pathlib import Path
import argparse
import csv
import json
import re

OUT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    read=lambda n:json.loads((OUT/n).read_text(encoding='utf-8'))
    with (OUT/'probe_metrics.csv').open(newline='',encoding='utf-8') as f:metrics=list(csv.DictReader(f))
    def row(target,view,population='periodic'):
        return next(r for r in metrics if r['role']=='execution' and r['population']==population and r['target']==target and r['view']==view and r['stratum']=='all' and r['calibration']=='selection_calibrated')
    def val(target,view,key,population='periodic'):return float(row(target,view,population)[key])
    paired=read('paired_group_intervals.json')['intervals']
    def contrast(target,metric,population='periodic',candidate='HM',reference='H'):
        return next(r for r in paired if (r['target'],r['metric'],r['population'],r['candidate'],r['reference'])==(target,metric,population,candidate,reference))
    def interval(r):return f"{r['difference']:+.6f} [{r['CI95'][0]:+.6f}, {r['CI95'][1]:+.6f}]"
    pop=read('population_manifest.json');validation=read('target_replay_validation.json');supp=read('supplemental_review.json')
    cost=read('cost_feasibility.json');alignment=read('teacher_task_alignment.json')['rows']
    align=lambda t:next(r for r in alignment if (r['role'],r['population'],r['target'])==('execution','periodic',t))
    headroom=next(r for r in read('offline_headroom.json')['roles'] if r['role']=='execution')
    inc=lambda method,reference:next(r for r in cost['incremental'] if (r['method'],r['reference'])==(method,reference))
    timing=lambda method:next(r for r in cost['methods'] if r['method']==method)
    lines=['# Experiment22: reviewed exploratory findings','',
        '**Decision: SIGNAL_INFORMATION_NOT_ESTABLISHED.** Incremental task-benefit predictability from the existing32 cheap statistics is not established beyond history. This preserves a modest positive finding for history on immediate gain, and does not claim information-theoretic impossibility.','',
        'This is DEVELOPMENT / EXPLORATORY on already exposed UCF101 data. No official-test outcomes were opened. No new X3D forwards, GPU inference, decoder calls, recognition-head fitting or adaptive-controller training were performed. Only fixed small CPU logistic/ridge probes were fitted; all timings below are genuine PREVIOUS Experiment21 executions.','',
        '## Genuine population and targets','',
        f"Inherited counts calculated from authenticated membership: {json.dumps(pop['counts'])}. All101 classes are present. Every source belongs to original official split-1 TRAINING membership. Train/selection/execution groups and known byte/duplicate components are separated; unknown parent/pretraining relationships remain. Primary states are periodic B4 prefixes only. Training uses4848 discretionary states, selection2424; descriptive execution uses{row('local','H')['states']} states from400 clips/200 groups. Random4683 and actual H+M4740 discretionary states are separately labeled sensitivity populations on the same400 clips, not independent datasets. All{validation['states']} t=1..15 states are retained; {validation['discretionary_states']} are discretionary across the complete study. Counts are computational states, not independent samples.", '',
        'Target A = CE(last genuine cache feature)-CE(current genuine feature). Target B = min final C-pooled CE over feasible HOLD completions minus min CE over feasible REFRESH completions, preserving the complete recorded prefix and exact B4. Positive favors refresh. Full-information targets use labels/future dense arrays retrospectively; causal vectors never receive them. Target A is not final pooled utility; Target B is not a deployed policy or downstream treatment effect. Forced/exhausted states are retained but excluded from predictability; no gain-dependent exclusions.','',
        f"All455 schedules per clip and all conditional branches were checked against independent scalar pooling/combinations enumeration. Maximum scalar loss discrepancy {validation['all_clip_independent_scalar_C_pooling_loss_max_error']:.3g} nats, target discrepancy {validation['all_state_independent_prefix_combinations_target_max_error']:.3g}. Raw2048D pooling/head replay additionally checked all455 schedules on12 hash-selected real clips. Independent review reproduced4848 original periodic training input vectors exactly and the original actual H+M teacher-confusion counts. No missing vector was reconstructed from means.", '',
        '## Predictability on identical primary periodic states','',
        'Class-balanced, then group/clip/state-balanced metrics. PR AUC means weighted average precision. Classification columns below use selection-only calibration; raw scores, ROC, Brier, precision/recall, calibration bins, signed biases, error quantiles and strata are retained in machine-readable outputs.','',
        '|Target|View|PR AUC|ROC AUC|Brier|Signed-gain MAE|Gain Spearman|',
        '|---|---|---:|---:|---:|---:|---:|']
    for target in ['local','budget']:
        for view in ['constant','baseline','H','M','HM']:
            r=row(target,view);lines.append('|'+ '|'.join([target,view,*[f'{float(r[k]):.6f}' if r[k] else 'undefined' for k in ['PR_AUC','ROC_AUC','Brier','MAE','Spearman']]])+'|')
    lines.extend(['',f"Positive prevalence is{val('local','H','positive_prevalence'):.4%} for A and{val('budget','H','positive_prevalence'):.4%} for B. Primary execution has no exact/near-zero(abs<=1e-10) targets, but near-zero states in other roles/populations remain in all analyses. Selected threshold.25 labels all primary execution states positive for every view: recall1, precision equal to prevalence. This threshold is a diagnostic selected by F1, not a feasible four-refresh policy.", '',
        f"H minus time/age/budget PR AUC: A {interval(contrast('local','PR_AUC',candidate='H',reference='baseline'))}; B {interval(contrast('budget','PR_AUC',candidate='H',reference='baseline'))}. History therefore has modest immediate sign information, while richer history does not improve budget-sign classification over the structural baseline. A's H regression MAE is worse than the constant predictor ({val('local','H','MAE'):.6f} versus{val('local','constant','MAE'):.6f}); classification/ranking and gain magnitudes are different questions.", '',
        f"H+M minus H PR AUC: A {interval(contrast('local','PR_AUC'))}; B {interval(contrast('budget','PR_AUC'))}. Brier differences: A {interval(contrast('local','Brier'))}; B {interval(contrast('budget','Brier'))}. Signed-gain MAE differences: A {interval(contrast('local','MAE'))}; B {interval(contrast('budget','MAE'))}. Tiny/inconsistent score changes, no magnitude improvement, and M-only performance close to the constant baseline do not support added information of useful size. This conclusion does not rely only on CI crossings.", '',
        f"The same fixed probes transported to random states give H+M-H AP: A {interval(contrast('local','PR_AUC','random'))}; B {interval(contrast('budget','PR_AUC','random'))}. On actual H+M states: A {interval(contrast('local','PR_AUC','HM_executed'))}; B {interval(contrast('budget','PR_AUC','HM_executed'))}. The different populations are not pooled. Position/budget strata are descriptive; all retain101 classes and at least{min(r['groups'] for r in supp['coverage'])} groups, so none meets the prespecified sparse(<20 groups) flag. Strata do not select the result.", '',
        'Intervals use2000 saved paired PCG64 class-stratified GROUP bootstrap draws. They are marginal/unadjusted, conditional on fitted/selected/calibrated probes and a reused head. They omit fitting and prior-exposure uncertainty. BenchPress/JavelinThrow have only one execution group; their within-class group variance is unestimated. No equivalence or power claim follows.','',
        '## Proxy alignment and task headroom','',
        f"On primary states the Z_pool teacher agrees with A's beneficial-action sign on{align('local')['agreement']:.2%} and B on{align('budget')['agreement']:.2%}. Its refresh action accompanies negative budget gain on{align('budget')['teacher_refresh_task_harmful_mass']:.2%} of weighted states. Mean budget gain conditional on teacher refresh is{align('budget')['gain_if_teacher_refresh']:.6f} nats. The hold-label majority is{1-val('budget','H','positive_prevalence'):.2%}; that is a descriptive label frequency, NOT a newly executed hold policy. Conditional hindsight wrong-branch regret, confusion cells and group intervals are retained. Teacher imitation is not task-utility prediction.", '',
        f"Periodic final CE={headroom['periodic_CE']:.6f}; Z_pool={headroom['Z_pool_CE']:.6f}; true-label task oracle={headroom['task_oracle_CE']:.6f}; dense16={headroom['dense_CE']:.6f}. Perfect-hindsight task headroom is{headroom['perfect_hindsight_headroom']:.6f} nats, whereas Z_pool's reduction is only{headroom['proxy_reduction']:.6f}. The small fidelity-teacher advantage does NOT establish small task-scheduling headroom. The true-label optimum inspects455 schedules and is an optimistic privileged bound, not a reachable causal-performance promise.", '',
        '## Observed acquisition cost and competing explanations','',
        f"Reproduced actual end-to-end medians: periodic{1000*timing('periodic')['seconds']:.1f}ms, H{1000*timing('history')['seconds']:.1f}ms, H+M{1000*timing('history_motion')['seconds']:.1f}ms, periodic+unusedgate{1000*timing('periodic_unused_gate')['seconds']:.1f}ms. Paired median added latency is{1000*inc('history_motion','periodic')['paired_median_extra_seconds']:.1f}ms versus periodic and{1000*inc('history_motion','history')['paired_median_extra_seconds']:.1f}ms versus H. Unused gate alone adds{1000*inc('periodic_unused_gate','periodic')['paired_median_extra_seconds']:.1f}ms. H+M is slower on all30 timing groups. These use{cost['measured_trials']} measured trials from{cost['all_trials']} total trials, with repeats collapsed within video.", '',
        f"H+M median thumbnail conversion={1000*timing('history_motion')['cheap_thumbnail_seconds']:.1f}ms, cheap statistics={1000*timing('history_motion')['cheap_statistics_seconds']:.1f}ms, controller decide={1000*timing('history_motion')['controller_decide_seconds']:.1f}ms. Observation preparation={1000*timing('history_motion')['observation_preparation_seconds']:.1f}ms ALREADY includes thumbnail/statistics; do not add nested fields twice. Medians of components do not sum to the median total. The current H+M point is dominated by periodic on the matched30 quality/cost cohort and separately the400-quality/30-cost summary. This is observed implementation feasibility, not an optimized deployment frontier or a replay-derived speedup.", '',
        'H1 objective mismatch is supported by direct conditional proxy/task disagreement and the task-oracle versus Z_pool gap. H2 insufficient incremental information from these particular cheap statistics is consistent with all simple-probe comparisons, but information-theoretic absence is unproved. H3 failure to extract useful signal remains unresolved: history predicts local signs somewhat, but no missed cheap signal is established and the old neural learner optimized another target. H4 acquisition/controller overhead is directly observed; it cannot alone explain the original failure because recognition improvement was not established either. H5 a blanket lack of task-scheduling headroom is contradicted by the large true-label hindsight bound; limited fidelity-proxy headroom is observed. These explanations can coexist. See hypothesis_assessment.md.','',
        '## Integrity, exposure and next step','',
        'All15 new synthetic tests plus30 existing regression tests passed; genuine target replay and independent provenance/aggregation checks are separate evidence. Archived persistence/resume interruptions were resolved. A post-initial-analysis correction made constant Spearman undefined instead of a floating residual; all selected parameters, calibrators, thresholds and five complete prediction-vector files were independently verified bitwise unchanged. All original versions, errors/seals and partial evidence are retained. A supplemental exact-bit normalization check detected only1.63e-14 BLAS roundoff; the disclosed1e-12 check passes without changing observations or models. Passing tests is not scientific confirmation.','',
        'The recognition head fitted the probe-training population and selected alpha using all original development clips; its iteration cap and severe train/development loss-scale shift remain. Only one/two execution groups per class, unresolved parent/pretraining mappings, fixed periodic fitting and simple-model capacity constrain interpretation. No new independent confirmation or publication readiness is claimed.','',
        'Decision B: focus the manuscript on offline objective mismatch, conditional observability limitations and real overhead. Do not search another architecture or retune controllers on these outcomes. The next experiment is a fresh, authorized, input-only source-isolated population and separated-head/controller feasibility gate (Experiment23A), specified in next_experiment.md. No fresh dataset acquisition, GPU evaluation or controller experiment begins in this commit.','',
        '|Claim|Evidence|Limit|','|---|---|---|',
        '|Exact genuine task targets|target_replay_validation.json; original input/feature/trace hashes|Retrospective privileged targets, not treatment effects|',
        '|History has modest local sign information|H-baseline paired metrics|Magnitude prediction and budget utility do not follow|',
        '|Cheap incremental information unestablished|H+M-H and M-baseline across identical primary/secondary states|Limited linear probes and exposed population|',
        '|Fidelity/task mismatch and nontrivial task headroom|teacher_task_alignment.json; exact455 true-label oracle|Hindsight cannot be deployed|',
        '|Cheap pathway currently costly|cost_feasibility.json; original actual timing rows|30 groups, previous implementation, no optimized speedup claim|',
        '|Historical and official-test safety|historical_integrity.json; guards; verification.json|Unknown source/pretraining mappings remain|',''])
    findings='\n'.join(lines)
    # Readable prose around measured counts/units; preserve scientific e-notation,
    # named X3D/PCG64 models and H1-H5 hypothesis identifiers.
    findings=re.sub(r'(?<=[a-z])(?=[0-9])',' ',findings)
    findings=re.sub(r'(?<=[0-9])(?=[a-df-z])',' ',findings)
    if args.write:(OUT/'findings.md').write_text(findings,encoding='utf-8')
    else:print(findings)


if __name__=='__main__':main()
