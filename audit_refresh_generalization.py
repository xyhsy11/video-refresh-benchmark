"""Bounded independent audit of frozen single-move refresh diagnostics."""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import refresh_generalization_audit_utils as u

ROOT = Path(__file__).resolve().parent
OLD = ROOT/"results/two_target_refresh_predictability_v1"
HISTORY = ROOT/"results/cached_history_signal_diagnostic_v1_rerun1"
RECOGNITION = ROOT/"results/recognition_baseline_v2"
DECOMP = ROOT/"results/refresh_target_decomposition_v1"
EXPERIMENTS = ["task_signal_ablation_v2", "refresh_target_decomposition_v1",
               "two_target_refresh_predictability_v1", "cached_history_signal_diagnostic_v1_rerun1",
               "recognition_baseline_v2"]
SOURCES = ["audit_refresh_generalization.py", "refresh_generalization_audit_utils.py",
           "refresh_generalization_audit_report.py", "test_refresh_generalization_audit.py"]
MODEL_FILES = ["Direct_G_RF400_G.json", "Two_target_RF200x2_forward_B.json",
               "Two_target_RF200x2_forward_C.json", "C_plus_T_RF400.json"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def input_files() -> list[Path]:
    files = []
    for name in EXPERIMENTS:
        directory = ROOT/"results"/name
        for f in ["protocol.json", "findings.md", "verification.json", "status.json", "SHA256SUMS.txt"]:
            path = directory/f
            if path.exists(): files.append(path)
        assert u.read(directory/"verification.json")["passed"]
        assert u.read(directory/"status.json")["status"] == "complete"
    for directory, names in [
        (OLD, ["development_analysis_table.csv", "exploratory_analysis_table.csv", "fold_manifest.json",
               "out_of_fold_predictions.csv", "exploratory_predictions.csv", "per_video_metrics.csv", "frozen_models.json"]),
        (HISTORY, ["development_observations.csv", "exploratory_observations.csv", "fold_manifest.json",
                   "out_of_fold_predictions.csv", "exploratory_predictions.csv", "per_video_metrics.csv",
                   "cpu_microbenchmark.csv", "cpu_microbenchmark.metadata.json", "environment.json", "frozen_models.json"]),
        (DECOMP, ["per_opportunity_decomposition.csv"]),
        (RECOGNITION, ["dataset_manifest.json", "training_subsets.json", "exclusions.json", "frozen_models.json",
                       "regularization_grid.csv", "per_video_metrics.csv", "classifiers/20/parameters.json", "classifiers/20/selection.json"]),
    ]:
        files.extend(directory/name for name in names)
    for bundle in [f"fold_{i}" for i in range(5)]+["final_development"]:
        for name in MODEL_FILES:
            folder = (HISTORY if name == "C_plus_T_RF400.json" else OLD)/"models"/bundle
            state = u.read(folder/name)
            files.extend([folder/name, folder/state["model_file"]])
    for e in u.read(RECOGNITION/"dataset_manifest.json")["videos"]:
        # Reference arrays only; no video bytes, protected confirmation or motion tables.
        files.append(ROOT/e["reference_directory"]/"features"/(e["video_id"]+".npz"))
    files.extend(ROOT/name for name in ["task_signal_utils.py", "task_signal_ablation_utils.py",
        "two_target_refresh_utils.py", "cached_history_signal_utils.py", "task_aware_oracle_utils.py",
        "recognition_baseline_data.py", "run_recognition_baseline_v2.py", "collect_cache_drift.py",
        "motion_utils.py", "benchmark.py", "inspect_video.py"])
    return sorted(set(files))


def freeze(output: Path, seed: int, bootstrap: int) -> dict:
    paths = input_files()
    inputs = {relative(p): u.digest(p) for p in paths}
    hashes = {s: u.digest(ROOT/s) for s in SOURCES}
    if (output/"protocol.json").exists():
        protocol = u.read(output/"protocol.json")
        assert protocol["input_sha256"] == inputs and protocol["source_sha256"] == hashes
        assert protocol["seed"] == seed and protocol["bootstrap"]["resamples"] == bootstrap
        return protocol
    assert not output.exists() or not any(output.iterdir()), "Preserve existing nonempty directory;choose a fresh output"
    manifest = u.read(RECOGNITION/"dataset_manifest.json")
    membership = {r: sorted(e["source_video_id"] for e in manifest["videos"] if e["recognition_role"] == r)
                  for r in ["train", "development", "exploratory"]}
    entries = subprocess.check_output(["git", "ls-files", "-s"], cwd=ROOT, text=True)
    tracked = {line.split("\t", 1)[1]: line.split()[1] for line in entries.splitlines()
               if line.split("\t", 1)[1] not in ["README.md", ".gitignore"]}
    protocol = {"experiment": "refresh_generalization_audit_v1", "frozen_at_utc": now(),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "starting_worktree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "source_sha256": hashes, "input_sha256": inputs, "previous_tracked_blob_identities": tracked,
        "seed": seed, "class_order": manifest["classes"], "membership": membership,
        "repository_instructions": "No AGENTS.md found in repository or ancestor directories;README and relevant experiment protocols/findings/code inspected;no other running experiment interrupted",
        "RF_parameters_from_frozen_direct_model": u.read(OLD/'models/final_development/Direct_G_RF400_G.json')['parameters'],
        "head_artifact": 'results/recognition_baseline_v2/classifiers/20/parameters.json',
        "folds": u.read(OLD/"fold_manifest.json"), "target": "B=LA-LB;C=LC-LB;G=LA-LC=B-C;signed nats",
        "opportunities": "nine independent moves/source;r=4,8,12;t=r-3,r-2,r-1;reference[0,4,8,12];replace r with t",
        "methods": u.METHODS, "frozen_model_inputs": {"C": u.C_COLUMNS, "C_plus_T": u.C_COLUMNS+u.T_COLUMNS},
        "learned_acceptance": "predicted_G > 1e-12;no refitting or tuning",
        "motion_rule": "available current_p90_magnitude_px strictly greater than available-training-row median;fold train only;final development-only median for exploratory;missing/no available training motion => reject",
        "random_rule": "independent Bernoulli0.5;analytic expected counts and utility only;no sampled best seed",
        "utility": "mean accept_probability*G over all9 opportunities within source;then equal source mean",
        "metrics": "explicit selected/helpful/harmful/neutral counts,missed benefit and harmful rates with denominators;mean selected gain;source distribution;regression MAE vs training median;AUROC descriptive",
        "independence": "recognition head trained on400;regularization selected on100 development;exploratory40 previously inspected;controller folds do not make complete pipeline independent",
        "independent_check": "explicit tree traversal,float32 sklearn input convention;separate cache replay and stable CE;new denominator/grouping/bootstrap arithmetic;no original report calls",
        "training_diagnostics": "per fitted fold,training-source predictions;average four in-sample appearances within each source before summarizing100 sources;final model in-sample separate;not independent evidence",
        "bootstrap": {"resamples": bootstrap, "seed": seed, "rng": "numpy.Generator(PCG64)",
            "percentile": "linear 2.5/97.5", "unit": "whole original source;all9 opportunities together",
            "sharing": "same indices across methods/comparisons within cohort;cohorts separate",
            "comparisons": "absolute utilities and every pair of six predefined methods;paired differences secondary marginal unadjusted",
            "limits": "conditional on fitted models,reused cohorts;not full training,selection or repeated-research uncertainty;random intervals describe expected utility across sources,not Bernoulli decision variance"},
        "tolerance": {"sign": u.TOL, "arithmetic_absolute": u.ATOL, "arithmetic_relative": u.RTOL,
                      "saved_prediction_absolute": u.TOL},
        "timing": {"sample_sources": membership["development"][:5], "opportunities_per_pass": 45,
            "warmups": 2, "repeats": 5, "seed": seed, "threads": 1,
            "modes": ["motion_rule", *u.LEARNED], "boundary": "preloaded rows;batch1 scaling+saved sklearn forest predict or motion decision;excludes all observation construction,loading,fitting,logging,decoding,encoder,codec motion and full recognition"},
        "fresh_test": "No genuinely independent test was run. No eligible untouched saved cohort;do not parse protected confirmation outcomes or acquire another cohort.",
        "resource_scope": "bounded CPU saved-artifact audit;no architecture search,downloads,video decode,encoder or recognition/probe training",
        "newer_completed_experiments": "local results directory and Git/readme inspection found none newer than cached_history_signal_diagnostic_v1_rerun1;original incomplete history attempt preserved"}
    output.mkdir(parents=True, exist_ok=True)
    u.save(output/"protocol.json", protocol)
    return protocol


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for role in ["development", "exploratory"]:
        base = u.frame(OLD/f"{role}_analysis_table.csv", usecols=u.KEYS+u.C_COLUMNS+["cohort", "class", "r", "B", "C", "G"])
        obs = u.frame(HISTORY/f"{role}_observations.csv", usecols=["example_id", *u.C_COLUMNS, *u.T_COLUMNS])
        assert set(obs.example_id) == set(base.example_id)
        obs = obs.set_index("example_id").loc[base.example_id]
        np.testing.assert_array_equal(obs[u.C_COLUMNS], base[u.C_COLUMNS])
        np.testing.assert_allclose(u.independent_history(base), obs[u.T_COLUMNS], atol=u.TOL, rtol=0)
        for c in u.T_COLUMNS: base[c] = obs[c].to_numpy()
        base["cohort"] = "development_oof" if role == "development" else "exploratory"
        tables.append(base)
    return tables[0], tables[1]


def recognition_and_identity_check(protocol: dict, dev: pd.DataFrame, exploratory: pd.DataFrame) -> dict:
    manifest = u.read(RECOGNITION/"dataset_manifest.json")
    entries = manifest["videos"]
    assert len(entries) == len({e["source_video_id"] for e in entries}) == len({e["file_identity"]["sha256"] for e in entries}) == 540
    for e in entries:
        assert e["video_id"].rsplit("_", 2)[0] == e["source_video_id"]
        assert e["time_end"] > e["time_start"]
    membership = protocol["membership"]
    assert [len(membership[r]) for r in ["train", "development", "exploratory"]] == [400, 100, 40]
    assert set(membership['train']).isdisjoint(membership['development']+membership['exploratory'])
    assert set(membership['development']).isdisjoint(membership['exploratory'])
    for table, role, count in [(dev, "development", 900), (exploratory, "exploratory", 360)]:
        assert len(table) == table.example_id.nunique() == count
        assert sorted(table.source_video_id.unique()) == membership[role]
        assert table.groupby("source_video_id").size().eq(9).all()
    selection = u.read(RECOGNITION/"classifiers/20/selection.json")
    assert u.digest(OLD/'fold_manifest.json') == u.digest(HISTORY/'fold_manifest.json') == u.digest(ROOT/'results/task_signal_ablation_v2/fold_manifest.json')
    fold_videos = pd.DataFrame(protocol['folds']['videos'])
    assert len(fold_videos) == fold_videos.source_video_id.nunique() == 100
    assert fold_videos.groupby(['fold','label']).size().eq(1).all()
    for f in protocol['folds']['folds']:
        assert len(f['training_sources']) == 80 and len(f['validation_sources']) == 20
        assert set(f['training_sources']).isdisjoint(f['validation_sources'])
        assert set(f['validation_sources']) == set(fold_videos.loc[fold_videos.fold == f['fold'],'source_video_id'])
    head = u.read(RECOGNITION/"classifiers/20/parameters.json")
    expected = u.read(RECOGNITION/"frozen_models.json")["parameter_sha256"]["20"]
    assert u.digest(RECOGNITION/"classifiers/20/parameters.json") == expected == selection["parameters_sha256"]
    assert set(selection["training_source_ids"]) == set(membership["train"])
    assert u.read(RECOGNITION/"protocol.json")["classes"] == manifest["classes"]
    grid = selection["grid"]
    chosen = min(grid, key=lambda r: (-r["development_accuracy"], r["development_CE"], r["grid_order"]))
    assert chosen["grid_order"] == selection["selected_grid_order"] == 1
    assert chosen["regularization"] == selection["selected_regularization"] == .01
    references, pools = {}, {}
    for e in entries:
        name = f"features/{e['video_id']}.npz"
        path = ROOT/e["reference_directory"]/name
        assert u.digest(path) == e["reference_artifact_sha256"][name]
        with np.load(path, allow_pickle=False) as arrays:
            x = arrays["features"].astype(np.float64)
            assert x.shape == (16, 512) and np.isfinite(x).all()
            np.testing.assert_array_equal(arrays["frame_indices"], e["candidate_indices"])
            np.testing.assert_array_equal(arrays["timestamps_seconds"], e["candidate_timestamps_seconds"])
        pools[e["source_video_id"]] = x.mean(axis=0)
        if e["recognition_role"] != "train": references[e["source_video_id"]] = x
    tr = np.stack([pools[s] for s in selection["training_source_ids"]])
    means, std = tr.mean(0), tr.std(0, ddof=0)
    np.testing.assert_allclose(head["means"], means, atol=u.ATOL, rtol=u.RTOL)
    np.testing.assert_allclose(head["scales"], np.where(std <= u.TOL, 1., std), atol=u.ATOL, rtol=u.RTOL)
    original = u.frame(DECOMP/"per_opportunity_decomposition.csv").set_index("example_id")
    saved_head = u.frame(RECOGNITION/"per_video_metrics.csv")
    max_target, max_loss, checks = 0., 0., 0
    dense_correct, periodic_correct = {}, {}
    by_source = {e["source_video_id"]: e for e in entries}
    for cohort, table in [("development", dev), ("exploratory", exploratory)]:
        for source, group in table.groupby("source_video_id", sort=True):
            e = by_source[source]; x = references[source]
            logits = ((x-head["means"])/head["scales"])@np.asarray(head["weights"]).T+head["bias"]
            y = manifest["classes"].index(e["label"])
            assert group['class'].eq(e["label"]).all()
            for policy, positions in [("dense", list(range(16))), ("periodic", [0, 4, 8, 12])]:
                prediction = logits[u.cache(positions)].mean(0)
                loss = u.cross_entropy(prediction, y)
                row = saved_head[(saved_head.training_per_class == 20)&(saved_head.cohort == cohort)&(saved_head.source_video_id == source)&(saved_head.policy == policy)].iloc[0]
                np.testing.assert_allclose(loss, row.cross_entropy, atol=u.ATOL, rtol=u.RTOL)
                assert int(prediction.argmax()) == row.prediction_index
                (dense_correct if policy == 'dense' else periodic_correct).setdefault(cohort, []).append(int(prediction.argmax() == y))
            for row in group.itertuples():
                a = [0, 4, 8, 12]; t, r = int(row.t), int(row.r)
                b, c = sorted(a+[t]), sorted([v for v in a if v != r]+[t])
                assert len(a) == len(c) == 4 and len(b) == 5 and a[0] == c[0] == 0
                ca, cc = u.cache(a), u.cache(c)
                np.testing.assert_array_equal(ca[:t], cc[:t])
                la, lb, lc = [u.cross_entropy(logits[u.cache(s)].mean(0), y) for s in [a, b, c]]
                old = original.loc[row.example_id]
                for actual, saved in [(la, old.loss_A), (lb, old.loss_B), (lc, old.loss_C)]:
                    max_loss = max(max_loss, abs(actual-saved)); np.testing.assert_allclose(actual, saved, atol=u.ATOL, rtol=u.RTOL)
                for actual, saved in [(la-lb, row.B), (lc-lb, row.C), (la-lc, row.G), (row.B-row.C, row.G), (la-lc, old.original_saved_target)]:
                    max_target = max(max_target, abs(actual-saved)); np.testing.assert_allclose(actual, saved, atol=u.ATOL, rtol=u.RTOL)
                assert np.all(x[ca] == np.stack([x[max(p for p in a if p <= i)] for i in range(16)]))
                assert np.all(x[cc] == np.stack([x[max(p for p in c if p <= i)] for i in range(16)]))
                checks += 1
    return {"manifest_sources": 540, "encoded_hash_duplicates": 0, "same_parent_cross_cohort": 0,
        "adjacent_same_parent_windows": 0, "reference_arrays_hash_checked": 540,
        "head_sha256": expected, "head_regularization": .01,
        "head_selection_uses_development_outcomes": True, "head_train_normalization_reproduced": True,
        "dense_accuracy_reproduced": {c: float(np.mean(v)) for c, v in dense_correct.items()},
        "periodic_accuracy_reproduced": {c: float(np.mean(v)) for c, v in periodic_correct.items()},
        "independent_target_and_cache_replays": checks, "maximum_loss_discrepancy": max_loss,
        "maximum_target_discrepancy": max_target,
        "unverifiable": ["semantic/near duplicates or reuploads under distinct source IDs;no perceptual content analysis",
                         "independence beyond recorded source IDs/windows and encoded hashes;encoded video bytes not rehashed or decoded in this audit",
                         "complete human inspection/decision chronology outside saved protocols and Git"]}


def generate_predictions(output: Path, protocol: dict, dev: pd.DataFrame, exploratory: pd.DataFrame):
    parts, training_rows, model_checks, thresholds, frozen_timing_models = [], [], [], [], {}
    old_predictions = {c: u.frame(OLD/n) for c, n in [("development_oof", "out_of_fold_predictions.csv"), ("exploratory", "exploratory_predictions.csv")]}
    history_predictions = {c: u.frame(HISTORY/n) for c, n in [("development_oof", "out_of_fold_predictions.csv"), ("exploratory", "exploratory_predictions.csv")]}
    maximum_prediction = 0.
    bundles = [(f"fold_{f['fold']}", f, dev[dev.source_video_id.isin(f['validation_sources'])], "development_oof") for f in protocol['folds']['folds']]
    bundles.append(("final_development", None, exploratory, "exploratory"))
    for bundle, fold, held, cohort in bundles:
        tr = dev if fold is None else dev[dev.source_video_id.isin(fold['training_sources'])]
        assert set(tr.source_video_id).isdisjoint(held.source_video_id)
        threshold = u.motion_threshold(tr); median = float(tr.G.median())
        thresholds.append({"bundle": bundle, "training_sources": sorted(tr.source_video_id.unique()),
            "training_example_ids": tr.example_id.tolist(), "available_training_opportunities": int(tr.current_mv_available.sum()),
            "threshold": threshold, "comparison": "available && p90 > median;missing=>reject", "training_median_G": median})
        models, values_held, values_train = [], {}, {}
        for filename in MODEL_FILES:
            folder = (HISTORY if filename == "C_plus_T_RF400.json" else OLD)/"models"/bundle
            state, forest, check = u.load_forest(folder, filename, tr)
            for source in held.source_video_id.unique(): assert source not in state['training_source_ids']
            ph, dh = u.predict(held, state, forest)
            pt, dt = u.predict(tr, state, forest)
            model_checks.append({"bundle": bundle, "file": relative(folder/filename), "max_independent_vs_library_prediction": max(dh, dt), **check})
            models.append((state, forest)); values_held[filename], values_train[filename] = ph, pt
        for table, values, phase in [(held, values_held, "held_out"), (tr, values_train, "in_sample")]:
            learned = {"direct_RF400": values[MODEL_FILES[0]],
                "two_target_RF200x2": values[MODEL_FILES[1]]-values[MODEL_FILES[2]],
                "history_RF400": values[MODEL_FILES[3]]}
            for method in (u.METHODS if phase == "held_out" else u.LEARNED):
                p = learned.get(method)
                probability = (p > u.TOL).astype(float) if p is not None else np.zeros(len(table)) if method == "keep_periodic" else np.full(len(table), .5) if method == "random_move_expected" else u.accept_motion(table, threshold).astype(float)
                raw = table[[*u.KEYS, "class", "r", "G", "B", "C", "current_p90_magnitude_px", "current_mv_available"]].copy()
                raw['cohort'], raw['method'], raw['bundle'] = cohort, method, bundle
                raw['fold'] = -1 if fold is None else fold['fold']
                raw['predicted_G'] = p if p is not None else np.nan
                raw['training_median_G'], raw['motion_threshold'] = median, threshold
                raw['accept_probability'], raw['utility_contribution'] = probability, probability*raw.G
                raw['phase'] = phase
                raw['reference_schedule'] = '[0, 4, 8, 12]'
                raw['move_schedule'] = [json.dumps(sorted([v for v in [0, 4, 8, 12] if v != int(r)]+[int(t)])) for r,t in zip(raw.r,raw.t)]
                raw['simulated_refreshes_if_rejected'] = raw['simulated_refreshes_if_accepted'] = 4
                raw['candidate_outputs_each'] = 16
                if phase == 'held_out':
                    if p is not None:
                        if method == 'history_RF400':
                            saved = history_predictions[cohort]; saved = saved[saved.method == 'C_plus_T_RF400'].set_index('example_id').loc[table.example_id]
                            expected = saved.prediction.to_numpy()
                        else:
                            saved = old_predictions[cohort]; saved = saved[saved.method == ('Direct_G_RF400' if method == 'direct_RF400' else 'Two_target_RF200x2_forward')].set_index('example_id').loc[table.example_id]
                            expected = saved.prediction_G.to_numpy()
                        error = float(abs(p-expected).max()); maximum_prediction = max(maximum_prediction,error)
                        np.testing.assert_allclose(p, expected, atol=u.TOL, rtol=0)
                        np.testing.assert_array_equal(probability.astype(bool), saved.selected.to_numpy(dtype=bool))
                        np.testing.assert_allclose(raw.utility_contribution, saved.realized_gain, atol=u.TOL, rtol=0)
                    parts.append(raw)
                else:
                    raw['cohort'] = bundle
                    training_rows.append(u.per_source(raw))
        if fold is None:
            frozen_timing_models = {'direct_RF400': [models[0]], 'two_target_RF200x2': models[1:3], 'history_RF400': [models[3]]}
    decisions = pd.concat(parts, ignore_index=True)
    train_source = pd.concat(training_rows, ignore_index=True)
    u.csv(output/'opportunity_decisions.csv',decisions)
    u.csv(output/'in_sample_by_model_source.csv',train_source)
    u.save(output/'motion_thresholds.json',thresholds)
    return decisions, train_source, model_checks, maximum_prediction, frozen_timing_models


def timing(output: Path, protocol: dict, dev: pd.DataFrame, models: dict) -> None:
    selected = dev[dev.source_video_id.isin(protocol['timing']['sample_sources'])].sort_values(['source_video_id','t'])
    preloaded = {m: [(r[state['input_columns']].to_numpy(dtype=float), np.array(state['means']), np.array(state['scales']), forest)
                    for _,r in selected.iterrows() for state,forest in models[m]] for m in u.LEARNED}
    motion = list(zip(selected.current_p90_magnitude_px.to_numpy(),selected.current_mv_available.to_numpy()))
    threshold = u.motion_threshold(dev)
    def execute(mode):
        last = None
        if mode == 'motion_rule':
            for value,available in motion: last = available == 1 and threshold is not None and value > threshold
        elif mode == 'two_target_RF200x2':
            rows=preloaded[mode]
            for i in range(0,len(rows),2):
                a,m,s,f=rows[i];b,m2,s2,f2=rows[i+1]
                last=(f.predict(((a-m)/s).reshape(1,-1))-f2.predict(((b-m2)/s2).reshape(1,-1))) > u.TOL
        else:
            for row,mean,scale,forest in preloaded[mode]: last = forest.predict(((row-mean)/scale).reshape(1,-1)) > u.TOL
        return last
    rng=np.random.Generator(np.random.PCG64(protocol['seed'])); rows=[]; orders=[]
    for phase,repeats in [('warmup',2),('measured',5)]:
        for repeat in range(repeats):
            order=rng.permutation(protocol['timing']['modes']).tolist(); orders.append({'phase':phase,'repeat':repeat,'order':order})
            for ordinal,mode in enumerate(order):
                start=time.perf_counter(); execute(mode); milliseconds=(time.perf_counter()-start)*1000
                if phase=='measured': rows.append({'mode':mode,'repeat':repeat,'execution_ordinal':ordinal,'opportunities':len(selected),'RF_batch_one_calls':0 if mode=='motion_rule' else len(selected)*(2 if mode=='two_target_RF200x2' else 1),'total_ms':milliseconds,'ms_per_opportunity':milliseconds/len(selected)})
    u.csv(output/'cpu_component_timings.csv',pd.DataFrame(rows))
    u.save(output/'cpu_component_timing_metadata.json',{'recorded_at_utc':now(),'sample':protocol['timing'],'execution_order':orders,
        'models_resident_simultaneously':True,'same_400_total_trees_does_not_imply_equal_cost':True,
        'two_target_boundary':'two independent200-tree scaling/prediction calls,net subtraction and threshold decision included;no feature construction/history updates included',
        'baseline_evidence':'Existing history construction timing is separately cited,not added to these timings or historical GPU timings.'})


def unchanged(protocol: dict) -> None:
    for name,expected in protocol['input_sha256'].items(): assert u.digest(ROOT/name)==expected,name
    for name,expected in protocol['source_sha256'].items(): assert u.digest(ROOT/name)==expected,name
    # Read-only Git integrity check; no protected outcomes parsed by this audit.
    changed=subprocess.check_output(['git','diff','--name-only',protocol['starting_commit']],cwd=ROOT,text=True).splitlines()
    changed=[name for name in changed if name in protocol['previous_tracked_blob_identities']]
    assert not changed,changed


def ledger(output: Path) -> None:
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text(''.join(f'{u.digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files),encoding='utf-8')


def audit_saved(output: Path, protocol: dict) -> dict:
    unchanged(protocol)
    for line in (output/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1);assert u.digest(output/name)==expected,name
    decisions=u.frame(output/'opportunity_decisions.csv')
    source=u.per_source(decisions); stored=u.frame(output/'per_source_metrics.csv')
    u.compare_saved_csv(source,stored)
    summary,comparisons,matrices=u.summarize(decisions,source,protocol['seed'],protocol['bootstrap']['resamples'])
    u.compare_saved_csv(summary,u.frame(output/'summary.csv'))
    assert comparisons==u.read(output/'bootstrap_intervals.json')['paired_comparisons']
    with np.load(output/'bootstrap_indices.npz',allow_pickle=False) as stored:
        for name,indices in matrices.items():np.testing.assert_array_equal(indices,stored[name])
    return {'passed':True,'read_only':True,'saved_metrics_denominators_and_bootstraps_reproduced':True,'previous_artifacts_unchanged':True}


def run(args) -> None:
    output=(ROOT/args.output).resolve()
    if args.audit_only and not (output/'verification.json').exists():
        raise FileNotFoundError('Read-only audit requires a completed output directory')
    if output.exists() and any(output.iterdir()) and not (output/'protocol.json').exists():
        raise FileExistsError('Nonempty output preserved;choose a versioned fresh output')
    protocol=freeze(output,args.seed,args.bootstrap)
    if args.audit_only or (output/'verification.json').exists():
        print(json.dumps(audit_saved(output,protocol),indent=2));return
    u.save(output/'status.json',{'status':'running','started_at_utc':now()})
    dev,exploratory=load_tables()
    print('Protocol frozen;observation/target membership and independent history reconstruction checked.',flush=True)
    identity=recognition_and_identity_check(protocol,dev,exploratory)
    print('Recognition normalization,source identities and1260 explicit target/cache replays checked.',flush=True)
    # An audit must not refit even inadvertently. No torch/PyAV imported here.
    with patch.object(RandomForestRegressor,'fit',side_effect=AssertionError('No fitting permitted in audit')):
        decisions,training,models,prediction_error,timing_models=generate_predictions(output,protocol,dev,exploratory)
        sources=u.per_source(decisions)
        summary,comparisons,matrices=u.summarize(decisions,sources,args.seed,args.bootstrap)
        u.csv(output/'per_source_metrics.csv',sources);u.csv(output/'summary.csv',summary)
        np.savez_compressed(output/'bootstrap_indices.npz',**matrices)
        u.save(output/'bootstrap_intervals.json',{'seed':args.seed,'resamples':args.bootstrap,'rng':'PCG64','percentile':'linear','paired_comparisons':comparisons})
        u.save(output/'model_checks.json',{'models':models,'maximum_saved_prediction_discrepancy':prediction_error})
        print('Frozen forest predictions reproduced;new single-move baselines and source metrics saved.',flush=True)
        timing(output,protocol,dev,timing_models)
    packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    cpu=subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'],text=True)
    u.save(output/'environment.json',{'recorded_at_utc':now(),'os':platform.platform(),'python':platform.python_version(),'cpu':json.loads(cpu),'packages':packages,'threads':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},'RF_n_jobs':1,'dependencies_changed':False})
    from refresh_generalization_audit_report import report
    report(output,protocol,summary,training,identity)
    tests=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','unittest','test_refresh_generalization_audit','-v'],cwd=ROOT,capture_output=True,text=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    release=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'validate_release.py'],cwd=ROOT,capture_output=True,text=True)
    assert release.returncode==0,release.stdout+release.stderr
    unchanged(protocol)
    verification={'passed':True,'checked_at_utc':now(),'independent_identity_and_recognition_checks':identity,
        'maximum_saved_prediction_discrepancy':prediction_error,'model_scaler_checks':len(models),'bootstrap_root_checks':sum(r['bootstrap_root_checks'] for r in models),
        'sources_per_cohort':{'development':100,'exploratory':40},'opportunities_per_cohort':{'development':900,'exploratory':360},
        'source_grouping_and_training_only_motion_thresholds':True,'saved_metrics_and_denominators':True,'no_refitting_guard':True,
        'actual_encoder_calls':0,'video_decodes':0,'downloads':0,'recognition_training_calls':0,'probe_training_calls':0,
        'protected_confirmation_outcomes_parsed':0,'independent_test_run':False,'previous_artifacts_unchanged':True,
        'focused_tests':{'returncode':tests.returncode,'output':tests.stdout+tests.stderr},'release_validator':{'returncode':release.returncode,'output':release.stdout+release.stderr}}
    u.save(output/'verification.json',verification)
    ledger(output)
    audit_saved(output,protocol)
    u.save(output/'status.json',{'status':'complete','completed_at_utc':now(),'blocker':None})
    ledger(output)
    print('Complete:'+relative(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/refresh_generalization_audit_v1'))
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.bootstrap<1:parser.error('--bootstrap must be positive')
    run(args)
