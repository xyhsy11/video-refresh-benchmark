"""Exact-budget offline supervision; deployable scorers see only cheap inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from analyze_cache_drift import TARGET, fit_predict
from cache_drift_v2_utils import read_json, validate_split
from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS
from refresh_policy_utils import (CheapVideo, ReplayVideo, IntervalAccumulator, TOL,
    distance_matrix, exact_oracle, pair_inputs, protected_files)
from research_utils import ROOT, relative_path, sha256_file

VALUE_TARGET = "refresh_advantage"
BUDGET_INPUTS = ["remaining_refresh_count", "remaining_candidate_count", "remaining_refresh_fraction"]
VALUE_COLUMNS = {"value_age_budget": AGE_INPUTS+BUDGET_INPUTS,
                 "value_age_motion_budget": AGE_INPUTS+BUDGET_INPUTS+MOTION_INPUTS}
DRIFT_COLUMNS = {"age_only": AGE_INPUTS, "age_plus_motion": AGE_INPUTS+MOTION_INPUTS}
THRESHOLD_POLICIES = ["age_only", "age_plus_motion", "paced_age_plus_motion", "privileged_actual_drift"]
POLICIES = ["periodic", "random", *THRESHOLD_POLICIES, *VALUE_COLUMNS, "oracle"]
LAMBDAS = [0.0, 0.05, 0.10, 0.20]
EXPERIMENT = "budget_aware_refresh_v2"


def protected_inventory() -> dict:
    inventory = protected_files()
    for name in ["refresh_policy_utils.py", "replay_refresh_policies.py", "refresh_replay_report.py", "test_refresh_policies.py"]:
        inventory[name] = sha256_file(ROOT/name)
    # Preservation reads bytes for hashes, not original validation/test arrays or metrics.
    import subprocess
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name.startswith("results/adaptive_refresh_v1"):
            inventory[name] = sha256_file(ROOT/name)
    return dict(sorted(inventory.items()))


def make_development_split(manifest: dict, seed: int) -> dict:
    validate_split(manifest, check_files=False)
    originals = [e for e in manifest["videos"] if e["experiment_split"] == "train"]
    classes = sorted({e["label"] for e in originals})
    assert len(originals)==120 and len(classes)==20
    rows, counts, seeds = [], {}, {}
    for label in classes:
        entries = sorted([e for e in originals if e["label"]==label], key=lambda e:e["source_video_id"])
        assert len(entries)==6
        class_seed = int(hashlib.sha256(f"{EXPERIMENT}|split|{seed}|{label}".encode()).hexdigest()[:16],16)
        order = list(range(6)); random.Random(class_seed).shuffle(order)
        seeds[label] = class_seed
        training = set(order[:4])
        for i,e in enumerate(entries):
            rows.append({**e,"original_experiment_split":e["experiment_split"],
                         "development_role":"train" if i in training else "development"})
        counts[label] = {"train":4,"development":2}
    sources = {s:{e["source_video_id"] for e in rows if e["development_role"]==s} for s in ["train","development"]}
    assert len(sources["train"])==80 and len(sources["development"])==40
    assert sources["train"].isdisjoint(sources["development"])
    excluded = {e["source_video_id"] for e in manifest["videos"] if e["experiment_split"]!="train"}
    assert not (sources["train"]|sources["development"]) & excluded
    assert len({e["file_identity"]["sha256"] for e in rows})==120
    return {"seed":seed,"status":"frozen","selection_rule":"sort classes and six source IDs; SHA256 first 16 hex digits of budget_aware_refresh_v2|split|seed|class; random.Random.shuffle indices; first four train, last two development",
            "class_seeds":seeds,"class_counts":counts,"counts":{"train":80,"development":40},
            "excluded_original_validation_test_sources":sorted(excluded),"videos":sorted(rows,key=lambda e:e["video_id"])}


def load_training_artifacts(directory: Path, manifest: dict, split: dict) -> tuple[list, list, dict]:
    """Unlike v1's loader, never open original validation/test feature or motion data."""
    ledger_path = directory/"SHA256SUMS.txt"
    if not ledger_path.exists(): raise FileNotFoundError(f"Required identity ledger missing: {relative_path(ledger_path)}")
    ledger = {name:digest for digest,name in (line.split("  ",1) for line in ledger_path.read_text().splitlines())}
    metadata_path = directory/"collection_metadata.json"
    if not metadata_path.exists(): raise FileNotFoundError(f"Required collection metadata missing: {relative_path(metadata_path)}")
    assert sha256_file(metadata_path)==ledger["collection_metadata.json"]
    collection = read_json(metadata_path)
    assert collection["status"]=="collected"
    assert sha256_file(directory/"split_manifest.json")==collection["split_manifest_sha256"]==ledger["split_manifest.json"]
    original_train = {e["video_id"]:e for e in manifest["videos"] if e["experiment_split"]=="train"}
    opened, hashes, loaded = [], {}, []
    for selected in split["videos"]:
        video_id = selected["video_id"]
        entry = original_train[video_id]
        assert {k:selected[k] for k in entry}==entry
        names = [f"features/{video_id}.npz",f"features/{video_id}.json",f"motion_frames/{video_id}.csv"]
        for name in names:
            p=directory/name
            if not p.exists(): raise FileNotFoundError(f"Required original-training artifact missing: {relative_path(p)}")
            digest=sha256_file(p)
            if digest!=ledger.get(name): raise ValueError(f"Artifact identity mismatch: {relative_path(p)}")
            hashes[name]=digest; opened.append(name)
        side=read_json(directory/names[1])
        assert side["entry"]==entry
        with np.load(directory/names[0],allow_pickle=False) as data:
            f=data["features"].copy(); indices=data["frame_indices"].tolist(); stamps=data["timestamps_seconds"].tolist()
        assert f.shape==(16,512) and f.dtype==np.float32 and np.isfinite(f).all()
        assert np.min(np.linalg.norm(f.astype(np.float64),axis=1))>TOL
        assert indices==entry["candidate_indices"]==[c["decoded_frame_index"] for c in side["candidates"]]
        assert indices==sorted(set(indices)) and indices[0]==0 and indices[-1]==entry["decoded_frame_count"]-1
        assert stamps==entry["candidate_timestamps_seconds"]==[c["timestamp_seconds"] for c in side["candidates"]]
        assert np.isfinite(stamps).all() and (np.diff(stamps)>0).all()
        frames=pd.read_csv(directory/names[2],float_precision="round_trip")
        assert len(frames)==entry["decoded_frame_count"] and frames.decoded_frame_index.tolist()==list(range(len(frames)))
        assert set(frames.video_id)=={video_id}
        assert frames.frame_type.value_counts().to_dict()=={k:v for k,v in entry["frame_type_counts"].items() if v}
        assert (frames.mv_available==(frames.vector_count>0)).all()
        counts=frames[["vector_count","past_reference_vectors","future_reference_vectors","other_reference_vectors"]]
        assert np.isfinite(counts).all().all() and (counts>=0).all().all() and (counts%1==0).all().all()
        assert (counts.iloc[:,1:].sum(axis=1)==frames.vector_count).all()
        mag=frames[["mean_magnitude_px","p90_magnitude_px"]]
        assert np.isfinite(mag).all().all() and (mag>=0).all().all()
        assert (mag.loc[~frames.mv_available]==0).all().all()
        f.setflags(write=False)
        loaded.append(ReplayVideo(selected,CheapVideo(entry["source_video_id"],tuple(indices),tuple(stamps),tuple(frames.to_dict("records"))),f))
    train=[v for v in loaded if v.entry["development_role"]=="train"]
    dev=[v for v in loaded if v.entry["development_role"]=="development"]
    assert len(train)==80 and len(dev)==40
    assert all(Path(name).stem in original_train for name in opened)
    return train,dev,{"opened_artifacts":opened,"artifact_sha256":hashes,"original_validation_test_artifacts_opened":0,
        "reference_features_loaded":1920,"manifest_sha256":sha256_file(directory/"split_manifest.json"),
        "ledger_sha256":sha256_file(ledger_path),"collection_configuration":{k:collection[k] for k in ["model","numerical_settings","decoder","motion_descriptor_definition","environment"]}}


class ActionOracle:
    """None represents infeasibility; no infinite costs are serialized."""
    def __init__(self, distances: np.ndarray):
        self.d=np.asarray(distances,dtype=np.float64)
        self.n=len(self.d)
        assert self.d.shape==(self.n,self.n) and np.isfinite(self.d).all()
        self.value=lru_cache(None)(self._value)

    def q(self,t:int,r:int,b:int) -> dict:
        if not (0<=r<t<self.n and 0<=b<=self.n-t): raise ValueError("Infeasible decision state")
        reuse=refresh=None
        if b<=self.n-t-1:
            tail=self.value(t+1,r,b)
            if tail is not None: reuse=float(self.d[t,r]+tail)
        if b>=1:
            tail=self.value(t+1,t,b-1)
            if tail is not None: refresh=float(self.d[t,t]+tail)
        return {"reuse":reuse,"refresh":refresh}

    def _value(self,t:int,r:int,b:int) -> float | None:
        if t==self.n: return 0.0 if b==0 else None
        if b<0 or b>self.n-t: return None
        values=[v for v in self.q(t,r,b).values() if v is not None]
        return min(values) if values else None

    def reachable(self,budget:int) -> list[tuple]:
        assert 1<=budget<=self.n
        states={(1,0,budget-1)}; all_states=set()
        for t in range(1,self.n):
            following=set()
            for state in sorted(states):
                assert state[0]==t
                all_states.add(state)
                _,r,b=state
                q=self.q(t,r,b)
                if q["reuse"] is not None: following.add((t+1,r,b))
                if q["refresh"] is not None: following.add((t+1,t,b-1))
            states=following
        assert all(t==self.n and b==0 for t,r,b in states)
        return sorted(all_states)


def budget_inputs(b:int,t:int,n:int=16) -> dict:
    assert 0<=b<=n-t and t<n
    return dict(zip(BUDGET_INPUTS,[b,n-t,b/(n-t)]))


def training_tables(videos:list,budgets:list[int]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    drift,values,audits=[],[],[]
    for v in videos:
        assert v.entry["development_role"]=="train"
        d=distance_matrix(v.references); oracle=ActionOracle(d)
        inputs={(r,t):pair_inputs(v.cheap,r,t) for t in range(1,16) for r in range(t)}
        for (r,t),x in inputs.items():
            drift.append({"video_id":v.entry["video_id"],"source_video_id":v.cheap.source_video_id,"t":t,"r":r,**x,TARGET:float(d[t,r])})
        for budget in budgets:
            exact=exact_oracle(d,budget)
            initial=(float(d[0,0])+oracle.value(1,0,budget-1))/16
            assert abs(initial-exact["exact_minimum_error"])<=TOL
            reachable=oracle.reachable(budget); count=0
            for t,r,b in reachable:
                q=oracle.q(t,r,b)
                if q["reuse"] is None or q["refresh"] is None: continue
                count+=1
                values.append({"video_id":v.entry["video_id"],"source_video_id":v.cheap.source_video_id,
                    "budget":budget,"t":t,"r":r,"remaining_budget":b,**inputs[r,t],**budget_inputs(b,t),
                    "Q_reuse_sum":q["reuse"],"Q_refresh_sum":q["refresh"],"V_sum":oracle.value(t,r,b),
                    VALUE_TARGET:(q["reuse"]-q["refresh"])/16})
            audits.append({"video_id":v.entry["video_id"],"budget":budget,"reachable_states":len(reachable),
                "optional_supervised_states":count,"dp_initial_mean_error":initial,"exhaustive_mean_error":exact["exact_minimum_error"]})
    drift=pd.DataFrame(drift); values=pd.DataFrame(values)
    assert len(drift)==120*len(videos) and not drift.duplicated(["video_id","t","r"]).any()
    assert not values.duplicated(["video_id","budget","t","r","remaining_budget"]).any()
    assert np.isfinite(drift[AGE_INPUTS+MOTION_INPUTS+[TARGET]]).all().all()
    assert np.isfinite(values[AGE_INPUTS+BUDGET_INPUTS+MOTION_INPUTS+[VALUE_TARGET,"Q_reuse_sum","Q_refresh_sum"]]).all().all()
    counts={}
    for budget in budgets:
        sizes=values[values.budget==budget].groupby("video_id").size()
        assert len(sizes)==len(videos) and sizes.nunique()==1
        counts[str(budget)]={"states_per_training_video":int(sizes.iloc[0]),"total_states":int(sizes.sum())}
    return drift,values,{"counts_by_budget":counts,"initial_dp_exhaustive_checks":audits}


def fit_action_ridge(table:pd.DataFrame,columns:list[str]) -> dict:
    if columns not in list(VALUE_COLUMNS.values()): raise ValueError("Action-value input whitelist required")
    x=table[columns].to_numpy(dtype=np.float64); y=table[VALUE_TARGET].to_numpy(dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    means=x.mean(axis=0); raw_std=x.std(axis=0,ddof=0); scales=np.where(raw_std<=TOL,1.0,raw_std)
    z=(x-means)/scales; intercept=float(y.mean())
    lhs=z.T@z+np.eye(len(columns)); rhs=z.T@(y-intercept)
    coefficients=np.linalg.solve(lhs,rhs)
    np.testing.assert_allclose(lhs@coefficients,rhs,rtol=1e-8,atol=1e-9)
    return {"columns":columns,"target":VALUE_TARGET,"alpha":1.0,"training_rows":len(table),
        "means":means.tolist(),"raw_standard_deviations":raw_std.tolist(),"scales":scales.tolist(),
        "coefficients":coefficients.tolist(),"intercept":intercept,"intercept_penalized":False,
        "target_statistics":{"mean":intercept,"std_population":float(y.std()),"min":float(y.min()),"max":float(y.max())},
        "constant_columns":[columns[i] for i in np.flatnonzero(raw_std<=TOL)]}


def fit_predictors(drift:pd.DataFrame,values:pd.DataFrame,budgets:list[int]) -> dict:
    models={"drift":{},"action_value":{}}
    for name,columns in DRIFT_COLUMNS.items():
        _,state=fit_predict(drift,drift.iloc[:1],columns)
        models["drift"][name]=state
    for budget in budgets:
        table=values[values.budget==budget]
        models["action_value"][str(budget)]={name:fit_action_ridge(table,cols) for name,cols in VALUE_COLUMNS.items()}
    return models


def predict_array(table:pd.DataFrame,state:dict) -> np.ndarray:
    x=table[state["columns"]].to_numpy(dtype=np.float64)
    result=state["intercept"]+((x-np.asarray(state["means"]))/np.asarray(state["scales"]))@np.asarray(state["coefficients"])
    assert np.isfinite(result).all()
    return result


def controller_score(policy:str,inputs:dict,state:dict) -> tuple[float,float]:
    """No identities, classes, reference features, evaluator targets or future data."""
    allowed=VALUE_COLUMNS[policy] if policy in VALUE_COLUMNS else DRIFT_COLUMNS[policy]
    if set(inputs)!=set(allowed) or state["columns"]!=allowed: raise ValueError("Scorer inputs must exactly match explicit whitelist")
    if policy in VALUE_COLUMNS and state.get("target")!=VALUE_TARGET: raise ValueError("Distinct signed action-value target required")
    x=np.array([inputs[c] for c in allowed],dtype=np.float64)
    raw=float(state["intercept"]+((x-np.asarray(state["means"]))/np.asarray(state["scales"]))@np.asarray(state["coefficients"]))
    assert np.isfinite(raw)
    return raw,raw if policy in VALUE_COLUMNS else float(np.clip(raw,0,2))


def random_positions(source_id:str,budget:int,seed:int) -> list[int]:
    digest=hashlib.sha256(f"{EXPERIMENT}|random|{seed}|{source_id}".encode()).hexdigest()
    return [0,*sorted(random.Random(int(digest[:16],16)).sample(range(1,16),budget-1))]


def _trajectory(cheap:CheapVideo,policy:str,budget:int,score_fn=None,setting:dict|None=None,positions:list|None=None) -> list[dict]:
    r=0; b=budget-1; scheduled=set(positions or [])
    acc=IntervalAccumulator(cheap.motion,cheap.indices[0])
    rows=[{"candidate_ordinal":0,"cached_before_ordinal":None,"cached_output_ordinal":0,"refresh":True,
           "decision_reason":"mandatory_first","forced_refresh":False,"remaining_budget_before":budget,
           "remaining_budget_after":b,"score_raw":None,"score_decision":None,"threshold_t":None}]
    for t in range(1,16):
        elapsed=cheap.timestamps[t]-cheap.timestamps[r]
        x={"age_seconds":elapsed,"age_decoded_frames":cheap.indices[t]-cheap.indices[r],"age_candidate_positions":t-r,
           "age_seconds_squared":elapsed*elapsed,**acc.advance(cheap.indices[t]),**budget_inputs(b,t)}
        raw=score=None
        if score_fn is not None: raw,score=score_fn(x,r,t)
        threshold=None
        if setting and setting["kind"]=="threshold":
            threshold=setting["value"]+setting.get("lambda",0.0)*((budget-1)/15-b/(16-t))
        if b==0: action,reason=False,"budget_exhausted"
        elif b==16-t: action,reason=True,"forced_budget"
        elif policy in VALUE_COLUMNS:
            action=score>0; reason="positive_advantage" if action else "nonpositive_advantage"
        elif policy in THRESHOLD_POLICIES:
            action=setting["kind"]=="always" or (threshold is not None and score>=threshold)
            reason="threshold_refresh" if action else "threshold_reuse"
        else:
            action=t in scheduled; reason="scheduled_refresh" if action else "scheduled_reuse"
        previous=r; before=b
        if action: r=t; b-=1; acc.reset(cheap.indices[t])
        rows.append({"candidate_ordinal":t,"cached_before_ordinal":previous,"cached_output_ordinal":r,
            "refresh":bool(action),"decision_reason":reason,"forced_refresh":reason=="forced_budget",
            "remaining_budget_before":before,"remaining_budget_after":b,"score_raw":raw,"score_decision":score,
            "threshold_t":threshold,**x})
    assert b==0 and sum(row["refresh"] for row in rows)==budget
    return rows


def plan_deployable(cheap:CheapVideo,policy:str,budget:int,models:dict,setting:dict|None=None,seed:int|None=None) -> list[dict]:
    assert policy in ["periodic","random","dense","age_only","age_plus_motion","paced_age_plus_motion",*VALUE_COLUMNS]
    assert budget in [2,4,8,16]
    if policy in VALUE_COLUMNS:
        state=models["action_value"][str(budget)][policy]; columns=VALUE_COLUMNS[policy]
        callback=lambda x,r,t:controller_score(policy,{c:x[c] for c in columns},state)
    elif policy in ["age_only","age_plus_motion","paced_age_plus_motion"]:
        name="age_plus_motion" if policy=="paced_age_plus_motion" else policy
        state=models["drift"][name]; columns=DRIFT_COLUMNS[name]
        callback=lambda x,r,t:controller_score(name,{c:x[c] for c in columns},state)
    else: callback=None
    positions=list(range(16)) if policy=="dense" else list(range(0,16,16//budget)) if policy=="periodic" else random_positions(cheap.source_video_id,budget,seed) if policy=="random" else None
    if policy=="dense": assert budget==16
    return _trajectory(cheap,policy,budget,callback,setting,positions)


def plan_privileged(cheap:CheapVideo,budget:int,distances:np.ndarray,setting:dict) -> list[dict]:
    # Separate privileged interface; never invoked by deployable callbacks.
    return _trajectory(cheap,"privileged_actual_drift",budget,lambda x,r,t:(float(distances[t,r]),float(distances[t,r])),setting)


def plan_oracle(cheap:CheapVideo,budget:int,distances:np.ndarray) -> list[dict]:
    return _trajectory(cheap,"oracle",budget,positions=exact_oracle(distances,budget)["positions"])


def evaluate(cheap:CheapVideo,plan:list[dict],oracle:ActionOracle,budget:int) -> tuple[list[dict],dict]:
    rows=[]; latest=0; errors=[]; regrets=[]
    for row in plan:
        t=row["candidate_ordinal"]
        q={"reuse":None,"refresh":None}; regret=0.0
        if t:
            assert row["cached_before_ordinal"]==latest
            expected=pair_inputs(cheap,latest,t)
            for k,v in expected.items(): assert np.isclose(row[k],v,rtol=1e-12,atol=TOL),k
            b=row["remaining_budget_before"]; q=oracle.q(t,latest,b)
            chosen=q["refresh" if row["refresh"] else "reuse"]
            assert chosen is not None
            regret=(chosen-min(v for v in q.values() if v is not None))/16
            assert regret>=-TOL
            if b==0: assert not row["refresh"]
            if b==16-t: assert row["refresh"]
        else: assert row["refresh"]
        before=latest
        if row["refresh"]: latest=t
        assert row["cached_output_ordinal"]==latest
        error=float(oracle.d[t,latest]); errors.append(error); regrets.append(regret)
        if row["refresh"]: assert abs(error)<=TOL
        rows.append({**row,"evaluator_true_pre_decision_drift":float(oracle.d[t,before]) if t else None,
            "evaluator_Q_reuse_sum":q["reuse"],"evaluator_Q_refresh_sum":q["refresh"],
            "evaluator_reuse_feasible":q["reuse"] is not None,"evaluator_refresh_feasible":q["refresh"] is not None,
            "evaluator_action_regret":float(regret),"output_feature_error":error})
    positions=[row["candidate_ordinal"] for row in rows if row["refresh"]]
    mean=float(np.mean(errors)); optimum=(oracle.d[0,0]+oracle.value(1,0,budget-1))/16
    assert len(rows)==16 and len(positions)==budget and rows[-1]["remaining_budget_after"]==0
    assert abs(sum(regrets)-(mean-optimum))<=TOL
    return rows,{"mean_output_error":mean,"max_output_error":max(errors),"simulated_refreshes":len(positions),
        "cache_reuses":16-len(positions),"feature_outputs":16,"refresh_positions":positions,
        "forced_refreshes":sum(row["forced_refresh"] for row in rows),"last_refresh_position":positions[-1],
        "outputs_after_budget_exhausted":15-positions[-1],"sum_action_regret":float(sum(regrets)),"oracle_optimum_error":float(optimum)}


def threshold_grids(drift:pd.DataFrame,models:dict) -> dict:
    result={}
    for policy in THRESHOLD_POLICIES:
        name="age_plus_motion" if policy=="paced_age_plus_motion" else policy
        scores=drift[TARGET].to_numpy(dtype=np.float64) if policy=="privileged_actual_drift" else np.clip(predict_array(drift,models["drift"][name]),0,2)
        values=[]
        for q,value in zip(range(0,101,5),np.percentile(scores,np.arange(0,101,5),method="linear")):
            if not values or value!=values[-1]["value"]:
                values.append({"threshold_id":f"q{q:03d}","kind":"threshold","value":float(value),"first_quantile_percent":q})
        base=[{"threshold_id":"always","kind":"always","value":None},*values,{"threshold_id":"wait","kind":"wait","value":None}]
        result[policy]=[]
        for order,setting in enumerate(base):
            for lam in LAMBDAS if policy=="paced_age_plus_motion" and setting["kind"]=="threshold" else [0.0]:
                result[policy].append({**setting,"lambda":lam,"grid_order":order,"setting_id":f"{setting['threshold_id']}:lambda{lam:.2f}"})
    return result
