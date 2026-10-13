"""Offline policy decisions see only age/motion; RGB references belong to evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
import random

import numpy as np
import pandas as pd

from analyze_cache_drift import TARGET, fit_predict
from analyze_cache_drift_v2 import apply_fitted
from cache_drift_v2_utils import preservation_inventory, read_json, validate_split
from collect_cache_drift import AGE_INPUTS, MOTION_INPUTS, cosine_distance, interval_inputs
from research_utils import ROOT, relative_path, sha256_file

TOL = 1e-12
SCORED = ["simple_motion", "age_only", "motion_only", "age_plus_motion"]
LEARNED = {"age_only": AGE_INPUTS, "motion_only": MOTION_INPUTS,
           "age_plus_motion": AGE_INPUTS + MOTION_INPUTS}
POLICIES = ["periodic", "random", *SCORED, "oracle"]


def protocol(budgets: list[int], seed: int) -> dict:
    return {
        "experiment": "adaptive_refresh_v1", "scope": "offline saved-artifact replay; validation-tuned developmental results",
        "inputs": "120 training and 40 validation videos from frozen cache_drift_v2; no test data parsed or evaluated",
        "training": "all 0<=r<t<16 pairs; 120 pairs/video; 14,400 rows from 120 source videos",
        "age_inputs": AGE_INPUTS, "motion_inputs": MOTION_INPUTS, "ridge_alpha": 1.0,
        "standardization": "training-only population mean/std; std<=1e-12 uses scale 1",
        "target": "existing float64 1-clipped cosine(f_t,f_r), pre-decision; safe norm >1e-12",
        "motion_interval": "decoded frames strictly after most recent actual refresh through current frame; reset on refresh",
        "missing_motion": "zero magnitude plus explicit availability, never an exclusion",
        "scored_policies": SCORED, "budgets": budgets, "primary_budget": 4, "dense_budget": 16,
        "score": {"simple_motion": "current candidate p90 magnitude in codec-displacement pixels",
                  "regression": "raw recorded; clip predicted drift to [0,2] for all decisions"},
        "threshold_grid": "training-pair score quantiles at 0,5,...,100%, linear; remove duplicate values; add always and wait",
        "threshold_rule": "refresh iff clipped score >= numeric threshold, unless exact budget rule overrides",
        "extremes": "kind=always or wait; value=null (no Infinity/NaN)",
        "threshold_selection": "minimize equal-weight mean validation-video output error; ties within 1e-12 take earliest grid order: always, ascending unique thresholds, wait",
        "global_threshold": "one per scored policy and budget, not per video/class; models fit once, never changed by validation",
        "budget_rule": "mandatory refresh 0; reuse if remaining=0; refresh if remaining equals remaining candidate positions including current; otherwise policy rule",
        "periodic": "range(0,16,16//B)", "random_seeds": list(range(seed, seed+10)),
        "random_seed_derivation": "integer from first 16 SHA256 hex digits of adaptive_refresh_v1|seed|source_video_id; random.Random.sample(1..15,B-1)",
        "oracle": "exact enumerate all schedules including 0; minimize all-16-output mean cosine error; lexicographically earliest within 1e-12 of minimum",
        "oracle_scope": "offline diagnostic bound; unavailable to scored callbacks",
        "metric": "post-decision appearance-feature error, mean over all 16 outputs; pre-decision drift logged separately by evaluator",
        "aggregation": "random first averages all ten seeds within each video; then every video equal weight",
        "comparison": "error(periodic)-error(policy); percentages per video before aggregation; negative values preserved; percentage undefined if reference error<=1e-12",
        "oracle_gap": "(E_periodic-E_policy)/(E_periodic-E_oracle); undefined if denominator<=1e-12, no clipping",
        "example_selection": "B=4 age-plus-motion largest absolute gain and largest loss versus periodic; video-ID tie break; descriptive only",
        "numerical_tolerance": TOL,
        "limits": ["validation thresholds selected on evaluated validation videos", "not action-recognition accuracy",
                   "no decoding/controller/motion/GPU latency measurements", "future-reference codec vectors and full-clip candidate selection: offline"],
    }


def protected_files() -> dict:
    result = preservation_inventory()
    for name in ["prepare_cache_drift_v2.py", "collect_cache_drift_v2.py", "analyze_cache_drift_v2.py",
                 "cache_drift_v2_utils.py", "verify_cache_drift_v2.py"]:
        result[name] = sha256_file(ROOT/name)
    # Hashes are integrity checks only; no test arrays/observations are parsed.
    for p in (ROOT/"results/cache_drift_v2").rglob("*"):
        if p.is_file():
            result[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    return dict(sorted(result.items()))


def check_protected(expected: dict) -> None:
    if protected_files() != expected:
        raise AssertionError("Existing scripts or published snapshots changed")


@dataclass(frozen=True)
class CheapVideo:
    source_video_id: str
    indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    motion: tuple[dict, ...]


@dataclass(frozen=True)
class ReplayVideo:
    entry: dict
    cheap: CheapVideo
    references: np.ndarray


def load_development(directory: Path) -> tuple[list[ReplayVideo], list[ReplayVideo], dict]:
    manifest = read_json(directory/"split_manifest.json")
    metadata = read_json(directory/"collection_metadata.json")
    assert manifest["status"] == "frozen" and metadata["status"] == "collected"
    assert sha256_file(directory/"split_manifest.json") == metadata["split_manifest_sha256"]
    validate_split(manifest, check_files=False)
    groups = {s: {e["source_video_id"] for e in manifest["videos"] if e["experiment_split"] == s}
              for s in ["train", "validation", "test"]}
    assert all(groups[a].isdisjoint(groups[b]) for a,b in [("train","validation"),("train","test"),("validation","test")])
    progress = {v["entry"]["video_id"]: v for v in read_json(directory/"collection_progress.json")["videos"]}
    loaded, artifact_hashes = [], {}
    for entry in sorted(manifest["videos"], key=lambda e: e["video_id"]):
        if entry["experiment_split"] not in ["train", "validation"]:
            continue
        video_id = entry["video_id"]
        assert progress[video_id]["entry"] == entry
        names = [f"features/{video_id}.npz", f"features/{video_id}.json", f"motion_frames/{video_id}.csv"]
        for name in names:
            path = directory/name
            if not path.exists():
                raise FileNotFoundError(f"Required development artifact missing: {relative_path(path)}")
            digest = sha256_file(path)
            if digest != progress[video_id]["artifact_sha256"][name]:
                raise ValueError(f"Development artifact hash mismatch: {relative_path(path)}")
            artifact_hashes[name] = digest
        side = read_json(directory/names[1])
        assert side["entry"] == entry and len(side["candidates"]) == 16
        with np.load(directory/names[0], allow_pickle=False) as saved:
            features = saved["features"].copy()
            indices = saved["frame_indices"].tolist()
            stamps = saved["timestamps_seconds"].tolist()
        assert features.shape == (16,512) and features.dtype == np.float32 and np.isfinite(features).all()
        assert np.min(np.linalg.norm(features.astype(float), axis=1)) > TOL
        assert indices == entry["candidate_indices"] == [c["decoded_frame_index"] for c in side["candidates"]]
        assert indices == sorted(set(indices)) and indices[0] == 0 and indices[-1] == entry["decoded_frame_count"]-1
        assert stamps == entry["candidate_timestamps_seconds"] == [c["timestamp_seconds"] for c in side["candidates"]]
        assert np.isfinite(stamps).all() and (np.diff(stamps)>0).all()
        motion = pd.read_csv(directory/names[2], float_precision="round_trip")
        assert len(motion) == entry["decoded_frame_count"] and motion.decoded_frame_index.tolist() == list(range(len(motion)))
        assert set(motion.video_id) == {video_id} and motion.frame_type.value_counts().to_dict() == {k:v for k,v in entry["frame_type_counts"].items() if v}
        assert (motion.mv_available == (motion.vector_count > 0)).all()
        counts = motion[["vector_count", "past_reference_vectors", "future_reference_vectors", "other_reference_vectors"]]
        assert np.isfinite(counts).all().all() and (counts>=0).all().all() and (counts % 1 == 0).all().all()
        assert (counts.iloc[:,1:].sum(axis=1)==motion.vector_count).all()
        assert np.isfinite(motion[["mean_magnitude_px","p90_magnitude_px"]]).all().all()
        assert (motion[["mean_magnitude_px","p90_magnitude_px"]]>=0).all().all()
        assert (motion.loc[~motion.mv_available,["mean_magnitude_px","p90_magnitude_px"]]==0).all().all()
        features.setflags(write=False)
        loaded.append(ReplayVideo(entry, CheapVideo(entry["source_video_id"],tuple(indices),tuple(stamps),tuple(motion.to_dict("records"))),features))
    train = [v for v in loaded if v.entry["experiment_split"] == "train"]
    validation = [v for v in loaded if v.entry["experiment_split"] == "validation"]
    assert len(train) == 120 and len(validation) == 40
    assert {v.cheap.source_video_id for v in loaded}.isdisjoint(groups["test"])
    return train, validation, {"split_manifest_sha256": sha256_file(directory/"split_manifest.json"),
        "collection_metadata_sha256": sha256_file(directory/"collection_metadata.json"),
        "parsed_development_artifact_sha256": artifact_hashes, "parsed_test_artifact_count": 0,
        "training_sources": sorted(groups["train"]), "validation_sources": sorted(groups["validation"]),
        "train_validation_source_overlap": False, "test_sources_excluded": True,
        "note": "Other published files, including test files, are hashed only for preservation; no test data parsed for replay."}


def pair_inputs(cheap: CheapVideo, r: int, t: int) -> dict:
    assert 0 <= r < t < len(cheap.indices)
    elapsed = cheap.timestamps[t]-cheap.timestamps[r]
    return {"age_seconds": elapsed, "age_decoded_frames": cheap.indices[t]-cheap.indices[r],
            "age_candidate_positions": t-r, "age_seconds_squared": elapsed*elapsed,
            **interval_inputs(cheap.motion,cheap.indices[r],cheap.indices[t])}


def make_training_pairs(videos: list[ReplayVideo]) -> pd.DataFrame:
    rows=[]
    for video in videos:
        assert video.entry["experiment_split"] == "train"
        for t in range(1,16):
            for r in range(t):
                rows.append({"pair_id": f"{video.entry['video_id']}:{r:02d}:{t:02d}",
                    "video_id": video.entry["video_id"], "source_video_id": video.cheap.source_video_id,
                    "cached_candidate_ordinal": r, "candidate_ordinal": t,
                    "cached_frame_index": video.cheap.indices[r], "current_frame_index": video.cheap.indices[t],
                    **pair_inputs(video.cheap,r,t), TARGET: cosine_distance(video.references[t],video.references[r])})
    result=pd.DataFrame(rows)
    assert len(result)==120*len(videos) and not result.pair_id.duplicated().any()
    assert (result.groupby("video_id").size()==120).all()
    assert np.isfinite(result[AGE_INPUTS+MOTION_INPUTS+[TARGET]]).all().all()
    return result


def fit_models(train: pd.DataFrame) -> dict:
    result={}
    for name,columns in LEARNED.items():
        _,state=fit_predict(train,train.iloc[:1],columns)
        assert state["alpha"]==1.0 and state["training_rows"]==len(train)
        np.testing.assert_allclose(state["means"],train[columns].mean().to_numpy(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(state["raw_standard_deviations"],train[columns].std(ddof=0).to_numpy(),rtol=1e-12,atol=1e-12)
        result[name]=state
    return result


def threshold_grids(train: pd.DataFrame, models: dict) -> dict:
    grids={}
    for policy in SCORED:
        raw = train.current_p90_magnitude_px.to_numpy() if policy == "simple_motion" else apply_fitted(train,models[policy])
        scores = raw if policy == "simple_motion" else np.clip(raw,0,2)
        assert np.isfinite(scores).all()
        quantiles=np.percentile(scores,np.arange(0,101,5),method="linear")
        values=[]
        for q,value in zip(range(0,101,5),quantiles):
            if not values or float(value)!=values[-1]["value"]:
                values.append({"threshold_id": f"q{q:03d}", "kind": "threshold", "value": float(value), "first_quantile_percent": q})
        grids[policy]=[{"threshold_id":"always","kind":"always","value":None},*values,
                       {"threshold_id":"wait","kind":"wait","value":None}]
    return grids


class IntervalAccumulator:
    def __init__(self, motion: tuple[dict,...], start: int):
        self.motion=motion
        self.reset(start)

    def reset(self, start: int) -> None:
        self.start=self.cursor=start
        self.count=self.available=0
        self.magnitude_sum=self.maximum=0.0

    def advance(self, end: int) -> dict:
        assert self.cursor < end
        for index in range(self.cursor+1,end+1):
            frame=self.motion[index]
            assert frame["decoded_frame_index"]==index
            self.count+=1
            if frame["mv_available"]:
                self.available+=1
                self.magnitude_sum+=frame["mean_magnitude_px"]
                self.maximum=max(self.maximum,frame["p90_magnitude_px"])
        self.cursor=end
        current=self.motion[end]
        return {"current_mean_magnitude_px":current["mean_magnitude_px"], "current_p90_magnitude_px":current["p90_magnitude_px"],
                "current_mv_available":int(current["mv_available"]),
                "interval_mean_available_magnitude_px":self.magnitude_sum/self.available if self.available else 0.0,
                "interval_max_available_p90_px":self.maximum,
                "interval_mv_available_fraction":self.available/self.count,
                "interval_start_frame":self.start+1,"interval_end_frame":end,
                "interval_frame_count":self.count,"interval_available_frame_count":self.available}


def policy_score(policy: str, inputs: dict, state: dict | None) -> tuple[float,float]:
    assert set(inputs)==set(AGE_INPUTS+MOTION_INPUTS)
    if policy=="simple_motion":
        raw=float(inputs["current_p90_magnitude_px"])
        return raw,raw  # Pixel units, not drift; no [0,2] clipping for this policy.
    assert policy in LEARNED and state["columns"]==LEARNED[policy]
    x=np.array([inputs[name] for name in state["columns"]],dtype=np.float64)
    raw=float(state["intercept"]+((x-np.array(state["means"]))/np.array(state["scales"]))@np.array(state["coefficients"]))
    assert np.isfinite(raw)
    return raw,float(np.clip(raw,0,2))


def random_schedule(source_id: str, n: int, budget: int, seed: int) -> list[int]:
    digest=hashlib.sha256(f"adaptive_refresh_v1|{seed}|{source_id}".encode()).hexdigest()
    rng=random.Random(int(digest[:16],16))
    return [0,*sorted(rng.sample(range(1,n),budget-1))]


def plan_policy(cheap: CheapVideo, policy: str, budget: int, setting: dict | None = None,
                state: dict | None = None, seed: int | None = None) -> list[dict]:
    """No RGB references/true labels/oracle schedule may enter this interface."""
    n=len(cheap.indices)
    assert 1<=budget<=n and policy in ["dense","periodic","random",*SCORED]
    if policy=="dense":
        assert budget==n
        scheduled=set(range(n))
    elif policy=="periodic":
        assert n % budget==0
        scheduled=set(range(0,n,n//budget))
    elif policy=="random":
        scheduled=set(random_schedule(cheap.source_video_id,n,budget,seed))
    else:
        scheduled=set()
        assert setting["kind"] in ["always","wait","threshold"]
    return _plan(cheap,policy,budget,setting,state,scheduled)


def _plan(cheap: CheapVideo, policy: str, budget: int, setting: dict | None,
          state: dict | None, scheduled: set[int]) -> list[dict]:
    n=len(cheap.indices)
    r=0
    remaining=budget-1
    acc=IntervalAccumulator(cheap.motion,cheap.indices[0])
    rows=[{"candidate_ordinal":0,"cached_before_ordinal":None,"cached_output_ordinal":0,
           "refresh":True,"decision_reason":"mandatory_first","forced_refresh":False,
           "remaining_budget_before":budget,"remaining_budget_after":remaining,
           "score_raw":None,"score_clipped":None}]
    for t in range(1,n):
        elapsed=cheap.timestamps[t]-cheap.timestamps[r]
        inputs={"age_seconds":elapsed,"age_decoded_frames":cheap.indices[t]-cheap.indices[r],
                "age_candidate_positions":t-r,"age_seconds_squared":elapsed*elapsed,
                **acc.advance(cheap.indices[t])}
        raw=score=None
        if policy in SCORED:
            raw,score=policy_score(policy,{k:inputs[k] for k in AGE_INPUTS+MOTION_INPUTS},state)
        before=remaining
        if remaining==0:
            refresh,reason=False,"budget_exhausted"
        elif policy=="dense":
            refresh,reason=True,"dense"
        elif remaining==n-t:
            refresh,reason=True,"forced_budget"
        elif policy in SCORED:
            refresh=setting["kind"]=="always" or (setting["kind"]=="threshold" and score>=setting["value"])
            reason="threshold_refresh" if refresh else "threshold_reuse"
        else:
            refresh=t in scheduled
            reason="scheduled_refresh" if refresh else "scheduled_reuse"
        previous=r
        if refresh:
            r=t
            remaining-=1
            acc.reset(cheap.indices[t])
        rows.append({"candidate_ordinal":t,"cached_before_ordinal":previous,"cached_output_ordinal":r,
                     "refresh":refresh,"decision_reason":reason,"forced_refresh":reason=="forced_budget",
                     "remaining_budget_before":before,"remaining_budget_after":remaining,
                     "score_raw":raw,"score_clipped":score,**inputs})
    assert len(rows)==n and remaining==0 and sum(row["refresh"] for row in rows)==budget
    return rows


def distance_matrix(features: np.ndarray) -> np.ndarray:
    return np.array([[cosine_distance(a,b) for b in features] for a in features],dtype=np.float64)


def exact_oracle(distances: np.ndarray, budget: int) -> dict:
    n=len(distances)
    schedules=np.array([(0,*rest) for rest in combinations(range(1,n),budget-1)],dtype=int)
    marks=np.zeros((len(schedules),n),dtype=int)
    marks[np.arange(len(schedules))[:,None],schedules]=schedules
    caches=np.maximum.accumulate(marks,axis=1)
    errors=distances[np.arange(n)[None,:],caches]
    means=errors.mean(axis=1)
    minimum=float(means.min())
    chosen=int(np.flatnonzero(means<=minimum+TOL)[0])
    return {"positions":schedules[chosen].tolist(),"mean_error":float(means[chosen]),
            "enumerated_schedules":len(schedules),"exact_minimum_error":minimum}


def oracle_plan(cheap: CheapVideo, positions: list[int]) -> list[dict]:
    assert positions==sorted(set(positions)) and positions[0]==0
    return _plan(cheap,"oracle",len(positions),None,None,set(positions))


def evaluate_plan(distances: np.ndarray, plan: list[dict]) -> tuple[list[dict],dict]:
    rows=[]
    for row in plan:
        t=row["candidate_ordinal"]
        before=row["cached_before_ordinal"]
        rows.append({**row,"pre_decision_true_drift":float(distances[t,before]) if before is not None else None,
                     "output_feature_error":float(distances[t,row["cached_output_ordinal"]])})
    errors=[row["output_feature_error"] for row in rows]
    return rows,{"mean_output_error":float(np.mean(errors)),"max_candidate_error":float(max(errors)),
                 "simulated_refreshes":sum(row["refresh"] for row in rows),"reuses":sum(not row["refresh"] for row in rows),
                 "feature_outputs":len(rows),"refresh_positions":[row["candidate_ordinal"] for row in rows if row["refresh"]],
                 "forced_refreshes":sum(row["forced_refresh"] for row in rows)}


def verify_plan(cheap: CheapVideo, plan: list[dict], rows: list[dict], budget: int) -> None:
    assert len(plan)==len(rows)==len(cheap.indices) and plan[0]["refresh"]
    assert sum(row["refresh"] for row in plan)==budget
    latest=0
    for row,evaluated in zip(plan,rows):
        t=row["candidate_ordinal"]
        if t:
            assert row["cached_before_ordinal"]==latest
            expected=pair_inputs(cheap,latest,t)
            for key,value in expected.items():
                assert np.isclose(row[key],value,rtol=1e-12,atol=1e-12),key
            if row["remaining_budget_before"]==0:
                assert not row["refresh"]
            elif row["remaining_budget_before"]==len(plan)-t:
                assert row["refresh"]
        if row["refresh"]:
            latest=t
            assert abs(evaluated["output_feature_error"])<=TOL
        assert row["cached_output_ordinal"]==latest
    assert plan[-1]["remaining_budget_after"]==0
