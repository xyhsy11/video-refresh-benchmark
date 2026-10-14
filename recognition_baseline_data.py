"""Frozen source-disjoint expansion using verified caches and bounded HTTP bytes."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import urllib.request
import uuid
from unittest.mock import patch

import av

from fresh_confirmation_data import ReferenceCollector, strict_json
from fresh_confirmation_eval import load_references
from prepare_cache_drift_v2 import CACHE, BASE, download_shard, inspect_eligible
from research_utils import ROOT, now, relative_path, sha256_file

SEED = 20261016


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ByteBudget:
    """All GET body bytes, including failed/retried requests, charge the cap.

    Persist worst-case request reservations before network activity. After a
    process interruption, stale unread reservations are charged conservatively;
    they are distinguished from bytes actually returned by HTTP reads.
    HEAD bodies are zero; TCP headers/retransmissions are not body-byte evidence.
    """
    def __init__(self, path: Path, cap: int, opener=None):
        self.path = path; self.opener = opener or urllib.request.urlopen
        self.lock = threading.RLock()
        self.state = read(path) if path.exists() else {"cap_bytes": cap, "actual_body_bytes": 0,
            "uncertain_interruption_bytes_charged": 0, "inflight": {}, "requests": [], "started_at_utc": now()}
        assert self.state["cap_bytes"] == cap, "Resume download cap changed"
        if self.state["inflight"]:
            self.state["uncertain_interruption_bytes_charged"] += sum(self.state["inflight"].values())
            self.state["inflight"] = {}
        self.persist()

    def persist(self):
        strict_json(self.path, self.state)

    def used(self) -> int:
        return self.state["actual_body_bytes"] + self.state["uncertain_interruption_bytes_charged"]

    def open(self, request, *args, **kwargs):
        req = urllib.request.Request(request) if isinstance(request, str) else request
        if req.get_method() == "HEAD":
            return self.opener(req, *args, **kwargs)
        range_header = req.get_header("Range")
        if range_header:
            start, end = map(int, range_header.split("=", 1)[1].split("-")); expected = end-start+1
        else:
            with self.opener(urllib.request.Request(req.full_url, method="HEAD"), timeout=45) as head:
                expected = int(head.headers["Content-Length"])
        token = uuid.uuid4().hex
        with self.lock:
            if self.used() + sum(self.state["inflight"].values()) + expected > self.state["cap_bytes"]:
                raise RuntimeError("Download body-byte cap would be exceeded; preserve progress and resume with the same cap")
            self.state["inflight"][token] = expected
            event = {"url": req.full_url, "range": range_header, "expected_bytes": expected,
                     "actual_bytes": 0, "started_at_utc": now(), "status": "opened"}
            self.state["requests"].append(event); self.persist()
        try:
            response = self.opener(req, *args, **kwargs)
        except Exception:
            with self.lock:
                # No response reader means delivered bytes cannot be established.
                self.state["uncertain_interruption_bytes_charged"] += self.state["inflight"].pop(token)
                event["status"] = "open_failed_conservatively_charged"; self.persist()
            raise
        budget = self

        class Response:
            def __getattr__(self, name): return getattr(response, name)
            def __enter__(self): response.__enter__(); return self
            def __exit__(self, *exc): self.close(uncertain=exc[0] is not None); return False
            def charge(self, size):
                with budget.lock:
                    assert size <= budget.state["inflight"][token], "Response exceeded requested body bytes"
                    budget.state["inflight"][token] -= size
                    budget.state["actual_body_bytes"] += size; event["actual_bytes"] += size
                    budget.persist()
            def read(self, size=-1):
                with budget.lock: remaining = budget.state["inflight"][token]
                size = remaining if size is None or size < 0 else min(size, remaining)
                try: data = response.read(size)
                except http.client.IncompleteRead as exc:
                    self.charge(len(exc.partial)); raise
                self.charge(len(data)); return data
            def close(self, uncertain=False):
                response.close()
                with budget.lock:
                    if token in budget.state["inflight"]:
                        unused = budget.state["inflight"].pop(token)
                        if uncertain: budget.state["uncertain_interruption_bytes_charged"] += unused
                        event.update(status="closed_failed_conservatively_charged" if uncertain else "closed", completed_at_utc=now()); budget.persist()
        return Response()


def prior_memberships() -> tuple[dict, list]:
    """Canonical membership identities only; never parse fresh feature/performance arrays."""
    paths = ["results/cache_drift_v2/split_manifest.json", "results/budget_aware_refresh_v2/development_split.json",
             "results/task_aware_oracle_v1/dataset_split.json", "results/fresh_refresh_confirmation_v1/frozen_test_manifest.json",
             "results/pilot_2026_10_10/manifest.json"]
    ids, hashes, evidence = set(), set(), []
    for name in paths:
        data = read(ROOT/name); videos = data["videos"]
        for value in videos.values() if isinstance(videos, dict) else videos:
            entry = value.get("entry", value)
            ids.add(entry.get("source_video_id", entry["video_id"]))
            identity = entry.get("file_identity", {})
            if "sha256" in identity: hashes.add(identity["sha256"])
        evidence.append({"path": name, "sha256": sha256_file(ROOT/name)})
    return {"source_ids": sorted(ids), "file_hashes": sorted(hashes), "evidence": evidence,
            "rule": "All previous frozen memberships excluded from newly acquired sources. Only original80 exclusively-training references and the fixed40dev/40exploratory cohorts are reused in their designated roles."}, paths


def compatible_reuse() -> list:
    original = read(ROOT/"results/cache_drift_v2/split_manifest.json")
    split = read(ROOT/"results/budget_aware_refresh_v2/development_split.json")
    train = {e["video_id"] for e in split["videos"] if e["development_role"] == "train"}
    dev = {e["video_id"] for e in split["videos"] if e["development_role"] == "development"}
    ledger = dict((n, d) for d, n in (line.split("  ",1) for line in (ROOT/"results/cache_drift_v2/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()))
    selected = []
    for e in original["videos"]:
        role = "train" if e["video_id"] in train else "development" if e["video_id"] in dev else "exploratory" if e["experiment_split"] == "test" else None
        if role is None: continue
        names = [f"features/{e['video_id']}.npz",f"features/{e['video_id']}.json",f"motion_frames/{e['video_id']}.csv"]
        selected.append({**e, "recognition_role": role, "reference_directory": "results/cache_drift_v2",
                         "reference_artifact_sha256": {n: ledger[n] for n in names}, "reused_reference": True})
    assert Counter(e["recognition_role"] for e in selected) == {"train":80,"development":40,"exploratory":40}
    return selected


def validate_membership(entries: list, complete: bool = True) -> dict:
    assert len({e["source_video_id"] for e in entries}) == len(entries)
    assert len({e["file_identity"]["sha256"] for e in entries}) == len(entries)
    for e in entries:
        assert len(e["candidate_indices"]) == 16 and e["candidate_indices"] == sorted(set(e["candidate_indices"]))
        assert e["candidate_indices"][0] == 0 and e["candidate_indices"][-1] == e["decoded_frame_count"]-1
    counts = Counter(e["recognition_role"] for e in entries)
    if complete:
        assert counts == {"train":400,"development":100,"exploratory":40}
        for role, target in [("train",20),("development",5),("exploratory",2)]:
            classes = Counter(e["label"] for e in entries if e["recognition_role"] == role)
            assert len(classes) == 20 and set(classes.values()) == {target}
    return dict(counts)


def load_role(entries: list, role: str) -> list:
    groups = {}
    for entry in entries:
        if entry["recognition_role"] == role: groups.setdefault(entry["reference_directory"],[]).append(entry)
    videos = []
    original_model = read(ROOT/"results/cache_drift_v2/collection_metadata.json")["model"]
    for directory, group in groups.items():
        metadata = read(ROOT/directory/"collection_metadata.json")
        assert metadata["model"] == original_model, "Encoder/preprocessing provenance mismatch"
        hashes = {n:d for e in group for n,d in e["reference_artifact_sha256"].items()}
        videos.extend(load_references(group,ROOT/directory,hashes))
    return sorted(videos,key=lambda v:v.entry["source_video_id"])


def subsets(entries: list, classes: list) -> dict:
    result = {str(n):[] for n in [4,10,20]}
    for label in classes:
        rows = sorted([e for e in entries if e["recognition_role"] == "train" and e["label"] == label],
                      key=lambda e:(hashlib.sha256(f"recognition_baseline_v2|subset|{SEED}|{e['source_video_id']}".encode()).hexdigest(),e["source_video_id"]))
        assert len(rows) == 20
        for n in [4,10,20]: result[str(n)].extend(e["source_video_id"] for e in rows[:n])
    assert set(result["4"]) <= set(result["10"]) <= set(result["20"])
    return result


def initialize(output: Path, cap: float, sources: list[str]) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    protocol_path = output/"protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        assert protocol["max_download_bytes"] == int(cap*1024**3)
        assert protocol["implementation_sha256"] == {n:sha256_file(ROOT/n) for n in sources}, "Resume source identity changed"
        return protocol
    exclusions,_ = prior_memberships(); strict_json(output/"exclusions.json",exclusions)
    original = read(ROOT/"results/cache_drift_v2/split_manifest.json")
    reused = compatible_reuse(); strict_json(output/"reused_membership.json",{"videos":reused})
    budget = ByteBudget(output/"download_accounting.json",int(cap*1024**3))
    # Small official-source probe is counted before choosing a route; no training shards acquired.
    probe_path = ROOT/"data/source/recognition_baseline_v2/k400_train_path.txt"
    if not probe_path.exists():
        probe_path.parent.mkdir(parents=True,exist_ok=True)
        with budget.open(BASE+"train/k400_train_path.txt",timeout=45) as response: body=response.read()
        probe_path.write_bytes(body)
    train_urls = probe_path.read_text(encoding="utf-8").splitlines()
    assert train_urls and all("/train/" in u for u in train_urls)
    train_probe = {"archive_list_sha256":sha256_file(probe_path),"archive_count":len(train_urls),"cached_official_training_archives":0,
                   "selection":"custom CVDF validation-source protocol fixed before training; eleven verified reusable val archives and160 compatible references; no official-training archive cache; mixed-class training shards plus separate development acquisition have uncertain coverage under20GiB. This resource-conscious diagnostic is not an official recognition benchmark."}
    cached = []
    urls = (CACHE/"k400_val_path.txt").read_text(encoding="utf-8").splitlines()
    previous = read(ROOT/"results/cache_drift_v2/sampling_plan.json")
    assert sha256_file(CACHE/"val.csv") == previous["annotations_sha256"]
    assert sha256_file(CACHE/"k400_val_path.txt") == previous["shard_list_sha256"]
    for url in urls:
        path = CACHE/"shards"/PurePosixPath(url).name; info_path = path.with_suffix(".download.json")
        if path.exists() and info_path.exists():
            info = read(info_path); print(f"Preflight SHA256 verification: {path.name}",flush=True)
            assert path.stat().st_size == info["size_bytes"] and sha256_file(path) == info["sha256"]
            assert info["url"] == url
            cached.append({**info,"path":relative_path(path)})
    rank = lambda url: hashlib.sha256(f"recognition_baseline_v2|archive|{SEED}|{url}".encode()).hexdigest()
    cached.sort(key=lambda e:rank(e["url"]))
    uncached = sorted([u for u in urls if u not in {a["url"] for a in cached}],key=rank)
    rows = list(csv.DictReader((CACHE/"val.csv").open(encoding="utf-8",newline="")))
    candidates = []
    for row in rows:
        if row["label"] not in original["classes"] or row["youtube_id"] in exclusions["source_ids"]: continue
        name = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
        candidates.append({**row,"filename":name,"rank":hashlib.sha256(f"recognition_baseline_v2|candidate|{SEED}|{row['label']}|{name}".encode()).hexdigest()})
    assert len({r["filename"] for r in candidates}) == len(candidates)
    plan = {"frozen_at_utc":now(),"ordered_archives":[e["url"] for e in cached]+uncached,"cached_archives":cached,
            "candidate_rows":sorted(candidates,key=lambda e:(e["rank"],e["filename"])),"official_training_probe":train_probe,
            "annotation_sha256":previous["annotations_sha256"],"archive_list_sha256":previous["shard_list_sha256"]}
    strict_json(output/"sampling_plan.json",plan)
    protocol = {"experiment":"recognition_baseline_v2","frozen_at_utc":now(),"seed":SEED,"starting_commit":__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
                "implementation_sha256":{n:sha256_file(ROOT/n) for n in sources},"classes":original["classes"],
                "class_to_index":{c:i for i,c in enumerate(original["classes"])},"source_protocol":train_probe,
                "targets":{"train":400,"development":100,"exploratory":40},"per_class":{"train":20,"development":5,"exploratory":2},
                "reused_membership_sha256":sha256_file(output/"reused_membership.json"),"exclusions_sha256":sha256_file(output/"exclusions.json"),"sampling_plan_sha256":sha256_file(output/"sampling_plan.json"),
                "selection":"cached-first seeded archive order; within archive candidate SHA ranking; first3 eligible new clips/class augment development, next16 augment training. Existing80train/40dev/40exploratory fixed. Network/CUDA failures retry same clip; replacements only predefined input eligibility/duplicates.",
                "eligibility":"H264,5..15sec,>=16 actual decoded frames,16 uniform indices including endpoints,finite increasing timestamps,finite safely nonzero512features; motion missing is valid",
                "encoder":read(ROOT/"results/cache_drift_v2/collection_metadata.json")["model"],"numerical_settings":read(ROOT/"results/cache_drift_v2/collection_metadata.json")["numerical_settings"],
                "training_sizes_per_class":[4,10,20],"primary_training_videos_per_class":20,"regularization_grid":[.001,.01,.1,1.0],
                "classifier":"reuse task_aware_oracle_utils.fit_classifier exactly; training-only population normalization, mean pooling, float64 CPU, deterministic linear cross entropy+lambda/2 squared weights, unpenalized intercept; LBFGS max_iter400/max_eval800,strong_wolfe,grad1e-9/change1e-12,history50",
                "selection_rule":"highest dense development top1,lower CE,earliest grid order; freeze all selected models before exploratory arrays/metrics; no oracle-based selection",
                "policies":["dense","periodic","feature_oracle","task_oracle"],"budget":4,"dense_budget":16,"task_oracle":"exact455 schedules; true label and future references; lexicographically first within1e-12",
                "primary_comparison":["task_oracle","periodic"],"secondary_comparisons":[["feature_oracle","periodic"],["task_oracle","feature_oracle"],["periodic","dense"]],
                "bootstrap":{"resamples":10000,"seed":SEED,"rng":"PCG64","percentile_method":"linear","unit":"paired whole source videos; shared indices within cohort; fitted models fixed"},
                "max_download_bytes":int(cap*1024**3),"accounting":"all new GET response body bytes incl metadata/retries; stale interrupted reservations conservatively charged; verified preexisting cache bytes free. TCP/IP overhead not application-body measurements.",
                "limits":["custom diagnostic source split","exploratory cohort previously used","development tuned","label-guided future oracle optimism,not deployable accuracy or accuracy-maximizing bound","temporal weighting induced by cache repetition","no arbitrary accuracy publication gate","no streaming/latency/speedup/novelty claim"]}
    strict_json(protocol_path,protocol)
    candidates_count = Counter(r["label"] for r in candidates)
    missing = [{"label":c,"training_missing":16,"development_missing":3,"available_annotation_candidates":candidates_count[c]} for c in original["classes"]]
    estimated = []
    for url in uncached:
        with budget.open(urllib.request.Request(url,method="HEAD"),timeout=45) as response: size=int(response.headers["Content-Length"])
        estimated.append({"url":url,"size_bytes":size})
    strict_json(output/"preflight.json",{"checked_at_utc":now(),"compatible_saved_reference_videos":160,"new_reference_videos_needed":380,
                 "cached_archives":len(cached),"cached_compressed_bytes":sum(e["size_bytes"] for e in cached),"disk_free_bytes":shutil.disk_usage(ROOT).free,
                 "missing_by_class":missing,"remaining_uncached_archives":estimated,"all_remaining_archive_bytes":sum(e["size_bytes"] for e in estimated),
                 "estimated_new_download_range_bytes":[0,sum(e["size_bytes"] for e in estimated)],"coverage_unknown_until_cached_archive_traversal":True,"official_training_probe":train_probe})
    return protocol


def acquire(output: Path, protocol: dict, workers: int) -> list:
    plan = read(output/"sampling_plan.json"); exclusions = read(output/"exclusions.json")
    base = read(output/"reused_membership.json")["videos"]
    state_path = output/"acquisition.json"
    state = read(state_path) if state_path.exists() else {"started_at_utc":now(),"status":"incomplete","archives":[],"events":[]}
    strict_json(state_path,state)
    collector = ReferenceCollector(output,{"sampling_seed":SEED})
    budget = ByteBudget(output/"download_accounting.json",protocol["max_download_bytes"])
    selected = [*base]
    for item in collector.progress["videos"]:
        selected.append({**item["entry"],"reference_directory":relative_path(output),"reference_artifact_sha256":item["artifact_sha256"],"reused_reference":False})
    validate_membership(selected,False)
    counts = Counter((e["recognition_role"],e["label"]) for e in selected)
    ids={e["source_video_id"] for e in selected}; hashes={e["file_identity"]["sha256"] for e in selected}
    done={e["video_id"] for e in selected}
    rejected={e["video_id"] for e in state["events"] if e["status"] in ["input_ineligible","excluded_duplicate"]}
    lookup={r["filename"]:r for r in plan["candidate_rows"]}
    videos_dir=ROOT/"data/source/recognition_baseline_v2/videos"; videos_dir.mkdir(parents=True,exist_ok=True)
    try:
        for url in plan["ordered_archives"]:
            if all(counts[("train",c)]==20 and counts[("development",c)]==5 for c in protocol["classes"]): break
            old=next((a for a in state["archives"] if a["url"]==url),None)
            if old and old["status"]=="processed": continue
            archive=next((dict(a) for a in plan["cached_archives"] if a["url"]==url),{"url":url,"status":"pending"})
            destination=CACHE/"shards"/PurePosixPath(url).name
            if not archive.get("sha256"):
                print(f"Acquiring {destination.name}; charged {budget.used()/1024**3:.3f} GiB/{protocol['max_download_bytes']/1024**3:g}",flush=True)
                with patch('urllib.request.urlopen',budget.open): archive.update(download_shard(url,destination,workers))
                archive["path"]=relative_path(destination)
            else: assert sha256_file(destination)==archive["sha256"]
            archive["status"]="indexing"
            if old: state["archives"].remove(old)
            state["archives"].append(archive);strict_json(state_path,state)
            matches=[]; print(f"Streaming {destination.name} in frozen order",flush=True)
            with tarfile.open(destination,mode="r|gz") as tar:
                for member in tar:
                    name=PurePosixPath(member.name).name
                    if not member.isfile() or name not in lookup or name[:-4] in done|rejected: continue
                    row=lookup[name]
                    if counts[("train",row['label'])]>=20 and counts[("development",row['label'])]>=5: continue
                    target=videos_dir/name
                    member_hash=hashlib.sha256()
                    temporary=target.with_suffix('.mp4.part') if not target.exists() else None
                    dst=temporary.open('wb') if temporary else None
                    try:
                        with tar.extractfile(member) as src:
                            while block:=src.read(1024*1024):
                                member_hash.update(block)
                                if dst: dst.write(block)
                    finally:
                        if dst: dst.close()
                    if temporary: assert temporary.stat().st_size==member.size;temporary.replace(target)
                    assert sha256_file(target)==member_hash.hexdigest(),"Extracted encoding identity mismatch"
                    matches.append((row,target,member_hash.hexdigest()))
            for row,target,digest in sorted(matches,key=lambda e:(e[0]['rank'],e[0]['filename'])):
                label=row['label']
                role='development' if counts[('development',label)]<5 else 'train'
                limit=5 if role=='development' else 20
                if counts[(role,label)]>=limit: continue
                event={'video_id':target.stem,'source_video_id':row['youtube_id'],'label':label,'role':role,'archive_url':url,'rank':row['rank'],'encoded_sha256':digest,'at_utc':now()}
                if row['youtube_id'] in ids or digest in hashes or digest in exclusions['file_hashes']:
                    event.update(status='excluded_duplicate',reason='source/hash overlap');state['events'].append(event);rejected.add(target.stem);strict_json(state_path,state);continue
                try: entry=inspect_eligible(target,row,archive)
                except (ValueError,av.error.FFmpegError) as exc:
                    event.update(status='input_ineligible',reason=f'{type(exc).__name__}: input decode/metadata eligibility failed');state['events'].append(event);rejected.add(target.stem);strict_json(state_path,state);continue
                entry.update(experiment_split='train' if role=='train' else 'validation',recognition_role=role)
                collector.collect(entry)
                item=collector.progress['videos'][-1]
                selected.append({**item['entry'],'reference_directory':relative_path(output),'reference_artifact_sha256':item['artifact_sha256'],'reused_reference':False})
                ids.add(row['youtube_id']);hashes.add(digest);done.add(target.stem);counts[(role,label)]+=1
                event.update(status='collected');state['events'].append(event);strict_json(state_path,state)
                print(f"CUDA references retained: {len(selected)}/540; {label}: train {counts[('train',label)]}/20, dev {counts[('development',label)]}/5",flush=True)
            archive['status']='processed';strict_json(state_path,state)
        shortage={c:{'train':20-counts[('train',c)],'development':5-counts[('development',c)]} for c in protocol['classes']}
        state['shortfall_by_class']=shortage
        if any(v for r in shortage.values() for v in r.values()): raise RuntimeError('Incomplete dataset; exact shortages saved; no training/evaluation permitted')
        validate_membership(selected)
        state.update(status='complete',completed_at_utc=now(),counts={'train':400,'development':100,'exploratory':40})
        strict_json(output/'dataset_manifest.json',{'frozen_at_utc':now(),'source_protocol':'custom CVDF validation-source diagnostic','classes':protocol['classes'],'videos':selected})
        collector.metadata.update(status='collected',completed_at_utc=now());strict_json(output/'collection_metadata.json',collector.metadata)
        return selected
    finally:
        state['shortfall_by_class']={c:{'train':max(0,20-counts[('train',c)]),'development':max(0,5-counts[('development',c)])} for c in protocol['classes']}
        state['actual_new_reference_forward_calls']=collector.metadata['actual_forward_calls_entered']
        state['reused_reference_features']=2560;state['retained_new_reference_features']=16*len(collector.progress['videos'])
        state['actual_download_body_bytes']=budget.state['actual_body_bytes'];state['charged_download_bytes']=budget.used()
        strict_json(state_path,state)
