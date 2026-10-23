"""Fail-closed Experiment 27 runner: metadata preflight, real input screening,
fitting-only planning, sealed single-intervention evaluation, and read-only audit.

Archive acquisition is explicit, metered and input-only. Resume advances the
persistent stages only after their authenticated prerequisites pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

from native_head_value_confirmation import (
    STUDY, SEED, REVISION, CHECKPOINT_SHA, VIEWS, PERIODIC, ALIASES,
    class_mapping, consequences, derived_seed, folds, precision_plan,
    rank, read, save, sha, source_inference, trajectory, validate_components,feature_schema)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/STUDY
CACHE = ROOT/'data/source'/STUDY
ASSETS = ROOT/'data/source/temporal_refresh_generalization_v1'
REVIEWED = 'a9e1942f226f341071e1a990ea446dc717a070bb'
CAPS = dict(network_bytes=61*2**30, retained_bytes=20*2**30,
            collection_seconds=8*3600, cpu_seconds=4*3600, timing_seconds=2*3600)
CODE = ['native_head_value_confirmation.py', 'run_native_head_value_confirmation.py',
        'verify_native_head_value_confirmation.py', 'test_native_head_value_confirmation.py',
        'native_head_confirmation_runtime.py', 'executable_value_observability.py',
        'causal_observability_policy.py', 'research_utils.py', 'temporal_refresh_common.py',
        'native_head_confirmation_inputs.py', 'native_head_confirmation_state.py', 'native_head_confirmation_availability.py']
SOURCES = dict(
    cvdf_readme='https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/README.md',
    cvdf_revision='https://api.github.com/repos/cvdfoundation/kinetics-dataset/commits/main',
    native_classes='https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json',
    model_zoo='https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html',
    official_readme='https://s3.amazonaws.com/kinetics/400/readme.md',
    **{f'{split}_annotations': f'https://s3.amazonaws.com/kinetics/400/annotations/{split}.csv' for split in ('train', 'val', 'test')},
    **{f'{split}_archives': f'https://s3.amazonaws.com/kinetics/400/{split}/k400_{split}_path.txt' for split in ('val', 'test')})


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def event(kind, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'resource_exposure_ledger.jsonl').open('a', encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(at_utc=now(), kind=kind, **fields), allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def network_ledger():
    p = OUT/'network_ledger.json'
    if p.exists():
        state = read(p)
        state['cap_bytes'] = CAPS['network_bytes']
        return state
    # These three public metadata bodies were inspected before this runner existed.
    records = []
    for name, urlkey in [('cvdf_readme', 'cvdf_readme'), ('val_list', 'val_archives'), ('test_list', 'test_archives')]:
        path = CACHE/'inspection'/f'{name}.txt'
        if path.exists():
            records.append(dict(url=SOURCES[urlkey], bytes=path.stat().st_size, sha256=sha(path),
                                purpose='initial metadata inspection', path=rel(path)))
    return dict(cap_bytes=CAPS['network_bytes'], requests=records, inflight=None,
                scope='HTTP response bodies including failed attempts; Git control traffic is separate, unmetered')


def fetch(name):
    path = CACHE/'metadata'/name
    identity = path.with_name(path.name+'.identity.json')
    if path.exists() and identity.exists():
        old = read(identity)
        if old['url'] != SOURCES[name] or old['sha256'] != sha(path):
            raise RuntimeError('Metadata identity changed: '+name)
        return path
    state = network_ledger()
    if state['inflight']:
        raise RuntimeError('Interrupted metadata request requires explicit conservative byte reconciliation')
    used = sum(r['bytes'] for r in state['requests'])
    state['inflight'] = dict(url=SOURCES[name], at_utc=now())
    save(OUT/'network_ledger.json', state)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        request = urllib.request.Request(SOURCES[name], headers={'User-Agent': 'video-refresh-research/27'})
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = dict(response.headers)
            maximum = min(64*2**20, CAPS['network_bytes']-used)
            size = int(response.headers.get('Content-Length', 0))
            if size > maximum:
                raise RuntimeError('Metadata response exceeds transfer reservation')
            tmp = path.with_name(path.name+'.writing')
            with tmp.open('wb') as f:
                while True:
                    block = response.read(min(2**20, maximum-count+1))
                    if not block:
                        break
                    count += len(block)
                    state['inflight']['bytes_received'] = count
                    save(OUT/'network_ledger.json', state)
                    if count > maximum:
                        raise RuntimeError('Metadata transfer cap reached')
                    f.write(block)
            tmp.replace(path)
        record = dict(url=SOURCES[name], bytes=count, sha256=sha(path), path=rel(path),
                      headers=headers, at_utc=now(), status='complete')
        save(identity, record)
        state['requests'].append(record)
        state['inflight'] = None
        save(OUT/'network_ledger.json', state)
        return path
    except Exception as exc:
        state['requests'].append(dict(url=SOURCES[name], bytes=count, status='failed', error=str(exc)))
        state['inflight'] = None
        save(OUT/'network_ledger.json', state)
        raise


def historical_fingerprint():
    names = git('ls-files', '-z', 'results').split('\0')
    return {n: sha(ROOT/n) for n in names if n and not n.startswith(f'results/{STUDY}/') and (ROOT/n).is_file()}


def metadata_paths():
    """Inventory all studies, including failed/partial snapshots; no outcome arrays."""
    accept = re.compile(r'(manifest|membership|population|screening|source|exclusion|exposure|split|allocation|duplicate|pretraining|collection|eligib|data_plan)', re.I)
    deny = re.compile(r'(outcome|per_state|prediction|performance|metrics|bootstrap|timing|fit|originals|execution_history)', re.I)
    result = []
    for p in (ROOT/'results').rglob('*'):
        if not p.is_file() or STUDY in p.parts or not accept.search(p.name) or deny.search(p.name):
            continue
        if p.suffix == '.json' or p.name.endswith(('.json.gz','.jsonl.gz')) or p.suffix == '.jsonl':
            result.append(p)
    # Retain input-only inventories that never reached a tracked experiment.
    for p in (ROOT/'data/source/causal_observability_v1').glob('*population.json'):
        result.append(p)
    return sorted(set(result))


class Union:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def join(self, nodes):
        if not nodes:
            return
        roots = sorted({self.root(k) for k in nodes})
        for key in roots[1:]:
            self.parent[key] = roots[0]


def history_graph():
    from native_head_confirmation_inputs import historical_audit
    audit = historical_audit()
    return audit.graph, audit.exposed, audit.forbidden, audit.evidence


def annotations():
    rows = {}
    for split in ('train', 'val', 'test'):
        with (CACHE/'metadata'/f'{split}_annotations').open(encoding='utf-8-sig', newline='') as f:
            rows[split] = list(csv.DictReader(f))
    return rows


def population(rows, mapping):
    from native_head_confirmation_inputs import rebuild_population
    return rebuild_population()


def archive_inventory():
    urls = [(split, url) for split in ('val', 'test') for url in (CACHE/'metadata'/f'{split}_archives').read_text().splitlines() if url.strip()]

    def inspect(item):
        split, url = item
        name = url.rsplit('/', 1)[-1]
        record = dict(split=split, url=url, content_length=None, cached_path=None)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=20) as response:
                record.update(content_length=int(response.headers['Content-Length']), etag=response.headers.get('ETag'),
                              last_modified=response.headers.get('Last-Modified'), status=response.status)
        except Exception as exc:
            record['error'] = str(exc)
        locations = [CACHE/'archives'/split/name]
        if split == 'val':
            locations += [ROOT/'data/source/cache_drift_v2/shards'/name, ASSETS/'shards'/name]
        for path in locations:
            if path.is_file():
                # Hash archive bytes; no members, decoder or outcomes are opened.
                info = path.with_suffix('.download.json')
                actual = sha(path)
                if info.exists():
                    old = read(info)
                    if old.get('url') != url or old.get('sha256') != actual:
                        raise RuntimeError('Cached archive identity mismatch: '+rel(path))
                if record['content_length'] is not None and path.stat().st_size != record['content_length']:
                    raise RuntimeError('Cached archive size mismatch: '+rel(path))
                record.update(cached_path=rel(path), cached_bytes=path.stat().st_size, cached_sha256=actual)
                break
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(inspect, urls))
    save(OUT/'archive_inventory.json', records)
    known = [r for r in records if r['content_length'] is not None]
    uncached = [r for r in known if not r['cached_path']]
    new = sum(r['content_length'] for r in uncached)
    maximum = max((r['content_length'] for r in known), default=0)
    used = sum(r['bytes'] for r in network_ledger()['requests'])
    result = dict(at_utc=now(), basis='HEAD content-length of every listed archive; complete gzip shards, no byte-range extraction assumed',
        archive_counts={s:sum(r['split']==s for r in records) for s in ('val','test')},
        missing_remote_sizes=[r['url'] for r in records if r['content_length'] is None],
        all_archive_bytes=sum(r['content_length'] for r in known),
        verified_cached_archive_bytes=sum(r.get('cached_bytes',0) for r in records),
        conservative_new_complete_route_bytes=new, metadata_body_bytes=used,
        complete_route_with_metadata_bytes=new+used, largest_archive_scratch_bytes=maximum,
        available_disk_bytes=shutil.disk_usage(ROOT).free, caps=CAPS,
        exact_selected_shard_transfer_bytes=None,
        selected_shard_uncertainty='No authenticated public member-to-shard index retained; exact selected shard set cannot be certified without archive traversals.',
        conservative_route_exceeds_network_cap=new+used > CAPS['network_bytes'],
        proposed_network_cap_gib=int(np.ceil((new+used)*1.10/2**30)),
        resource_amendment='61 GiB authorized by resumed brief; cumulative prior bytes preserved. Stream bounded complete shards and gate retained/peak/throughput separately.',
        collection_projection_seconds=None, cpu_projection_seconds=None, timing_projection_seconds=None,
        throughput_status='NOT_MEASURED: certified fitting smoke pending; fitting and exact-m EVAL gated separately.',
        retained_projection_bytes=None, scratch_status='Largest compressed shard is only a lower bound; expanded members/decode buffers require a separately measured peak.',
        reference_calls_floor=16*(3200+3200), reference_calls_ceiling=16*(3200+8000),
        timing_calls_planned=60*7*(16+4+4+4), no_bulk_acquisition=True)
    save(OUT/'resource_plan.json', result)
    return result


def freeze_config():
    config = dict(study=STUDY, seed=SEED, derived_seeds={k:derived_seed(k) for k in ('bootstrap','timing','smoke')},
        generators='SHA256 UTF-8 ranks; numpy Generator(PCG64) bootstrap/timing', classes=400,
        N=16, K=4, periodic=list(PERIODIC), states=list(range(1,13)), pooling='C candidate reuse count /16 on native logits',
        primary='V(HGH)-V(HG), twelve independently reset complete-schedule CE reductions/source',
        practical_nats=.01, precision_halfwidth=.005, fit_components_per_class=8, folds=4,
        eval_m_range=[8,20], variance_inflation=1.5, variance_sensitivities=[1.,2.],
        views=VIEWS, head='unchanged native400, no recognition-head fitting/scaler/temperature',
        checkpoint_sha256=CHECKPOINT_SHA, pytorchvideo_revision=REVISION, aliases=ALIASES,
        ridge=dict(alpha=1, intercept='unpenalized', objective='normalized weighted half-MSE + alpha/2 L2',
                   scaling='training-only weighted mean/population SD; SD<=1e-12 ->1; mean-one weights'),
        decision='>1e-12 refresh; <-1e-12 hold; otherwise preserve original Periodic action',
        continuation='E25 deterministic periodic repair from original Periodic prefixes',
        grid=dict(offsets_seconds=[-47/30,-32/30,-16/30,0], candidates=16, tolerance_seconds=1e-9,
                  pts='greatest actual presentation timestamp <= requested +1e-9; four distinct frames; no padding',
                  endpoints='16 desired endpoints linearly spaced between first and last technically supported actual PTS, then floor resolve',
                  crop=182, short_side=182, mean=[.45]*3, std=[.225]*3, model_zoo='4x12 nominal; historical uniform4 over48 virtual frames gives [0,15,31,47]/30'),
        bootstrap=dict(draws=10000, percentile=[.025,.975], method='linear', unit='whole independent component within native class', paired=True),
        analytic='unbiased within-class variance; Welch-Satterthwaite t sensitivity',
        uncertainty='Approximate conditional95; fixed checkpoint/classes/final probes; excludes fit variation/unknown duplicates',
        timing=dict(classes=60, components=60, warmups=2, repetitions=5, threads=1,
                    arms=['Dense16','PeriodicB4','PeriodicB4_unused_HG','PeriodicB4_unused_HGH'],
                    boundary='encoded bytes in memory -> fresh full decode/grid -> RGB prep/transfers -> encoder/native head -> pooling -> CUDA completion'),
        caps=CAPS, no_simulated_data=True, synthetic_fixture_amendment='Direct user request overrides attached synthetic-fixture instruction; exact combinatorial/structural checks and real inputs only',
        implementation={n:sha(ROOT/n) for n in CODE if (ROOT/n).is_file()},
        preparation_start_head=read(OUT/'checkout_audit.json')['head'],
        actual_start_head=read(OUT/'resumed_checkout_audit.json')['head'] if (OUT/'resumed_checkout_audit.json').exists() else git('rev-parse','HEAD'),
        evaluation_sealed=False)
    if (OUT/'population_rule.json').exists():
        config.update(population_rule='archive_available_v1',population_rule_sha256=sha(OUT/'population_rule.json'))
    save(OUT/'frozen_config.json', config)


def preflight(online=False):
    started = time.perf_counter()
    from native_head_confirmation_state import state, initialize
    initialize()
    if state()['stage'] != 'PREPARED' or (OUT/'evaluation_seal.json').exists():
        return audit()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (OUT/'checkout_audit.json').exists():
        save(OUT/'checkout_audit.json', dict(at_utc=now(), head=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
            status=git('status','--short'), reviewed=REVIEWED,
            reviewed_tree=git('rev-parse', REVIEWED+'^{tree}'), actual_tree=git('rev-parse','HEAD^{tree}'),
            reconciliation=git('log','--oneline','HEAD..'+REVIEWED),
            tracked_tree_difference=git('diff','--stat','HEAD',REVIEWED),
            AGENTS=[rel(p) for p in [ROOT/'AGENTS.md',ROOT/'results/AGENTS.md',OUT/'AGENTS.md'] if p.exists()],
            authority='Direct user authorizes commit/push; no PR/merge; preserve unrelated presentation and seminar directories'))
        save(OUT/'historical_fingerprint.json', historical_fingerprint())
    freeze_config()
    versions = {}
    for name in ('numpy','scipy','torch','torchvision','pytorchvideo','av','Pillow','threadpoolctl','scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save(OUT/'environment.json', dict(python=sys.version, versions=versions, no_package_changes=True))
    checkpoint = ASSETS/'X3D_XS.pyth'
    model = dict(path=rel(checkpoint), exists=checkpoint.exists(), expected_sha256=CHECKPOINT_SHA,
                 actual_sha256=sha(checkpoint) if checkpoint.exists() else None)
    if model['actual_sha256'] != CHECKPOINT_SHA:
        raise RuntimeError('BLOCKED_INPUT: retained native checkpoint mismatch/missing')
    upstream = ASSETS/f'pytorchvideo-{REVISION}'
    model['upstream_source_hashes'] = {rel(upstream/n):sha(upstream/n) for n in
        ('pytorchvideo/models/hub/x3d.py','pytorchvideo/models/x3d.py','pytorchvideo/models/head.py','pytorchvideo/transforms/functional.py')}
    model['native_validation_scope'] = 'Historical real native-logit verification retained; fresh fitting-only native validation pending'
    save(OUT/'model_identity.json', model)
    if online:
        for name in SOURCES:
            print('Metadata', name, flush=True)
            fetch(name)
    missing = [name for name in SOURCES if not (CACHE/'metadata'/name).is_file()]
    if missing:
        status = dict(technical='BLOCKED_INPUT', scientific='NOT_EVALUATED', missing_metadata=missing,
                      message='Use preflight --online for capped public metadata retrieval; no model exposure')
    else:
        rows = annotations()
        labels = {r['label'] for r in rows['val']}
        mapping = class_mapping(read(CACHE/'metadata/native_classes'), labels)
        if any(set(r['label'] for r in rows[s]) != labels for s in ('train','test')):
            raise RuntimeError('BLOCKED_INPUT: split label sets differ')
        save(OUT/'class_mapping.json', dict(annotation_to_native=mapping, explicit_aliases=ALIASES,
            native_source_sha256=sha(CACHE/'metadata/native_classes')))
        ordered = sorted(mapping,key=mapping.get)
        save(OUT/'observation_schema.json',dict(classes=ordered,dimensions=VIEWS,columns=feature_schema(ordered),
            probability_source='genuine native400 logits at actually earlier refreshed anchors'))
        pop = population(rows, mapping)
        resources = archive_inventory() if online or not (OUT/'resource_plan.json').exists() else read(OUT/'resource_plan.json')
        if resources['missing_remote_sizes']:
            technical = 'BLOCKED_INPUT'
        elif resources['conservative_route_exceeds_network_cap']:
            technical = 'BLOCKED_RESOURCE'
        else:
            technical = 'BLOCKED_INPUT'
        status = dict(technical=technical, scientific='NOT_EVALUATED', eval_model_calls=0, probe_fit_model_calls=0,
            population_minima=pop['minimum_components'],
            blockers=['No complete technically verified/canonical all400 source manifests or evaluation seal',
                      'No authenticated member-to-shard index; full public archive route exceeds cap' if resources['conservative_route_exceeds_network_cap'] else 'Original RGB acquisition/input screening pending',
                      'Scoped programmatic source audit generated; selected-byte/PTS/duplicate certification pending',
                      'Fitting-only smoke throughput, OOF precision, final probes and genuine timing pending'],
            primary_estimate=None, primary_interval=None, matched_opportunity=None, drift_contrast=None,
            no_outcomes_exposed=True)
    save(OUT/'status.json', status)
    event('preflight_complete', elapsed_seconds=time.perf_counter()-started, status=status['technical'],
          encoder_starts=0, encoder_completions=0, decodes=0, probe_fits=0)
    print(json.dumps(status, indent=2))
    return status


def audit():
    from verify_native_head_value_confirmation import verify
    result = verify(OUT, write=False)
    print(json.dumps(result, indent=2))
    return result


def dry_run():
    from native_head_confirmation_inputs import reserve_members
    from native_head_confirmation_runtime import local_path
    split='test' if (OUT/'precision_plan.json').exists() else 'val'
    m=read(OUT/'precision_plan.json')['selected_m'] if split=='test' else 8
    rows=reserve_members(split,m) if m is not None else []
    result=dict(status=read(OUT/'status.json'),next_split=split,selected_m=m,
        prospective_prefix_and_reserve_rows=len(rows),
        missing_needed_or_reserve_bytes=sum(local_path(r['filename'],split) is None for r in rows),
        population_rule='archive_available_v1',
        technical_eligibility='Confirmed absence from the complete pinned release excludes a clip; PRESENT uncached and UNKNOWN required clips remain unresolved.',
        all_lower_ranked_locality_required=False,model_calls=0,decodes=0,network_bytes=0,read_only=True)
    print(json.dumps(result,indent=2))
    return result


@contextmanager
def run_lock():
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/'run.lock'
    if path.exists():
        from native_head_confirmation_state import recover_lock
        recover_lock(path)
    handle = os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        with os.fdopen(handle,'w',encoding='utf-8') as f:
            from native_head_confirmation_state import process_identity
            json.dump(dict(pid=os.getpid(),process_created=process_identity(os.getpid()),started_at_utc=now()),f)
        yield
    finally:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['status','preflight','dry-run','acquire','certify','plan','run','resume','verify','audit','finalize'])
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--online', action='store_true', help='Public metadata/HEAD requests only, no archives/model calls')
    args = parser.parse_args()
    if args.mode == 'status':
        status=read(OUT/'status.json')
        if (OUT/'population_rule.json').exists():
            from native_head_confirmation_availability import VERSION
            status.update(population_rule=VERSION,archive_population=read(OUT/'archive_population_attrition.json')['splits'])
            status['archive_population']={s:dict(complete=r['complete'],states=r['states']) for s,r in status['archive_population'].items()}
        print(json.dumps(status,indent=2))
        return 0
    if args.mode == 'dry-run':
        dry_run();return 0
    if args.mode == 'audit':
        return 0 if audit()['passed'] else 1
    with run_lock():
        from native_head_confirmation_state import initialize, metered
        initialize()
        if read(OUT/'status.json').get('technical') == 'COMPLETE':
            return 0 if audit()['passed'] else 1
        if (OUT/'evaluation_seal.json').exists() and args.mode in ('preflight','dry-run','acquire','certify','plan'):
            return 0 if audit()['passed'] else 1
        if args.mode in ('acquire','certify'):
            from native_head_confirmation_inputs import acquire, certify
            from native_head_confirmation_runtime import execute_input_stage
            return execute_input_stage(args.mode,args.split)
        if args.mode in ('preflight','dry-run'):
            with metered('cpu','metadata-preflight'):
                result = preflight(args.online)
            return 0 if result.get('technical') == 'COMPLETE' else 2
        if args.mode == 'verify':
            from verify_native_head_value_confirmation import verify
            from native_head_confirmation_state import snapshot_accounting
            with metered('cpu','independent-verification'):
                result = verify(OUT, write=True)
            snapshot_accounting()
            print(json.dumps(result, indent=2))
            return 0 if result['passed'] else 1
        from native_head_confirmation_runtime import execute
        return execute(args.mode)


if __name__ == '__main__':
    raise SystemExit(main())
