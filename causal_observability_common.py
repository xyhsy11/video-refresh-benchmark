"""Experiment 19 identities, atomic checkpoints, bounded acquisition and seals."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/causal_observability_v1'
DATA = ROOT/'data/source/causal_observability_v1'
SEED = 20261019
REVIEWED = '017fe79843428127be55ebf0b8819a9921b46d6e'
REVISION = '1f1e6df3fccdfe34a230e60c00a799b82905b35f'
BASE = f'https://huggingface.co/datasets/Serrelab/hmdb51/resolve/{REVISION}/'
SOURCES = {
    'hmdb51_org.rar': (BASE+'hmdb51_org.rar', '9e714a0d8b76104d76e932764a7ca636f929fff66279cda3f2e326fa912a328e', 2200*2**20),
    'test_train_splits.rar': (BASE+'test_train_splits.rar', '229c94f845720d01eb3946d39f39292ea962d50a18136484aa47c1eba251d2b7', 2**20),
}


class Blocker(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for retry in range(100):
        try:
            tmp.replace(path); return
        except PermissionError:
            if retry == 99: raise
            time.sleep(.05)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def rank(*parts):
    return hashlib.sha256(('20261019|'+'|'.join(map(str, parts))).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def state(stage, status, **details):
    path = OUT/'status.json'
    s = read(path) if path.exists() else dict(stages={})
    for name in ['review', 'data', 'eligibility', 'tests', 'GPU_smoke', 'training', 'freeze', 'test_rollouts', 'diagnostics', 'timing', 'verification', 'figures', 'final']:
        s['stages'].setdefault(name, dict(status='pending'))
    s['stages'][stage] = dict(status=status, at_utc=now(), **details)
    s.update(current_stage=stage, status=status, updated_at_utc=now(),
        resume_command=r'& .\.venv_temporal\Scripts\python.exe .\run_causal_observability.py --resume',
        audit_command=r'& .\.venv_temporal\Scripts\python.exe -B .\run_causal_observability.py --audit-only')
    save(path, s)


def verify(records):
    for name, digest in records.items():
        if not (ROOT/name).is_file() or sha(ROOT/name) != digest:
            raise Blocker('Missing or changed identity: '+name)


def seal_check(exposed=False):
    path = OUT/'evaluation_seal.json'
    if not path.exists(): raise Blocker('Test access prohibited: no evaluation seal')
    seal = read(path); verify(seal['sha256'])
    if exposed:
        p = OUT/'evaluation_exposure.json'
        if not p.exists() or read(p)['seal_sha256'] != sha(path):
            raise Blocker('Exposure ledger does not match original seal')
    return seal


def checkpoint(path, identity, compute):
    """Resume only identical provenance and byte-stable completed payloads."""
    path=Path(path)
    if path.exists():
        r=read(path)
        digest=hashlib.sha256(json.dumps(r['payload'],sort_keys=True,allow_nan=False).encode()).hexdigest()
        if r['identity']!=identity or r['payload_sha256']!=digest:
            raise Blocker('Checkpoint provenance/payload changed: '+rel(path))
        return r['payload']
    payload=compute();digest=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(path,dict(identity=identity,payload=payload,payload_sha256=digest,completed_at_utc=now()))
    return payload


def acquire(url, path, limit, expected=None):
    """Verified TLS, no automatic retries; interrupted unknown bodies charged in full."""
    path = Path(path); identity = path.with_name(path.name+'.identity.json')
    if identity.exists() and path.exists():
        m = read(identity)
        if m['url'] != url or m['sha256'] != sha(path) or (expected and m['sha256'] != expected):
            raise Blocker('Cached download identity mismatch: '+path.name)
        return path
    ledger = OUT/'network_accounting.json'
    n = read(ledger) if ledger.exists() else dict(cap_bytes=20*2**30, charged_bytes=8*2**20, requests=[],
        preflight_reserve_bytes=8*2**20, note='Git metadata/web/HEAD overhead conservatively reserved; completed bodies plus 1MiB/request, failures charged full response limit')
    for r in n['requests']:
        if r['status'] == 'inflight':
            n['charged_bytes'] += r['reserve']; r['status'] = 'interrupted_charged_full'
    reserve = limit+2**20
    if n['charged_bytes']+reserve > n['cap_bytes']: raise Blocker('20GiB network cap would be exceeded')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+f'.attempt{len(n["requests"]):03d}.partial')
    headers = partial.with_name(partial.name+'.headers')
    event = dict(url=url, path=rel(path), at_utc=now(), status='inflight', reserve=reserve)
    n['requests'].append(event); save(ledger, n)
    r = subprocess.run(['curl.exe', '--fail', '--location', '--max-redirs', '5', '--connect-timeout', '30',
        '--max-time', '1800', '--max-filesize', str(limit), '--silent', '--show-error',
        '--dump-header', str(headers), '--output', str(partial), url], capture_output=True, text=True)
    size = partial.stat().st_size if partial.exists() else 0
    event.update(returncode=r.returncode, actual_file_bytes=size, stderr=r.stderr, finished_at_utc=now())
    if r.returncode:
        event['status'] = 'failed_charged_full'; n['charged_bytes'] += reserve; save(ledger, n)
        raise Blocker('Official acquisition failed; attempt retained: '+path.name+' '+r.stderr.strip())
    n['charged_bytes'] += size+2**20
    digest = sha(partial)
    if expected and digest != expected:
        event['status'] = 'hash_mismatch'; save(ledger, n); raise Blocker('Original archive SHA256 mismatch: '+path.name)
    event['status'] = 'complete'; event['sha256'] = digest; save(ledger, n)
    partial.replace(path)
    save(identity, dict(url=url, bytes=size, sha256=digest, acquired_at_utc=now(), headers_sha256=sha(headers)))
    return path
