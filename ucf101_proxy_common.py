"""Frozen constants, atomic ledgers and bounded Windows-native HTTPS acquisition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/ucf101_proxy_replication_v1'
DATA = ROOT / 'data/source/ucf101_proxy_replication_v1'
SEED = 20261026
REVISION = 'f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0'
CHECKPOINT_SHA = '6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86'
REVIEWED = '247ba4ed6efbc91cd69d88f26f9b7552b7b3ca28'
OFFICIAL = 'https://www.crcv.ucf.edu/data/'
DATA_URL = OFFICIAL + 'UCF101/UCF101.rar'
SPLIT_URL = OFFICIAL + 'UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
PAGE_URL = OFFICIAL + 'UCF101.php'
CAP = 20 * 1024**3


class Blocker(RuntimeError):
    """A real access/population/resource gate; scientific stages remain pending."""


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(100):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def rank(*parts):
    # Explicit UTF-8, no delimiters inside UCF class/group names. Separate clip namespace.
    return hashlib.sha256(''.join(map(str, (SEED, *parts))).encode('utf-8')).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def verify_hashes(records):
    for name, digest in records.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise Blocker(f'Missing or changed frozen artifact: {name}')


def seal_check(require_exposure=False):
    path = OUT/'evaluation_seal.json'
    if not path.exists():
        raise Blocker('Test inference locked: evaluation seal missing')
    seal = read(path)
    amendment_path = OUT/'post_exposure_amendment.json'
    if amendment_path.exists():
        # Explicit post-exposure audit correction; original seal/exposure are immutable.
        amendment = read(amendment_path)
        allowed = {'ucf101_proxy_common.py', 'ucf101_proxy_verify.py'}
        if (set(amendment['changed_frozen_files']) != allowed
                or amendment['original_seal_sha256'] != sha(path)
                or amendment['original_exposure_sha256'] != sha(OUT/'evaluation_exposure.json')):
            raise Blocker('Invalid post-exposure amendment linkage or scope')
        verify_hashes({k: v for k, v in seal['sha256'].items() if k not in allowed})
        for name in allowed:
            record = amendment['changed_frozen_files'][name]
            if record['original_sha256'] != seal['sha256'][name]:
                raise Blocker('Amendment does not preserve original sealed code')
            verify_hashes({record['archived_original']: record['original_sha256'],
                           name: record['amended_sha256']})
        verify_hashes(amendment['unchanged_scientific_outputs'])
    else:
        verify_hashes(seal['sha256'])
    if require_exposure:
        exposure = OUT/'evaluation_exposure.json'
        if not exposure.exists() or read(exposure)['seal_sha256'] != sha(path):
            raise Blocker('Missing valid exposure record')
    return seal


class Network:
    """No implicit retry or disabled TLS. Windows curl uses the OS trust store.

    Body bytes plus 1 MiB/request conservative transport overhead are charged.
    Incomplete requests reserve their entire allowed size on crash/resume.
    The cap includes failed/retried requests. Refuse untracked preexisting files.
    """
    def __init__(self):
        self.path = OUT/'network_accounting.json'
        self.state = read(self.path) if self.path.exists() else dict(cap_bytes=CAP, charged_bytes=0, requests=[])
        if not self.state.get('other_preflight_network_reserve_bytes'):
            # Git fetch/ls-remote, official HEAD probes and browsing used before acquisition.
            self.state['other_preflight_network_reserve_bytes'] = 8*1024**2
            self.state['charged_bytes'] += 8*1024**2
        for event in self.state['requests']:
            if event['status'] == 'inflight':
                event['status'] = 'interrupted_charged_in_full'
                self.state['charged_bytes'] += event['reserve_bytes']
        save(self.path, self.state)

    def get(self, url, target, limit, byte_range=None):
        target = Path(target)
        identity = target.with_name(target.name+'.identity.json')
        if target.exists() and identity.exists():
            meta = read(identity)
            if meta['url'] != url or meta.get('byte_range') != byte_range or meta['sha256'] != sha(target):
                raise Blocker(f'Cached acquisition identity mismatch: {target}')
            return target
        remaining = CAP-self.state['charged_bytes']-1024**2
        limit = min(int(limit), remaining)
        if limit <= 0:
            raise Blocker('20 GiB NEW network cap exhausted')
        target.parent.mkdir(parents=True, exist_ok=True)
        event = dict(url=url, byte_range=byte_range, started_at_utc=now(), status='inflight', reserve_bytes=limit+1024**2)
        self.state['requests'].append(event)
        save(self.path, self.state)
        part = target.with_name(target.name+'.partial')
        # No account creation, consent POST, certificate bypass or external mirror.
        cmd = ['curl.exe', '--fail', '--location', '--silent', '--show-error',
               '--connect-timeout', '30', '--max-time', '7200', '--max-filesize', str(limit),
               '--output', str(part), '--write-out', '%{size_download}', url]
        headers = target.with_name(target.name+'.headers')
        if byte_range is not None:
            cmd[1:1] = ['--range', byte_range, '--dump-header', str(headers)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            body = int(float(result.stdout.strip()))
        except ValueError:
            body = limit
        charge = (body if result.returncode == 0 else limit) + 1024**2
        self.state['charged_bytes'] += charge
        event.update(status='complete' if result.returncode == 0 else 'failed_charged_in_full',
                     body_bytes_reported=body, charged_bytes=charge, finished_at_utc=now(),
                     error=result.stderr[-1500:], returncode=result.returncode)
        save(self.path, self.state)
        if result.returncode:
            raise Blocker(f'Official download failed (curl {result.returncode}): {url}: {result.stderr[-500:]}')
        if byte_range is not None:
            import re
            h = headers.read_text(encoding='utf-8', errors='replace')
            if not re.search(r'Content-Range:\s*bytes\s+'+re.escape(byte_range)+r'/', h, re.I):
                raise Blocker('Server did not honor exact range; untrusted segment preserved')
            lo, hi = map(int, byte_range.split('-'))
            if part.stat().st_size != hi-lo+1:
                raise Blocker('Incorrect ranged response byte count')
        part.replace(target)
        save(identity, dict(url=url, byte_range=byte_range, sha256=sha(target), bytes=target.stat().st_size, downloaded_at_utc=now(),
                            transport='Windows curl verified TLS; no new terms accepted'))
        return target
