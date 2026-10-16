"""Temporal-snippet contracts and bounded, resumable experiment I/O."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'data/source/temporal_refresh_generalization_v1'
REVISION = 'f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth'
SEED = 20261024
TIME_TOL = 1e-9
TOL = 1e-12
REPS = ['resnet18_endpoint', 'resnet18_four_frame_mean', 'x3d_xs']


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    # Windows indexers/antivirus may briefly hold a JSON file without delete
    # sharing. Retry the atomic replace, never truncate an existing ledger.
    for attempt in range(100):
        try:
            temporary.replace(path); break
        except PermissionError:
            if attempt==99: raise
            time.sleep(.05)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def rank(kind, value): return hashlib.sha256(f'temporal_refresh_generalization_v1|{SEED}|{kind}|{value}'.encode()).hexdigest()
def commit(): return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()


def require_evaluation_seal(output):
    path=output/'evaluation_seal.json'
    if not path.exists(): raise RuntimeError('Evaluation locked: freeze protocol,heads,cohort and evaluator before inference')
    seal=read(path)
    for name,digest in seal['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Frozen identity changed: {name}; preserve exposed results and document amendment')
    exposure=output/'evaluation_exposure.json'
    if not exposure.exists() or read(exposure)['seal_sha256']!=sha(path):
        raise RuntimeError('Evaluation exposure must be recorded against the valid pre-inference seal')


class Budget:
    """HTTP response-body accounting, including retries; no historical env writes.

    Unknown-length metadata is read only up to remaining budget. Interrupted
    known-size requests are conservatively charged in full on resume.
    """
    def __init__(self, path, cap):
        self.path=Path(path); self.lock=threading.RLock()
        self.state=read(path) if self.path.exists() else dict(cap_bytes=int(cap), actual_body_bytes=0,
            interruption_charge_bytes=0, requests=[], inflight={})
        assert self.state['cap_bytes']==int(cap)
        self.state['interruption_charge_bytes']+=sum(self.state['inflight'].values())
        self.state['inflight']={}; self.persist()
    def persist(self): save(self.path,self.state)
    def remaining(self):
        return self.state['cap_bytes']-self.state['actual_body_bytes']-self.state['interruption_charge_bytes']-sum(self.state['inflight'].values())
    def download(self,url,path):
        path=Path(path); info=path.with_name(path.name+'.identity.json')
        if path.exists() and info.exists():
            old=read(info); assert old['url']==url and old['sha256']==sha(path); return path
        path.parent.mkdir(parents=True,exist_ok=True)
        # A failed transfer is retried as the same candidate, never a replacement.
        for attempt in range(3):
            event={'url':url,'started_at_utc':now(),'attempt':attempt+1,'actual_bytes':0}
            token=f'{len(self.state["requests"])}'; self.state['requests'].append(event)
            try:
                with urllib.request.urlopen(url,timeout=60) as response:
                    expected=int(response.headers.get('Content-Length',0))
                    if expected and expected>self.remaining(): raise RuntimeError('20 GiB download cap would be exceeded')
                    self.state['inflight'][token]=expected; self.persist()
                    with path.with_name(path.name+'.partial').open('wb') as dest:
                        while True:
                            if not expected and self.remaining()<=0: raise RuntimeError('Download cap exhausted')
                            amount=1024*1024 if expected else min(1024*1024,self.remaining())
                            data=response.read(amount)
                            if not data: break
                            dest.write(data); event['actual_bytes']+=len(data)
                            self.state['actual_body_bytes']+=len(data)
                            self.state['inflight'][token]=max(0,expected-event['actual_bytes']); self.persist()
                    if expected: assert event['actual_bytes']==expected
                self.state['inflight'].pop(token,None); event['status']='complete'; self.persist()
                path.with_name(path.name+'.partial').replace(path)
                save(info,{'url':url,'sha256':sha(path),'bytes':path.stat().st_size,'downloaded_at_utc':now()})
                return path
            except Exception as exc:
                # A failed read may have consumed bytes inside the HTTP reader
                # before raising. Charge its entire unread known-size reserve.
                uncertain=self.state['inflight'].pop(token,None) or 0
                self.state['interruption_charge_bytes']+=uncertain
                event.update(status='failed',error=str(exc),uncertain_bytes_charged=uncertain); self.persist()
                if isinstance(exc,RuntimeError) or attempt==2: raise
                time.sleep(2**attempt)


def snippet_grid(timestamps):
    """Resolve four trailing requests by floor PTS, never nearest/future frames."""
    pts=np.asarray(timestamps,dtype=np.float64)
    if len(pts)<16 or not np.isfinite(pts).all() or not (np.diff(pts)>0).all():
        raise ValueError('Need finite strictly increasing presentation timestamps')
    offsets=np.array([47/30,32/30,16/30,0.])
    valid=[]
    for i,u in enumerate(pts):
        requested=u-offsets
        indices=np.searchsorted(pts,requested+TIME_TOL,side='right')-1
        if min(indices)>=0 and len(set(indices))==4 and indices[-1]==i: valid.append(i)
    if len(valid)<16: raise ValueError('Fewer than16 complete distinct-frame trailing snippets')
    desired=np.linspace(pts[valid[0]],pts[valid[-1]],16)
    endpoints=np.searchsorted(pts,desired+TIME_TOL,side='right')-1
    if len(set(endpoints))!=16 or not set(endpoints).issubset(valid):
        raise ValueError('Uniform endpoint resolution lacks distinct complete support')
    requested=pts[endpoints,None]-offsets
    indices=np.searchsorted(pts,requested+TIME_TOL,side='right')-1
    assert np.all(pts[indices]<=requested+TIME_TOL)
    assert all(len(set(row))==4 for row in indices)
    return {'desired_endpoint_timestamps':desired.tolist(),'endpoint_indices':endpoints.tolist(),
        'endpoint_timestamps':pts[endpoints].tolist(),'requested_timestamps':requested.tolist(),
        'frame_indices':indices.tolist(),'actual_timestamps':pts[indices].tolist(),
        'timestamp_tolerance_seconds':TIME_TOL,'rule':'greatest decoded PTS <= requested+1e-9 sec; no padding'}


def operational_grid(timestamps):
    """Same sampling on prevalidated clips, without timed diagnostic assertions."""
    pts=np.asarray(timestamps);offset=np.array([47/30,32/30,16/30,0.])
    first=int(np.searchsorted(pts,pts[0]+47/30-TIME_TOL,side='left'))
    while True:
        indices=np.searchsorted(pts,pts[first]-offset+TIME_TOL,side='right')-1
        if min(indices)>=0 and len(set(indices))==4 and indices[-1]==first:break
        first+=1
    desired=np.linspace(pts[first],pts[-1],16)
    endpoints=np.searchsorted(pts,desired+TIME_TOL,side='right')-1
    return (np.searchsorted(pts,pts[endpoints,None]-offset+TIME_TOL,side='right')-1).tolist()


def inspect_encoded(path):
    import av
    from research_utils import configure_decoder
    timestamps=[]; types=Counter(); start=time.perf_counter()
    with av.open(str(path)) as container:
        stream=container.streams.video[0]; configure_decoder(stream)
        codec=stream.codec_context.name
        resolution={'width':stream.codec_context.width,'height':stream.codec_context.height}
        for frame in container.decode(stream):
            if frame.pts is None: raise ValueError('Missing PTS')
            timestamps.append(float(frame.pts*frame.time_base))
            types[av.video.frame.PictureType(frame.pict_type).name]+=1
        fps=float(stream.average_rate) if stream.average_rate else 0
    if not timestamps: raise ValueError('No decoded frames')
    duration=timestamps[-1]-timestamps[0]+(1/fps if fps else 0)
    if codec!='h264' or not 5<=duration<=15: raise ValueError(f'Ineligible codec/duration {codec}/{duration}')
    grid=snippet_grid(timestamps)
    return {'codec':codec,'resolution':resolution,'reported_fps':fps,'decoded_frame_count':len(timestamps),'duration_seconds':duration,
        'duration_method':'last-first decoded PTS + inverse reported FPS','frame_types':dict(types),
        'decoder':{'thread_count':1,'thread_type':'NONE','skip_frame':'NONE','hardware':False},
        'snippet_grid':grid,'inspection_seconds':time.perf_counter()-start}
