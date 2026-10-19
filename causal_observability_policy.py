"""Causal controller interface: no video, labels or uncomputed features in observations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import time
from causal_observability_common import rank

METHODS = ['periodic', 'RandomB4_seeded', 'motion', 'history', 'history_motion']


def affine(f, head):
    return ((np.asarray(f, dtype=np.float64)-head['means'])/head['scales']) @ np.asarray(head['weights']).T+head['bias']


def softmax(z):
    p = np.exp(z-np.max(z)); return p/p.sum()


def cheap_signal(current, previous):
    """32 values from actual 32x32 grayscale observations encountered so far."""
    a = np.asarray(current, dtype=np.float64)/255
    if a.shape != (32,32): raise ValueError('Expected an actual 32x32 grayscale image')
    b = a if previous is None else np.asarray(previous,dtype=float)/255
    h = np.histogram(a,bins=8,range=(0,1))[0]/1024
    hp = np.histogram(b,bins=8,range=(0,1))[0]/1024
    diff = abs(a-b)
    return np.r_[h,a.reshape(4,8,4,8).mean((1,3)).ravel(),a.mean(),a.std(),
                 diff.mean(),np.sqrt(np.mean(diff**2)),np.max(diff),np.mean(diff>.1),abs(h-hp).sum(),float(diff.mean()>.2)]


@dataclass(frozen=True)
class Observation:
    t: int
    age: int
    remaining: int
    history: tuple
    current_signal: tuple

    @property
    def label(self): raise RuntimeError('Labels are prohibited in causal observations')
    @property
    def future_features(self): raise RuntimeError('Future features are prohibited')
    @property
    def current_feature(self): raise RuntimeError('Current uncomputed feature is prohibited')


def history_vector(obs, head):
    if not obs.history: raise ValueError('Mandatory first refresh must precede decisions')
    f = np.array(obs.history,dtype=float)
    z = affine(f[-1],head); p = softmax(z)
    margins=[]
    for row in f:
        prob=softmax(affine(row,head)); top=np.partition(prob,-2)[-2:];margins.append(top.max()-top.min())
    prev=f[-2] if len(f)>1 else f[-1]
    norm=np.linalg.norm(f[-1]);pn=np.linalg.norm(prev)
    cosine=float(f[-1] @ prev/max(norm*pn,1e-12))
    entropy=-float(np.sum(p*np.log(np.maximum(p,1e-300))))
    top=np.partition(z,-2)[-2:]
    return np.r_[obs.t/15,obs.age/15,obs.remaining/4,p,entropy,p.max(),top.max()-top.min(),
        norm/np.sqrt(f.shape[1]),cosine,np.linalg.norm(f[-1]-prev)/np.sqrt(f.shape[1]),
        len(f)/4,np.mean(margins),np.std(margins)]


def vector(obs, head, with_motion):
    h=history_vector(obs,head)
    # H-only never reads the cheap observations, including for normalization.
    m=np.asarray(obs.current_signal,dtype=float) if with_motion else np.zeros(32)
    if m.shape != (32,): raise ValueError('Cheap observation shape mismatch')
    return np.r_[h,m]


class Policy:
    def __init__(self,method,head,model=None,threshold=.5,clip=''):
        self.method,self.head,self.model,self.threshold=method,head,model,threshold
        rng=np.random.Generator(np.random.PCG64(int(rank('random_policy',clip)[:16],16)))
        self.random_positions={0,*map(int,rng.choice(np.arange(1,16),3,replace=False))}

    @property
    def needs_cheap(self):return self.method in ['motion','history_motion','periodic_unused_gate']

    def decide(self,obs):
        if self.method in ['periodic','periodic_unused_gate']:return obs.t in [0,4,8,12]
        if self.method=='RandomB4_seeded':return obs.t in self.random_positions
        if self.method=='motion':return obs.current_signal[26]>=self.threshold
        x=vector(obs,self.head,self.method=='history_motion');m=self.model
        x=(x-np.asarray(m['means']))/np.asarray(m['scales'])
        h=np.maximum(0,np.asarray(m['w1']) @ x+np.asarray(m['b1']))
        logit=float(np.asarray(m['w2']) @ h+float(m['b2']))
        probability=float(1/(1+np.exp(-np.clip(logit,-700,700))))
        return probability>=self.threshold


def execute(provider,policy,head,on_decision=None):
    """The provider exposes only the current candidate and executed refresh calls.

    Exactly four actual provider.refresh invocations in live inference; explicitly
    labeled training/unit replay providers are separate from empirical execution.
    """
    history=[];cached=[];positions=[];trace=[];previous=None;last=-1
    for t in range(16):
        provider.advance(t)
        decision_start=time.perf_counter()
        signal=()
        if policy.needs_cheap:
            current=provider.thumbnail();signal=tuple(cheap_signal(current,previous));previous=current
        remaining=4-len(positions)
        obs=Observation(t,t-last,remaining,tuple(tuple(row) for row in history),signal)
        forced=t==0 or remaining==16-t
        selected=bool(remaining and (forced or policy.decide(obs)))
        decision_seconds=time.perf_counter()-decision_start
        if on_decision is not None:on_decision(obs,selected,forced)
        trace.append(dict(t=t,remaining_before=remaining,refresh_age=t-last,selected=selected,forced=forced,
            available_refreshes=positions.copy(),cheap_observed=policy.needs_cheap,cpu_decision_wall_seconds=decision_seconds))
        if selected:
            f=np.asarray(provider.refresh(),dtype=np.float64)
            history.append(f.copy());positions.append(t);last=t
        if not history:raise RuntimeError('No initial cache')
        cached.append(history[-1])
    if len(positions)!=4 or positions[0]!=0 or provider.forward_completions!=4:
        raise RuntimeError('Genuine four-forward budget contract violated')
    features=np.asarray(cached)
    z=affine(features.mean(0),head)
    return dict(positions=positions,trace=trace,logits=z.tolist(),prediction=int(z.argmax()),
        cpu_decision_wall_seconds=sum(r['cpu_decision_wall_seconds'] for r in trace),
        cache_indices=np.maximum.accumulate([t if t in positions else 0 for t in range(16)]).tolist())


class TrainingReplayProvider:
    """OFFLINE TRAINING/unit replay only; never a live or timed inference provider."""
    def __init__(self,features,thumbs):
        self._features=np.asarray(features);self._thumbs=thumbs;self.t=-1;self.forward_completions=0;self._used=False
    def advance(self,t):
        if t!=self.t+1:raise ValueError('Strict chronological order required')
        self.t=t;self._used=False
    def thumbnail(self):return self._thumbs[self.t].copy()
    def refresh(self):
        if self._used:raise ValueError('Repeated current refresh prohibited')
        self._used=True;self.forward_completions+=1;return self._features[self.t].copy()
