"""Input-only temporal support descriptions for the frozen populations."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    probabilities=[0,.01,.05,.25,.5,.75,.95,.99,1]
    def describe(rows):
        spans=[];gaps=[];clips=[]
        for r in rows:
            g=r['grid'];ts=np.array(g['actual_timestamps']);ep=np.array(g['endpoint_timestamps'])
            spans.extend((ts[:,-1]-ts[:,0]).tolist());gaps.extend(np.diff(ep).tolist());clips.append(float(r['duration']))
        return dict(clips=len(rows),snippets=len(spans),candidate_gaps=len(gaps),
            weighting='unweighted input-metadata description of all included clips/snippets/gaps; no outcome selection',
            quantile_probabilities=probabilities,snippet_span_seconds=np.quantile(spans,probabilities).tolist(),
            candidate_gap_seconds=np.quantile(gaps,probabilities).tolist(),clip_PTS_span_seconds=np.quantile(clips,probabilities).tolist())
    path=ROOT/'results/ucf101_causal_development_v1/population_manifest.json'
    expected=read(ROOT/'results/safe_schedule_density_v1/input_inventory.json')[path.relative_to(ROOT).as_posix()]['sha256']
    assert sha(path)==expected
    wanted={r['clip'] for r in read(OUT/'ucf_manifest.json')['clips']}
    uc=[dict(grid=r['snippet_grid'],duration=r['duration_pts_span']) for r in read(path)['videos'] if r['clip'] in wanted]
    assert len(uc)==400
    path=ROOT/'results/causal_observability_population_v2/screening_evidence.jsonl.gz'
    assert sha(path)==read(OUT/'hmdb_input_manifest.json')[path.relative_to(ROOT).as_posix()]
    with gzip.open(path,'rt') as f:screen={r['key']:r for r in map(json.loads,f)}
    result=dict(UCF=describe(uc),input_sha256={'UCF':expected,'HMDB':sha(path)},
        interpretation='HMDB adaptive frame-index support differs from UCF requested physical-time offsets; differences cannot be attributed solely to semantic domain.')
    pop=read(OUT/'hmdb_population.json')
    for role in ['HEAD_TRAIN','EVAL']:
        rr=[dict(grid=screen[m['clip']]['grid'],duration=screen[m['clip']]['pts_span']) for m in pop['videos'] if m['role']==role]
        result['HMDB_'+role]=describe(rr)
    (OUT/'temporal_support.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
