"""Independent class-label/index/head/table identity checks; no model calls."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    uc=read(OUT/'ucf_manifest.json');reuse=read(ROOT/'results/ucf101_causal_development_v1/recognition_head_reuse.json')
    hp=read(ROOT/reuse['path']);hm=read(OUT/'hmdb_population.json');hh=read(OUT/'hmdb_head.json')
    # HMDB canonical order comes independently from the actual official class annotation file names.
    files=sorted((ROOT/'data/source/causal_observability_v1/splits').rglob('*_test_split1.txt'))
    classes=[p.name.removesuffix('_test_split1.txt') for p in files]
    assert len(classes)==51 and classes==hm['classes']==hh['classes']
    assert uc['classes']==hp['classes']==reuse['classes'] and len(uc['classes'])==101
    result=dict(passed=True,datasets={})
    for name,head,cl in [('ucf',hp,uc['classes']),('hmdb',hh,classes)]:
        h={k:np.array(v,dtype=np.float64) for k,v in head['parameters'].items()}
        for k,shape in [('means',(2048,)),('scales',(2048,)),('weights',(len(cl),2048)),('bias',(len(cl),))]:
            assert h[k].shape==shape and np.isfinite(h[k]).all()
        assert (h['scales']>0).all()
        if name=='hmdb':
            for m in hm['videos']:
                assert m['label']==m['clip'].split('/')[0] and m['y']==cl.index(m['label'])
        dst=OUT/name
        if (dst/'population.json').exists():
            meta=read(dst/'population.json')
            for m in meta:assert m['y']==cl.index(m['label'])
            if name=='hmdb':
                expected=[m for m in hm['videos'] if m['role']=='EVAL']
                assert [(m['clip'],m['label'],m['family'],m['y']) for m in meta]==[(m['clip'],m['label'],m['family'],m['y']) for m in expected]
            else:assert [m['clip'] for m in meta]==[m['clip'] for m in uc['clips']]
            with np.load(dst/'outcome_table.npz') as z:
                assert z['loss'].shape==z['prediction'].shape==(len(meta),455)
                assert np.isfinite(z['loss']).all() and ((z['prediction']>=0)&(z['prediction']<len(cl))).all()
                if len(z['candidate_logits']):
                    assert z['candidate_logits'].shape==(len(meta),16,len(cl)) and np.isfinite(z['candidate_logits']).all()
            result['datasets'][name]=dict(classes=len(cl),clips=len(meta),tables_checked=True)
        else:result['datasets'][name]=dict(classes=len(cl),tables_checked=False,reason='collection underway')
    (OUT/'class_order_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
