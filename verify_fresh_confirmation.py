"""Focused confirmation software checks; synthetic fixtures are not evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import fresh_confirmation_data as data
from fresh_confirmation_eval import per_video,statistics


def software_checks(protocol:dict) -> dict:
    checks={}
    with tempfile.TemporaryDirectory() as name:
        output=Path(name); target=output/'unicode.json'
        original_replace=Path.replace; attempts=[0]
        def transient(self,dest):
            attempts[0]+=1
            if attempts[0]<=2: raise PermissionError('synthetic sharing violation')
            return original_replace(self,dest)
        with patch.object(Path,'replace',transient): data.strict_json(target,{'value':'中文 / UTF-8'})
        assert json.loads(target.read_text(encoding='utf-8'))['value']=='中文 / UTF-8' and attempts[0]==3
        checks['UTF8_and_transient_atomic_replace_retries']=True
        prior={'source_ids':protocol['excluded_source_ids'],'file_hashes':protocol['excluded_file_SHA256']}
        data.strict_json(output/'exclusions.json',prior)
        with patch.object(Path,'rglob',side_effect=AssertionError('Resume must not rescan prior memberships')):
            assert data.prior_exclusions(output,protocol)==prior
        checks['resume_exclusions_frozen_no_self_exclusion_rescan']=True
        response=SimpleNamespace(headers={'Content-Length':str(1024**3),'ETag':'fixed'})
        response.__enter__=lambda:response
        class Head:
            headers=response.headers
            def __enter__(self): return self
            def __exit__(self,*args): return False
        methods=[]
        def request(req,**kwargs): methods.append(req.get_method()); return Head()
        plan={'ordered_archives':['https://example.invalid/shard.tar.gz'],'verified_cached_archives':[],'candidate_rows':[]}
        fake=SimpleNamespace(progress={'videos':[]})
        with patch('urllib.request.urlopen',request):
            try: data.acquire_and_collect(output,protocol,plan,prior,fake,.5,8)
            except RuntimeError as exc: assert 'Incomplete cohort' in str(exc)
            else: raise AssertionError('Cap failed to block full-size reservation')
        state=json.loads((output/'acquisition.json').read_text(encoding='utf-8'))
        assert methods==['HEAD'] and state['newly_reserved_archive_bytes']==0 and state['transfer_response_body_bytes']==0
        assert all(v==5 for v in state['shortfall_per_class'].values())
        checks['full_archive_reservation_cap_blocks_GET_and_reports_exact_shortfall']=True
    # Synthetic metric fixture: 100 sources, full policy/seed schema, no real frames.
    raw=[]
    for i in range(100):
        base=.1+i*.0001 if i else 0.0
        for B in [*protocol['budgets'],16]:
            for policy in ['dense'] if B==16 else protocol['policies']:
                for s in range(protocol['sampling_seed'],protocol['sampling_seed']+10) if policy=='random' else [-1]:
                    advantage=(.002+(i%5-2)*.0002) if policy=='value_age_motion_budget' else .001 if policy=='value_age_budget' else .003 if policy=='paced_age_plus_motion' else .03 if policy=='oracle' else 0.0
                    error=0.0 if policy=='dense' or i==0 else base-advantage+((s-protocol['sampling_seed']-4.5)*1e-6 if policy=='random' else 0)
                    positions=list(range(0,16,16//B))
                    raw.append({'video_id':f'fixture_{i:03d}','source_video_id':f'fixture_{i:03d}','label':protocol['classes'][i//5],
                        'budget':B,'policy':policy,'seed_id':s,'privileged':policy in ['oracle','privileged_actual_drift'],
                        'simulated_refreshes':B,'cache_reuses':16-B,'feature_outputs':16,'mean_output_error':error,'max_output_error':error,
                        'forced_refreshes':0,'last_refresh_position':positions[-1],'outputs_after_budget_exhausted':15-positions[-1],
                        'sum_action_regret':0,'refresh_positions':json.dumps(positions)})
    raw=pd.DataFrame(raw); table=per_video(raw,100,protocol)
    assert len(raw)==5500 and len(table)==2800
    for B in protocol['budgets']:
        r=table[(table.budget==B)&(table.policy=='random')&(table.video_id=='fixture_050')].iloc[0]
        assert abs(r.mean_output_error-.105)<1e-12 and r.trajectory_count==10
    checks['all_ten_random_seeds_averaged_within_video_and_expected_counts']=True
    a,p,ci,classes=statistics(table,raw,protocol)
    b,p2,ci2,classes2=statistics(table,raw,protocol)
    assert a==b and ci==ci2 and len(ci['comparisons'])==12
    assert len({r['resampling_indices_sha256'] for r in ci['comparisons']})==1
    idx=np.random.Generator(np.random.PCG64(protocol['sampling_seed'])).integers(0,100,size=(10000,100),dtype=np.int64)
    primary=next(r for r in ci['comparisons'] if r['comparison_type']=='primary')
    pivot=table[table.budget==4].pivot(index='source_video_id',columns='policy',values='mean_output_error').sort_index()
    deltas=(pivot.periodic-pivot.value_age_motion_budget).to_numpy()
    low,high=np.quantile(np.array([sum(deltas[sampled])/100 for sampled in idx]),[.025,.975],method='linear')
    np.testing.assert_allclose([low,high],[primary['CI95_lower'],primary['CI95_upper']],atol=1e-12,rtol=1e-12)
    assert p[p.source_video_id=='fixture_000'].percentage_improvement.isna().all()
    assert Counter(classes[classes.budget==4].groupby('label').videos.first())==Counter({5:20})
    json.dumps({'aggregate':a,'intervals':ci},allow_nan=False)
    checks['whole_video_bootstrap_reproducible_shared_indices_independent_loop']=True
    checks['zero_denominators_missing_with_reasons_not_fabricated_zero']=True
    checks['balanced_class_descriptives_and_finite_scientific_JSON']=True
    return {'passed':True,'scope':'synthetic software fixtures only; never acquisition/collection/evaluation evidence','checks':checks}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-run',type=Path,default=Path('results/budget_aware_refresh_v2'))
    args=parser.parse_args()
    protocol=json.loads((args.frozen_run/'fresh_test_protocol.json').read_text(encoding='utf-8'))
    result=software_checks(protocol)
    print(f"Additional acquisition/statistics software checks PASS: {len(result['checks'])}")
