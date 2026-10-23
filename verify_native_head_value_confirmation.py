"""Independent read-only empirical/identity verifier. No model, fit or generator.

Scalar schedule/branch calculations below do not call the runner's metric or
branch implementations. A blocked preflight can pass identity verification
without passing the empirical study or being relabeled COMPLETE.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent


def read(path):
    import gzip
    path=Path(path)
    if path.name.endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def scalar_loss(z, schedule, label):
    cached = 0
    totals = [[] for _ in range(400)]
    for t in range(16):
        if t in schedule:
            cached = t
        for c in range(400):
            totals[c].append(float(z[cached,c])/16)
    pooled = [math.fsum(values) for values in totals]
    maximum = max(pooled)
    ce = maximum+math.log(math.fsum(math.exp(v-maximum) for v in pooled))-pooled[label]
    return ce, max(range(400),key=lambda c:pooled[c])


def independent_repair(t, action):
    chosen = [u for u in (0,4,8,12) if u < t]
    if action:
        chosen.append(t)
    for u in range(t+1,16):
        left = 4-len(chosen)
        anchors = sum(v>=u for v in (4,8,12))
        if left and ((u in (4,8,12) and anchors<=left) or (u not in (4,8,12) and anchors<left)):
            chosen.append(u)
    return tuple(chosen)


def independent_grid_check(row):
    from bisect import bisect_right
    from fractions import Fraction
    pts=[float(int(p['pts'])*Fraction(p['time_base'])) for p in row['pts']]
    if any(x!=p['seconds'] for x,p in zip(pts,row['pts'])) or len(pts)<16 or any(b<=a for a,b in zip(pts,pts[1:])):
        return False
    offsets=(47/30,32/30,16/30,0.)
    valid=[]
    for i,u in enumerate(pts):
        indices=[bisect_right(pts,u-o+1e-9)-1 for o in offsets]
        if min(indices)>=0 and len(set(indices))==4 and indices[-1]==i:valid.append(i)
    if len(valid)<16:return False
    wanted=np.linspace(pts[valid[0]],pts[valid[-1]],16).tolist()
    endpoints=[bisect_right(pts,u+1e-9)-1 for u in wanted]
    indices=[[bisect_right(pts,pts[i]-o+1e-9)-1 for o in offsets] for i in endpoints]
    grid=row['grid']
    return (len(set(endpoints))==16 and set(endpoints).issubset(valid) and endpoints==grid['endpoint_indices']
        and indices==grid['frame_indices'] and [[pts[j] for j in ii] for ii in indices]==grid['actual_timestamps']
        and [[pts[i]-o for o in offsets] for i in endpoints]==grid['requested_timestamps']
        and wanted==grid['desired_endpoint_timestamps'] and grid['timestamp_tolerance_seconds']==1e-9)


def empirical_source(row, feature_file, predictions, table_file=None):
    schedules = [(0,*tail) for tail in combinations(range(1,16),3)]
    if table_file is not None:
        with np.load(table_file,allow_pickle=False) as table:
            losses = table['loss'].tolist()
        if len(losses)!=455 or not np.isfinite(losses).all():
            raise ValueError('Invalid sealed schedule loss table')
    else:
        with np.load(feature_file,allow_pickle=False) as data:
            z = data['logits'].astype(float)
        losses = [scalar_loss(z,schedule,row['y'])[0] for schedule in schedules]
    periodic = losses[schedules.index((0,4,8,12))]
    result = {}
    branch = []
    for t in range(1,13):
        branch.append([losses[schedules.index(independent_repair(t,a))] for a in (0,1)])
    for view,pred in predictions.items():
        action = [1 if p>1e-12 else 0 if p< -1e-12 else int(t in (4,8,12)) for t,p in enumerate(pred,1)]
        result['V_'+view] = math.fsum(periodic-branch[t][a] for t,a in enumerate(action))/12
    result['primary'] = result['V_HGH']-result['V_HG']
    result['opportunity'] = math.fsum(periodic-min(pair) for pair in branch)/12
    return result,losses


def verify_source_certificate(out,scope='fitting'):
    """Rebuild graph connectivity and role sets; never trust success booleans."""
    import gzip
    import csv
    out=Path(out);errors=[]
    cert=read(out/f'{scope}_source_certificate.json')
    if cert.get('input_audit_implementation_sha256')!=digest(ROOT/'native_head_confirmation_inputs.py'):
        errors.append('Source audit implementation changed; pre-exposure recertification required')
    for name,expected in cert['sha256'].items():
        if digest(ROOT/name)!=expected:errors.append('Certificate artifact changed: '+name)
    artifact=ROOT/cert['artifact_directory']
    identity=read(artifact/'source_audit_identity.json');path=ROOT/cert['graph_path']
    if digest(path)!=cert['graph_sha256']:errors.append('Source graph identity changed')
    with gzip.open(path,'rt',encoding='utf-8') as f:graph=json.load(f)
    import hashlib
    with gzip.open(ROOT/cert['candidate_path'],'rb') as f:
        if hashlib.sha256(f.read()).hexdigest()!=cert['candidate_uncompressed_sha256']:errors.append('Public candidate inventory differs from authenticated raw identity')
    parent={}
    def root(n):
        parent.setdefault(n,n)
        while parent[n]!=n:
            parent[n]=parent[parent[n]];n=parent[n]
        return n
    for edge in graph['edges']:
        ns=edge['nodes']
        if ns:
            a=root(ns[0])
            for node in ns[1:]:parent[root(node)]=a
        if not edge.get('file') or 'pointer' not in edge:errors.append('Unattributed source edge')
    reserved=set();unknown=set();affirmed=set();exposed=set()
    for record in graph['decisions']:
        ns={root(n) for n in record['nodes']}
        decision=record['decision']
        if decision=='reserved':reserved.update(ns)
        elif decision in ('unresolved','unresolved_union'):unknown.update(ns)
        elif decision=='nonreserved':affirmed.update(ns)
        elif decision in ('exposed','model_exposed'):exposed.update(ns)
    forbidden=reserved|(unknown-affirmed)
    canonical={}
    for node in graph['component_by_node']:
        component=root(node);priority=(0 if node.startswith('yt:') else 1 if node.startswith('family:') else 2,node)
        if component not in canonical or priority<canonical[component]:canonical[component]=priority
    if any(value!=canonical[root(node)][1] for node,value in graph['component_by_node'].items()):
        errors.append('Published component IDs differ from independently reconstructed canonical identities')
    # Independent official parent/train lookup, using actual CSV rows.
    train=set()
    with (artifact/'metadata/train_annotations').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):train.add(root('yt:'+r['youtube_id']))
    evidence=read(artifact/'historical_source_evidence.json')
    for item in evidence:
        if digest(ROOT/item['path'])!=item['sha256']:errors.append('Historical input evidence changed: '+item['path'])
    archive_inventory={r['url']:r for r in read(artifact/'archive_inventory.json')}
    authentic_members={}
    for catalog_file in (artifact/'member_catalogs').glob('*.json'):
        if len(catalog_file.stem)!=64:continue
        catalog=read(catalog_file)
        if 'archive_sha256' not in catalog or 'members' not in catalog:continue
        inv=archive_inventory.get(catalog['url'])
        if inv is None:errors.append('Member catalog URL is outside official archive inventory');continue
        if inv.get('cached_path'):
            expected=inv['cached_sha256']
        else:
            receipt=artifact/'transfer_receipts'/inv['split']/(catalog['url'].rsplit('/',1)[-1]+'.transfer.json')
            if not receipt.exists():errors.append('No complete original archive transfer receipt');continue
            response=read(receipt)
            if response.get('status')!='complete' or response.get('url')!=catalog['url'] or response.get('etag')!=inv['etag'] or response.get('expected_bytes')!=inv['content_length']:
                errors.append('Archive remote-representation receipt mismatch');continue
            expected=response['sha256']
        if catalog['archive_sha256']!=expected:errors.append('Archive/member catalog integrity identity mismatch');continue
        catalog_sha=digest(catalog_file)
        for member in catalog['members']:
            authentic_members[(member['filename'],member['encoded_sha256'])]=dict(url=catalog['url'],catalog_sha256=catalog_sha)
    historical_origins=set()
    historical_names={r['path'] for r in evidence}
    for edge in graph['edges']:
        if edge['file'] in historical_names:
            clips=[n.split(':',2)[2] for n in edge['nodes'] if n.startswith('clip:kinetics:')]
            hashes=[n[6:] for n in edge['nodes'] if n.startswith('bytes:')]
            historical_origins.update((c,h) for c in clips for h in hashes)
    roles=['val'] if scope=='fitting' else ['val','test'];components={}
    for split in roles:
        rows=read(artifact/f'{split}_manifest.json.gz');quota=8 if split=='val' else read(artifact/'precision_plan.json')['selected_m']
        with (artifact/'metadata'/f'{split}_annotations').open(encoding='utf-8-sig',newline='') as f:
            official={f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}.mp4":r['label'] for r in csv.DictReader(f)}
        native_map=read(artifact/'class_mapping.json')['annotation_to_native']
        published=read(artifact/'metadata/native_classes')
        published={(n[1:-1] if n.startswith('"') and n.endswith('"') else n):int(i) for n,i in published.items()}
        if len(native_map)!=400 or any(published.get(name)!=index for name,index in native_map.items()):
            errors.append('Certificate native class indices differ from authenticated published ordering')
        counts={c:0 for c in range(400)};seen=set()
        candidate=read(ROOT/cert['candidate_path'])
        groups={}
        for r in candidate:
            if r['split']==split:
                if r['component']!=canonical[root('yt:'+r['youtube_id'])][1]:errors.append('Candidate component/rank identity mismatch')
                expected_rank=hashlib.sha256(f'native_head_value_confirmation_v1|20261023|canonical|{r["filename"]}'.encode()).hexdigest()
                if r['canonical_rank']!=expected_rank:errors.append('Candidate canonical rank differs from frozen seed')
                groups.setdefault(r['component'],[]).append(r)
        selected={(r['component'],r['filename']) for r in rows}
        prefix=read(artifact/f'{split}_prefix_certificate.json')
        prefix_rows={(r['component'],r['filename']):r['reason'] for r in prefix['prefix']}
        # Every preceding component whose canonical class remains relevant has
        # a resolved technical receipt; irrelevant lower ranks need no bytes.
        order=lambda v:hashlib.sha256(f'native_head_value_confirmation_v1|20261023|component|{v}'.encode()).hexdigest()
        import hashlib
        earlier={c:0 for c in range(400)}
        for component,members in sorted(groups.items(),key=lambda kv:order(kv[0])):
            if all(earlier[r['y']]>=quota for r in members):continue
            chosen=None
            for r in sorted(members,key=lambda r:r['canonical_rank']):
                reason=prefix_rows.get((component,r['filename']))
                if reason=='technical_ineligible':
                    key=hashlib.sha256(f'native_head_value_confirmation_v1|20261023|screen|{split}|{r["filename"]}'.encode()).hexdigest()
                    receipt=read(ROOT/'data/source/native_head_value_confirmation_v1/screening'/(key+'.json'))
                    if receipt['eligible']:errors.append('Eligible preceding source was skipped')
                    continue
                if reason not in ('selected','canonical_class_already_full'):
                    errors.append('Unresolved preceding canonical candidate: '+r['filename']);break
                chosen=r;break
            if chosen and earlier[chosen['y']]<quota:
                if (component,chosen['filename']) not in selected:errors.append('Selected cohort is not the deterministic complete prefix')
                earlier[chosen['y']]+=1
            if min(earlier.values())>=quota:break
        for row in rows:
            component=root('yt:'+row['youtube_id'])
            if row['component']!=canonical[component][1] or row['youtube_id']!=row['filename'][:11]:errors.append('Selected component/parent identity mismatch')
            if official.get(row['filename'])!=row['label'] or native_map.get(row['label'])!=row['y']:
                errors.append('Selected source label/native index differs from official annotation')
            if component in seen:errors.append('Shared selected source component')
            seen.add(component);counts[row['y']]+=1
            if component in forbidden or component in train or (split=='test' and component in exposed):
                errors.append('Forbidden source role: '+row['filename'])
            if root('bytes:'+row['encoded_sha256'])!=component:errors.append('Selected encoded identity disconnected')
            if (row['filename'],row['encoded_sha256']) not in authentic_members and (row['filename'],row['encoded_sha256']) not in historical_origins:
                errors.append('Selected bytes have no authenticated archive or historical input origin: '+row['filename'])
            if digest(ROOT/row['path'])!=row['encoded_sha256']:errors.append('Selected video identity changed')
            if not independent_grid_check(row):errors.append('Independent real PTS/grid identity mismatch: '+row['filename'])
        if set(counts.values())!={quota}:errors.append('All400 source quotas not satisfied')
        components[split]=seen
    if scope=='joint' and components['val']&components['test']:errors.append('Cross-role source collision')
    return dict(passed=not errors,errors=errors,independent_connectivity=True,evidence_files=len(evidence),
        selected_sources={s:len(v) for s,v in components.items()},authenticated_archive_members=len(authentic_members),no_model_calls=True)


def acquisition_inventory(out,write=False):
    """Authenticate real retained RGB against complete validated member catalogs."""
    cache=ROOT/'data/source/native_head_value_confirmation_v1'
    inventory={r['url']:r for r in read(out/'archive_inventory.json')}
    members={};errors=[];catalogs=[]
    for p in sorted((cache/'catalogs').glob('*.json')):
        catalog=read(p);inv=inventory.get(catalog['url'])
        if inv is None:errors.append('Unofficial member catalog URL');continue
        if inv.get('cached_path'):
            expected=inv['cached_sha256']
        else:
            receipt=cache/'scratch'/inv['split']/(catalog['url'].rsplit('/',1)[-1]+'.transfer.json')
            if not receipt.exists():errors.append('Missing original archive transfer receipt');continue
            response=read(receipt)
            if response.get('status')!='complete' or response.get('url')!=catalog['url'] or response.get('etag')!=inv['etag'] or response.get('expected_bytes')!=inv['content_length']:
                errors.append('Original archive receipt identity mismatch');continue
            expected=response['sha256']
        if catalog['archive_sha256']!=expected:errors.append('Archive/member identity mismatch');continue
        catalogs.append(dict(url=catalog['url'],members=len(catalog['members']),archive_sha256=expected,catalog_sha256=digest(p)))
        for member in catalog['members']:
            members[inv['split'],member['filename']]=dict(member,url=catalog['url'],archive_sha256=expected)
    rows=[]
    for path in sorted((cache/'videos').glob('*/*.mp4')):
        key=path.parent.name,path.name;origin=members.get(key)
        actual=digest(path)
        if origin is None or actual!=origin['encoded_sha256'] or path.stat().st_size!=origin['bytes']:
            errors.append('Uncertified or changed retained original: '+path.name);continue
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),filename=path.name,split=path.parent.name,
            youtube_parent=path.name[:11],encoded_sha256=actual,bytes=path.stat().st_size,
            archive_url=origin['url'],archive_sha256=origin['archive_sha256'],role='input-only candidate/reserve, not a certified fitting/evaluation cohort'))
    screen=[]
    for path in (cache/'screening').glob('*.json'):screen.append(read(path))
    result=dict(passed=not errors,errors=errors,validated_archives=len(catalogs),
        cataloged_original_members=sum(c['members'] for c in catalogs),retained_originals=len(rows),
        retained_original_bytes=sum(r['bytes'] for r in rows),screened_inputs=len(screen),
        technically_eligible_receipts=sum(bool(r['eligible']) for r in screen),
        technically_ineligible_receipts=sum(not r['eligible'] for r in screen),
        publisher_sha256_available=False,hash_scope='Locally computed integrity identities of official HTTPS archives and their actual members',
        no_encoder_calls=True,scope='Acquisition and partial technical-input verification only; no all-class membership/isolation certificate')
    if write:
        from native_head_value_confirmation import save
        save(out/'input_byte_inventory.json.gz',rows)
        save(out/'acquisition_inventory_verification.json',dict(result,archives=catalogs))
    return result


def verify(out, write=False):
    out = Path(out)
    checks = {}
    errors = []
    for name,expected in read(out/'historical_fingerprint.json').items():
        if not (ROOT/name).is_file() or digest(ROOT/name)!=expected:
            errors.append('Historical file changed/missing: '+name)
    checks['historical_files'] = len(read(out/'historical_fingerprint.json'))
    config = read(out/'frozen_config.json')
    for name,expected in config['implementation'].items():
        if digest(ROOT/name)!=expected:
            errors.append('Configuration code identity changed: '+name)
    model = read(out/'model_identity.json')
    if digest(ROOT/model['path'])!=model['expected_sha256']:
        errors.append('Checkpoint hash differs')
    for name,expected in model.get('upstream_source_hashes',{}).items():
        if digest(ROOT/name)!=expected:errors.append('Pinned native model/preprocessing source changed: '+name)
    checks['checkpoint_authenticated'] = not any('Checkpoint' in e for e in errors)
    mapping_path = out/'class_mapping.json'
    if mapping_path.exists():
        mapping = read(mapping_path)['annotation_to_native']
        if len(mapping)!=400 or set(mapping.values())!=set(range(400)):
            errors.append('Class mapping not bijective')
        checks['native_classes'] = len(mapping)
        native_path=ROOT/'data/source/native_head_value_confirmation_v1/metadata/native_classes'
        official=read(native_path)
        published={(name[1:-1] if name.startswith('"') and name.endswith('"') else name):int(index) for name,index in official.items()}
        aliases=config.get('aliases',{})
        if any(published.get(aliases.get(name,name))!=index for name,index in mapping.items()):
            errors.append('Native index ordering differs from published class metadata')
        if read(mapping_path)['native_source_sha256']!=digest(native_path):errors.append('Native class-map source identity differs')
    if (out/'native_smoke_verification.json').exists():
        import torch
        state=torch.load(ROOT/model['path'],map_location='cpu',weights_only=False)['model_state']
        weight=state['blocks.5.proj.weight'].numpy().astype(np.float64)
        bias=state['blocks.5.proj.bias'].numpy().astype(np.float64)
        maximum=0.
        for row in read(out/'smoke_manifest.json'):
            import hashlib
            key=hashlib.sha256(f'native_head_value_confirmation_v1|20261023|feature|{row["filename"]}'.encode()).hexdigest()
            with np.load(ROOT/'data/source/native_head_value_confirmation_v1/features'/(key+'.npz'),allow_pickle=False) as saved:
                f=saved['features'].astype(np.float64);z=saved['logits'].astype(np.float64)
            predicted=f@weight.T+bias
            maximum=max(maximum,float(abs(predicted-z).max()))
            if not np.allclose(predicted,z,atol=2e-5,rtol=2e-5):errors.append('Independent genuine feature/native-head relationship mismatch')
        checks['native_smoke_affine_max_error']=maximum
    for identity in (ROOT/'data/source/native_head_value_confirmation_v1/metadata').glob('*.identity.json'):
        obj = read(identity)
        path = ROOT/obj['path']
        if digest(path)!=obj['sha256'] or path.stat().st_size!=obj['bytes']:
            errors.append('Official metadata identity differs: '+str(path.name))
    if (out/'archive_inventory.json').exists():
        archives = read(out/'archive_inventory.json')
        resources = read(out/'resource_plan.json')
        total = sum(r['content_length'] for r in archives if r['content_length'] is not None and not r['cached_path'])
        if total!=resources['conservative_new_complete_route_bytes']:
            errors.append('Archive full-transfer accounting differs')
        checks['archives_priced'] = len(archives)
        checks['complete_uncached_route_bytes'] = total
        acquired=acquisition_inventory(out,write)
        checks['acquisition_inventory']=acquired;errors.extend(acquired['errors'])
    original_ledger=out/'preparation_a9e1942/resource_exposure_ledger.jsonl'
    if original_ledger.exists() and not (out/'resource_exposure_ledger.jsonl').read_bytes().startswith(original_ledger.read_bytes()):
        errors.append('Original exposure/resource ledger prefix was modified')
    if (out/'source_audit_identity.json').exists():
        audit=read(out/'source_audit_identity.json')
        if digest(ROOT/audit['path'])!=audit['sha256']:errors.append('Input-only source graph identity changed')
        if audit.get('public_snapshot') and digest(ROOT/audit['public_snapshot'])!=audit['sha256']:errors.append('Public input-only graph snapshot differs')
        checks['source_audit_edges']=audit['edges'];checks['source_evidence_files']=audit['evidence_files']
    status = read(out/'status.json')
    for scope in ('fitting','joint'):
        if (out/f'{scope}_source_certificate.json').exists():
            source_check=verify_source_certificate(out,scope)
            checks[scope+'_source_certificate']=source_check
            errors.extend(source_check['errors'])
    empirical = (out/'evaluation_exposure.json').exists()
    if not empirical:
        if status['scientific']!='NOT_EVALUATED' or status.get('primary_estimate') is not None:
            errors.append('Unexposed study mislabeled or contains an empirical estimate')
        if any((out/n).exists() for n in ('summary.json','eval_source_summaries.json','bootstrap_draws.npz')):
            errors.append('Unexpected outcome artifacts before recorded exposure')
        checks['new_eval_exposure_absent'] = True
    else:
        seal = read(out/'evaluation_seal.json')
        for name,expected in seal['sha256'].items():
            if digest(ROOT/name)!=expected:
                errors.append('Evaluation seal mismatch: '+name)
        if not (out/'eval_source_summaries.json').exists():
            errors.append('Empirical stage incomplete')
        else:
            # Use saved state predictions as evidence, never refit/reconstruct.
            states = out/'eval_state_predictions.npz'
            if not states.exists():
                errors.append('Independent verification needs saved genuine state predictions')
            else:
                rows = read(out/'test_manifest.json.gz')
                summaries = read(out/'eval_source_summaries.json')
                if len({r['component'] for r in rows})!=len(rows):
                    errors.append('Shared source component, including across classes')
                with np.load(states,allow_pickle=False) as stored:
                    pp = {v:stored['advantage_'+v] for v in ('S','H','HG','HGH')}
                    if any(not np.isfinite(stored[k]).all() for k in stored.files):errors.append('Nonfinite stored state predictions')
                for i,row in enumerate(rows):
                    import hashlib
                    key = hashlib.sha256(f'native_head_value_confirmation_v1|20261023|feature|{row["filename"]}'.encode()).hexdigest()
                    fp = ROOT/'data/source/native_head_value_confirmation_v1/features'/(key+'.npz')
                    table = out/'schedule_tables'/(key+'.npz')
                    independently,_ = empirical_source(row,fp,{v:a[i*12:(i+1)*12] for v,a in pp.items()},table)
                    for key,value in independently.items():
                        if abs(value-summaries[i][key])>1e-10:
                            errors.append(f'Source scalar consequence mismatch: {i}/{key}')
                checks['independent_scalar_sources'] = len(rows)
                # Full scalar 455-schedule loss checks on preselected genuine
                # fitting smoke sources; EVAL consequences replay saved tables.
                smoke = read(out/'smoke_manifest.json')
                for row in smoke:
                    import hashlib
                    key = hashlib.sha256(f'native_head_value_confirmation_v1|20261023|feature|{row["filename"]}'.encode()).hexdigest()
                    fp = ROOT/'data/source/native_head_value_confirmation_v1/features'/(key+'.npz')
                    with np.load(fp,allow_pickle=False) as stored:
                        z = stored['logits'].astype(float)
                    with np.load(out/'fit_schedule_tables'/(key+'.npz'),allow_pickle=False) as stored:
                        losses = stored['loss']
                    scalar = [scalar_loss(z,s,row['y'])[0] for s in [(0,*tail) for tail in combinations(range(1,16),3)]]
                    if not np.allclose(losses,scalar,atol=1e-10,rtol=0):
                        errors.append('Fitting smoke scalar native loss mismatch')
                checks['scalar_smoke_sources_all455'] = len(smoke)
                summary = read(out/'summary.json')
                names = summary['columns']
                x = np.array([[r[n] for n in names] for r in summaries])
                groups = [np.array([i for i,r in enumerate(rows) if r['y']==c]) for c in range(400)]
                with np.load(out/'bootstrap_draws.npz',allow_pickle=False) as stored:
                    draws = stored['indices']
                boot = np.zeros((10000,len(names)))
                offset = 0
                vc = []
                import hashlib
                seed=int(hashlib.sha256(b'native_head_value_confirmation_v1|20261023|seed|bootstrap').hexdigest()[:16],16)
                generator=np.random.Generator(np.random.PCG64(seed))
                for group in groups:
                    picked = draws[:,offset:offset+len(group)]
                    expected_draws=generator.choice(group,size=(10000,len(group)),replace=True)
                    if not np.array_equal(picked,expected_draws):errors.append('Saved bootstrap draws differ from frozen PCG64 stream')
                    if not np.isin(picked,group).all():
                        errors.append('Bootstrap component crosses native class')
                    boot += x[picked].mean(1)/400
                    vc.append(np.var(x[group],axis=0,ddof=1)/(400**2*len(group)))
                    offset += len(group)
                theta = np.mean([x[g].mean(0) for g in groups],axis=0)
                variance = np.sum(vc,axis=0)
                denominator = np.sum([v*v/(len(g)-1) for v,g in zip(vc,groups)],axis=0)
                df = np.divide(variance**2,denominator,out=np.full_like(variance,np.inf),where=denominator>0)
                half = student_t.ppf(.975,df)*np.sqrt(variance)
                endpoints = np.quantile(boot,[.025,.975],axis=0,method='linear').T
                for actual,expected in [(theta,summary['inference']['mean']),(endpoints,summary['inference']['bootstrap95']),
                                        (np.array([theta-half,theta+half]).T,summary['inference']['analytic95'])]:
                    if not np.allclose(actual,expected,atol=1e-10,rtol=0):
                        errors.append('Independent aggregation/interval endpoint mismatch')
                checks['bootstrap_draws'] = 10000
                timing_path = out/'timing_trials.jsonl'
                if not timing_path.exists():
                    errors.append('Genuine complete-system timing absent')
                else:
                    trials = [json.loads(s) for s in timing_path.read_text().splitlines()]
                    if len(trials)!=60*7*4:
                        errors.append('Timing trial count mismatch')
                    for trial in trials:
                        expected_calls = 16 if trial['arm']=='Dense16' else 4
                        if trial['encoder_calls']!=expected_calls:
                            errors.append('Timing encoder budget mismatch')
                    paired = {(r['component'],r['repetition'],r['arm']):r for r in trials}
                    for row in read(out/'timing_manifest.json'):
                        for rep in range(-2,5):
                            base = paired[row['component'],rep,'PeriodicB4']['logits']
                            if any(paired[row['component'],rep,a]['logits']!=base for a in ('PeriodicB4_unused_HG','PeriodicB4_unused_HGH')):
                                errors.append('Timing unused-observation prediction equality fails')
    result = dict(passed=not errors,scope='identity/resource audit' if not empirical else 'independent genuine scalar replay',
        technical_status=status['technical'],scientific_status=status['scientific'],empirical_verification='NOT_RUN' if not empirical else 'REPLAYED',
        checks=checks,errors=errors,no_encoder_or_decode_calls=True,saved_feature_affine_replay=(out/'native_smoke_verification.json').exists(),synthetic_data_used=False,
        completion_note='A passed blocked-state audit is not technical completion of the experiment or scientific evidence.')
    if write:
        from native_head_value_confirmation import save
        save(out/'independent_verification.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/native_head_value_confirmation_v1')
    p.add_argument('--write',action='store_true')
    args = p.parse_args()
    result = verify(args.output,args.write)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
