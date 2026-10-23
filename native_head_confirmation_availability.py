"""Pinned archive population membership, without model or outcome access."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import shutil
import subprocess

from native_head_value_confirmation import read, save, sha, rank

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/native_head_value_confirmation_v1'
CACHE = ROOT/'data/source/native_head_value_confirmation_v1'
VERSION = 'archive_available_v1'
REVIEWED = 'd97802601500101a94c91de2a546769e9fae29ab'
_indexes = {}


def freeze():
    """Preserve the actual unexposed attempt before changing derived records."""
    from native_head_confirmation_state import totals, metered, snapshot_accounting
    from run_native_head_value_confirmation import event, now
    if (OUT/'population_rule.json').exists():
        return read(OUT/'population_rule.json')
    if (OUT/'evaluation_seal.json').exists() or (OUT/'evaluation_exposure.json').exists():
        raise RuntimeError('Population amendment forbidden after cohort sealing/exposure')
    counts=totals()
    if counts['encoder_starts'] or counts['probe_fits'] or counts['uncertain_encoder_calls']:
        raise RuntimeError('Population amendment requires preserved pre-model state')
    with metered('cpu','archive-population-amendment-freeze'):
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        files={}
        history=CACHE/'attempt_history'/REVIEWED
        paths=list(OUT.rglob('*'))+[ROOT/n for n in read(OUT/'frozen_config.json')['implementation']]
        paths += list((CACHE/'catalogs').glob('*.json'))+[CACHE/'candidate_manifest.json']
        for p in paths:
            if not p.is_file() or p.name=='run.lock':continue
            name=p.relative_to(ROOT).as_posix()
            target=history/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(p,target)
            if sha(p)!=sha(target):raise RuntimeError('Blocked attempt snapshot identity differs: '+name)
            files[name]=dict(sha256=sha(p),bytes=p.stat().st_size,snapshot=target.relative_to(ROOT).as_posix())
        save(OUT/'blocked_d978026/snapshot_identity.json',dict(reviewed_commit=REVIEWED,actual_head=head,
            at_utc=now(),files=files,counters=counts,exposure_absent=True,
            recovery='Immutable Git commit plus byte-identical local snapshots; old failures remain failures.'))
        inventory=read(OUT/'archive_inventory.json')
        releases={}
        for split,expected in [('val',20),('test',39)]:
            p=CACHE/'metadata'/f'{split}_archives'
            urls=[u.strip() for u in p.read_text().splitlines() if u.strip()]
            rows=[r for r in inventory if r['split']==split]
            if len(urls)!=expected or len(set(urls))!=len(urls) or set(urls)!={r['url'] for r in rows} or len(rows)!=len(urls):
                raise RuntimeError('Pinned archive list/inventory is incomplete or duplicated')
            releases[split]=dict(list_sha256=sha(p),urls=urls,representations={r['url']:r for r in rows},
                annotation_sha256=sha(CACHE/'metadata'/f'{split}_annotations'))
        legacy={p.name:sha(p) for p in (CACHE/'catalogs').glob('*.json')}
        rule=dict(version=VERSION,at_utc=now(),reviewed_commit=REVIEWED,actual_head=head,
            previous_config_sha256=sha(OUT/'frozen_config.json'),snapshot_sha256=sha(OUT/'blocked_d978026/snapshot_identity.json'),
            releases=releases,legacy_complete_catalog_sha256=legacy,
            legacy_enumeration_evidence=dict(commit=REVIEWED,implementation_sha256=files['native_head_confirmation_inputs.py']['sha256'],
                procedure='catalog_archive streamed every regular tar member and drained gzip through CRC before atomic catalog save; reviewed committed catalog identities pinned here'),
            matching='Exact official youtube_id and integer start/end filename; all duplicate occurrences must agree in encoded SHA256 and byte count; all annotation mappings must be unambiguous',
            population='Eligible archive-available official K400 validation/test clips in the fixed 400 native classes',
            absence='Zero matches only after every unique pinned archive representation has a verified complete catalog',
            uncertainty='Availability is not assumed missing at random; inaccessible annotations are outside the target population',
            exposure_before_amendment=counts)
        save(OUT/'population_rule.json',rule)
        event('population_amendment_frozen',version=VERSION,sha256=sha(OUT/'population_rule.json'),
            reviewed_commit=REVIEWED,encoder_starts=0,eval_exposure=False)
    snapshot_accounting()
    return rule


def index(split, refresh=False):
    if split in _indexes and not refresh:return _indexes[split]
    rule=read(OUT/'population_rule.json')
    if rule['version']!=VERSION:raise RuntimeError('Unknown population rule')
    release=rule['releases'][split]
    if sha(CACHE/'metadata'/f'{split}_archives')!=release['list_sha256'] or sha(CACHE/'metadata'/f'{split}_annotations')!=release['annotation_sha256']:
        raise RuntimeError('BLOCKED_INPUT: pinned release metadata changed')
    urls=[u.strip() for u in (CACHE/'metadata'/f'{split}_archives').read_text().splitlines() if u.strip()]
    if len(set(urls))!=len(urls) or urls!=release['urls'] or set(urls)!=set(release['representations']):
        raise RuntimeError('BLOCKED_INPUT: pinned release list changed/duplicated')
    members=defaultdict(list);coverage=[]
    for url in urls:
        inv=release['representations'][url];p=CACHE/'catalogs'/(rank('archive',url)+'.json')
        evidence=dict(url=url,etag=inv['etag'],bytes=inv['content_length'],complete=False)
        if p.exists():
            c=read(p);digest=sha(p)
            expected=inv.get('cached_sha256')
            if not expected:
                receipt=CACHE/'scratch'/split/(url.rsplit('/',1)[-1]+'.transfer.json')
                if receipt.exists():
                    t=read(receipt)
                    if t.get('status')=='complete' and t.get('url')==url and t.get('etag')==inv['etag'] and t.get('expected_bytes')==inv['content_length']:
                        expected=t['sha256'];evidence['transfer_sha256']=sha(receipt)
            enumeration=(rule['legacy_complete_catalog_sha256'].get(p.name)==digest or
                c.get('enumeration')==dict(complete=True,gzip_crc_drained=True,etag=inv['etag'],bytes=inv['content_length'],list_sha256=release['list_sha256']))
            if expected and c.get('url')==url and c.get('archive_sha256')==expected and enumeration:
                evidence.update(complete=True,catalog_sha256=digest,archive_sha256=expected,members=len(c['members']))
                for member in c['members']:
                    members[member['filename']].append(dict(member,url=url,catalog_sha256=digest,archive_sha256=expected))
            else:evidence['reason']='stale_or_unverified_catalog'
        else:evidence['reason']='catalog_not_yet_complete'
        coverage.append(evidence)
    complete=all(r['complete'] for r in coverage)
    with (CACHE/'metadata'/f'{split}_annotations').open(encoding='utf-8-sig',newline='') as f:annotations=list(csv.DictReader(f))
    labels=defaultdict(set)
    for a in annotations:labels[filename(a)].add(a['label'])
    rows={}
    for a in annotations:
        name=filename(a);origins=sorted(members.get(name,[]),key=lambda r:(r['url'],r['member']))
        identities={(r['encoded_sha256'],r['bytes']) for r in origins}
        ambiguous=len(identities)>1 or len(labels[name])!=1
        state='UNKNOWN' if ambiguous else 'PRESENT' if origins else 'CONFIRMED_ABSENT' if complete else 'UNKNOWN'
        rows[name]=dict(filename=name,label=a['label'],state=state,
            reason='conflicting_member_or_annotation_identity' if ambiguous else 'verified_member' if origins else 'zero_members_in_complete_pinned_release' if complete else 'incomplete_release_coverage',
            origins=origins)
    result=dict(version=VERSION,split=split,population_rule_sha256=sha(OUT/'population_rule.json'),
        complete=complete,coverage=coverage,rows=rows,states=dict(Counter(r['state'] for r in rows.values())))
    _indexes[split]=result
    return result


def filename(a):
    return f"{a['youtube_id']}_{int(a['time_start']):06d}_{int(a['time_end']):06d}.mp4"


def membership(split, name):
    return index(split)['rows'].get(name,dict(filename=name,state='UNKNOWN',reason='not_an_official_annotation',origins=[]))


def selectable(row):
    return membership(row['split'],row['filename'])['state']!='CONFIRMED_ABSENT'


def remaining_release_transfer(split):
    release=read(OUT/'population_rule.json')['releases'][split]
    return sum(release['representations'][r['url']]['content_length'] for r in index(split)['coverage']
        if not r['complete'] and not release['representations'][r['url']].get('cached_path'))


def publish():
    """Publish all-annotation availability; no RGB or outcome evaluation."""
    reports={}
    for split in ('val','test'):
        obj=index(split,refresh=True)
        save(OUT/f'{split}_archive_availability.json.gz',obj)
        counts=defaultdict(Counter)
        for row in obj['rows'].values():counts[row['label']][row['state']]+=1
        reports[split]=dict(states=obj['states'],complete=obj['complete'],
            per_class={k:dict(v) for k,v in sorted(counts.items())},availability_sha256=sha(OUT/f'{split}_archive_availability.json.gz'))
    save(OUT/'archive_population_attrition.json',dict(version=VERSION,splits=reports))
    return reports


def selection_attrition():
    """Observed per-class stages; unscreened entries remain explicitly unknown."""
    mapping=read(OUT/'class_mapping.json')['annotation_to_native']
    exclusions=read(OUT/'source_exclusion_ledger.json')
    result={}
    for split in ('val','test'):
        av=index(split);per={label:dict(y=y,annotation_records=0,PRESENT=0,CONFIRMED_ABSENT=0,UNKNOWN=0,
            source_exclusions={},screened_eligible=0,screened_ineligible=0,selected=0) for label,y in mapping.items()}
        for r in av['rows'].values():per[r['label']]['annotation_records']+=1;per[r['label']][r['state']]+=1
        for r in exclusions:
            if r['split']!=split or r['filename'] not in av['rows']:continue
            label=av['rows'][r['filename']]['label'];counts=per[label]['source_exclusions']
            counts[r['reason']]=counts.get(r['reason'],0)+1
        for p in (CACHE/'screening').glob('*.json'):
            r=read(p);name=r.get('row',{}).get('filename')
            if not name:
                # Negative receipts predate explicit filename fields; their
                # frozen filename-derived keys can still be joined exactly.
                continue
            if r['row']['split']==split and name in av['rows']:
                per[av['rows'][name]['label']]['screened_eligible' if r['eligible'] else 'screened_ineligible']+=1
        negative={p.stem for p in (CACHE/'screening').glob('*.json') if not read(p)['eligible']}
        for name,r in av['rows'].items():
            if rank('screen',split+'|'+name) in negative:per[r['label']]['screened_ineligible']+=1
        manifest=OUT/f'{split}_manifest.json.gz'
        if manifest.exists():
            for r in read(manifest):per[r['label']]['selected']+=1
        result[split]=per
    save(OUT/'per_class_population_attrition.json',dict(version=VERSION,splits=result,
        note='Screening counts are actual observed candidate receipts, not screening of every annotation; selected does not imply model exposure.'))
    changes=read(OUT/'population_selection_changes.json')
    history=CACHE/'attempt_history'/REVIEWED
    old=read(history/'data/source/native_head_value_confirmation_v1/candidate_manifest.json')
    current=read(CACHE/'candidate_manifest.json')
    canonical={}
    for label,rows in [('before',old),('after',current)]:
        groups=defaultdict(list)
        for r in rows:groups[r['split'],r['component']].append(r)
        canonical[label]={sp+'|'+c:min(v,key=lambda r:r['canonical_rank'])['filename'] for (sp,c),v in groups.items()}
    changes['canonical_changes']=[dict(component=k,before=canonical['before'].get(k),after=canonical['after'].get(k))
        for k in sorted(set(canonical['before'])|set(canonical['after'])) if canonical['before'].get(k)!=canonical['after'].get(k)]
    changes['prefixes']={}
    from native_head_confirmation_inputs import reserve_members
    for split in ('val','test'):
        prior=history/'results/native_head_value_confirmation_v1'/f'{split}_prefix_certificate.json'
        active=OUT/f'{split}_prefix_certificate.json'
        changes['prefixes'][split]=dict(prior_sha256=sha(prior) if prior.exists() else None,
            active_sha256=sha(active) if active.exists() else None,
            prior_certified=read(prior).get('complete',False) if prior.exists() else False,
            active_certified=read(active).get('complete',False) if active.exists() else False)
        m=8 if split=='val' else read(OUT/'precision_plan.json')['selected_m'] if (OUT/'precision_plan.json').exists() else None
        if m is not None:
            reserves=reserve_members(split,m);save(OUT/f'{split}_prospective_reserves.json.gz',reserves)
            changes['prefixes'][split].update(per_class=m,reserve_extra=2,reserve_manifest_sha256=sha(OUT/f'{split}_prospective_reserves.json.gz'),reserve_rows=len(reserves))
    save(OUT/'population_selection_changes.json',changes)


def remaining_work_projection(m, events):
    """Explicit workload projections from actual completed fitting/input receipts."""
    starts={r['operation_id']:i for i,r in enumerate(events) if r['kind']=='operation_start'}
    inventory={r['url']:r for r in read(OUT/'archive_inventory.json')}
    catalog_seconds=0.;catalog_bytes=0
    for i,r in enumerate(events):
        if r['kind']!='operation_finish' or not r.get('success') or r.get('label')!='acquire-val':continue
        segment=events[starts[r['operation_id']]:i]
        urls={e['url'] for e in segment if e['kind']=='archive_catalog_complete'}
        if not urls:continue
        transfer=sum(e['acquisition_seconds'] for e in segment if e['kind']=='archive_transfer')
        catalog_seconds+=max(0,r['collection_seconds']-transfer)
        catalog_bytes+=sum(inventory[url]['content_length'] for url in urls)
    remaining_bytes=sum(inventory[r['url']]['content_length'] for r in index('test')['coverage'] if not r['complete'])
    screened=0;screened_bytes=0
    for p in (CACHE/'screening').glob('*.json'):
        receipt=read(p)
        if receipt.get('row',{}).get('split')=='val':
            screened+=1;screened_bytes+=receipt['row']['encoded_bytes']
    screen_seconds=sum(r.get('collection_seconds',0) for r in events if r['kind']=='operation_finish' and r.get('label')=='certify-val')
    if not catalog_bytes or not screened:raise RuntimeError('BLOCKED_RESOURCE: real catalog/screen projection receipts missing')
    # Actual certification wall includes cache authentication, failed attempts
    # and conservatively recovered interruption time. No counter is subtracted.
    test_screened=sum(read(p).get('row',{}).get('split')=='test' for p in (CACHE/'screening').glob('*.json'))
    technical=max(0,(m+2)*400-test_screened)*screen_seconds/screened*1.5
    catalog=remaining_bytes*catalog_seconds/catalog_bytes*1.5
    duplicate=sum(r.get('collection_seconds',0) for r in events if r['kind']=='operation_finish' and r.get('label')=='input-duplicate-screen')
    scalar=sum(r.get('cpu_seconds',0) for r in events if r['kind']=='operation_finish' and r.get('label')=='smoke-scalar-verification')
    audits=sum(r.get('cpu_seconds',0) for r in events if r['kind']=='operation_finish' and r.get('label') in ('independent-verification','independent-fitting-source-check'))
    return dict(test_screening_projection_seconds=technical,test_catalog_projection_seconds=catalog,
        test_duplicate_projection_seconds=duplicate*4,independent_verification_projection_seconds=scalar/8*(400*m)*1.5+audits*2,
        measured_fitting_certification_seconds=screen_seconds,measured_eligible_fitting_screens=screened,
        measured_successful_catalog_nontransfer_seconds=catalog_seconds,measured_catalog_archive_bytes=catalog_bytes,
        safety_factor=1.5,projected_test_original_bytes=max(0,(m+2)*400-len(list((CACHE/'videos/test').glob('*.mp4'))))*screened_bytes/screened*1.3,
        projection_scope='Measured input-only/genuine fitting work extrapolated conservatively; projections are not observed test measurements. Reference collection separately includes actual decode, hashing, RGB preprocessing, GPU synchronization and feature persistence.')


if __name__=='__main__':
    freeze()
    from native_head_confirmation_state import metered
    with metered('cpu','publish-archive-availability'):print({s:r['states'] for s,r in publish().items()})
