"""Read-only complete known-edge/group graph review; --write retains receipt."""
from pathlib import Path
import argparse
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import task_utility_observability_study as s


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    s.GUARD.install();manifest,*_=s.load_inputs();audit=s.read(s.OLD/'duplicate_and_pretraining_audit.json')
    parent={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):parent[root(a)]=root(b)
    for e in audit['edges']:join(e['a'],e['b'])
    for v in manifest['videos']:root(v['clip'])
    bygroup={}
    for name in list(parent):
        match=re.search(r'(?:^|/)(v_(.+)_g(\d+)_c\d+\.avi)$',name)
        if match:
            key=(match[2],match[3])
            if key in bygroup:join(name,bygroup[key])
            else:bygroup[key]=name
    roles={};members={}
    for v in manifest['videos']:
        r=root(v['clip']);roles.setdefault(r,set()).add(v['study_role']);members.setdefault(r,[]).append(v['clip'])
    conflicts=[dict(roles=sorted(r),selected_members=members[k]) for k,r in roles.items() if len(r)>1]
    receipt=dict(passed=not conflicts,known_duplicate_edges=len(audit['edges']),complete_edge_graph_nodes=len(parent),
        selected_clips=len(manifest['videos']),cross_role_components=conflicts,
        transitive_bridges_through_nonselected_members_included=True,group_links_class_scoped=True,
        original_duplicate_audit_sha256=s.sha(s.OLD/'duplicate_and_pretraining_audit.json'),
        unknown_parent_pretraining_mappings_unresolved=True,source_code_sha256=s.sha(Path(__file__)))
    if args.write:s.save(s.OUT/'full_duplicate_component_validation.json',receipt)
    print(json.dumps(receipt,indent=2));assert not conflicts


if __name__=='__main__':main()
