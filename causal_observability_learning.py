"""New recognition head and equal-budget compact imitation learners; train/dev only."""
from __future__ import annotations
import copy
import numpy as np
import refresh_pooling_utils as pool
from causal_observability_common import *
from causal_observability_policy import *
from causal_observability_runtime import features


def class_weights(rows,classes):
    return np.array([1/(len(classes)*sum(r['label']==v['label'] for r in rows)) for v in rows])


def metrics(z,y,rows,classes):
    from scipy.special import logsumexp
    weights=class_weights(rows,classes)
    loss=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
    return dict(CE=float(weights @ loss),accuracy=float(weights @ (z.argmax(1)==y)))


def fit_head(x,y,rows,classes,alpha):
    import torch
    torch.manual_seed(SEED);torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    sw=class_weights(rows,classes);mean=sw @ x;std=np.sqrt(sw @ (x-mean)**2);scale=np.where(std<=1e-12,1,std)
    a=torch.tensor((x-mean)/scale,dtype=torch.float64);target=torch.tensor(y);weight=torch.tensor(sw)
    model=torch.nn.Linear(x.shape[1],len(classes),dtype=torch.float64)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=400,max_eval=800,history_size=50,
        tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad();loss=(torch.nn.functional.cross_entropy(model(a),target,reduction='none')*weight).sum()+alpha/2*model.weight.square().sum()
        loss.backward();calls+=1;return loss
    optimizer.step(closure);objective=float(closure().detach())
    iterations=int(optimizer.state[model.weight]['n_iter'])
    return dict(means=mean.tolist(),scales=scale.tolist(),weights=model.weight.detach().numpy().tolist(),bias=model.bias.detach().numpy().tolist()),dict(
        alpha=alpha,objective=objective,iterations=iterations,at_iteration_cap=iterations>=400,closure_calls=calls,
        gradient_max_abs=max(float(v.grad.abs().max()) for v in model.parameters()),constant_columns=int(sum(std<=1e-12)))


def recognition(manifest):
    target=OUT/'head_selection.json'
    if target.exists():
        r=read(target);verify({r['path']:r['sha256']});payload=read(ROOT/r['path'])
        if payload['manifest_sha256']!=sha(OUT/'eligible_manifest.json'):raise Blocker('Selected head population changed')
        return payload['parameters']
    classes=manifest['classes'];entries={role:[v for v in manifest['videos'] if v['role']==role] for role in ['head_train','development']}
    arrays={r:np.stack([features(v,r)[0].mean(0,dtype=float) for v in entries[r]]) for r in entries}
    labels={r:np.array([classes.index(v['label']) for v in entries[r]]) for r in entries}
    candidates=[]
    for order,alpha in enumerate([.001,.01,.1,1]):
        path=OUT/f'head/grid_{order}.json'
        if path.exists():
            cached=read(path)
            if cached['manifest_sha256']!=sha(OUT/'eligible_manifest.json'):raise Blocker('Head training population changed')
            record=cached['fit']
        else:
            head,record=fit_head(arrays['head_train'],labels['head_train'],entries['head_train'],classes,alpha)
            for r in entries:record[r]=metrics(affine(arrays[r],head),labels[r],entries[r],classes)
            record['grid_order']=order
            save(path,dict(parameters=head,fit=record,classes=classes,manifest_sha256=sha(OUT/'eligible_manifest.json')))
        candidates.append(record)
    best=min(candidates,key=lambda r:(r['development']['CE'],-r['development']['accuracy'],r['grid_order']))
    payload=read(OUT/f'head/grid_{best["grid_order"]}.json');save(OUT/'head/parameters.json',payload)
    save(OUT/'head_grid.json',candidates);save(target,dict(**best,path=rel(OUT/'head/parameters.json'),sha256=sha(OUT/'head/parameters.json')))
    return payload['parameters']


def teacher(features,head,trace,t,precomputed=None):
    """Full-information TRAINING/retrospective teacher, never called by live policies."""
    table=pool.schedules(16,4)
    if precomputed is None:
        z=affine(features,head);z=z-z.mean(1,keepdims=True)
        objective=np.square(table['cache_weights'] @ z-z.mean(0)).mean(1)
    else:objective=precomputed
    previous={v['t'] for v in trace if v['selected'] and v['t']<t}
    allowed=np.array([{int(j) for j in pos if j<t}==previous for pos in table['positions']])
    ids=np.flatnonzero(allowed)
    if not len(ids):raise ValueError('Infeasible causal prefix')
    j=int(ids[pool.first_tie(objective[ids])['index']])
    return int(t in table['positions'][j])


def examples(entry,f,thumbs,head,policies):
    rows=[]
    z=affine(f,head);z-=z.mean(1,keepdims=True)
    objective=np.square(pool.schedules(16,4)['cache_weights'] @ z-z.mean(0)).mean(1)
    for policy in policies:
        observations=[]
        # Compute the same encountered thumbnail statistics for both training views.
        # The history branch is then masked, not given future observations.
        class Collector:
            needs_cheap=True
            def decide(self,obs):return policy.decide(obs)
        result=execute(TrainingReplayProvider(f,thumbs),Collector(),head,
            on_decision=lambda obs,selected,forced:observations.append((obs,selected,forced)))
        for obs,selected,forced in observations:
            if obs.t==0 or forced or obs.remaining==0:continue
            rows.append(dict(label=entry['label'],source=entry['clip'],t=obs.t,age=obs.age,remaining=obs.remaining,
                input=vector(obs,head,True).tolist(),target=teacher(f,head,result['trace'],obs.t,objective),behavior_selected=selected))
    return rows


def train_network(rows,with_motion,epochs=(20,40,80)):
    import torch
    torch.set_num_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    x=np.array([r['input'] for r in rows]);y=np.array([r['target'] for r in rows],dtype=float)
    if not with_motion:x[:,-32:]=0
    classes=sorted({r['label'] for r in rows})
    sources={c:sorted({r['source'] for r in rows if r['label']==c}) for c in classes}
    counts=Counter((r['label'],r['source']) for r in rows)
    sw=np.array([1/(len(classes)*len(sources[r['label']])*counts[r['label'],r['source']]) for r in rows])
    mean=sw @ x;scale=np.sqrt(sw @ (x-mean)**2);scale=np.where(scale<=1e-12,1,scale)
    a=torch.tensor((x-mean)/scale,dtype=torch.float64);target=torch.tensor(y,dtype=torch.float64);weights=torch.tensor(sw,dtype=torch.float64)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(32,1,dtype=torch.float64))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.001)
    rng=np.random.Generator(np.random.PCG64(SEED));checkpoints={};log=[]
    for epoch in range(1,max(epochs)+1):
        for start in range(0,len(rows),256):
            if start==0:order=rng.permutation(len(rows))
            ids=order[start:start+256];optimizer.zero_grad()
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(a[ids]).ravel(),target[ids],reduction='none')*weights[ids]).sum()*len(rows)/len(ids)
            loss.backward();optimizer.step()
        if epoch in epochs:
            with torch.no_grad():training_loss=float((torch.nn.functional.binary_cross_entropy_with_logits(model(a).ravel(),target,reduction='none')*weights).sum())
            log.append(dict(epoch=epoch,weighted_training_BCE=training_loss,examples=len(rows)))
            checkpoints[epoch]=dict(means=mean.tolist(),scales=scale.tolist(),w1=model[0].weight.detach().numpy().tolist(),b1=model[0].bias.detach().numpy().tolist(),w2=model[2].weight.detach().numpy()[0].tolist(),b2=float(model[2].bias.detach()[0]))
    return checkpoints,log


from collections import Counter


def controllers(manifest,head):
    path=OUT/'controller_selection.json'
    if path.exists():
        identity=read(OUT/'controller_training_identity.json')
        if identity['head_sha256']!=sha(OUT/'head/parameters.json') or identity['manifest_sha256']!=sha(OUT/'eligible_manifest.json'):
            raise Blocker('Controller training head/population changed')
        verify({identity['path']:identity['sha256']});return read(path)
    training=[v for v in manifest['videos'] if v['role']=='controller_train']
    development=[v for v in manifest['videos'] if v['role']=='development']
    base=[]
    for v in training:
        f,thumbs=features(v,'controller_train')
        policies=[Policy('periodic',head,clip=v['clip'])]+[Policy('RandomB4_seeded',head,clip=v['clip']+f'/rollout{j}') for j in range(3)]
        base.extend(examples(v,f,thumbs,head,policies))
    preliminary={}
    for method in ['history','history_motion']:
        checkpoints,log=train_network(base,method=='history_motion',(40,));preliminary[method]=checkpoints[40]
        save(OUT/f'controllers/preliminary_{method}.json',dict(model=checkpoints[40],training=log))
    augmented=base.copy()
    for v in training:
        f,thumbs=features(v,'controller_train')
        policies=[Policy(m,head,preliminary[m],.35,clip=v['clip']) for m in preliminary]
        augmented.extend(examples(v,f,thumbs,head,policies))
    training_file=DATA/'controller_training_examples.json';save(training_file,dict(base_examples=len(base),rows=augmented,scope='Genuine chronological controller_train feature replay and privileged conditional teacher labels; not live inference'))
    save(OUT/'controller_training_identity.json',dict(path=rel(training_file),sha256=sha(training_file),base_examples=len(base),augmented_examples=len(augmented),sources=len(training),head_sha256=sha(OUT/'head/parameters.json'),manifest_sha256=sha(OUT/'eligible_manifest.json')))
    import pandas as pd
    pd.concat([pd.DataFrame(base).assign(stage='periodic_random'),pd.DataFrame(augmented[len(base):]).assign(stage='on_policy_augmentation')]).groupby(['stage','age','remaining']).size().rename('examples').reset_index().to_csv(OUT/'training_state_shift.csv',index=False)
    selection={};records=[]
    for method in ['history','history_motion']:
        checkpoints,training_log=train_network(augmented,method=='history_motion')
        candidates=[]
        for epoch,model in checkpoints.items():
            save(OUT/f'controllers/{method}_epoch{epoch}.json',model)
            for threshold in [.2,.35,.5]:
                logits=[]
                for v in development:
                    f,thumbs=features(v,'development')
                    logits.append(execute(TrainingReplayProvider(f,thumbs),Policy(method,head,model,threshold,v['clip']),head)['logits'])
                y=np.array([manifest['classes'].index(v['label']) for v in development])
                candidates.append(dict(epoch=epoch,threshold=threshold,grid_order=len(candidates),**metrics(np.array(logits),y,development,manifest['classes'])))
        best=min(candidates,key=lambda r:(r['CE'],-r['accuracy'],r['grid_order']))
        source=OUT/f'controllers/{method}_epoch{best["epoch"]}.json'
        selection[method]=dict(**best,path=rel(source),sha256=sha(source));records.append(dict(method=method,training=training_log,candidates=candidates))
    motion=[]
    for v in training:
        _,thumbs=features(v,'controller_train')
        motion.extend(cheap_signal(thumbs[t],thumbs[t-1])[26] for t in range(1,16))
    candidates=[]
    for quantile in [.5,.75,.9]:
        threshold=float(np.quantile(motion,quantile));logits=[]
        for v in development:
            f,thumbs=features(v,'development');logits.append(execute(TrainingReplayProvider(f,thumbs),Policy('motion',head,threshold=threshold),head)['logits'])
        y=np.array([manifest['classes'].index(v['label']) for v in development])
        candidates.append(dict(quantile=quantile,threshold=threshold,grid_order=len(candidates),**metrics(np.array(logits),y,development,manifest['classes'])))
    selection['motion']=min(candidates,key=lambda r:(r['CE'],-r['accuracy'],r['grid_order']))
    save(OUT/'controller_training_development.json',dict(learned=records,motion=candidates));save(path,selection)
    return selection
