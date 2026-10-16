"""Unmodified native X3D path with read-only pre-projection feature capture."""
from __future__ import annotations
import importlib.util
import io
import time
from collections import Counter

import av
import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms._functional_video import center_crop, normalize

from temporal_refresh_common import *
from research_utils import configure_decoder


class Encoders:
    def __init__(self,output,device='cuda'):
        from benchmark import configure_numerics
        from pytorchvideo.models.hub.x3d import x3d_xs
        if device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU substitution')
        self.output=output; self.device=torch.device(device); self.counts=Counter(); self.phase='setup'
        self.numerics=configure_numerics(SEED)
        self.rn=resnet18(weights=None)
        rn_path=ROOT/'data/source/torch_hub/checkpoints/resnet18-f37072fd.pth'
        assert sha(rn_path)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
        self.rn.load_state_dict(torch.load(rn_path,map_location='cpu',weights_only=True),strict=True)
        self.rn.fc=torch.nn.Identity(); self.rn.eval().to(device=self.device,dtype=torch.float32)
        self.preprocess=ResNet18_Weights.DEFAULT.transforms()
        self.x3d=x3d_xs(pretrained=False) # Factory crop160, clip4; no architecture overrides.
        checkpoint=ASSETS/'X3D_XS.pyth'
        identity=read(output/'dependency_setup.json')
        assert sha(checkpoint)==identity['checkpoint_sha256']
        loaded=self.x3d.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        self.x3d.eval().to(device=self.device,dtype=torch.float32)
        self.capture={}
        def capture(module,args,result):
            self.capture={'input':args[0],'preactivation':result}
        self.x3d.blocks[-1].proj.register_forward_hook(capture)
        upstream=ASSETS/f'pytorchvideo-{REVISION}'
        spec=importlib.util.spec_from_file_location('pinned_video_functional',upstream/'pytorchvideo/transforms/functional.py')
        self.functional=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.functional)
        self.audit_handle=(output/'encoder_execution_events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.log_calls=True
        self.provenance={'rn18_weights':'ResNet18_Weights.IMAGENET1K_V1(DEFAULT)','rn18_checkpoint_sha256':sha(rn_path),
            'rn18_preprocess':str(self.preprocess),'x3d_revision':REVISION,'x3d_checkpoint_sha256':sha(checkpoint),
            'x3d_factory':'x3d_xs(pretrained=False), unchanged input_clip_length4/input_crop_size160',
            'x3d_preprocessing':'uint8 RGB C,T,H,W /255; normalize mean0.45/std0.225; official bilinear ShortSideScale182 align_corners=False; torchvision CenterCropVideo182',
            'x3d_native_head':str(self.x3d.blocks[-1]),'x3d_feature':'read-only proj input hook, mean all nonbatch/nonchannel axes; pre-projection and pre-classification-activation',
            'upstream_code_hashes':{relative(p):sha(p) for p in [upstream/'pytorchvideo/models/hub/x3d.py',upstream/'pytorchvideo/models/x3d.py',upstream/'pytorchvideo/models/head.py',upstream/'pytorchvideo/transforms/functional.py']},
            'numerics':self.numerics,'upstream_license':'Apache-2.0; source/LICENSE retained; dataset/weights separate terms'}
        if not (output/'encoder_provenance.json').exists(): save(output/'encoder_provenance.json',self.provenance)
        else: assert read(output/'encoder_provenance.json')==self.provenance

    def record(self,model):
        self.counts[f'{self.phase}:{model}']+=1
        if self.log_calls:
            self.audit_handle.write(json.dumps({'phase':self.phase,'model':model,'event':'forward_start','timestamp':now()})+'\n')

    def rn_feature(self,rgb):
        tensor=self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device,dtype=torch.float32,non_blocking=True)
        self.record('resnet18'); return self.rn(tensor)

    def x3d_input(self,rgbs):
        tensor=torch.from_numpy(np.stack(rgbs)).permute(3,0,1,2).to(dtype=torch.float32)/255.
        tensor=normalize(tensor,[.45]*3,[.225]*3)
        return center_crop(self.functional.short_side_scale(tensor,182),[182,182]).unsqueeze(0)

    def x3d_feature(self,rgbs):
        tensor=self.x3d_input(rgbs).to(self.device,non_blocking=True)
        self.record('x3d_xs'); self.native=self.x3d(tensor)
        return self.capture['input'].mean(dim=tuple(range(1,self.capture['input'].ndim-1)))

    def features(self,rgbs,representation):
        if representation==REPS[0]: return self.rn_feature(rgbs[-1])
        if representation==REPS[1]: return torch.stack([self.rn_feature(rgb) for rgb in rgbs]).mean(0)
        return self.x3d_feature(rgbs)

    def native_check(self,rgbs):
        with torch.inference_mode():
            feature=self.x3d_feature(rgbs)
            head=self.x3d.blocks[-1]
            projected=self.capture['preactivation'].mean(dim=(1,2,3))
            projection_shape=list(self.capture['input'].shape)
            affine=torch.nn.functional.linear(feature,head.proj.weight,head.proj.bias)
            torch.testing.assert_close(affine,projected,atol=2e-5,rtol=2e-5)
            # Pinned factory has activation=None. Native global pooling averages
            # pre-activation400 logits, followed by softmax once outside model.
            assert head.activation is None
            torch.testing.assert_close(self.native,projected,atol=2e-5,rtol=2e-5)
            probability=torch.softmax(self.native,dim=1)
            torch.testing.assert_close(probability.sum(1),torch.ones(1,device=self.device))
            return {'strict_loading':True,'feature_dimension':feature.shape[-1],
                'projection_input_shape':projection_shape,'native_activation':None,
                'native_400_logits_shape':list(self.native.shape),'softmax_applied_once_outside':True,
                'maximum_affine_mean_difference':float((affine-projected).abs().max().cpu()),
                'native_mean_logit_difference':float((self.native-projected).abs().max().cpu()),
                'factory_crop160_input_crop182':'factory pooling kernel(4,5,5) unchanged;182 input leaves1x2x2 projection axes, averaged'}

    def verify_preprocessing(self):
        path=self.output/'preprocessing_verification.json'
        if path.exists():return
        records=[]
        for value in [0,255]:
            rgb=np.full((217,301,3),value,dtype=np.uint8)
            tensor=self.x3d_input([rgb]*4);expected=(value/255.-.45)/.225
            assert list(tensor.shape)==[1,3,4,182,182] and tensor.dtype==torch.float32
            torch.testing.assert_close(tensor,torch.full_like(tensor,expected),atol=2e-6,rtol=0)
            records.append({'uint8_RGB':value,'expected_normalized_value':expected,'observed_min':float(tensor.min()),'observed_max':float(tensor.max())})
        assert self.preprocess.mean==[.485,.456,.406] and self.preprocess.std==[.229,.224,.225]
        assert self.preprocess.crop_size==[224] and self.preprocess.resize_size==[256]
        save(path,{'X3D_constant_pixel_fixtures':records,'RN18_original_checkpoint_transform_verified':True,
            'official_recipe':'https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/',
            'native_factory_crop160_unchanged':True,'inference_short_side_and_crop':182,
            'sampling_offsets':'uniform_subsample4 on48 virtual frames -> [0,15,31,47]',
            'no_encoder_forwards_in_this_check':True})


def load_rgb_frames(encoded,entry):
    needed=set(sum(entry['snippet_grid']['frame_indices'],[])); frames={}; count=0
    with av.open(io.BytesIO(encoded),mode='r') as container:
        stream=container.streams.video[0]; configure_decoder(stream)
        for index,frame in enumerate(container.decode(stream)):
            count+=1
            if index in needed: frames[index]=frame.to_ndarray(format='rgb24')
    assert count==entry['decoded_frame_count'] and set(frames)==needed
    return frames


def collect(output,entries,encoders,phase):
    if phase=='evaluation':
        require_evaluation_seal(output)
        frozen=read(output/'evaluation_manifest.json')['videos']
        assert entries==frozen and len(entries)==100,'Collect only the full frozen cohort; no substitutions'
    encoders.phase=phase; ledger_path=output/f'{phase}_collection.json'
    ledger=read(ledger_path) if ledger_path.exists() else {'videos':{},'phase':phase}
    for i,v in enumerate(entries):
        identity=v['source_video_id']; target=ASSETS/'features'/phase/f'{identity}.npz'
        if identity in ledger['videos']:
            assert sha(target)==ledger['videos'][identity]['sha256']
            assert sha(ROOT/v['relative_path'])==v['file_identity']['sha256']
            assert ledger['videos'][identity]['input_sha256']==v['file_identity']['sha256']
            continue
        start=time.perf_counter(); raw=(ROOT/v['relative_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==v['file_identity']['sha256']
        frames=load_rgb_frames(raw,v); arrays={}; begin=dict(encoders.counts)
        with torch.inference_mode():
            for rep in REPS:
                values=[]
                for indices in v['snippet_grid']['frame_indices']:
                    # Encoder batch size remains1. Retain only small output
                    # embeddings, then copy once per representation; no
                    # per-candidate synchronization is needed for collection.
                    values.append(encoders.features([frames[idx] for idx in indices],rep))
                array=torch.cat(values,dim=0).cpu().numpy()
                assert array.shape==(16,512 if rep!=REPS[2] else 2048) and array.dtype==np.float32
                assert np.isfinite(array).all() and (np.linalg.norm(array,axis=1)>1e-12).all()
                arrays[rep]=array
        target.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(target,**arrays)
        ledger['videos'][identity]={'path':relative(target),'sha256':sha(target),'input_sha256':v['file_identity']['sha256'],
            'snippet_grid_sha256':hashlib.sha256(json.dumps(v['snippet_grid'],sort_keys=True).encode()).hexdigest(),
            'encoder_provenance_sha256':sha(output/'encoder_provenance.json'),'collected_at_utc':now(),
            'seconds':time.perf_counter()-start,'forward_calls':{k:encoders.counts[k]-begin.get(k,0) for k in encoders.counts}}
        save(ledger_path,ledger)
        if (i+1)%5==0 or i==0: print(f'{phase} reference collection {i+1}/{len(entries)}',flush=True)
    return ledger


def verify_real_isolation(output,entries,encoders):
    """Real training pixels: perturb every unobserved frame, not selected inputs."""
    path=output/'real_input_isolation.json'
    if path.exists(): return read(path)
    v=min([x for x in entries if x['recognition_role']=='train'],key=lambda x:rank('input-isolation',x['source_video_id']))
    frames=load_rgb_frames((ROOT/v['relative_path']).read_bytes(),v)
    indices=v['snippet_grid']['frame_indices'][3]
    changed={idx:(rgb.copy() if idx in indices else np.bitwise_xor(rgb,255)) for idx,rgb in frames.items()}
    original=[frames[idx] for idx in indices]; mutated=[changed[idx] for idx in indices]
    for a,b in zip(original,mutated): np.testing.assert_array_equal(a,b)
    torch.testing.assert_close(encoders.x3d_input(original),encoders.x3d_input(mutated),atol=0,rtol=0)
    phase=encoders.phase; encoders.phase='training_verification'; maximum={}
    with torch.inference_mode():
        for rep in REPS:
            a=encoders.features(original,rep)
            ledger=read(output/'train_development_collection.json')['videos'][v['source_video_id']]
            with np.load(ROOT/ledger['path']) as saved:
                np.testing.assert_array_equal(a.cpu().numpy()[0],saved[rep][3])
            # Intervening unrelated input cannot create hidden temporal state.
            encoders.features([frames[i] for i in v['snippet_grid']['frame_indices'][10]],rep)
            b=encoders.features(mutated,rep)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            maximum[rep]=float((a-b).abs().max().cpu())
    encoders.phase=phase
    value={'source_video_id':v['source_video_id'],'role':'training','candidate_ordinal':3,'selected_frame_indices':indices,
        'mutated_nonselected_frame_count':len(frames)-4,'future_and_nonselected_pixels_input_and_feature_invariant':True,
        'maximum_feature_differences':maximum,'intervening_unrelated_snippet_state_isolation':True,
        'saved_collection_outputs_exactly_reproduced':True,
        'encoder_interface':'only selected RGB arrays, no full-video/labels/future frame argument','verified_at_utc':now()}
    save(path,value); return value
