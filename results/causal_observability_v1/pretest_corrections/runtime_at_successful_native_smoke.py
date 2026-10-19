"""Real native decoding, chronological observations, and instrumented X3D execution."""
from __future__ import annotations
import importlib.util
import io
import time
import numpy as np
from causal_observability_common import *
from causal_observability_policy import execute,affine


class Encoder:
    def __init__(self):
        import torch,pytorchvideo,zipfile
        from pytorchvideo.models.hub.x3d import x3d_xs
        from benchmark import configure_numerics
        self.torch=torch;self.numerics=configure_numerics(SEED)
        if not torch.cuda.is_available():raise Blocker('CUDA unavailable; no simulated timing permitted')
        revision='f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0'
        source=ROOT/f'data/source/temporal_refresh_generalization_v1/pytorchvideo-{revision}'
        if not Path(pytorchvideo.__file__).resolve().is_relative_to(source.resolve()):raise Blocker('Unpinned PyTorchVideo import')
        checkpoint=source.parent/'X3D_XS.pyth'
        expected='6dafc96144c88f800b190d2cb179ff69b8fd48c40afd4deca6c1e61f76fdfe86'
        if sha(checkpoint)!=expected:raise Blocker('Checkpoint SHA mismatch')
        setup=read(ROOT/'results/temporal_refresh_generalization_v1/dependency_setup.json')
        archive=source.parent/'pytorchvideo.zip'
        if sha(archive)!=setup['upstream_zip_sha256']:raise Blocker('Upstream archive mismatch')
        with zipfile.ZipFile(archive) as z:
            members=[n for n in z.namelist() if '/pytorchvideo/' in n and n.endswith('.py')]
            for n in members:
                if sha(source.parent/n)!=hashlib.sha256(z.read(n)).hexdigest():raise Blocker('Modified factory source')
        self.model=x3d_xs(pretrained=False)
        loaded=self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:raise Blocker('Strict loading failed')
        self.model.eval().to('cuda',dtype=torch.float32)
        self.model.blocks[-1].proj.register_forward_hook(self.capture)
        spec=importlib.util.spec_from_file_location('causal_pinned_video_functional',source/'pytorchvideo/transforms/functional.py')
        self.functional=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.functional)
        self.starts=self.completions=0;self.phase='setup';self.clip=None;self.log=True
        self.identity=dict(revision=revision,checkpoint=rel(checkpoint),checkpoint_sha256=expected,
            upstream_zip_sha256=sha(archive),upstream_python_files=len(members),
            preprocessing='unchanged factory; 4RGB /255 mean.45 std.225; bilinear shortside182 and crop182; 2048D preprojection mean',
            native_projection='400-class projection/averaging executes; activation=None; included in every actual call',
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),numerics=self.numerics)
        p=OUT/'encoder_identity.json'
        if p.exists() and read(p)!=self.identity:raise Blocker('Encoder identity changed')
        if not p.exists():save(p,self.identity)

    def capture(self,module,args,output):
        self.projection_input=args[0];self.projection_output=output

    def prepare(self,rgbs):
        from ucf101_proxy_inference import X3D
        return X3D.prepare(self,rgbs)

    def event(self,event,t):
        if self.log:
            with (OUT/'encoder_events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(at_utc=now(),phase=self.phase,clip=self.clip,event=event,endpoint=t))+'\n')

    def feature(self,rgbs,t):
        tensor=self.prepare(rgbs).to('cuda',non_blocking=True)
        self.starts+=1;self.event('start',t)
        with self.torch.inference_mode():
            self.native=self.model(tensor)
            f=self.projection_input.mean(dim=tuple(range(1,self.projection_input.ndim-1)))
            if f.shape!=(1,2048):raise Blocker('Wrong preprojection feature shape')
            values=f.cpu().numpy()[0].copy()
        self.torch.cuda.synchronize();self.completions+=1;self.event('complete',t)
        return values

    def native_check(self):
        from ucf101_proxy_inference import X3D
        return X3D.native_check(self)


def decode(encoded,entry):
    import av
    from research_utils import configure_decoder
    from temporal_refresh_common import snippet_grid
    from ucf101_proxy_data import presentation_order
    frames,pts,ids=[],[],[]
    with av.open(io.BytesIO(encoded)) as c:
        stream=c.streams.video[0];configure_decoder(stream)
        for i,f in enumerate(c.decode(stream)):
            if f.pts is None or f.time_base is None:raise Blocker('Missing actual PTS')
            frames.append(f);pts.append(float(f.pts*f.time_base));ids.append(dict(index=i,pts=int(f.pts),time_base=str(f.time_base)))
        frames,pts,ids,_=presentation_order(frames,pts,ids)
        grid=snippet_grid(pts);grid['frame_identities']=[[ids[j] for j in row] for row in grid['frame_indices']]
    if grid!=entry['snippet_grid'] or len(frames)!=entry['decoded_frames']:raise Blocker('Runtime PTS/identity differs from screened inputs')
    return frames,grid


class LiveProvider:
    """No embeddings are loaded. Only refresh() invokes the expensive model."""
    def __init__(self,frames,grid,encoder):
        self._frames,self._grid,self._encoder=frames,grid,encoder
        self.t=-1;self._used=False;self.forward_completions=0;self.grayscale_conversions=0;self.RGB_conversions=0
        self.features={};self.cheap_seconds=0.;self.thumbnail_resizes=0
    def advance(self,t):
        if t!=self.t+1:raise ValueError('Nonchronological candidate access')
        self.t=t;self._used=False
    def thumbnail(self):
        tick=time.perf_counter()
        f=self._frames[self._grid['endpoint_indices'][self.t]]
        image=f.reformat(width=32,height=32,format='gray').to_ndarray().copy()
        self.grayscale_conversions+=1;self.thumbnail_resizes+=1;self.cheap_seconds+=time.perf_counter()-tick
        return image
    def refresh(self):
        if self._used:raise ValueError('A current feature cannot be requested twice')
        self._used=True
        rgbs=[self._frames[j].to_ndarray(format='rgb24') for j in self._grid['frame_indices'][self.t]]
        self.RGB_conversions+=len(rgbs)
        f=self._encoder.feature(rgbs,self.t);self.forward_completions+=1;self.features[self.t]=f.copy()
        return f


def live(encoded,entry,encoder,policy,head):
    """Encoded in-memory bytes -> complete fresh decode -> completed prediction."""
    from temporal_refresh_timing import working_set
    torch=encoder.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    resident=torch.cuda.memory_allocated();working_before=working_set()
    before_start,before_complete=encoder.starts,encoder.completions
    tick=time.perf_counter();frames,grid=decode(encoded,entry)
    provider=LiveProvider(frames,grid,encoder)
    cpu_start=time.perf_counter()
    result=execute(provider,policy,head)
    decision_and_execution=time.perf_counter()-cpu_start
    torch.cuda.synchronize();result['seconds']=time.perf_counter()-tick
    result.update(decoded_frames=len(frames),grayscale_conversions=provider.grayscale_conversions,
        thumbnail_resizes=provider.thumbnail_resizes,RGB_conversions=provider.RGB_conversions,
        cheap_conversion_seconds=provider.cheap_seconds,decision_and_encoder_seconds=decision_and_execution,
        encoder_starts=encoder.starts-before_start,encoder_completions=encoder.completions-before_complete,
        resident_cuda_bytes=resident,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_cuda_bytes=max(0,torch.cuda.max_memory_allocated()-resident),
        process_working_set_before=working_before,process_working_set_after=working_set())
    # Reference checks happen after measured completed-output boundary.
    expected=affine(np.array([provider.features[j] for j in result['cache_indices']]).mean(0,dtype=np.float64),head)
    np.testing.assert_allclose(result['logits'],expected,atol=1e-10,rtol=1e-10)
    if result['encoder_starts']!=4 or result['encoder_completions']!=4:raise Blocker('Actual four-forward budget failed')
    result['refreshed_features']={str(k):v.tolist() for k,v in provider.features.items()}
    return result


def collect_dense(entries,role,encoder):
    ledger_path=OUT/f'{role}_feature_collection.json';ledger=read(ledger_path) if ledger_path.exists() else {}
    for i,e in enumerate(entries):
        key=e['label']+'/'+e['clip']
        if key in ledger:
            verify({ledger[key]['path']:ledger[key]['sha256']});continue
        if role=='test':
            seal_check(True)
            complete=read(OUT/'test_rollouts.json')
            if not complete.get('all_deployable_rollouts_complete'):raise Blocker('Dense test diagnostics prohibited before all actual policy decisions')
        raw=(ROOT/e['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise Blocker('Input bytes changed')
        frames,grid=decode(raw,e);provider=LiveProvider(frames,grid,encoder)
        encoder.phase=role+'_dense';encoder.clip=key
        f=[];thumbs=[]
        for t in range(16):
            provider.advance(t);thumbs.append(provider.thumbnail());f.append(provider.refresh())
        p=DATA/'features'/role/(rank(key)+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,features=np.asarray(f,dtype=np.float32),thumbs=np.asarray(thumbs,dtype=np.uint8))
        ledger[key]=dict(path=rel(p),sha256=sha(p),input_sha256=e['sha256'],encoder_sha256=sha(OUT/'encoder_identity.json'),starts=16,completions=16,at_utc=now())
        save(ledger_path,ledger)
        if (i+1)%25==0:print(f'Real dense {role} features {i+1}/{len(entries)}',flush=True)
    return ledger


def features(entry,role):
    r=read(OUT/f'{role}_feature_collection.json')[entry['label']+'/'+entry['clip']]
    if r['input_sha256']!=entry['sha256']:raise Blocker('Feature source mismatch')
    verify({r['path']:r['sha256']})
    with np.load(ROOT/r['path'],allow_pickle=False) as a:return a['features'].copy(),a['thumbs'].copy()
