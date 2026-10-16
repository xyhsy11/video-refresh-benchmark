"""Create an isolated add-on environment, preserving the tested CUDA stack."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from temporal_refresh_common import ASSETS, ROOT, REVISION, CHECKPOINT_URL, Budget, read, save, sha

PACKAGES={'fvcore':'0.1.5.post20221221','iopath':'0.1.10','yacs':'0.1.8',
          'termcolor':'3.1.0','portalocker':'3.2.0','einops':'0.8.1','tqdm':'4.67.1',
          'tabulate':'0.9.0','PyYAML':'6.0.3'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/temporal_refresh_generalization_v1')
    parser.add_argument('--max-download-gib',type=float,default=20); args=parser.parse_args()
    output=ROOT/args.output; output.mkdir(parents=True,exist_ok=True)
    budget=Budget(output/'download_accounting.json',int(args.max_download_gib*1024**3))
    env=ROOT/'.venv_temporal'; python=env/'Scripts/python.exe'
    if not python.exists(): subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
    # Reuse binary packages without copying/upgrading/reinstalling historical CUDA.
    site=env/'Lib/site-packages'; old=ROOT/'.venv/Lib/site-packages'
    (site/'historical_cuda_stack.pth').write_text(str(old.resolve())+'\n',encoding='utf-8')
    sourcezip=budget.download(f'https://codeload.github.com/facebookresearch/pytorchvideo/zip/{REVISION}',ASSETS/'pytorchvideo.zip')
    upstream=ASSETS/f'pytorchvideo-{REVISION}'
    if not upstream.exists():
        with zipfile.ZipFile(sourcezip) as archive: archive.extractall(ASSETS)
    (site/'pinned_pytorchvideo.pth').write_text(str(upstream.resolve())+'\n',encoding='utf-8')
    identities=[]
    for name,version in PACKAGES.items():
        metadata=budget.download(f'https://pypi.org/pypi/{name}/{version}/json',ASSETS/f'dependencies/{name}-{version}.json')
        choices=read(metadata)['urls']
        compatible=[e for e in choices if e['filename'].endswith('py3-none-any.whl') or
                    ('cp312-cp312-win_amd64.whl' in e['filename'])]
        item=compatible[0] if compatible else next(e for e in choices if e['packagetype']=='sdist')
        path=budget.download(item['url'],ASSETS/'dependencies'/item['filename'])
        assert sha(path)==item['digests']['sha256']
        subprocess.run([str(python),'-m','pip','install','--no-deps','--no-build-isolation','--no-index',str(path)],check=True)
        identities.append({'package':name,'version':version,'sha256':sha(path),'filename':path.name})
    checkpoint=budget.download(CHECKPOINT_URL,ASSETS/'X3D_XS.pyth')
    # Official training annotations are identity/provenance input, never videos.
    budget.download('https://s3.amazonaws.com/kinetics/400/annotations/train.csv',ASSETS/'k400_train.csv')
    save(output/'dependency_setup.json',{'python':'.venv_temporal/Scripts/python.exe',
        'historical_environment_modified':False,'packages':identities,'upstream_revision':REVISION,
        'upstream_zip_sha256':sha(sourcezip),'license':(upstream/'LICENSE').read_text(encoding='utf-8'),
        'checkpoint_url':CHECKPOINT_URL,'checkpoint_sha256':sha(checkpoint),
        'environment_note':'Isolated add-on venv references unchanged historical binary packages via local .pth; all new dependency GET bodies counted.'})

if __name__=='__main__': main()
