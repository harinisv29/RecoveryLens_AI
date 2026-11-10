"""Complete a protected expanded experiment after the live full download finishes."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from utils.experiments import run_paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); parser.add_argument('--torch-site',required=True); args=parser.parse_args(); paths=run_paths(args.run)
    state_path=paths['results']/'pipeline_status.json'
    active_path=ROOT/'models/active_model.json'
    old_active=active_path.read_bytes() if active_path.exists() else None
    activated=False
    def state(stage,**extra):
        state_path.write_text(json.dumps({'stage':stage,'run':args.run,**extra},indent=2)); print(stage,extra,flush=True)
    def execute(label,arguments):
        state(label)
        with (paths['results']/f'{label}.log').open('w',encoding='utf-8') as log:
            process=subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode: raise RuntimeError(f'{label} failed with exit code {process.returncode}; see {label}.log')
    try:
        state('waiting_for_full_download')
        while True:
            progress_path=ROOT/'data/metadata/full_download_progress.json'
            if progress_path.exists():
                progress=json.loads(progress_path.read_text())
                if progress.get('complete'): break
            time.sleep(10)
        execute('prepare',['scripts/prepare_dataset.py','--run',args.run])
        cuda=['scripts/run_cuda.py','--torch-site',args.torch_site]
        execute('train_transformer',cuda+['training/train.py','--run',args.run,'--model','transformer','--epochs','40'])
        execute('train_gru',cuda+['training/train.py','--run',args.run,'--model','gru','--epochs','40'])
        execute('train_xgboost',cuda+['training/train_baselines.py','--run',args.run])
        execute('select_checkpoint',['scripts/select_checkpoint.py','--run',args.run])
        activated=True
        execute('compare',['scripts/compare_experiments.py','--run',args.run])
        execute('tests',['-m','pytest','-q'])
        state('complete_pending_browser_verification')
    except Exception as exc:
        if activated:
            if old_active is None: active_path.unlink(missing_ok=True)
            else: active_path.write_bytes(old_active)
        state('failed',error=str(exc)); raise
if __name__=='__main__': main()
