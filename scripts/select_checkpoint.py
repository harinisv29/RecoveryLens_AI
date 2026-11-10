"""Select exclusively by validation AUROC, tie-break by validation Brier score."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from utils.experiments import run_paths
from inference.predict import load_checkpoint

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args(); paths=run_paths(args.run)
    candidates={name:json.loads((paths['results']/f'metrics/{name}.json').read_text()) for name in ['transformer','gru','xgboost']}
    selected=max(candidates,key=lambda name:(candidates[name]['validation']['auroc'],-candidates[name]['validation']['brier_score']))
    checkpoint=paths['checkpoints']/(selected+('.json' if selected=='xgboost' else '.pt'))
    load_checkpoint(checkpoint)  # Verify serialization before changing application selection.
    selection={'run':args.run,'model':selected,'checkpoint':str(checkpoint.relative_to(ROOT)),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selection_policy':'Highest validation AUROC; tie-break lowest validation Brier score; test results not used','validation_candidates':{name:{key:value['validation'][key] for key in ['auroc','auprc','brier_score']} for name,value in candidates.items()}}
    active=ROOT/'models/active_model.json'
    if active.exists(): shutil.copy2(active,active.with_name(f'active_model.backup_{time.time_ns()}.json'))
    temporary=active.with_suffix('.part'); temporary.write_text(json.dumps(selection,indent=2)); temporary.replace(active)
    (paths['results']/'model_selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps(selection,indent=2))
if __name__=='__main__': main()
