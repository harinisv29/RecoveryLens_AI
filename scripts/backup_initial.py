"""Preserve initial experiment before full-cohort changes; never overwrite a backup."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT

def main():
    destination=ROOT/'results/runs/initial_127'
    if (destination/'backup_manifest.json').exists():
        print('Initial backup already exists:',destination); return
    destination.mkdir(parents=True,exist_ok=True)
    paths=['models/checkpoints','data/processed','data/metadata/cohort.json','data/metadata/outcome_audit.json','data/metadata/outcome_exclusions.csv','results/metrics','results/plots','results/explanations','results/execution_summary.json','results/browser_verification.json','results/tests.txt','results/RESULTS.md','README.md']
    for name in paths:
        source=ROOT/name; target=destination/name
        if not source.exists(): continue
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir(): shutil.copytree(source,target,dirs_exist_ok=True)
        else: shutil.copy2(source,target)
    manifest={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
    (destination/'backup_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Backed up',len(manifest),'files:',destination)
if __name__=='__main__': main()
