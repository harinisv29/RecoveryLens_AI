"""Named experiment directories protect working artifacts from retraining."""
import re
from config import ROOT

def run_paths(run=None):
    if run and not re.fullmatch(r'[A-Za-z0-9_-]+',run): raise ValueError('Invalid experiment name')
    paths={'processed':ROOT/'data/processed','metadata':ROOT/'data/metadata','checkpoints':ROOT/'models/checkpoints','results':ROOT/'results'}
    if run:
        paths={key:(value/'runs'/run if key=='results' else value/run) for key,value in paths.items()}
    for path in paths.values(): path.mkdir(parents=True,exist_ok=True)
    for name in ['metrics','plots','explanations']: (paths['results']/name).mkdir(exist_ok=True)
    return paths
