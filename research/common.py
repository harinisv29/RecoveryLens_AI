"""Additive research evaluation. Original artifacts are read-only inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from config import ROOT

RUN='advanced_20261110'
SOURCE='full_20261110'
OUT=ROOT/'results/statistical_validation'/RUN
CACHE=ROOT/'data/processed'/RUN
CALIBRATORS=ROOT/'models/calibration'/RUN

def setup():
    for path in [OUT,CACHE,CALIBRATORS]: path.mkdir(parents=True,exist_ok=True)
    return OUT

def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False),encoding='utf-8')

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_audit():
    path=OUT/'cohort_integrity_verified.json'
    audit=read_json(path if path.exists() else OUT/'cohort_audit.json')
    if not audit['passed']: raise RuntimeError('Cohort integrity audit has not passed')
    return audit

def predictions(split='test'):
    import pandas as pd
    folder='predictions_verified' if (OUT/'cohort_integrity_verified.json').exists() else 'predictions'
    frames={name:pd.read_csv(OUT/f'{folder}/{name}_{split}.csv') for name in ['transformer','gru','xgboost']}
    reference=frames['transformer'][['caseid','subjectid','actual']]
    for frame in frames.values():
        if not frame[['caseid','subjectid','actual']].equals(reference): raise RuntimeError('Predictions not aligned on identical cases, patients and labels')
    return frames
