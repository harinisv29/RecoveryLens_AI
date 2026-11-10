"""Resumable public numeric-track downloads; no waveforms required."""
import argparse
import io
import json
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import requests
from config import ROOT, TRACKS
from utils.outcomes import audit_outcomes

def fetch(url, path, columns):
    if path.exists():
        try:
            frame=pd.read_csv(path)
            if set(columns).issubset(frame.columns) and len(frame):
                return frame
        except Exception:
            pass
    for attempt in range(4):
        try:
            response=requests.get(url,timeout=(15,90))
            response.raise_for_status()
            frame=pd.read_csv(io.StringIO(response.text))
            if not set(columns).issubset(frame.columns) or frame.empty:
                raise ValueError('Unexpected or empty CSV schema')
            temporary=path.with_suffix('.part')
            temporary.write_text(response.text,encoding='utf-8')
            temporary.replace(path)
            return frame
        except Exception:
            if attempt==3:
                raise
            time.sleep(2**attempt)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=32); args=p.parse_args()
    for folder in ['data/raw','data/metadata','data/processed','data/sample','results/metrics','results/plots','results/explanations','models/checkpoints']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=ROOT/'data/metadata/download.log',level=logging.INFO)
    cases=fetch('https://api.vitaldb.net/cases',ROOT/'data/metadata/cases.csv',['caseid'])
    tracks=fetch('https://api.vitaldb.net/trks',ROOT/'data/metadata/tracks.csv',['caseid','tname','tid'])
    cases=audit_outcomes(cases)
    cases.to_csv(ROOT/'data/metadata/clinical_derived.csv',index=False)
    eligible=cases[cases.outcome_eligible]
    successes=0
    for row in eligible.itertuples():
        if args.limit and successes>=args.limit: break
        selected=tracks[tracks.caseid==row.caseid]
        manifest={}
        def download_channel(item):
            feature,name=item
            found=selected[selected.tname==name]
            if found.empty:
                logging.warning('case %s missing %s',row.caseid,name); return None
            tid=str(found.iloc[0].tid)
            try:
                fetch('https://api.vitaldb.net/'+tid,ROOT/f'data/raw/{tid}.csv',['Time',name])
                return feature,tid
            except Exception as exc:
                logging.error('case %s track %s failed: %s',row.caseid,name,exc)
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(download_channel,TRACKS.items()):
                if result: manifest[result[0]]=result[1]
        if len(manifest)>=3:
            (ROOT/f'data/raw/case_{row.caseid}.json').write_text(json.dumps(manifest))
            successes+=1
            if successes%16==0: print(f'Verified {successes} real cases',flush=True)
        else: logging.warning('case %s excluded: fewer than 3 channels',row.caseid)
    print(f'Downloaded/verified {successes} cases')

if __name__=='__main__': main()
