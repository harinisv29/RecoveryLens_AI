"""Bounded parallel, resumable retrieval of the seven configured numeric channels."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import requests
from config import ROOT,TRACKS
from utils.outcomes import audit_outcomes

local=threading.local()

def session():
    if not hasattr(local,'session'):
        local.session=requests.Session()
        local.session.headers['User-Agent']='RecoveryLens-research-downloader/2.0'
    return local.session

def valid_frame(content,name):
    frame=pd.read_csv(io.BytesIO(content))
    if list(frame.columns)!=['Time',name] or frame.empty: raise ValueError('Unexpected or empty numeric CSV')
    for column in frame:
        frame[column]=pd.to_numeric(frame[column],errors='raise')
    if not np.isfinite(frame.Time).all() or not np.isfinite(frame[name]).any(): raise ValueError('No finite measurements or invalid timestamps')
    return frame

def retrieve(item):
    cid,feature,name,tid=item; path=ROOT/f'data/raw/{tid}.csv'
    result={'caseid':int(cid),'feature':feature,'track':name,'tid':tid}
    try:
        if path.exists():
            content=path.read_bytes(); valid_frame(content,name)
            return {**result,'status':'cached','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    except Exception:
        # A bad existing file is preserved separately for audit before replacement.
        path.rename(path.with_suffix('.invalid.'+str(time.time_ns())))
    for attempt in range(5):
        try:
            response=session().get('https://api.vitaldb.net/'+tid,timeout=(15,60))
            if response.status_code==429:
                time.sleep(min(float(response.headers.get('Retry-After','15')),60))
            response.raise_for_status()
            content=response.content; valid_frame(content,name)
            temporary=path.with_suffix('.part'); temporary.write_bytes(content); temporary.replace(path)
            return {**result,'status':'downloaded','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
        except Exception as exc:
            error=str(exc)
            if isinstance(exc,requests.HTTPError) and exc.response.status_code in (400,401,403,404): break
            if attempt<4: time.sleep(min(2**attempt,16))
    return {**result,'status':'failed','error':error}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=12); args=p.parse_args()
    metadata=ROOT/'data/metadata'; (ROOT/'data/raw').mkdir(parents=True,exist_ok=True)
    cases=audit_outcomes(pd.read_csv(metadata/'cases.csv'))
    cases.to_csv(metadata/'clinical_derived.csv',index=False)
    eligible=cases[cases.outcome_eligible]
    tracks=pd.read_csv(metadata/'tracks.csv')
    lookup={(int(row.caseid),row.tname):str(row.tid) for row in tracks.itertuples() if row.tname in TRACKS.values()}
    tasks=[]; missing=[]
    for row in eligible.itertuples():
        for feature,name in TRACKS.items():
            tid=lookup.get((int(row.caseid),name))
            if tid: tasks.append((row.caseid,feature,name,tid))
            else: missing.append({'caseid':int(row.caseid),'feature':feature,'track':name,'reason':'not_in_track_index'})
    if len({x[3] for x in tasks})!=len(tasks): raise RuntimeError('Duplicate track IDs: refusing concurrent duplicate writes')
    (metadata/'full_missing_channels.json').write_text(json.dumps(missing,indent=2))
    results=[]; manifests={}; started=time.perf_counter()
    counts={'cached':0,'downloaded':0,'failed':0}
    def progress(complete=False):
        report={'complete':complete,'eligible_outcome_cases':len(eligible),'requested_tracks':len(tasks),'finished_tracks':len(results),'missing_channel_slots':len(missing),'counts':counts,'cases_with_at_least_three_tracks':sum(len(x)>=3 for x in manifests.values()),'elapsed_seconds':time.perf_counter()-started,'workers':args.workers}
        temporary=metadata/'full_download_progress.part'; temporary.write_text(json.dumps(report,indent=2)); temporary.replace(metadata/'full_download_progress.json')
        print(json.dumps(report),flush=True)
    progress()
    with (metadata/'full_download_tracks.jsonl').open('w',encoding='utf-8') as log,ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(retrieve,item):item for item in tasks}
        for future in as_completed(futures):
            result=future.result(); results.append(result); counts[result['status']]+=1
            log.write(json.dumps(result)+'\n')
            if result['status']!='failed':
                mapping=manifests.setdefault(result['caseid'],{}); mapping[result['feature']]=result['tid']
            if len(results)%250==0: log.flush(); progress()
    # Only publish completed manifests after all channels have been attempted.
    for cid,mapping in manifests.items():
        if len(mapping)>=3:
            path=ROOT/f'data/raw/case_{cid}.json'; tmp=path.with_suffix('.part'); tmp.write_text(json.dumps(mapping)); tmp.replace(path)
    failed=[r for r in results if r['status']=='failed']
    (metadata/'full_download_failures.json').write_text(json.dumps(failed,indent=2))
    (metadata/'full_case_inventory.json').write_text(json.dumps([{'caseid':int(row.caseid),'downloaded_channels':len(manifests.get(int(row.caseid),{})),'eligible_for_signal_screening':len(manifests.get(int(row.caseid),{}))>=3} for row in eligible.itertuples()],indent=2))
    progress(True)
if __name__=='__main__': main()
