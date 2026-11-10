import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from config import ROOT, SEED
from utils.preprocessing import resample, fit_stats, transform
from utils.experiments import run_paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run'); args=parser.parse_args(); paths=run_paths(args.run)
    if args.run and (paths['processed']/'dataset.pt').exists(): raise RuntimeError('Named dataset already exists; use a new run name to protect existing splits')
    cases=pd.read_csv(ROOT/'data/metadata/clinical_derived.csv').set_index('caseid')
    sequences=[]; rows=[]; excluded=[]
    for path in sorted((ROOT/'data/raw').glob('case_*.json')):
        cid=int(path.stem.split('_')[1]); row=cases.loc[cid]
        if not row.outcome_eligible:
            excluded.append({'caseid':cid,'reason':str(row.exclusion_reasons)}); continue
        frame=pd.DataFrame({'time_min':np.arange(0,(row.opend-row.opstart)/60,1)})
        for feature,tid in json.loads(path.read_text()).items():
            track=pd.read_csv(ROOT/f'data/raw/{tid}.csv')
            track.columns=['Time','Value']
            track=track[(track.Time>=row.opstart)&(track.Time<row.opend)].copy()
            track['time_min']=np.floor((track.Time-row.opstart)/60)
            frame=frame.merge(track.groupby('time_min').Value.median().rename(feature),on='time_min',how='left')
        try: values=resample(frame)
        except ValueError as exc: excluded.append({'caseid':cid,'reason':str(exc)}); continue
        if len(values)<5 or (np.isfinite(values).mean(axis=0)>=.2).sum()<3:
            excluded.append({'caseid':cid,'reason':'Signal quality: require >=5 minute bins and >=3 channels each >=20% observed'}); continue
        sequences.append(values); rows.append({'caseid':cid,'subjectid':int(row.subjectid) if pd.notna(row.subjectid) else None,'los':float(row.los_postop),'observed_fraction':float(np.isfinite(values).mean()),'minutes':len(values)})
        if len(rows)%500==0: print(f'Preprocessed {len(rows)} eligible cases; excluded {len(excluded)}',flush=True)
    if len(rows)<20: raise RuntimeError('At least 20 eligible real cases required for a pipeline experiment')
    cohort=pd.DataFrame(rows)
    if cohort.subjectid.isna().any(): raise RuntimeError('Missing patient identifiers: cannot guarantee patient-independent splits')
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED)
    prior_path=ROOT/'results/runs/initial_127/data/metadata/cohort.json'
    if args.run and prior_path.exists():
        # All previously examined patients stay in development/training, so expanded
        # validation/test patients have not been used by the initial experiment.
        prior=json.loads(prior_path.read_text())
        exposed={row['subjectid'] for row in prior['cohort']}
        prior_indices=np.flatnonzero(cohort.subjectid.isin(exposed))
        fresh_indices=np.flatnonzero(~cohort.subjectid.isin(exposed))
        a,b=next(splitter.split(cohort.iloc[fresh_indices],groups=cohort.iloc[fresh_indices].subjectid))
        trainval,test=fresh_indices[a],fresh_indices[b]
        a,b=next(splitter.split(cohort.iloc[trainval],groups=cohort.iloc[trainval].subjectid))
        train,val=np.concatenate([prior_indices,trainval[a]]),trainval[b]
        split_policy='Seed 42 patient-group split; all initial-experiment subjects restricted to training; fresh subjects split 64/16/20'
    else:
        trainval,test=next(splitter.split(cohort,groups=cohort.subjectid))
        a,b=next(splitter.split(cohort.iloc[trainval],groups=cohort.iloc[trainval].subjectid))
        train,val=trainval[a],trainval[b]
        split_policy='Seed 42 patient-group split, approximately 64/16/20'
    threshold=float(cohort.iloc[train].los.quantile(.75))
    stats=fit_stats([sequences[i] for i in train])
    labels=(cohort.los.to_numpy()>threshold).astype(np.float32)
    splits={'train':train.tolist(),'val':val.tolist(),'test':test.tolist()}
    for name,indices in splits.items():
        if len(np.unique(labels[indices]))<2: raise RuntimeError(f'{name} has one class; download more cases, do not tune splits on test outcomes')
    import torch
    torch.save({'x':[transform(s,stats) for s in sequences],'y':labels,'splits':splits,'stats':stats,'target_days':threshold,'cohort':rows,'run':args.run,'split_policy':split_policy},paths['processed']/'dataset.pt')
    (paths['processed']/'preprocessing.json').write_text(json.dumps(stats,indent=2))
    (paths['metadata']/'cohort.json').write_text(json.dumps({'cohort':rows,'splits':splits,'excluded':excluded,'threshold_days':threshold,'positive_rule':'los_postop > training 75th percentile','positive_counts':{k:int(labels[v].sum()) for k,v in splits.items()},'quality_policy':'At least 5 minute bins and at least 3 channels each >=20% observed','population':'Known in-hospital survivors only'},indent=2))
    (paths['metadata']/'split_policy.json').write_text(json.dumps({'policy':split_policy,'seed':SEED},indent=2))
    print('Prepared',len(rows),'cases; target >',threshold,'days', {k:len(v) for k,v in splits.items()})
if __name__=='__main__': main()
