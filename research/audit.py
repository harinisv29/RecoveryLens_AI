"""Gate all advanced analyses on independent source/label/split checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from config import ROOT,FEATURES
from research.common import OUT,CACHE,SOURCE,setup,save_json,read_json,digest
from utils.outcomes import audit_outcomes
from utils.preprocessing import resample,fit_stats,transform,engineered
from inference.predict import load_checkpoint
from training.train import collate
from training.evaluate import metrics

def main():
    setup()
    if (OUT/'cohort_audit.json').exists(): raise RuntimeError('Audit exists; preserve it and choose a new research RUN before repeating')
    protected=[]
    for folder in ['models/checkpoints','data/processed/full_20261110','data/metadata/full_20261110','results/runs/full_20261110','templates','static','inference','utils','training']:
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    protected += [ROOT/'app.py',ROOT/'config.py',ROOT/'README.md',ROOT/'models/active_model.json']
    save_json(OUT/'preserved_manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in protected})
    cases=pd.read_csv(ROOT/'data/metadata/cases.csv'); audited=audit_outcomes(cases).set_index('caseid')
    data=torch.load(ROOT/f'data/processed/{SOURCE}/dataset.pt',weights_only=False)
    cohort=pd.DataFrame(data['cohort']); inventory=pd.DataFrame(read_json(ROOT/'data/metadata/full_case_inventory.json'))
    cid=cohort.caseid.to_numpy(); eligible=audited.loc[cid]
    checks={'unique_cohort_cases':not cohort.caseid.duplicated().any(),'survivors_only':bool((eligible.death_inhosp==0).all()),'valid_outcomes':bool(eligible.outcome_eligible.all()),'finite_nonnegative_los':bool(np.isfinite(eligible.los_postop).all() and (eligible.los_postop>=0).all()),'los_matches_source':bool(np.allclose(cohort.los,eligible.los_postop,rtol=0,atol=1e-10)),'patient_ids_match':bool(np.array_equal(cohort.subjectid,eligible.subjectid))}
    groups={name:set(cohort.iloc[ids].subjectid) for name,ids in data['splits'].items()}
    checks['patient_disjoint']=not any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
    all_indices=sum(data['splits'].values(),[])
    checks['complete_disjoint_case_partition']=sorted(all_indices)==list(range(len(cohort)))
    threshold=float(np.quantile(cohort.iloc[data['splits']['train']].los,.75))
    checks['threshold_training_only']=threshold==data['target_days']
    checks['labels_match']=bool(np.array_equal(data['y'],(cohort.los.to_numpy()>threshold).astype(np.float32)))
    prior=read_json(ROOT/'results/runs/initial_127/data/metadata/cohort.json')
    exposed={row['subjectid'] for row in prior['cohort']}
    checks['initial_patients_excluded_from_new_holdout']=not bool(exposed&(groups['val']|groups['test']))
    # Rebuild every included operation directly from shared-reference raw timestamps.
    raw=[]; mismatches=[]
    for index,row in enumerate(cohort.itertuples()):
        clinical=audited.loc[row.caseid]
        frame=pd.DataFrame({'time_min':np.arange(0,(clinical.opend-clinical.opstart)/60,1)})
        mapping=read_json(ROOT/f'data/raw/case_{row.caseid}.json')
        for feature,tid in mapping.items():
            track=pd.read_csv(ROOT/f'data/raw/{tid}.csv'); track.columns=['Time','Value']
            track=track[(track.Time>=clinical.opstart)&(track.Time<clinical.opend)].copy()
            track['time_min']=np.floor((track.Time-clinical.opstart)/60)
            frame=frame.merge(track.groupby('time_min').Value.median().rename(feature),on='time_min',how='left')
        values=resample(frame); raw.append(values)
        if len(values)<5 or (np.isfinite(values).mean(0)>=.2).sum()<3: mismatches.append(int(row.caseid))
        if not np.allclose(transform(values,data['stats']),data['x'][index],rtol=1e-5,atol=1e-5): mismatches.append(int(row.caseid))
        if (index+1)%1000==0: print('Independently rebuilt',index+1,'cases',flush=True)
    recomputed=fit_stats([raw[i] for i in data['splits']['train']])
    checks['raw_sequences_and_quality_match']=not mismatches
    checks['normalization_training_only']=bool(np.allclose(recomputed['mean'],data['stats']['mean'],rtol=1e-6) and np.allclose(recomputed['std'],data['stats']['std'],rtol=1e-6))
    torch.save(raw,CACHE/'raw_sequences.pt')
    # Recompute all held-out probabilities from exact frozen checkpoints; never retrain.
    torch.set_num_threads(4); metric_checks={}
    for name in ['transformer','gru','xgboost']:
        checkpoint=ROOT/f'models/checkpoints/{SOURCE}'/(name+('.json' if name=='xgboost' else '.pt'))
        model,metadata=load_checkpoint(checkpoint)
        saved=read_json(ROOT/f'results/runs/{SOURCE}/metrics/{name}.json')
        checks[name+'_dataset_hash']=saved['dataset_sha256']==digest(ROOT/f'data/processed/{SOURCE}/dataset.pt')
        for split in ['val','test']:
            ids=data['splits'][split]; probabilities=[]
            if name=='xgboost': probabilities=model.predict_proba(np.stack([engineered(data['x'][i]) for i in ids]))[:,1]
            else:
                with torch.no_grad():
                    for start in range(0,len(ids),32):
                        batch=ids[start:start+32]; x,m,_=collate([(data['x'][i],data['y'][i]) for i in batch])
                        probabilities.extend(torch.sigmoid(model(x,m)).tolist())
            frame=cohort.iloc[ids].copy(); frame['actual']=data['y'][ids].astype(int); frame['probability']=probabilities
            (OUT/'predictions').mkdir(exist_ok=True); frame.to_csv(OUT/f'predictions/{name}_{split}.csv',index=False)
            result=metrics(frame.actual,frame.probability,metadata['threshold'])
            original=saved['validation' if split=='val' else 'test']
            delta={key:float(result[key]-original[key]) for key in ['auroc','auprc','f1','sensitivity','specificity','brier_score']}
            metric_checks[name+'_'+split]={'recomputed':result,'saved':original,'delta':delta}
            checks[name+'_'+split+'_metrics_match']=all(abs(v)<1e-4 for v in delta.values())
            if split=='test':
                recorded=pd.read_csv(ROOT/f'results/runs/{SOURCE}/errors/{name}_case_predictions.csv')
                checks[name+'_saved_prediction_alignment']=bool(np.array_equal(recorded.caseid,frame.caseid) and np.array_equal(recorded.actual,frame.actual) and np.allclose(recorded.probability,frame.probability,atol=1e-5))
    counts=cohort.subjectid.value_counts()
    flow=[['Original cases',len(cases)],['Excluded deaths OR invalid outcomes',int((~audited.outcome_eligible).sum())],['Outcome-eligible survivors',int(audited.outcome_eligible.sum())],['With at least one downloaded configured track',int((inventory.downloaded_channels>0).sum())],['No configured tracks',int((inventory.downloaded_channels==0).sum())],['With only one or two tracks',int(inventory.downloaded_channels.between(1,2).sum())],['At least three tracks',int((inventory.downloaded_channels>=3).sum())],['Fail signal-quality policy',int((inventory.downloaded_channels>=3).sum()-len(cohort))],['Final eligible cases',len(cohort)]]
    checks['cohort_reconciles']=len(cases)==82+127+60+len(cohort)
    report={'passed':bool(all(checks.values())),'checks':checks,'flow':flow,'original_exclusions_overlapping':{reason:int(audited.exclusion_reasons.str.contains(reason).sum()) for reason in ['negative_postoperative_interval','missing_or_nonfinite_discharge','in_hospital_death','invalid_operation_timing','unknown_mortality_status']},'negative_and_death_overlap':int(((audited.los_postop<0)&(audited.death_inhosp==1)).sum()),'unique_patients':int(cohort.subjectid.nunique()),'patients_with_multiple_cases':int((counts>1).sum()),'extra_cases_beyond_one_per_patient':int(len(cohort)-cohort.subjectid.nunique()),'max_cases_per_patient':int(counts.max()),'splits':{name:{'cases':len(ids),'patients':len(groups[name]),'positive':int(data['y'][ids].sum())} for name,ids in data['splits'].items()},'threshold_days':threshold,'timestamp_source':'https://vitaldb.net/dataset/','population':'Survivor-only: known in-hospital survivors. Excluding deaths changes the estimand; not an all-patient clinical outcome model.','metric_checks':metric_checks,'raw_mismatches':mismatches,'limitations':['Cannot independently verify source EMR dates.','Artifact rejection is applied to per-minute medians, so within-minute transient artifacts may remain.','Recordings are capped at 720 minutes.','Original holdout has already been examined; these are secondary retrospective analyses, not a new independent validation.']}
    save_json(OUT/'cohort_audit.json',report)
    lines=['# Cohort integrity audit','','| Stage | Cases |','| --- | --- |']+[f'| {label} | {value} |' for label,value in flow]
    lines += ['',f"Passed all checks: {report['passed']}. Threshold: {threshold:.10f} days (training only).",'',report['population'],'',f"Unique patients: {report['unique_patients']}; multiple-case patients: {report['patients_with_multiple_cases']}; additional cases: {report['extra_cases_beyond_one_per_patient']}; maximum cases/patient: {report['max_cases_per_patient']}.",'', 'All frozen-model test and validation metrics and saved test predictions were recomputed and checked. Details and numerical deltas are in cohort_audit.json.']
    (OUT/'COHORT_FLOW.md').write_text('\n'.join(lines),encoding='utf-8')
    print('AUDIT PASSED:',report['passed'],flow,flush=True)
    if not report['passed']: raise RuntimeError('Audit failed; advanced analysis is gated off')
if __name__=='__main__': main()
