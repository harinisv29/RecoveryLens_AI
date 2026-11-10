"""Deterministically export one genuine held-out case, then independently infer.

Selection uses test membership and the minimum case ID only. No predictions are
read until the case is fixed and the actual source CSV has been exported.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT,FEATURES,LIMITS
from scripts.verify_independent_prediction import IndependentTransformer,independently_prepare,independent_ig
from utils.preprocessing import resample,transform
from utils.validation import validate_csv

OUT=ROOT/'results/real_patient_test'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection_and_inference.json').exists() or (OUT/'vitaldb_heldout_test.csv').exists(): raise RuntimeError('Preserve prior patient evidence; do not overwrite it')
    protected=[ROOT/'data/processed/full_20261110/dataset.pt',ROOT/'data/processed/full_20261110/preprocessing.json',ROOT/'data/metadata/full_20261110/cohort.json',ROOT/'data/metadata/cases.csv',ROOT/'models/checkpoints/full_20261110/transformer.pt',ROOT/'models/active_model.json',ROOT/'inference/live_transformer.py',ROOT/'utils/preprocessing.py',ROOT/'inference/explain.py',ROOT/'transformer_app.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    data=torch.load(ROOT/'data/processed/full_20261110/dataset.pt',weights_only=False)
    index=min(data['splits']['test'],key=lambda i:data['cohort'][i]['caseid'])
    fixed=data['cohort'][index];cid=int(fixed['caseid']);subject=int(fixed['subjectid'])
    memberships={name:[int(data['cohort'][i]['caseid']) for i in ids if data['cohort'][i]['subjectid']==subject] for name,ids in data['splits'].items()}
    assert not memberships['train'] and not memberships['val'] and cid in memberships['test']
    assert cid==min(data['cohort'][i]['caseid'] for i in data['splits']['test'])
    save(OUT/'selection_fixed_before_inference.json',{'caseid':cid,'dataset_index':int(index),'selection_rule':'Lowest case ID among saved eligible test indices; fixed before any prediction is evaluated','patient_present_in_train':False,'patient_present_in_validation':False,'patient_test_case_ids':memberships['test'],'subject_identifier_disclosure':'Subject identifier is used locally for membership verification but omitted from exported CSV, dashboard, PDF and this evidence','subject_identifier_sha256':hashlib.sha256(str(subject).encode()).hexdigest(),'split_sizes':{k:len(v) for k,v in data['splits'].items()},'unique_test_patients':len({data['cohort'][i]['subjectid'] for i in data['splits']['test']})})
    clinical=pd.read_csv(ROOT/'data/metadata/cases.csv').set_index('caseid').loc[cid]
    assert clinical.subjectid==subject and clinical.death_inhosp==0
    assert np.isfinite([clinical.dis,clinical.opend,clinical.opstart]).all() and clinical.opend>clinical.opstart and clinical.dis>=clinical.opend
    elapsed_days=float((clinical.dis-clinical.opend)/86400)
    stay_cutoff=float(data['target_days']);truth=int(elapsed_days>stay_cutoff)
    assert abs(elapsed_days-fixed['los'])<1e-10 and truth==int(data['y'][index])
    # Identical operation window and one-minute medians to the saved training export.
    frame=pd.DataFrame({'time_min':np.arange(0,(clinical.opend-clinical.opstart)/60,1)})
    mapping=json.loads((ROOT/f'data/raw/case_{cid}.json').read_text()); track_info=[]
    for feature,tid in mapping.items():
        path=ROOT/f'data/raw/{tid}.csv'; hashes[str(path.relative_to(ROOT))]=sha(path)
        track=pd.read_csv(path);original_name=track.columns[1];track.columns=['Time','Value']
        window=track[(track.Time>=clinical.opstart)&(track.Time<clinical.opend)].copy()
        window['time_min']=np.floor((window.Time-clinical.opstart)/60)
        frame=frame.merge(window.groupby('time_min').Value.median().rename(feature),on='time_min',how='left')
        track_info.append({'feature':feature,'track_name':original_name,'tid':tid,'source_sha256':sha(path),'source_measurements':len(track),'intraoperative_measurements':len(window),'first_intraoperative_timestamp_seconds':float(window.Time.min()) if len(window) else None,'last_intraoperative_timestamp_seconds':float(window.Time.max()) if len(window) else None})
    for feature in FEATURES:
        if feature not in frame:frame[feature]=np.nan
    frame=frame[['time_min']+FEATURES]
    csv=OUT/'vitaldb_heldout_test.csv';frame.to_csv(csv,index=False)
    assert csv.stat().st_size<5*1024*1024
    assert set(frame)==set(['time_min']+FEATURES)
    loaded=validate_csv(csv); deployed_values=resample(loaded)
    supported=int((np.isfinite(deployed_values).mean(0)>=.2).sum());assert len(deployed_values)>=5 and supported>=3
    normalized=transform(deployed_values,data['stats'])
    assert np.array_equal(normalized,data['x'][index]),'Exported CSV must reproduce the original saved held-out tensor'
    # Only now load the frozen model. Selection and exported data are already fixed.
    torch.set_num_threads(4);cp_path=ROOT/'models/checkpoints/full_20261110/transformer.pt'
    cp=torch.load(cp_path,map_location='cpu',weights_only=False);stats=json.loads((ROOT/'data/processed/full_20261110/preprocessing.json').read_text())
    assert cp['stats']==stats==data['stats'] and cp['target_days']==stay_cutoff
    x,mask,independent_values,independent_tensor,artifacts=independently_prepare(loaded,stats,LIMITS)
    assert np.array_equal(independent_tensor,normalized)
    model=IndependentTransformer(cp['architecture'],cp['state_dict']['embedding.weight'].shape[1],cp['state_dict']['position'].shape)
    model.load_state_dict(cp['state_dict'],strict=True);model.eval();model.requires_grad_(False)
    with torch.inference_mode(): logit=float(model(x,mask).item());probability=float(torch.sigmoid(torch.tensor(logit)).item())
    predicted=int(probability>=cp['threshold']);correct=predicted==truth
    attributions,ig=independent_ig(model,x,mask)
    observed=int(np.isfinite(deployed_values).sum());missing=int(deployed_values.size-observed)
    result={'caseid':cid,'dataset_index':int(index),'selection_rule':'Lowest eligible case ID in the saved held-out test split; no prediction-based selection','split':'Held-out test','test_cases':len(data['splits']['test']),'test_unique_patients':len({data['cohort'][i]['subjectid'] for i in data['splits']['test']}),'patient_in_training':False,'patient_in_validation':False,'same_patient_test_case_ids':memberships['test'],'csv_path':str(csv.resolve()),'csv_sha256':sha(csv),'csv_bytes':csv.stat().st_size,'csv_columns':frame.columns.tolist(),'rows':len(frame),'first_time_min':float(frame.time_min.iloc[0]),'last_time_min':float(frame.time_min.iloc[-1]),'operation_duration_minutes':float((clinical.opend-clinical.opstart)/60),'model_minute_bins':len(deployed_values),'input_tensor_shape':list(x.shape),'supported_channels':supported,'exported_observed_values':int(frame[FEATURES].notna().sum().sum()),'exported_missing_values':int(frame[FEATURES].isna().sum().sum()),'model_observed_values':observed,'model_missing_values':missing,'model_missing_per_channel':{f:int((~np.isfinite(deployed_values[:,j])).sum()) for j,f in enumerate(FEATURES)},'exported_signal_stats':{f:{'observed':int(frame[f].notna().sum()),'missing':int(frame[f].isna().sum()),'min':float(frame[f].min()) if frame[f].notna().any() else None,'max':float(frame[f].max()) if frame[f].notna().any() else None,'mean':float(frame[f].mean()) if frame[f].notna().any() else None} for f in FEATURES},'timestamps':{'casestart_seconds':float(clinical.casestart),'opstart_seconds':float(clinical.opstart),'opend_seconds':float(clinical.opend),'dis_seconds':float(clinical.dis),'source':'Official VitalDB field dictionary: https://vitaldb.net/dataset/; opend and dis are seconds from casestart','validation_limitation':'Shared units/reference are documented; original EMR dates cannot be independently verified'},'postoperative_stay_days':elapsed_days,'saved_training_only_stay_threshold_days':stay_cutoff,'ground_truth_label':truth,'in_hospital_death':0,'raw_logit':logit,'sigmoid_probability':probability,'percentage':100*probability,'probability_cutoff':float(cp['threshold']),'predicted_label':predicted,'prediction_correct':bool(correct),'checkpoint_path':str(cp_path.relative_to(ROOT)),'checkpoint_sha256':sha(cp_path),'input_sha256':hashlib.sha256(independent_tensor.tobytes()).hexdigest(),'saved_test_tensor_bitwise_match':True,'independent_vs_deployment_tensor_bitwise_match':True,'feature_order':FEATURES,'artifact_values_excluded':artifacts,'source_tracks':track_info,'independent_ig':ig,'independent_feature_attribution':np.abs(attributions).sum(0).tolist(),'independent_temporal_attribution':attributions.tolist(),'export_policy':'Real operation-only one-minute medians using the original training aggregation. No extra smoothing, value editing or imputation before export; missing observations stay blank. Original high-frequency records remain unchanged.','identity_policy':'CSV and its generic filename contain no case/subject identifiers or outcome fields. The local evaluation report includes the requested deidentified case ID; subject ID is omitted.','limitations':['One case cannot establish clinical performance, calibration, safety or generalization.','This is a previously evaluated single-source survivor-only holdout, not external validation.','Retrospective statistical prolonged-stay threshold; surgery/discharge practices can confound outcomes.'],'protected_files_before':hashes}
    assert all(sha(ROOT/name)==expected for name,expected in hashes.items())
    save(OUT/'selection_and_inference.json',result)
    print(json.dumps({key:result[key] for key in ['caseid','split','rows','model_observed_values','model_missing_values','postoperative_stay_days','ground_truth_label','raw_logit','sigmoid_probability','probability_cutoff','predicted_label','prediction_correct','csv_path']},indent=2))

if __name__=='__main__':main()
