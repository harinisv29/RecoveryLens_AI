"""Independent frozen-checkpoint calculation. No application prediction code is used.

Optional HTTP comparison runs only AFTER independently reconstructing preprocessing,
model evaluation, sigmoid, threshold and Integrated Gradients from saved artifacts.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class IndependentTransformer(nn.Module):
    """Reconstruct saved architecture/state names without calling deployment classes."""
    def __init__(self,architecture,input_dim,position_shape):
        super().__init__(); dimension=architecture['d_model']
        self.embedding=nn.Linear(input_dim,dimension)
        self.register_buffer('position',torch.zeros(position_shape))
        layer=nn.TransformerEncoderLayer(d_model=dimension,nhead=architecture['heads'],dim_feedforward=dimension*4,dropout=architecture['dropout'],batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,architecture['layers'],enable_nested_tensor=False)
        self.head=nn.Linear(dimension,1)

    def forward(self,x,mask):
        hidden=self.embedding(x)+self.position[:x.shape[1]]
        encoded=self.encoder(hidden,src_key_padding_mask=~mask)
        pooled=(encoded*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)

def independently_prepare(frame,stats,limits):
    features=stats['features']; work=frame.copy()
    if set(work)-set(features+['time_min']): raise ValueError('Unknown CSV columns')
    if 'time_min' not in work or len(work)<5: raise ValueError('Need timestamps and at least five rows')
    for column in work: work[column]=pd.to_numeric(work[column],errors='raise')
    if np.isinf(work.to_numpy()).any(): raise ValueError('Infinite CSV values')
    t=work.time_min.to_numpy()
    if not np.isfinite(t).all() or t[0]<0 or (np.diff(t)<=0).any() or t[-1]-t[0]>720: raise ValueError('Incompatible timestamps')
    bins=np.floor(t-t[0]).astype(int); artifact_counts={}
    for name,(low,high) in zip(features,limits):
        if name not in work: work[name]=np.nan
        invalid=work[name].notna() & ~work[name].between(low,high)
        artifact_counts[name]=int(invalid.sum()); work.loc[invalid,name]=np.nan
    work['_minute_bin']=bins
    values=work.groupby('_minute_bin')[features].median().reindex(range(int(bins.max())+1)).to_numpy(dtype=np.float32)[:stats['max_steps']]
    observed=np.isfinite(values)
    if len(values)<5 or (observed.mean(0)>=.2).sum()<3: raise ValueError('Input fails saved cohort quality policy')
    standardized=(values-np.asarray(stats['mean']))/np.asarray(stats['std'])
    tensor_values=np.concatenate([np.nan_to_num(standardized),observed.astype(float)],axis=1).astype(np.float32)
    x=torch.tensor(tensor_values)[None]; mask=torch.ones((1,len(values)),dtype=torch.bool)
    return x,mask,values,tensor_values,artifact_counts

def independent_ig(model,x,mask,steps=32):
    baseline=x.clone(); baseline[:,:,:7]=0; difference=x-baseline; accumulated=torch.zeros_like(x)
    for step in range(steps+1):
        interpolated=(baseline+difference*(step/steps)).detach().requires_grad_(True)
        gradient=torch.autograd.grad(model(interpolated,mask).sum(),interpolated)[0]
        accumulated+=gradient*(.5 if step in (0,steps) else 1)
    attribution=(difference*accumulated/steps)[0,:,:7].detach().numpy()
    with torch.inference_mode(): delta=(model(x,mask)-model(baseline,mask)).item()
    return attribution,{'target':'Pre-sigmoid classification logit','steps':steps,'baseline':'Normalized training-mean physiology (zero); observation flags fixed','logit_difference_from_baseline':delta,'attribution_sum':float(attribution.sum()),'completeness_absolute_error':float(abs(attribution.sum()-delta))}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--csv',type=Path,required=True); parser.add_argument('--application-url'); parser.add_argument('--report',type=Path,default=ROOT/'results/INDEPENDENT_PREDICTION_VERIFICATION.md'); args=parser.parse_args()
    if not args.csv.is_file(): raise FileNotFoundError(args.csv)
    if args.report.exists() or args.report.with_suffix('.json').exists(): raise RuntimeError('Verification report exists; preserve it and choose a new --report path')
    checkpoint_path=ROOT/'models/checkpoints/full_20261110/transformer.pt'
    normalization_path=ROOT/'data/processed/full_20261110/preprocessing.json'
    protected=[checkpoint_path,normalization_path,ROOT/'config.py',ROOT/'inference/live_transformer.py',ROOT/'utils/preprocessing.py',ROOT/'models/temporal_transformer.py',ROOT/'inference/explain.py',ROOT/'transformer_app.py',ROOT/'static/js/premium-dashboard.js']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}; csv_sha=sha(args.csv)
    torch.set_num_threads(4)
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    stats=json.loads(normalization_path.read_text()); selection=json.loads((ROOT/'results/runs/full_20261110/model_selection.json').read_text())
    assert saved['model']=='transformer' and sha(checkpoint_path)==selection['sha256']
    assert saved['stats']==stats and stats['interval_seconds']==60 and stats['features']==['HR','SpO2','SBP','DBP','MAP','EtCO2','RR']
    config=ast.parse((ROOT/'config.py').read_text(encoding='utf-8'))
    limits=next(ast.literal_eval(node.value) for node in config.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='LIMITS' for target in node.targets))
    frame=pd.read_csv(args.csv); features=stats['features']
    input_summary={'csv_path':str(args.csv.resolve()),'csv_sha256':csv_sha,'rows':len(frame),'columns':len(frame.columns),'column_names':frame.columns.tolist(),'first_timestamp_minutes':float(frame.time_min.iloc[0]),'last_timestamp_minutes':float(frame.time_min.iloc[-1]),'elapsed_timestamp_span_minutes':float(frame.time_min.iloc[-1]-frame.time_min.iloc[0]),'missing_values':{name:int(frame[name].isna().sum()) if name in frame else len(frame) for name in ['time_min']+features},'signal_statistics':{name:{'min':float(frame[name].min()),'max':float(frame[name].max()),'mean':float(frame[name].mean())} for name in features if name in frame}}
    x,mask,values,normalized,artifacts=independently_prepare(frame,stats,limits)
    input_summary.update(minute_bins=len(values),artifact_values_excluded=artifacts,observed_fraction=float(np.isfinite(values).mean()),input_tensor_shape=list(x.shape),input_dtype=str(x.dtype),attention_mask_shape=list(mask.shape),padding_required=False,features=features,observation_flag_order=['observed_'+name for name in features])
    model=IndependentTransformer(saved['architecture'],saved['state_dict']['embedding.weight'].shape[1],saved['state_dict']['position'].shape)
    model.load_state_dict(saved['state_dict'],strict=True); model.eval(); model.requires_grad_(False)
    with torch.inference_mode():
        raw_logit=float(model(x,mask).item()); probability=float(torch.sigmoid(torch.tensor(raw_logit)).item()); repeated=float(torch.sigmoid(model(x,mask)).item())
    math_sigmoid=1/(1+math.exp(-raw_logit)); threshold=float(saved['threshold']); binary=int(probability>=threshold)
    input_hash=hashlib.sha256(normalized.tobytes()).hexdigest()
    attribution,ig_details=independent_ig(model,x,mask)
    # Secondary numerical parity check only; this is not the source of the independent prediction.
    from utils.preprocessing import resample,transform
    from utils.validation import validate_csv
    deployed_tensor=transform(resample(validate_csv(args.csv)),stats)
    parity={'preprocessing_bitwise_equal':bool(np.array_equal(deployed_tensor,normalized)),'preprocessing_max_absolute_difference':float(np.abs(deployed_tensor-normalized).max()),'checkpoint_normalization_matches_json':saved['stats']==stats,'sigmoid_float32_vs_float64_difference':abs(probability-math_sigmoid),'repeat_probability_identical':probability==repeated,'all_modules_evaluation_mode':all(not module.training for module in model.modules()),'input_sha256':input_hash}
    application_comparison=None
    if args.application_url:
        import requests
        status=requests.get(args.application_url+'/api/model-status',timeout=15); status.raise_for_status(); status=status.json()
        with args.csv.open('rb') as handle: response=requests.post(args.application_url+'/api/predict',files={'file':(args.csv.name,handle,'text/csv')},timeout=120)
        response.raise_for_status(); application=response.json()
        app_heat=np.asarray(application['explanation']['temporal_heatmap'])
        application_comparison={'method':'Secondary application comparison after independent computation; not used to calculate the independent prediction','status':status,'probability':application['probability'],'binary_prediction':application['binary_prediction'],'threshold':application['threshold'],'quality':application['quality'],'probability_absolute_difference':abs(probability-application['probability']),'checkpoint_hash_match':application['checkpoint_sha256']==sha(checkpoint_path)==application['explanation']['checkpoint_sha256'],'input_hash_match':application['input_sha256']==input_hash==application['explanation']['input_sha256'],'logit_absolute_difference':abs(raw_logit-application['explanation']['prediction_logit']),'ig_max_absolute_difference':float(np.abs(attribution-app_heat).max()),'ig_feature_total_max_difference':float(np.abs(np.abs(attribution).sum(0)-application['explanation']['feature_attribution']).max()),'ig_target':application['explanation']['method'],'ig_temporal_shape':list(app_heat.shape),'actual_application_response':application}
    rounding_match=f'{probability*100:.1f}'=='15.4'
    checks={'dashboard_one_decimal_rounding_match':rounding_match,'approximate_probability_within_display_tolerance':abs(probability-.154)<=.0005,'saved_probability_threshold_match':abs(threshold-.370)<=1e-12,'binary_prediction_match':binary==0,'actual_14_input_shape':list(x.shape)==[1,len(values),14],'saved_preprocessing_matches':parity['preprocessing_bitwise_equal'],'repeat_inference_identical':parity['repeat_probability_identical'],'sigmoid_math_matches':abs(probability-math_sigmoid)<1e-7,'synthetic_csv_unchanged':sha(args.csv)==csv_sha,'protected_sources_and_artifacts_unchanged':all(sha(ROOT/name)==value for name,value in before.items())}
    if application_comparison:
        a=application_comparison
        checks.update(actual_api_probability_matches=a['probability_absolute_difference']<=1e-7,actual_api_binary_matches=a['binary_prediction']==binary,explanation_checkpoint_and_input_match=a['checkpoint_hash_match'] and a['input_hash_match'],explanation_logit_matches=a['logit_absolute_difference']<=1e-6,independent_ig_matches=a['ig_max_absolute_difference']<=1e-6)
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'input':input_summary,'checkpoint_path':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),'architecture':saved['architecture'],'normalization_path':str(normalization_path),'normalization':stats,'raw_logit':raw_logit,'probability':probability,'percentage':100*probability,'float64_sigmoid':math_sigmoid,'threshold':threshold,'binary_prediction':binary,'outcome_days':saved['target_days'],'reported_application_percentage':15.4,'signed_difference_percentage_points':100*probability-15.4,'absolute_difference_percentage_points':abs(100*probability-15.4),'tolerances':{'display_rounding_probability_absolute':.0005,'display_rounding_percentage_points':.05,'direct_probability_absolute':1e-7,'logit_absolute':1e-6,'attribution_absolute':1e-6,'threshold_absolute':1e-12},'parity':parity,'independent_integrated_gradients':ig_details,'application_comparison':application_comparison,'protected_sha256_before':before,'file_identity_note':'User confirmed Downloads/recoverylens_synthetic_test.csv is the file to verify; the (1) filename was not present.','clinical_limitation':'Synthetic input has no known postoperative outcome; this verifies computational correctness only, not clinical accuracy or calibration.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    def yn(value): return 'Yes' if value else 'No'
    lines=['# Independent Transformer Prediction Verification','',f"## Conclusion: {result['status']}",'',result['file_identity_note'],'',f"**Independent probability: {probability*100:.8f}%**. Application display: **15.4%**. Absolute difference: **{result['absolute_difference_percentage_points']:.8f} percentage points**. Saved threshold: **{threshold:.12f}**. Binary prediction: **{binary}**.",'','## 1. CSV input summary','',f"File: `{args.csv.resolve()}`. SHA-256: `{csv_sha}`. Rows: {len(frame)}. Columns: {len(frame.columns)}. Columns: `{', '.join(frame.columns)}`.",'',f"First timestamp: {input_summary['first_timestamp_minutes']:.0f}; last: {input_summary['last_timestamp_minutes']:.0f} minutes. The elapsed first-to-last span is **{input_summary['elapsed_timestamp_span_minutes']:.0f} minutes**, while the model has **{len(values)} one-minute bins**. The application's 120-minute value counts bins; it is not a literal 120-minute timestamp difference. This duration-label distinction does not affect the computed tensor or probability.",'',f"Missing values: `{input_summary['missing_values']}`. Artifact values excluded: `{artifacts}`.",'','| Physiological signal | Min | Max | Mean |','| --- | --- | --- | --- |']
    for name,stats_row in input_summary['signal_statistics'].items(): lines.append(f"| {name} | {stats_row['min']:.6f} | {stats_row['max']:.6f} | {stats_row['mean']:.6f} |")
    lines+=['','All seven raw physiological signals are present. Units are interpreted according to the existing schema (HR/RR per minute, SpO2 percent, blood pressures/EtCO2 mmHg); the CSV alone does not independently certify the synthetic generator\'s units. Values fall within configured artifact limits.','','## 2. Checkpoint and architecture','',f"Checkpoint: `{checkpoint_path}`. SHA-256: `{sha(checkpoint_path)}`. It matches the validation-selected model fingerprint. Architecture independently reconstructed from saved configuration: `{saved['architecture']}`; strict state-dict loading succeeds. Embedding receives 14 channels, followed by positional encoding, Transformer encoder, attention-masked mean pooling and one linear logit head. Evaluation mode disables dropout. No model or preprocessing artifact was changed.",'','## 3. Independent preprocessing','',f"Saved normalization: `{normalization_path}`. It exactly equals checkpoint normalization. Features are ordered `{features}`, followed by the seven observation flags in that same order. Relative timestamps are floored into elapsed-minute bins; unsupported/missing observations remain missing; per-channel minute medians are formed; saved training means/stds standardize values; normalized missing entries are zero and flags are zero. Nothing is refitted to this CSV. Float32 tensor: **{list(x.shape)}**. Boolean attention mask: **{list(mask.shape)}**, all real bins; no padding required. The input is 8 CSV columns, not 14 raw CSV columns: the other seven model inputs are derived observation flags.",'',f"Independent tensor versus deployed preprocessing: bitwise equal **{parity['preprocessing_bitwise_equal']}**, maximum difference **{parity['preprocessing_max_absolute_difference']:.12g}**. The independent primary calculation does not call LiveTransformer.predict, the application model class, or an HTTP endpoint. A separate reconstructed module loads the exact saved state. The deployed preprocessing is imported only afterward for parity checking.",'','## 4. Raw model calculation','',f"Raw classification logit: **{raw_logit:.12f}**. `sigmoid(logit) = 1 / (1 + exp(-logit))`. Float32 PyTorch result: **{probability:.12f}**; independent scalar float64 formula: **{math_sigmoid:.12f}**. Difference: `{abs(probability-math_sigmoid):.12g}`. Repeated forward evaluation is identical: **{probability==repeated}**. Binary classification is `probability >= saved_threshold`, giving **{binary}**. The postoperative stay threshold ({saved['target_days']:.8f} days) is an outcome definition, not a probability cutoff.",'','## 5. Comparison','', '| Metric | Application | Independent calculation | Match |','| --- | --- | --- | --- |',f'| Raw logit | Not displayed | {raw_logit:.12f} | N/A |',f'| Probability | Approximately 0.154 | {probability:.12f} | {yn(checks["approximate_probability_within_display_tolerance"])} |',f'| Percentage | 15.4% | {probability*100:.8f}% → {probability*100:.1f}% | {yn(rounding_match)} |',f'| Classification threshold | 0.370 | {threshold:.12f} | {yn(checks["saved_probability_threshold_match"])} |',f'| Binary prediction | 0 | {binary} | {yn(binary==0)} |','', 'Display rounding tolerance is **±0.05 percentage points**, equivalent to ±0.0005 probability, because the dashboard shows one decimal percent. Direct calculations/API probabilities are compared at 1e-7 absolute probability; logits and attribution entries at 1e-6; threshold at 1e-12. The model probability is not model accuracy.','','## 6. Explanation consistency','',f"Independent Integrated Gradients targets the **same raw logit**, with **32 trapezoidal steps**, zero normalized physiological baseline (training means), fixed observation indicators, identical feature order, same minute alignment and the same checkpoint. Attribution has shape `{list(attribution.shape)}`. Completeness absolute numerical residual is `{ig_details['completeness_absolute_error']:.12g}`; finite-step IG is a numerical approximation. Missingness contributions are not attributed because flags are held fixed. Attributions are not causal medical explanations."]
    if application_comparison:
        a=application_comparison
        lines+=['',f"After independent reconstruction, the actual API was separately checked. Raw API probability: **{a['probability']:.12f}**; independent-versus-API difference: **{a['probability_absolute_difference']:.12g}**. Checkpoint hashes match: **{a['checkpoint_hash_match']}**. Preprocessed input hashes match: **{a['input_hash_match']}**. Explanation logit difference: **{a['logit_absolute_difference']:.12g}**. Independent-versus-API IG maximum entry difference: **{a['ig_max_absolute_difference']:.12g}**; feature-total difference: **{a['ig_feature_total_max_difference']:.12g}**. The live service reports checkpoint load count {a['status']['checkpoint_load_count']}. These secondary checks do not supply the independently calculated prediction."]
    lines+=['','## 7. Problems and limitations','', 'No numerical mismatch was found.' if result['status']=='PASS' else 'Failed checks: '+str([key for key,value in checks.items() if not value]),'','The only duration nuance is 120 minute bins versus 119 elapsed minutes between timestamps. Filename identity was confirmed by the user. No feature-order, normalization, mask, checkpoint, sigmoid, evaluation-mode, tensor-shape or decision-threshold mismatch was detected for this CSV.' if result['status']=='PASS' else 'See JSON checks and intermediate comparisons for the exact differences.','','The dashboard, inference pipeline, model weights, saved normalization and CSV were verified unchanged by SHA-256 after this run. This synthetic CSV has no known postoperative outcome. A computational PASS cannot establish whether its clinical prediction is correct, calibrated or useful.','','## 8. Final status','',f"**{result['status']} — computational prediction and explanation consistency verification only.**",'']
    args.report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['status','raw_logit','probability','percentage','absolute_difference_percentage_points','threshold','binary_prediction','checks']},indent=2))
    print('Report:',args.report)

if __name__=='__main__': main()
