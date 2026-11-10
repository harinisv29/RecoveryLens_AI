"""One immutable validation-selected Transformer for the live application."""
import hashlib
import json
import numpy as np
import torch
from config import ROOT,FEATURES,LIMITS,MAX_STEPS,DISCLAIMER
from models.temporal_transformer import TemporalTransformer
from utils.preprocessing import resample,transform
from inference.explain import integrated_gradients

class LiveTransformer:
    def __init__(self,checkpoint_path=None):
        selection=json.loads((ROOT/'results/runs/full_20261110/model_selection.json').read_text())
        if selection['model']!='transformer': raise ValueError('Validated selection is not a Transformer; refuse fallback')
        self.path=checkpoint_path or ROOT/selection['checkpoint']
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256!=selection['sha256']: raise ValueError('Transformer checkpoint fingerprint does not match validated selection')
        self.checkpoint=torch.load(self.path,map_location='cpu',weights_only=False)
        c=self.checkpoint; stats=c['stats']
        if c.get('model')!='transformer': raise ValueError('Only the selected Transformer can serve live predictions')
        if stats['features']!=FEATURES or len(stats['mean'])!=len(FEATURES) or len(stats['std'])!=len(FEATURES): raise ValueError('Checkpoint feature schema is incompatible')
        if not np.isfinite(stats['mean']).all() or not np.isfinite(stats['std']).all() or np.any(np.asarray(stats['std'])<=0): raise ValueError('Invalid saved normalization statistics')
        if stats['interval_seconds']!=60 or stats['max_steps']!=MAX_STEPS: raise ValueError('Checkpoint resampling or sequence limit is incompatible')
        if c['state_dict']['embedding.weight'].shape[1]!=2*len(FEATURES): raise ValueError('Expected physiology plus missingness inputs')
        self.metrics=json.loads((ROOT/'results/runs/full_20261110/metrics/transformer.json').read_text())
        for key,expected in {'auroc':.8639,'auprc':.7060,'f1':.6531,'sensitivity':.6632,'specificity':.8804}.items():
            if abs(self.metrics['test'][key]-expected)>5e-5: raise ValueError('Saved evaluation does not match reported experiment')
        if c['target_days']!=self.metrics['target_days'] or c['threshold']!=self.metrics['validation']['threshold']: raise ValueError('Outcome or probability threshold differs from validation artifacts')
        torch.set_num_threads(4)
        self.model=TemporalTransformer(**c['architecture'])
        self.model.load_state_dict(c['state_dict']); self.model.eval(); self.model.requires_grad_(False)
        self.load_count=1

    def status(self):
        c=self.checkpoint
        return {'available':True,'model_type':'transformer','version':c['version'],'cohort_size':c['cohort_size'],'target_days':c['target_days'],'decision_threshold':c['threshold'],'features':c['stats']['features'],'input_dimension':2*len(FEATURES),'checkpoint_sha256':self.sha256,'checkpoint_load_count':self.load_count,'probability_label':'Estimated model probability (not clinically calibrated)','live_model_policy':'Transformer only; baselines and early-window ablations are retrospective research','limitations':DISCLAIMER,'metrics':self.metrics['test']}

    def predict(self,frame,progress=None):
        notify=progress or (lambda *args,**kwargs:None)
        notify('preprocessing','running',{})
        values=resample(frame); observed=np.isfinite(values)
        supported=int((observed.mean(0)>=.2).sum())
        if len(values)<5 or supported<3:
            raise ValueError('Insufficient data quality: require at least 5 minute bins and 3 physiological channels each at least 20% observed, matching the training cohort')
        c=self.checkpoint; normalized=transform(values,c['stats'])
        x=torch.tensor(normalized)[None]; mask=torch.ones((1,len(values)),dtype=torch.bool)
        notify('preprocessing','completed',{'minute_bins':len(values),'observed_fraction':float(observed.mean()),'supported_channels':supported,'missing_channels':[f for j,f in enumerate(FEATURES) if not observed[:,j].any()],'model_input_shape':list(x.shape),'normalization':'Saved training statistics; not refitted'})
        notify('inference','running',{'model':'Temporal Transformer','version':c['version'],'checkpoint_sha256':self.sha256,'checkpoint_load_count':self.load_count})
        with torch.inference_mode():
            logit=float(self.model(x,mask).item()); probability=float(torch.sigmoid(torch.tensor(logit)).item())
        notify('inference','completed',{'model':'Temporal Transformer','output':'Single model logit computed; result withheld until analysis completes'})
        # Explanations require gradients; only the scalar prediction uses inference mode.
        notify('explainability','running',{'method':'Integrated Gradients','target':'Model logit','steps':32})
        with torch.enable_grad(): explanation=integrated_gradients(self.model,x,mask)
        case_hash=hashlib.sha256(normalized.tobytes()).hexdigest()
        explanation.update(checkpoint_sha256=self.sha256,input_sha256=case_hash,prediction_logit=logit)
        notify('explainability','completed',{'features':len(explanation['feature_attribution']),'temporal_bins':len(explanation['temporal_heatmap']),'method':explanation['method']})
        notify('results','running',{'work':'Assembling input quality, signal series and prediction summary'})
        invalid={f:int((frame[f].notna() & ~frame[f].between(lo,hi)).sum()) if f in frame else 0 for f,(lo,hi) in zip(FEATURES,LIMITS)}
        binary=int(probability>=c['threshold']); label='Prolonged stay predicted' if binary else 'Prolonged stay not predicted'
        return {'probability':probability,'probability_label':'Estimated model probability (not clinically calibrated)','binary_prediction':binary,'risk_category':label,'threshold':c['threshold'],'target_days':c['target_days'],'outcome_definition':f"Postoperative stay strictly greater than {c['target_days']:.8f} days; elapsed (dis - opend) / 86400; training-only 75th percentile; in-hospital survivors only",'version':c['version'],'model_type':'transformer','cohort_size':c['cohort_size'],'checkpoint_sha256':self.sha256,'input_sha256':case_hash,'quality':{'minutes':len(values),'observed_fraction':float(observed.mean()),'missing_features':[f for j,f in enumerate(FEATURES) if not observed[:,j].any()],'supported_channels':supported,'artifact_values_by_feature':invalid,'input_time_origin_min':float(frame.time_min.iloc[0]),'normalization':'Saved training-only statistics; never fitted on uploads'},'signals':{f:[float(v) if np.isfinite(v) else None for v in values[:,j]] for j,f in enumerate(FEATURES)},'explanation':explanation,'limitations':DISCLAIMER}
