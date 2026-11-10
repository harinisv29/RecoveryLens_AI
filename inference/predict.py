import torch
import json
from pathlib import Path
import numpy as np
from config import ROOT,FEATURES,DISCLAIMER
from models.temporal_transformer import TemporalTransformer
from models.gru_baseline import GRUClassifier
from utils.preprocessing import resample,transform,engineered
from inference.explain import integrated_gradients

def load_checkpoint(path=None):
    if path is None:
        active=ROOT/'models/active_model.json'
        if active.exists():
            selection=json.loads(active.read_text()); path=ROOT/selection['checkpoint']
            if not path.resolve().is_relative_to((ROOT/'models/checkpoints').resolve()): raise ValueError('Active checkpoint must remain inside models/checkpoints')
        else: path=ROOT/'models/checkpoints/transformer.pt'
    path=Path(path)
    if not path.exists(): raise FileNotFoundError('No trained Transformer checkpoint. Train on eligible real data first.')
    if path.suffix=='.json':
        from xgboost import XGBClassifier
        checkpoint=json.loads((path.parent/'xgboost_metadata.json').read_text())
        model=XGBClassifier(); model.load_model(path); model.set_params(device='cpu')
        return model,checkpoint
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=GRUClassifier() if checkpoint.get('model')=='gru' else TemporalTransformer(**checkpoint['architecture']); model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model,checkpoint

def predict(df):
    model,c=load_checkpoint(); values=resample(df)
    x=torch.tensor(transform(values,c['stats']))[None]; mask=torch.ones((1,len(values)),dtype=torch.bool)
    if c.get('model')=='xgboost':
        normalized=x[0].numpy(); probability=float(model.predict_proba(engineered(normalized)[None])[:,1][0])
        explanation=xgboost_occlusion(model,normalized,probability)
    else:
        with torch.no_grad(): probability=torch.sigmoid(model(x,mask)).item()
        explanation=integrated_gradients(model,x,mask)
    return {'probability':probability,'risk_category':'Above validation threshold' if probability>=c['threshold'] else 'Below validation threshold','threshold':c['threshold'],'target_days':c['target_days'],'version':c['version'],'model_type':c.get('model','transformer'),'cohort_size':c['cohort_size'],'quality':{'minutes':len(values),'observed_fraction':float(np.isfinite(values).mean()),'missing_features':[f for i,f in enumerate(FEATURES) if not np.isfinite(values[:,i]).any()]},'signals':{f:[float(v) if np.isfinite(v) else None for v in values[:,i]] for i,f in enumerate(FEATURES)},'explanation':explanation,'limitations':DISCLAIMER}

def xgboost_occlusion(model,normalized,probability):
    """Signed prediction sensitivity to replacing one minute/channel with train mean."""
    batch=[]; coordinates=[]
    for minute in range(len(normalized)):
        for feature in range(7):
            if normalized[minute,feature+7] and normalized[minute,feature]!=0:
                changed=normalized.copy(); changed[minute,feature]=0
                batch.append(engineered(changed)); coordinates.append((minute,feature))
    attribution=np.zeros((len(normalized),7))
    if batch:
        alternative=model.predict_proba(np.stack(batch))[:,1]
        for (minute,feature),p in zip(coordinates,alternative): attribution[minute,feature]=probability-float(p)
    return {'feature_attribution':np.abs(attribution).sum(0).tolist(),'temporal_heatmap':attribution.tolist(),'important_intervals':np.argsort(np.abs(attribution).sum(1))[-5:][::-1].tolist(),'method':'Single-minute feature occlusion; signed probability change against training-mean replacement, fixed missingness. Nonadditive sensitivity, not causal.','units':'probability change'}
