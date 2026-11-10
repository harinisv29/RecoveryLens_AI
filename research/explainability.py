"""Frozen-checkpoint IG completeness, numerical convergence and stability checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from config import FEATURES,LIMITS
from research.common import ROOT,OUT,SOURCE,require_audit,predictions,read_json,save_json,digest
from inference.predict import load_checkpoint
from inference.explain import integrated_gradients

def ig_tensor(model,x,mask,steps):
    baseline=x.clone(); baseline[:,:,:7]=0; difference=x-baseline; gradients=torch.zeros_like(x)
    for i in range(steps+1):
        z=(baseline+difference*(i/steps)).detach().requires_grad_(True)
        grad=torch.autograd.grad(model(z,mask).sum(),z)[0]
        gradients+=grad*(.5 if i in (0,steps) else 1)
    attribution=difference*gradients/steps
    with torch.no_grad(): delta=(model(x,mask)-model(baseline,mask)).item()
    residual=float(attribution.sum().item()-delta)
    return attribution.detach(),{'steps':steps,'target':'Pre-sigmoid model logit','baseline_logit_difference':delta,'attribution_sum':float(attribution.sum()),'completeness_absolute_error':abs(residual),'completeness_relative_error':abs(residual)/max(abs(delta),.01)}

def correlation(a,b):
    result=spearmanr(np.asarray(a).ravel(),np.asarray(b).ravel()).statistic
    return float(result) if np.isfinite(result) else None

def main():
    require_audit()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for attribution experiments')
    torch.set_num_threads(4)
    data=torch.load(ROOT/f'data/processed/{SOURCE}/dataset.pt',weights_only=False)
    checkpoint=ROOT/f'models/checkpoints/{SOURCE}/transformer.pt'; model,c=load_checkpoint(checkpoint); model=model.cuda().eval()
    frame=predictions()['transformer']; threshold=c['threshold']; decision=frame.probability>=threshold
    frame['category']=np.select([decision&(frame.actual==1),decision&(frame.actual==0),~decision&(frame.actual==1)],['TP','FP','FN'],default='TN')
    chosen=[]
    for category in ['TP','FP','FN','TN']: chosen.extend(frame[frame.category==category].sample(n=min(3,int((frame.category==category).sum())),random_state=42).caseid.tolist())
    by_id={row['caseid']:i for i,row in enumerate(data['cohort'])}; output=OUT/'explainability'; output.mkdir(exist_ok=True)
    rng=np.random.default_rng(42); records=[]
    for cid in chosen:
        i=by_id[cid]; x=torch.tensor(data['x'][i],device='cuda')[None]; mask=torch.ones((1,x.shape[1]),dtype=torch.bool,device='cuda')
        application=integrated_gradients(model,x,mask,steps=32)
        a32,d32=ig_tensor(model,x,mask,32); a128,d128=ig_tensor(model,x,mask,128); a256,d256=ig_tensor(model,x,mask,256)
        alignment=np.max(np.abs(np.asarray(application['temporal_heatmap'])-a32[0,:,:7].cpu().numpy()))
        stabilities=[]
        for trial in range(3):
            noise=rng.normal(size=(len(data['x'][i]),7))*np.array([1,.2,2,2,2,.5,.5])
            values=x[0,:,:7].detach().cpu().numpy()*np.asarray(c['stats']['std'])+np.asarray(c['stats']['mean'])
            perturbed=np.clip(values+noise,np.array(LIMITS)[:,0],np.array(LIMITS)[:,1]); z=x.clone()
            normalized=(perturbed-np.asarray(c['stats']['mean']))/np.asarray(c['stats']['std'])
            z[0,:,:7]=torch.tensor(normalized,dtype=z.dtype,device='cuda')*z[0,:,7:]
            attrib,detail=ig_tensor(model,z,mask,128)
            a=a128[0,:,:7].abs().cpu().numpy(); b=attrib[0,:,:7].abs().cpu().numpy()
            with torch.no_grad(): shift=float((torch.sigmoid(model(z,mask))-torch.sigmoid(model(x,mask))).item())
            stabilities.append({'feature_rank_spearman':correlation(a.sum(0),b.sum(0)),'temporal_rank_spearman':correlation(a.sum(1),b.sum(1)),'probability_shift':shift,'completeness':detail})
        missing=(x[0,:,7:]==0); missing_attr=float(a128[0,:,:7][missing].abs().max()) if missing.any() else 0.
        pad_test=None
        if x.shape[1]+5<=720:
            padded=torch.cat([x,torch.ones((1,5,14),device='cuda')],1); padmask=torch.cat([mask,torch.zeros((1,5),dtype=torch.bool,device='cuda')],1)
            with torch.no_grad(): difference=float(abs(model(padded,padmask).item()-model(x,mask).item()))
            pad_test={'added_masked_minutes':5,'logit_difference':difference,'passed':difference<1e-5}
        record={'caseid':cid,'category':frame.set_index('caseid').loc[cid,'category'],'minutes':x.shape[1],'application_32_matches_reimplementation_max_abs':float(alignment),'integration':[d32,d128,d256],'missing_value_attribution_max_abs':missing_attr,'padding':pad_test,'perturbations':stabilities,'feature_attribution_128':a128[0,:,:7].abs().sum(0).cpu().tolist()}
        records.append(record)
        save_json(output/f'case_{cid}.json',record)
        if len(records)<=4:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6)); ax1.bar(FEATURES,record['feature_attribution_128']); ax1.set(ylabel='Absolute logit attribution',title=f'Case {cid}: {record["category"]}, IG 128 steps')
            heat=a128[0,:,:7].cpu().numpy().T; scale=max(float(np.abs(heat).max()),1e-8); image=ax2.imshow(heat,aspect='auto',cmap='coolwarm',vmin=-scale,vmax=scale); ax2.set(yticks=range(7),yticklabels=FEATURES,xlabel='Elapsed intraoperative minute bin'); fig.colorbar(image,ax=ax2,label='Signed logit attribution'); fig.tight_layout(); fig.savefig(output/f'case_{cid}_attribution.png',dpi=140); plt.close(fig)
        print('Attribution validated case',cid,'relative completeness errors',[round(d['completeness_relative_error'],4) for d in [d32,d128,d256]],flush=True)
    report={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':digest(checkpoint),'normalization_matches_audited_dataset':c['stats']==data['stats'],'target':'Uncalibrated Transformer logit, not probability or calibrated output','baseline':'Training-mean physiology, fixed observed/missingness indicators. Conditional physiology attribution; missingness contribution is deliberately not measured.','alignment':'One row per elapsed intraoperative minute; same audited raw timestamps, feature order and padding mask','sample_policy':'12 seeded held-out cases, 3 each TP/FP/FN/TN; descriptive, not exhaustive','perturbation_policy':'Three seeded Gaussian perturbations per case: SD HR 1 bpm, SpO2 .2 percentage points, BP 2 mmHg, EtCO2/RR .5; observed entries only, clipped to configured validity limits','records':records,'summary':{f'median_relative_completeness_{steps}':float(np.median([row['integration'][j]['completeness_relative_error'] for row in records])) for j,steps in enumerate([32,128,256])},'limitations':'Attribution does not establish clinical causality. Mean-baseline reference is mathematical, not a validated clinical counterfactual. Small convenience sample; gradients can be locally sensitive and rankings can change.'}
    report['summary']['median_feature_spearman']=float(np.median([p['feature_rank_spearman'] for r in records for p in r['perturbations'] if p['feature_rank_spearman'] is not None]))
    report['summary']['median_temporal_spearman']=float(np.median([p['temporal_rank_spearman'] for r in records for p in r['perturbations'] if p['temporal_rank_spearman'] is not None]))
    report['summary']['all_padding_checks_pass']=all(r['padding'] is None or r['padding']['passed'] for r in records)
    save_json(output/'summary.json',report)
    print(report['summary'],flush=True)
if __name__=='__main__': main()
