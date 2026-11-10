"""Retrain selected Transformer family for predeclared feature/window ablations."""
import copy
import random
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import ROOT,FEATURES,SEED
from models.temporal_transformer import TemporalTransformer
from training.train import collate
from training.evaluate import metrics,tune_threshold
from utils.preprocessing import fit_stats,transform
from research.common import OUT,CACHE,RUN,SOURCE,require_audit,save_json,digest
from research.subgroups import clustered_intervals

VARIANTS=[('hr_only',[0],True,None),('bp_only',[2,3,4],True,None),('spo2_only',[1],True,None),('hr_bp_spo2',[0,1,2,3,4],True,None),('all_features',list(range(7)),True,None),('no_missingness_flags',list(range(7)),False,None),('first_30',list(range(7)),True,30),('first_60',list(range(7)),True,60),('first_90',list(range(7)),True,90)]

def main():
    require_audit()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for deep-learning experiments')
    torch.set_num_threads(4)
    data=torch.load(ROOT/f'data/processed/{SOURCE}/dataset.pt',weights_only=False)
    raw=torch.load(CACHE/'raw_sequences.pt',weights_only=False)
    folder=OUT/'ablations'; folder.mkdir(exist_ok=True); checkpoints=ROOT/f'models/checkpoints/{RUN}/ablations'; checkpoints.mkdir(parents=True,exist_ok=True)
    results=[]
    for variant,features,indicators,window in VARIANTS:
        path=folder/f'{variant}.json'
        if path.exists():
            import json
            results.append(json.loads(path.read_text())); print('Preserving completed ablation',variant,flush=True); continue
        checkpoint_path=checkpoints/f'{variant}.pt'
        if checkpoint_path.exists(): raise RuntimeError('Partial checkpoint exists: preserve and select new version before rerun')
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        sequences=[a[:window].copy() if window else a.copy() for a in raw]
        stats=fit_stats([sequences[i] for i in data['splits']['train']])
        columns=features+[i+7 for i in features] if indicators else features
        x=[transform(a,stats)[:,columns] for a in sequences]
        loaders={name:DataLoader([(x[i],data['y'][i]) for i in ids],batch_size=16,shuffle=name=='train',collate_fn=collate) for name,ids in data['splits'].items()}
        architecture={'input_dim':len(columns),'d_model':64,'heads':4,'layers':2,'dropout':.1}
        model=TemporalTransformer(**architecture).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); lossfn=torch.nn.BCEWithLogitsLoss(); scaler=torch.amp.GradScaler('cuda')
        best=float('inf'); stale=0; history=[]; state=None; torch.cuda.synchronize(); started=time.perf_counter()
        for epoch in range(40):
            losses={}
            for phase in ['train','val']:
                model.train(phase=='train'); total=0
                for a,m,y in loaders[phase]:
                    a,m,y=a.cuda(),m.cuda(),y.cuda(); optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(phase=='train'),torch.autocast('cuda'):
                        loss=lossfn(model(a,m),y)
                    if phase=='train':
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(optimizer); scaler.update()
                    total+=loss.item()*len(y)
                losses[phase]=total/len(loaders[phase].dataset)
            history.append(losses)
            if losses['val']<best:
                best=losses['val']; stale=0; state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
            else: stale+=1
            if (epoch+1)%5==0: print(variant,'epoch',epoch+1,losses,flush=True)
            if stale>=7: break
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started; model.load_state_dict(state); model.eval()
        def predict(split):
            probabilities=[]; ys=[]
            with torch.no_grad():
                for a,m,y in loaders[split]:
                    probabilities.extend(torch.sigmoid(model(a.cuda(),m.cuda())).cpu().tolist()); ys.extend(y.tolist())
            return np.asarray(ys),np.asarray(probabilities)
        vy,vp=predict('val'); cutoff=tune_threshold(vy,vp)
        y,p=predict('test'); test=metrics(y,p,cutoff)
        subjects=[data['cohort'][i]['subjectid'] for i in data['splits']['test']]
        test['bootstrap_95_ci']=clustered_intervals(y,p,subjects,cutoff,repetitions=2000)
        result={'variant':variant,'features':[FEATURES[i] for i in features],'explicit_missingness_flags':indicators,'maximum_observed_minutes':window or 720,'prediction_timing':f'Up to first {window} minutes; shorter operations use only their available prefix' if window else 'End of available operation recording, capped at 720 minutes','n_shorter_than_window':sum(len(a)<window for a in raw) if window else None,'normalization_policy':'Training-only observed values, refitted using only this variant observation window','cohort_n':len(x),'split_sizes':{name:len(ids) for name,ids in data['splits'].items()},'target_days':data['target_days'],'validation':metrics(vy,vp,cutoff),'test':test,'training_seconds':elapsed,'epochs_completed':len(history),'selected_epoch':min(range(len(history)),key=lambda i:history[i]['val'])+1,'history':history,'gpu':torch.cuda.get_device_name(0),'seed':SEED,'checkpoint':str(checkpoint_path.relative_to(ROOT)),'limitations':'One-seed exploratory Transformer ablation, not all three model families. Same previously examined holdout. Explicit flags removed in no-flag model; mean imputation still encodes implicit missingness.'}
        torch.save({'state_dict':state,'architecture':architecture,'stats':stats,'features':features,'columns':columns,'window':window,'threshold':cutoff,'target_days':data['target_days'],'version':RUN+'_'+variant,'model':'transformer_ablation','dataset_sha256':digest(ROOT/f'data/processed/{SOURCE}/dataset.pt')},checkpoint_path)
        import pandas as pd
        frame=pd.DataFrame([data['cohort'][i] for i in data['splits']['test']]); frame['actual']=y.astype(int); frame['probability']=p; frame.to_csv(folder/f'{variant}_test.csv',index=False)
        save_json(path,result); results.append(result)
        print('Completed',variant,'seconds',elapsed,'AUROC',test['auroc'],flush=True)
        del model; torch.cuda.empty_cache()
    save_json(folder/'summary.json',{'variants':results,'scope':'Nine separately retrained Transformer experiments: six full-window feature sets and three all-feature early prefixes. Not a 6x4 factorial design. No ablation model is promoted.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5)); points=[r['test']['auroc'] for r in results]; intervals=[r['test']['bootstrap_95_ci']['auroc'] for r in results]
    ax.errorbar(points,range(len(results)),xerr=np.array([[p-a,p-b] for p,(a,b) in zip(points,intervals)]).T*np.array([[1],[-1]]),fmt='o',color='#277f99')
    ax.set(yticks=range(len(results)),yticklabels=[r['variant'] for r in results],xlabel='Test AUROC (patient-bootstrap 95% CI)',title='Separately retrained exploratory Transformer ablations'); fig.tight_layout(); fig.savefig(folder/'ablation_auroc.png',dpi=140); plt.close(fig)
if __name__=='__main__': main()
