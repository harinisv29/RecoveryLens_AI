import json
import argparse
import time
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from xgboost import XGBClassifier
from config import ROOT,SEED
from utils.preprocessing import engineered
from training.evaluate import metrics,tune_threshold,bootstrap_intervals,save_error_analysis
from utils.visualization import evaluation_plots
from utils.experiments import run_paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run'); args=parser.parse_args(); paths=run_paths(args.run)
    checkpoint_path=paths['checkpoints']/'xgboost.json'
    if args.run and checkpoint_path.exists(): raise RuntimeError('Named checkpoint exists; use a new run name')
    d=torch.load(paths['processed']/'dataset.pt',weights_only=False)
    x=np.stack([engineered(s) for s in d['x']]); y=d['y']; a,b,c=[d['splits'][k] for k in ['train','val','test']]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,random_state=SEED,n_jobs=4,eval_metric='logloss',early_stopping_rounds=15,tree_method='hist',device=device)
    started=time.perf_counter()
    model.fit(x[a],y[a],eval_set=[(x[b],y[b])],verbose=False)
    elapsed=time.perf_counter()-started
    threshold=tune_threshold(y[b],model.predict_proba(x[b])[:,1])
    p=model.predict_proba(x[c])[:,1]
    result={'validation':metrics(y[b],model.predict_proba(x[b])[:,1],threshold),'test':metrics(y[c],p,threshold),'cohort_size':len(y),'target_days':d['target_days'],'training_seconds':elapsed,'device':device,'run':args.run,'checkpoint':str(checkpoint_path.relative_to(ROOT))}
    result['test']['bootstrap_95_ci']=bootstrap_intervals(y[c],p,[d['cohort'][i]['subjectid'] for i in c])
    result.update(seed=SEED,best_iteration=int(model.best_iteration),gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,dataset_sha256=hashlib.sha256((paths['processed']/'dataset.pt').read_bytes()).hexdigest(),training_config={'n_estimators':300,'max_depth':3,'learning_rate':.03,'early_stopping_rounds':15,'tree_method':'hist'})
    model.save_model(checkpoint_path)
    (paths['checkpoints']/'xgboost_metadata.json').write_text(json.dumps({'stats':d['stats'],'threshold':threshold,'target_days':d['target_days'],'model':'xgboost','version':args.run or 'research-v1','cohort_size':len(y)},indent=2))
    (paths['results']/'metrics/xgboost.json').write_text(json.dumps(result,indent=2))
    save_error_analysis('xgboost',d,c,p,threshold,paths['results']/'errors')
    evaluation_plots('xgboost',[],y[c],p,result['test'],paths['results']/'plots')
if __name__=='__main__': main()
