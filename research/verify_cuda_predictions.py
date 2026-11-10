"""Resolve CPU/GPU numeric differences using the original frozen-model runtime."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from research.common import OUT,ROOT,SOURCE,read_json,save_json
from inference.predict import load_checkpoint
from training.train import collate
from training.evaluate import metrics

def main():
    if not torch.cuda.is_available(): raise RuntimeError('Original CUDA runtime required')
    report=read_json(OUT/'cohort_audit.json')
    if set(k for k,v in report['checks'].items() if not v)-{'gru_saved_prediction_alignment'}: raise RuntimeError('Unexpected audit failures: cannot repair via GPU compatibility verification')
    target=OUT/'predictions_verified'; target.mkdir(exist_ok=True)
    verified_path=OUT/'cohort_integrity_verified.json'
    if verified_path.exists(): raise RuntimeError('Verified audit already exists')
    data=torch.load(ROOT/f'data/processed/{SOURCE}/dataset.pt',weights_only=False)
    details={}
    for name in ['transformer','gru','xgboost']:
        checkpoint=ROOT/f'models/checkpoints/{SOURCE}'/(name+('.json' if name=='xgboost' else '.pt'))
        model,metadata=load_checkpoint(checkpoint)
        if name!='xgboost': model=model.cuda().eval()
        for split in ['val','test']:
            frame=pd.read_csv(OUT/f'predictions/{name}_{split}.csv'); ids=data['splits'][split]
            if name!='xgboost':
                probabilities=[]
                with torch.no_grad():
                    for start in range(0,len(ids),16):
                        batch=ids[start:start+16]; x,m,_=collate([(data['x'][i],data['y'][i]) for i in batch])
                        probabilities.extend(torch.sigmoid(model(x.cuda(),m.cuda())).cpu().tolist())
                frame['probability']=probabilities
            if split=='test':
                archived=pd.read_csv(ROOT/f'results/runs/{SOURCE}/errors/{name}_case_predictions.csv')
                maximum=float(np.abs(frame.probability-archived.probability).max())
                cpu=pd.read_csv(OUT/f'predictions/{name}_test.csv')
                details[name]={'cuda_vs_archived_max_absolute_difference':maximum,'cpu_vs_archived_max_absolute_difference':float(np.abs(cpu.probability-archived.probability).max())}
                if maximum>1e-6: raise RuntimeError(f'{name}: original GPU runtime failed reproduction: {maximum}')
                # Use exact archived test scores after successful frozen GPU reproduction.
                frame['probability']=archived.probability.to_numpy()
                report['checks'][name+'_saved_prediction_alignment']=True
            frame.to_csv(target/f'{name}_{split}.csv',index=False)
            computed=metrics(frame.actual,frame.probability,metadata['threshold'])
            saved=read_json(ROOT/f'results/runs/{SOURCE}/metrics/{name}.json')['validation' if split=='val' else 'test']
            delta={k:float(computed[k]-saved[k]) for k in ['auroc','auprc','f1','sensitivity','specificity','brier_score']}
            if not all(abs(value)<1e-6 for value in delta.values()): raise RuntimeError(str(delta))
            report['metric_checks'][name+'_'+split]['verified_gpu_delta']=delta
    report.update(passed=bool(all(report['checks'].values())),runtime_verification=details,verification_resolution='Original CUDA torch 2.5.1 runtime and batch size 16 reproduce archived test scores. CPU GRU roundoff differs; exact archived test scores used after reproduction. First CPU audit preserved unchanged.',runtime=torch.__version__,gpu=torch.cuda.get_device_name(0))
    save_json(verified_path,report)
    (OUT/'COHORT_FLOW_VERIFIED.md').write_text((OUT/'COHORT_FLOW.md').read_text(encoding='utf-8').replace('Passed all checks: False','Passed all checks: True')+'\n\nGPU reproduction resolved CPU GRU numerical differences; see cohort_integrity_verified.json. No cohort correction or retraining was necessary.\n',encoding='utf-8')
    print('VERIFIED:',report['passed'],details,flush=True)
if __name__=='__main__': main()
