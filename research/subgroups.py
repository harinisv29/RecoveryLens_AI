"""Exploratory subgroup evaluation with explicit stability/suppression rules."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from research.common import ROOT,OUT,SOURCE,require_audit,predictions,read_json,save_json
from research.calibration import calibration_metrics

def clustered_intervals(y,p,subjects,threshold,repetitions=1000,seed=42):
    """Vectorized multinomial patient-cluster bootstrap, with tie-correct ROC/AP."""
    y,p,subjects=np.asarray(y),np.asarray(p),np.asarray(subjects)
    unique,inverse=np.unique(subjects,return_inverse=True); rng=np.random.default_rng(seed)
    multiplicities=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repetitions)
    weights=multiplicities[:,inverse].astype(float)
    positive=(weights*y).sum(1); negative=(weights*(1-y)).sum(1); valid=(positive>0)&(negative>0)
    weights=weights[valid]; positive=positive[valid]; negative=negative[valid]
    order=np.argsort(-p,kind='stable'); ys=y[order]; ps=p[order]; w=weights[:,order]
    ends=np.r_[np.flatnonzero(np.diff(ps)),len(ps)-1]
    tp=np.cumsum(w*ys,axis=1)[:,ends]; fp=np.cumsum(w*(1-ys),axis=1)[:,ends]
    tpr=np.column_stack([np.zeros(len(w)),tp/positive[:,None]]); fpr=np.column_stack([np.zeros(len(w)),fp/negative[:,None]])
    auroc=((tpr[:,1:]+tpr[:,:-1])*.5*np.diff(fpr,axis=1)).sum(1)
    ap=(np.diff(tpr,axis=1)*np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)).sum(1)
    decision=(p>=threshold)
    sensitivity=(weights*y*decision).sum(1)/positive
    specificity=(weights*(1-y)*(~decision)).sum(1)/negative
    brier=(weights*(y-p)**2).sum(1)/weights.sum(1)
    return {'repetitions':repetitions,'valid':int(valid.sum()),'patient_clusters':len(unique),'seed':seed,'method':'Patient-cluster percentile bootstrap',**{key:np.quantile(value,[.025,.975]).tolist() for key,value in [('auroc',auroc),('auprc',ap),('sensitivity',sensitivity),('specificity',specificity),('brier',brier)]}}

def definitions(frame):
    return {
        'age':pd.cut(frame.age,[-np.inf,40,60,75,np.inf],right=False,labels=['<40','40-59','60-74','75+']).astype('object').fillna('Unknown'),
        'sex':frame.sex.fillna('Unknown'),
        'surgery_type':frame.optype.fillna('Unknown'),
        'surgery_duration_minutes':pd.cut((frame.opend-frame.opstart)/60,[0,60,120,240,np.inf],include_lowest=True,labels=['<=60','60-120','120-240','>240']).astype('object').fillna('Unknown'),
        'asa':frame.asa.map(lambda x:'Unknown' if pd.isna(x) else ('4+' if x>=4 else str(int(x)))),
        'urgency':frame.emop.map({0:'Elective',1:'Emergency'}).fillna('Unknown')}

def main():
    require_audit(); frames=predictions(); metadata=pd.read_csv(ROOT/'data/metadata/cases.csv')
    target=OUT/'subgroups'; target.mkdir(exist_ok=True); results=[]
    for name,frame in frames.items():
        frame=frame.merge(metadata[['caseid','age','sex','optype','opstart','opend','asa','emop']],on='caseid',validate='one_to_one')
        calibrated=pd.read_csv(OUT/f'calibration/{name}_test.csv').set_index('caseid').loc[frame.caseid].calibrated_probability.to_numpy()
        threshold=read_json(ROOT/f'results/runs/{SOURCE}/metrics/{name}.json')['test']['threshold']
        for variable,series in definitions(frame).items():
            for level in sorted(series.unique()):
                take=series.to_numpy()==level; group=frame.loc[take]; y=group.actual.to_numpy(); p=group.probability.to_numpy(); positives=int(y.sum()); negatives=len(y)-positives
                row={'model':name,'attribute':variable,'group':str(level),'n':len(group),'patients':int(group.subjectid.nunique()),'positive':positives,'negative':negatives,'stable_metrics_reported':len(group)>=50 and min(positives,negatives)>=10}
                if row['stable_metrics_reported']:
                    decision=p>=threshold
                    row.update(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),sensitivity=float(decision[y==1].mean()),specificity=float((~decision[y==0]).mean()),calibration_raw=calibration_metrics(y,p),calibration_adjusted=calibration_metrics(y,calibrated[take]))
                    if len(group)>=100 and min(positives,negatives)>=20 and group.subjectid.nunique()>=30:
                        row['ci95']=clustered_intervals(y,p,group.subjectid.to_numpy(),threshold)
                    else: row['ci_note']='CI withheld: require n>=100, >=20 in each class and >=30 patients'
                else: row['suppression_reason']='Metrics withheld: require n>=50 and at least 10 events and 10 non-events; counts retained'
                results.append(row)
    report={'analysis':'Exploratory, multiple overlapping subgroup analyses; no confirmatory fairness or superiority inference','metric_policy':'Raw discrimination and original validation cutoff; raw and calibrated probability diagnostics separately','ci_policy':'1000 patient-cluster percentile replicates for sufficiently supported subgroups; no multiplicity adjustment','available_attributes':['age','sex','optype','opstart','opend','asa','emop'],'missing_attributes_in_test':{key:int(metadata.set_index('caseid').loc[frames['transformer'].caseid,key].isna().sum()) for key in ['age','sex','optype','asa','emop']},'results':results}
    save_json(target/'summary.json',report)
    pd.json_normalize(results,sep='_').to_csv(target/'subgroups.csv',index=False)
    print('Subgroup cells:',len(results),'reported:',sum(r['stable_metrics_reported'] for r in results),flush=True)
if __name__=='__main__': main()
