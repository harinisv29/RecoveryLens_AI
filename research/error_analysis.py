"""Describe observed physiological associations with FP/FN, without causal claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from research.common import ROOT,OUT,CACHE,SOURCE,require_audit,predictions,read_json,save_json
from config import FEATURES

def main():
    require_audit(); data=torch.load(ROOT/f'data/processed/{SOURCE}/dataset.pt',weights_only=False)
    raw=torch.load(CACHE/'raw_sequences.pt',weights_only=False); rows=[]
    for i in data['splits']['test']:
        a=raw[i]; row={'caseid':data['cohort'][i]['caseid']}
        row.update(missing_fraction=float((~np.isfinite(a)).mean()),fraction_hr_gt100=float(np.mean(a[:,0][np.isfinite(a[:,0])]>100)) if np.isfinite(a[:,0]).any() else None,fraction_map_lt65=float(np.mean(a[:,4][np.isfinite(a[:,4])]<65)) if np.isfinite(a[:,4]).any() else None,fraction_spo2_lt94=float(np.mean(a[:,1][np.isfinite(a[:,1])]<94)) if np.isfinite(a[:,1]).any() else None)
        for j,feature in enumerate(FEATURES):
            finite=a[:,j][np.isfinite(a[:,j])]
            adjacent=np.isfinite(a[1:,j])&np.isfinite(a[:-1,j]); differences=np.diff(a[:,j])[adjacent]
            row[feature+'_median']=float(np.median(finite)) if len(finite) else None
            row[feature+'_variability_iqr']=float(np.quantile(finite,.75)-np.quantile(finite,.25)) if len(finite) else None
            row[feature+'_minute_change_median']=float(np.median(np.abs(differences))) if len(differences) else None
        rows.append(row)
    physiology=pd.DataFrame(rows); metadata=pd.read_csv(ROOT/'data/metadata/cases.csv')[['caseid','optype']]
    target=OUT/'errors'; target.mkdir(exist_ok=True); reports={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,frame in predictions().items():
        threshold=read_json(ROOT/f'results/runs/{SOURCE}/metrics/{name}.json')['test']['threshold']
        frame=frame.merge(physiology,on='caseid',validate='one_to_one').merge(metadata,on='caseid',validate='one_to_one')
        decision=frame.probability>=threshold; frame['error_type']=np.select([decision&(frame.actual==1),decision&(frame.actual==0),~decision&(frame.actual==1)],['TP','FP','FN'],default='TN')
        frame['los_range']=pd.cut(frame.los,[0,3,7.396099537037037,14,30,np.inf],include_lowest=True,labels=['<=3','3-to-training-cutoff','cutoff-to14','14-to30','>30'])
        frame.to_csv(target/f'{name}_all_cases.csv',index=False)
        frame[frame.error_type=='FN'].to_csv(target/f'{name}_false_negatives.csv',index=False)
        frame[frame.error_type=='FP'].to_csv(target/f'{name}_false_positives.csv',index=False)
        features=[c for c in physiology if c!='caseid']
        summaries={category:{'n':len(group),'median_physiology':{key:float(group[key].median()) if group[key].notna().any() else None for key in features},'substantial_missingness_gt50pct':int((group.missing_fraction>.5).sum())} for category,group in frame.groupby('error_type')}
        counts={key:pd.crosstab(frame[key],frame.error_type).to_dict('index') for key in ['optype','los_range']}
        reports[name]={'decision_threshold':threshold,'groups':summaries,'category_counts':{k:{str(a):b for a,b in value.items()} for k,value in counts.items()},'caution':'Descriptive associations on the test set, no adjustment for case mix, confounding or multiplicity. Missingness and physiological instability are not proven causes of outcomes or errors. Clinical thresholds here are exploratory pattern descriptors, not treatment recommendations.'}
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for category,group in frame.groupby('error_type'): axes[0].hist(group.probability,bins=np.linspace(0,1,21),alpha=.45,label=category)
        axes[0].legend(); axes[0].set(xlabel='Raw probability',ylabel='Cases',title=name+' error distributions')
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(frame.actual,decision,ax=axes[1],colorbar=False)
        fig.tight_layout(); fig.savefig(target/f'{name}_error_patterns.png',dpi=140); plt.close(fig)
    save_json(target/'summary.json',reports)
    print('Error analysis complete',flush=True)
if __name__=='__main__': main()
