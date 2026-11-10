"""Predeclared paired patient bootstrap comparisons on identical frozen cases."""
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from research.common import OUT,require_audit,predictions,save_json

def paired_bootstrap(frames,repetitions=2000,seed=20261110):
    reference=next(iter(frames.values())); y=reference.actual.to_numpy(); subjects=reference.subjectid.to_numpy()
    groups=[np.flatnonzero(subjects==subject) for subject in np.unique(subjects)]
    names=list(frames); rng=np.random.default_rng(seed); values=[]; skipped=0
    for _ in range(repetitions):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))<2: skipped+=1; continue
        row=[]
        for name in names:
            p=frames[name].probability.to_numpy()[indices]
            row += [roc_auc_score(y[indices],p),average_precision_score(y[indices],p)]
        values.append(row)
    distributions=np.asarray(values).reshape(-1,len(names),2)
    result={'method':'Paired patient-cluster percentile bootstrap; same resampled patients for all models','seed':seed,'requested_resamples':repetitions,'valid_resamples':len(values),'single_class_skipped':skipped,'patient_clusters':len(groups),'comparisons':[],'limitations':'Conditional on fitted models and a previously examined holdout. Marginal 95% intervals, no multiplicity adjustment; no confirmatory superiority claim.'}
    exports={}
    for a,b in combinations(range(len(names)),2):
        item={'first':names[a],'second':names[b]}
        for j,(metric,fn) in enumerate([('auroc',roc_auc_score),('auprc',average_precision_score)]):
            delta=distributions[:,a,j]-distributions[:,b,j]; ci=np.quantile(delta,[.025,.975])
            point=fn(y,frames[names[a]].probability)-fn(y,frames[names[b]].probability)
            item[metric]={'difference':point,'ci95':ci,'includes_zero':bool(ci[0]<=0<=ci[1]),'bootstrap_fraction_nonpositive':float(np.mean(delta<=0)),'interpretation':'Interval overlaps zero; direction uncertain' if ci[0]<=0<=ci[1] else 'Marginal interval excludes zero; exploratory conditional evidence only'}
            exports[names[a]+'_minus_'+names[b]+'_'+metric]=delta
        result['comparisons'].append(item)
    return result,pd.DataFrame(exports)

def main():
    require_audit(); result,distributions=paired_bootstrap(predictions())
    target=OUT/'paired'; target.mkdir(exist_ok=True)
    save_json(target/'comparison.json',result); distributions.to_csv(target/'bootstrap_distributions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,9))
    for ax,column in zip(axes.flat,distributions):
        ax.hist(distributions[column],bins=45,color='#277f99',alpha=.8); ax.axvline(0,color='black',linestyle='--'); ax.set(title=column.replace('_',' '),xlabel='Paired difference',ylabel='Bootstrap samples')
    fig.tight_layout(); fig.savefig(target/'paired_bootstrap.png',dpi=140); plt.close(fig)
    print(result,flush=True)
if __name__=='__main__': main()
