import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

def metrics(y,p,threshold):
    pred=np.asarray(p)>=threshold
    observed, predicted=calibration_curve(y,p,n_bins=5,strategy='quantile')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    extra={'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None}
    return {**extra,'n':len(y),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'brier_score':float(brier_score_loss(y,p)),'threshold':float(threshold),'calibration':{'observed':observed.tolist(),'predicted':predicted.tolist()}}

def tune_threshold(y,p):
    candidates=np.linspace(.05,.95,91)
    return float(max(candidates,key=lambda t:f1_score(y,np.asarray(p)>=t,zero_division=0)))

def bootstrap_intervals(y,p,subjects,repetitions=2000,seed=42):
    """Patient-cluster percentile bootstrap, keeping all observations per sampled patient."""
    y,p,subjects=np.asarray(y),np.asarray(p),np.asarray(subjects)
    unique=np.unique(subjects); groups=[np.flatnonzero(subjects==subject) for subject in unique]
    rng=np.random.default_rng(seed); values=[]; skipped=0
    for _ in range(repetitions):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[indices]))<2: skipped+=1; continue
        values.append([roc_auc_score(y[indices],p[indices]),average_precision_score(y[indices],p[indices])])
    if not values: raise ValueError('No valid two-class bootstrap resamples')
    bounds=np.quantile(np.asarray(values),[.025,.975],axis=0)
    return {'method':'Patient-cluster percentile bootstrap; conditional on fitted model and fixed endpoint','confidence_level':.95,'seed':seed,'requested':repetitions,'valid_resamples':len(values),'single_class_skipped':skipped,'patient_clusters':len(unique),'auroc':bounds[:,0].tolist(),'auprc':bounds[:,1].tolist()}

def save_error_analysis(name,data,indices,probabilities,threshold,output):
    import pandas as pd
    import json
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,probability in zip(indices,probabilities):
        row=dict(data['cohort'][index]); actual=int(data['y'][index]); predicted=int(probability>=threshold)
        row.update(actual=actual,predicted=predicted,probability=float(probability),error_type=('TP' if actual else 'FP') if predicted else ('FN' if actual else 'TN'),absolute_error=abs(actual-probability))
        rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(output/f'{name}_case_predictions.csv',index=False)
    group_reports={}
    for key,series in [('duration_minutes',pd.cut(frame.minutes,[0,60,120,240,720],include_lowest=True)),('observed_fraction',pd.cut(frame.observed_fraction,[0,.5,.8,1],include_lowest=True))]:
        reports=[]
        for value,group in frame.groupby(series,observed=True):
            reports.append({'group':str(value),'n':len(group),'positive_count':int(group.actual.sum()),'error_count':int((group.actual!=group.predicted).sum()),'brier':float(((group.actual-group.probability)**2).mean())})
        group_reports[key]=reports
    report={'threshold':threshold,'counts':frame.error_type.value_counts().to_dict(),'subgroups':group_reports,'largest_probability_errors':frame.nlargest(20,'absolute_error').to_dict('records'),'limitations':'Exploratory error descriptions only. No model tuning based on test subgroups. Small cells are unstable.'}
    (output/f'{name}_error_analysis.json').write_text(json.dumps(report,indent=2))
    lines=[f'# {name}: held-out error analysis','',f"Decision threshold fixed by validation F1: {threshold:.4f}. Counts: {report['counts']}.",'','| Case | Outcome | Predicted | Probability | Error | LOS days | Minutes | Observed fraction |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in report['largest_probability_errors']:
        lines.append(f"| {row['caseid']} | {row['actual']} | {row['predicted']} | {row['probability']:.4f} | {row['error_type']} | {row['los']:.2f} | {row['minutes']} | {row['observed_fraction']:.2f} |")
    lines+=['',report['limitations'],'','Full predictions and exploratory subgroup summaries are saved in the neighboring CSV and JSON files. Test errors do not alter model selection, endpoint, or cutoff.']
    (output/f'{name}_error_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
