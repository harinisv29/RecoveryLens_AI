"""Validation-only calibration selection and separate test diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit,logit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import brier_score_loss
from research.common import OUT,CALIBRATORS,SOURCE,ROOT,require_audit,predictions,save_json,digest

def logits(p): return logit(np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6))

def logistic_fit(y,p):
    y=np.asarray(y,dtype=float); z=logits(p); design=np.column_stack([np.ones(len(z)),z])
    def objective(beta):
        eta=design@beta
        return np.mean(np.logaddexp(0,eta)-y*eta),design.T@(expit(eta)-y)/len(y)
    fit=minimize(objective,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-8})
    return {'intercept':float(fit.x[0]),'slope':float(fit.x[1]),'optimizer_success':bool(fit.success),'gradient_norm':float(np.linalg.norm(fit.jac))}

def calibration_metrics(y,p,bins=10):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    categories=np.minimum((p*bins).astype(int),bins-1)
    rows=[]
    for k in range(bins):
        take=categories==k
        if take.any(): rows.append({'bin':k,'n':int(take.sum()),'predicted':float(p[take].mean()),'observed':float(y[take].mean())})
    result={'n':len(y),'events':int(y.sum()),'brier':float(np.mean((y-p)**2)),'ece':float(sum(r['n']/len(y)*abs(r['predicted']-r['observed']) for r in rows)),'mean_predicted':float(p.mean()),'observed_fraction':float(y.mean()),'mean_prediction_minus_observed':float(p.mean()-y.mean()),'bins':rows,'binning':'10 fixed equal-width probability bins; empty bins omitted'}
    if len(np.unique(y))==2:
        regression=logistic_fit(y,p)
        z=logits(p)
        offset=minimize_scalar(lambda a:np.mean(np.logaddexp(0,z+a)-y*(z+a)),bounds=(-20,20),method='bounded')
        result.update(calibration_intercept=regression['intercept'],calibration_slope=regression['slope'],calibration_in_the_large=float(offset.x),diagnostic_fit=regression)
    else: result.update(calibration_intercept=None,calibration_slope=None,calibration_in_the_large=None)
    return result

def fit_calibrator(method,y,p):
    if method=='identity': return {'method':'identity'}
    if method=='platt': return {'method':'platt',**logistic_fit(y,p)}
    estimator=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(p,y)
    return {'method':'isotonic','x':estimator.X_thresholds_.tolist(),'y':estimator.y_thresholds_.tolist()}

def apply_calibrator(config,p):
    if config['method']=='identity': return np.asarray(p,dtype=float)
    if config['method']=='platt': return expit(config['intercept']+config['slope']*logits(p))
    return np.interp(p,config['x'],config['y'])

def main():
    require_audit(); val=predictions('val'); test=predictions(); target=OUT/'calibration'; target.mkdir(exist_ok=True)
    reports={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,frame in val.items():
        y=frame.actual.to_numpy(); p=frame.probability.to_numpy(); folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
        oof={method:np.zeros(len(frame)) for method in ['identity','platt','isotonic']}; assignments=np.zeros(len(frame),dtype=int)
        for number,(a,b) in enumerate(folds.split(p,y,frame.subjectid)):
            assert not set(frame.iloc[a].subjectid)&set(frame.iloc[b].subjectid)
            assignments[b]=number
            for method in oof: oof[method][b]=apply_calibrator(fit_calibrator(method,y[a],p[a]),p[b])
        scores={method:float(brier_score_loss(y,values)) for method,values in oof.items()}
        selected=min(scores,key=scores.get); config=fit_calibrator(selected,y,p)
        base_path=ROOT/f'models/checkpoints/{SOURCE}'/(name+('.json' if name=='xgboost' else '.pt'))
        config.update(model=name,source_run=SOURCE,checkpoint_sha256=digest(base_path),selection='Lowest five-fold patient-group cross-fitted validation Brier; identity included',validation_oof_brier=scores,fit_cases=len(frame),fit_patients=int(frame.subjectid.nunique()),fit_scope='Validation only; validation also used during base-model selection, so CV is not an independent outer validation')
        path=CALIBRATORS/f'{name}.json'
        if path.exists(): raise RuntimeError('Calibrator exists; refusing overwrite')
        save_json(path,config)
        calibrated=apply_calibrator(config,test[name].probability)
        output=test[name].copy(); output['calibrated_probability']=calibrated; output.to_csv(target/f'{name}_test.csv',index=False)
        pd.DataFrame({'caseid':frame.caseid,'subjectid':frame.subjectid,'actual':y,'fold':assignments,**oof}).to_csv(target/f'{name}_validation_oof.csv',index=False)
        raw=calibration_metrics(output.actual,output.probability); adjusted=calibration_metrics(output.actual,calibrated)
        reports[name]={'selected_method':selected,'validation_oof_brier':scores,'test_raw':raw,'test_calibrated':adjusted,'test_brier_change':adjusted['brier']-raw['brier'],'diagnostic_note':'Test intercept/slope are descriptive diagnostic regressions only; never deployed as calibration transforms. All deployable calibration fitting and method choice used validation patients only.'}
        fig,(ax,hist)=plt.subplots(1,2,figsize=(10,4))
        ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
        for label,m in [('Raw',raw),(selected,adjusted)]: ax.plot([r['predicted'] for r in m['bins']],[r['observed'] for r in m['bins']],'o-',label=label)
        ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction',title=name+' held-out reliability'); ax.legend()
        hist.hist(output.probability,bins=20,alpha=.5,label='Raw'); hist.hist(calibrated,bins=20,alpha=.5,label=selected); hist.legend(); hist.set(xlabel='Probability',ylabel='Cases')
        fig.tight_layout(); fig.savefig(target/f'{name}_reliability.png',dpi=140); plt.close(fig)
        print(name,selected,scores,'test Brier',raw['brier'],adjusted['brier'],flush=True)
    save_json(target/'summary.json',reports)
if __name__=='__main__': main()
