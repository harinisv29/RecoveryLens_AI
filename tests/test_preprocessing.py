import io
import numpy as np
import pandas as pd
import pytest
from utils.validation import validate_csv
from utils.preprocessing import resample,fit_stats,transform

def test_invalid_and_missing_values():
    a=resample(pd.DataFrame({'time_min':range(7),'HR':[80,90,999,np.nan,100,90,80]}))
    stats=fit_stats([a]); x=transform(a,stats)
    assert x.shape==(7,14) and np.isfinite(x).all()
    assert x[2,7]==0 and x[0,7]==1
    assert np.all(x[:,8:]==0)

@pytest.mark.parametrize('csv',['time_min,HR\n0,80\n0,80\n2,80\n3,80\n4,80','time_min,HR\n0,x\n1,80\n2,80\n3,80\n4,80','foo\n1\n2\n3\n4\n5'])
def test_reject_invalid(csv):
    with pytest.raises(ValueError): validate_csv(io.StringIO(csv))

def test_resample_uses_elapsed_time():
    a=resample(pd.DataFrame({'time_min':[0,.5,2,3,4,5],'HR':[80,100,90,90,90,90]}))
    assert a[0,0]==90 and np.isnan(a[1,0])

def test_real_dataset_if_available():
    from config import ROOT
    path=ROOT/'data/metadata/cases.csv'
    if not path.exists(): pytest.skip('Real dataset not downloaded')
    data=pd.read_csv(path)
    assert len(data)==6388 and {'opstart','opend','dis','subjectid'}.issubset(data)

def test_outcome_exclusions_and_reference():
    from utils.outcomes import audit_outcomes
    frame=pd.DataFrame({'dis':[172800,-1,np.nan,86400],'opend':[86400,100,100,100],'opstart':[100,10,10,10],'adm':[-100]*4,'death_inhosp':[0,0,0,1]})
    audit=audit_outcomes(frame)
    assert audit.los_postop.iloc[0]==1
    assert audit.outcome_eligible.tolist()==[True,False,False,False]
    assert 'negative' in audit.exclusion_reasons.iloc[1]
    assert 'discharge' in audit.exclusion_reasons.iloc[2]
    assert 'death' in audit.exclusion_reasons.iloc[3]

def test_saved_cohort_has_no_patient_leakage():
    from config import ROOT
    import torch
    path=ROOT/'data/processed/dataset.pt'
    if not path.exists(): pytest.skip('Prepared dataset unavailable')
    data=torch.load(path,weights_only=False)
    groups=[{data['cohort'][i]['subjectid'] for i in data['splits'][name]} for name in ['train','val','test']]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    expected=np.quantile([data['cohort'][i]['los'] for i in data['splits']['train']],.75)
    assert data['target_days']==expected
    normalized=np.concatenate([data['x'][i] for i in data['splits']['train']])
    for feature in range(7):
        observed=normalized[normalized[:,feature+7]==1,feature]
        if len(observed): assert abs(observed.mean())<1e-4
