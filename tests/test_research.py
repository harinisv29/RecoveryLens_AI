"""Research checks use real frozen artifacts; software tests never supply training outcomes."""
import json
import numpy as np
import pandas as pd
import pytest
from research.common import ROOT,OUT,CALIBRATORS,require_audit,read_json,digest,predictions
from research.calibration import apply_calibrator
from research.subgroups import clustered_intervals

def test_verified_cohort_and_preserved_artifacts():
    audit=require_audit()
    assert audit['passed'] and audit['splits']['test']['cases']==1186
    manifest=read_json(OUT/'preserved_manifest.json')
    # The final application task explicitly authorizes an updated README. Its
    # original bytes must still be preserved; all other research inputs stay fixed.
    authorized=read_json(ROOT/'results/transformer_final_preservation.json').get('authorized_changes',{}) if (ROOT/'results/transformer_final_preservation.json').exists() else {}
    for name,value in manifest.items():
        if name in authorized:
            change=authorized[name]
            assert name=='README.md' and change['original_sha256']==value
            assert digest(ROOT/change['backup'])==value
            assert digest(ROOT/name)==change['updated_sha256']
        else: assert digest(ROOT/name)==value

def test_paired_bootstrap_cases_and_distribution():
    frames=predictions(); assert len(frames['transformer'])==1186
    report=read_json(OUT/'paired/comparison.json')
    distribution=pd.read_csv(OUT/'paired/bootstrap_distributions.csv')
    assert report['valid_resamples']>=2000 and len(distribution)==report['valid_resamples']
    for item in report['comparisons']:
        for metric in ['auroc','auprc']:
            column=f"{item['first']}_minus_{item['second']}_{metric}"
            assert np.allclose(np.quantile(distribution[column],[.025,.975]),item[metric]['ci95'])

def test_vectorized_bootstrap_matches_sklearn_with_patient_weights():
    from sklearn.metrics import roc_auc_score,average_precision_score
    frame=predictions()['transformer'].iloc[:100]
    y=frame.actual.to_numpy(); p=frame.probability.to_numpy(); unique,inverse=np.unique(frame.subjectid,return_inverse=True)
    rng=np.random.default_rng(9); samples=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=30); values=[]
    for counts in samples:
        w=counts[inverse]
        if (w*y).sum() and (w*(1-y)).sum(): values.append([roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)])
    expected=np.quantile(values,[.025,.975],axis=0)
    actual=clustered_intervals(y,p,frame.subjectid,.37,repetitions=30,seed=9)
    assert np.allclose(actual['auroc'],expected[:,0])
    assert np.allclose(actual['auprc'],expected[:,1])

def test_calibration_fitted_only_on_validation_and_reproducible():
    val=predictions('val'); test=predictions()
    for name,frame in val.items():
        config=read_json(CALIBRATORS/f'{name}.json')
        assert config['fit_cases']==len(frame)==952 and not set(frame.subjectid)&set(test[name].subjectid)
        folds=pd.read_csv(OUT/f'calibration/{name}_validation_oof.csv')
        assert folds.groupby('subjectid').fold.nunique().max()==1
        saved=pd.read_csv(OUT/f'calibration/{name}_test.csv')
        assert np.allclose(apply_calibrator(config,test[name].probability),saved.calibrated_probability)

def test_subgroup_small_cells_suppressed():
    results=read_json(OUT/'subgroups/summary.json')['results']
    assert any(not r['stable_metrics_reported'] for r in results)
    for row in results:
        if not row['stable_metrics_reported']: assert 'auroc' not in row and 'suppression_reason' in row
        if 'ci95' in row: assert row['n']>=100 and row['positive']>=20 and row['negative']>=20

def test_retrained_ablation_features_and_prefixes():
    path=OUT/'ablations/summary.json'
    if not path.exists(): pytest.skip('Ablations still running')
    report=read_json(path); assert len(report['variants'])==9
    import torch
    for row in report['variants']:
        checkpoint=torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False)
        assert checkpoint['architecture']['input_dim']==len(checkpoint['columns'])
        assert row['target_days']==require_audit()['threshold_days']
        assert row['split_sizes']=={'train':3981,'val':952,'test':1186}
        assert row['epochs_completed']>0 and row['gpu']=='NVIDIA GeForce RTX 4070 SUPER'
        if checkpoint['window']: assert checkpoint['window'] in [30,60,90]

def test_explanation_numeric_checks():
    path=OUT/'explainability/summary.json'
    if not path.exists(): pytest.skip('Explainability verification still running')
    report=read_json(path)
    assert report['normalization_matches_audited_dataset']
    assert report['summary']['all_padding_checks_pass']
    for row in report['records']:
        assert row['application_32_matches_reimplementation_max_abs']<1e-5
        assert row['missing_value_attribution_max_abs']<1e-7

def test_additive_dashboard_calibration_and_pdf():
    from research_app import app
    import io
    from pypdf import PdfReader
    client=app.test_client()
    assert b'research-validation' in client.get('/').data
    summary=client.get('/api/research/summary')
    if summary.status_code==503: pytest.skip('Research dashboard summary not built yet')
    assert summary.json['cohort_cases']==6119
    sample=(ROOT/'data/sample/real_recording.csv').read_bytes()
    result=client.post('/api/predict',data={'file':(io.BytesIO(sample),'real.csv')})
    assert result.status_code==200 and result.json['version']=='full_20261110'
    assert result.json['research_calibration']['method']=='identity'
    assert not result.json['research_calibration']['applied']
    pdf=client.post('/api/report',data={'file':(io.BytesIO(sample),'real.csv')})
    assert pdf.status_code==200 and pdf.data.startswith(b'%PDF')
    text='\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf.data)).pages)
    assert 'Research validation and probability disclosure' in text and 'identity' in text
