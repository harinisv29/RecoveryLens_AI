import numpy as np
import pytest
from training.evaluate import bootstrap_intervals
from utils.experiments import run_paths

def test_cluster_bootstrap_reproducible_and_bounded():
    y=[0,0,1,1,0,1,0,1]; p=[.1,.3,.6,.9,.2,.7,.4,.8]; subjects=[1,1,2,3,4,4,5,6]
    a=bootstrap_intervals(y,p,subjects,repetitions=100)
    b=bootstrap_intervals(y,p,subjects,repetitions=100)
    assert a==b and a['patient_clusters']==6
    assert a['valid_resamples']+a['single_class_skipped']==100
    assert all(0<=lo<=hi<=1 for lo,hi in [a['auroc'],a['auprc']])

def test_named_paths_cannot_escape():
    with pytest.raises(ValueError): run_paths('../escape')

def test_frozen_initial_checkpoint_matches_backup():
    from config import ROOT
    import hashlib
    original=ROOT/'models/checkpoints/transformer.pt'
    backup=ROOT/'results/runs/initial_127/models/checkpoints/transformer.pt'
    if not backup.exists(): pytest.skip('No initial experiment backup')
    assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(backup.read_bytes()).digest()

def test_expanded_cohort_if_available():
    import torch
    import json
    from config import ROOT
    active=ROOT/'models/active_model.json'
    if not active.exists(): pytest.skip('Expanded model not activated')
    selection=json.loads(active.read_text()); run=selection['run']
    data=torch.load(ROOT/f'data/processed/{run}/dataset.pt',weights_only=False)
    initial=json.loads((ROOT/'results/runs/initial_127/data/metadata/cohort.json').read_text())
    exposed={r['subjectid'] for r in initial['cohort']}
    subjects={name:{data['cohort'][i]['subjectid'] for i in ids} for name,ids in data['splits'].items()}
    assert not (subjects['train']&subjects['val'] or subjects['train']&subjects['test'] or subjects['val']&subjects['test'])
    assert not exposed&(subjects['val']|subjects['test'])
    expected=np.quantile([data['cohort'][i]['los'] for i in data['splits']['train']],.75)
    assert data['target_days']==expected
    values=np.concatenate([data['x'][i] for i in data['splits']['train']])
    for feature in range(7):
        observed=values[values[:,feature+7]==1,feature]
        if len(observed): assert abs(observed.mean())<1e-4

@pytest.mark.parametrize('kind',['gru','xgboost'])
def test_actual_baseline_can_serve_explanations(kind,tmp_path,monkeypatch):
    """Use only previously trained real-data artifacts; do not fit synthetic models."""
    import json
    import shutil
    import torch
    from config import ROOT
    import inference.predict as inference
    from utils.validation import validate_csv
    sample=ROOT/'data/sample/real_recording.csv'
    source=ROOT/'models/checkpoints'/('gru.pt' if kind=='gru' else 'xgboost.json')
    if not source.exists() or not sample.exists(): pytest.skip('Initial real model/sample unavailable')
    if kind=='xgboost':
        initial=torch.load(ROOT/'models/checkpoints/transformer.pt',map_location='cpu',weights_only=False)
        metrics=__import__('json').loads((ROOT/'results/metrics/xgboost.json').read_text())
        metadata={key:initial[key] for key in ['stats','target_days','version','cohort_size']}
        metadata.update(model='xgboost',threshold=metrics['test']['threshold'])
        shutil.copy2(source,tmp_path/'xgboost.json')
        (tmp_path/'xgboost_metadata.json').write_text(json.dumps(metadata))
        source=tmp_path/'xgboost.json'
    model,checkpoint=inference.load_checkpoint(source)
    monkeypatch.setattr(inference,'load_checkpoint',lambda:(model,checkpoint))
    result=inference.predict(validate_csv(sample))
    assert 0<=result['probability']<=1 and result['model_type']==kind
    assert len(result['explanation']['temporal_heatmap'])==result['quality']['minutes']
    assert len(result['explanation']['feature_attribution'])==7
    assert np.isfinite(result['explanation']['temporal_heatmap']).all()
