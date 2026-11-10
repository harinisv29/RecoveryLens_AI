import io
import numpy as np
import pytest
import torch
from config import ROOT
from utils.validation import validate_csv
from utils.preprocessing import resample,transform

@pytest.fixture(scope='module')
def module():
    import transformer_app
    return transformer_app

def test_live_checkpoint_schema_and_cached_loading(module,monkeypatch):
    assert module.engine is not None
    assert module.engine.checkpoint['model']=='transformer'
    assert module.engine.model.embedding.in_features==14
    def forbidden(*args,**kwargs): raise AssertionError('Checkpoint reloaded after startup')
    monkeypatch.setattr(torch,'load',forbidden)
    client=module.app.test_client()
    for _ in range(2):
        status=client.get('/api/model-status').json
        assert status['model_type']=='transformer' and status['checkpoint_load_count']==1
    sample=(ROOT/'data/sample/real_recording.csv').read_bytes()
    result=client.post('/api/predict',data={'file':(io.BytesIO(sample),'real.csv')})
    assert result.status_code==200 and result.json['model_type']=='transformer'

def test_exact_training_transform_and_reproducibility(module):
    frame=validate_csv(ROOT/'data/sample/real_recording.csv')
    result=module.engine.predict(frame); repeated=module.engine.predict(frame)
    x=torch.tensor(transform(resample(frame),module.engine.checkpoint['stats']))[None]
    mask=torch.ones(x.shape[:2],dtype=torch.bool)
    with torch.inference_mode(): expected=torch.sigmoid(module.engine.model(x,mask)).item()
    assert result['probability']==repeated['probability']==expected
    assert result['explanation']==repeated['explanation']
    assert result['explanation']['input_sha256']==result['input_sha256']
    assert result['checkpoint_sha256']==result['explanation']['checkpoint_sha256']
    assert result['target_days']>7 and 0<result['threshold']<1
    assert result['binary_prediction']==int(result['probability']>=result['threshold'])

def test_example_invalid_quality_and_pdf(module):
    client=module.app.test_client(); example=client.get('/api/example-csv')
    assert example.status_code==200 and 'attachment' in example.headers['Content-Disposition']
    assert validate_csv(io.BytesIO(example.data)).shape[0]>=5
    bad=b'time_min,HR\n0,80\n1,81\n2,82\n3,83\n4,84'
    assert client.post('/api/predict',data={'file':(io.BytesIO(bad),'thin.csv')}).status_code==400
    assert client.post('/api/predict',data={'file':(io.BytesIO(b'bad'),'x.exe')}).status_code==400
    pdf=client.post('/api/report',data={'file':(io.BytesIO(example.data),'example.csv')})
    assert pdf.status_code==200 and pdf.data.startswith(b'%PDF')
    assert pdf.headers['X-Live-Model']=='Temporal Transformer'
    from pypdf import PdfReader
    text='\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf.data)).pages)
    assert '7.40' in text and 'Research validation and probability disclosure' in text

def test_no_baseline_or_ablation_live_switch(module):
    client=module.app.test_client()
    assert client.get('/api/model-status?model=gru').json['model_type']=='transformer'
    assert client.get('/api/model-status?model=xgboost').json['model_type']=='transformer'
    assert client.get('/api/model-status?model=first_30').json['model_type']=='transformer'
    assert client.get('/api/research/summary').status_code==200
