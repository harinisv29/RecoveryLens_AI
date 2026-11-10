import io
import pytest
import app as module
from config import ROOT

@pytest.fixture
def client():
    module.app.config['TESTING']=True
    return module.app.test_client()

def test_health_and_dashboard(client):
    assert client.get('/api/health').json['status']=='ok'
    assert client.get('/').status_code==200
    assert 'available' in client.get('/api/model-status').json

def test_invalid_upload(client):
    assert client.post('/api/predict').status_code==400
    assert client.post('/api/predict',data={'file':(io.BytesIO(b'hello'),'x.exe')}).status_code==400

def test_missing_checkpoint(client,monkeypatch):
    def unavailable(*a,**k): raise FileNotFoundError('No trained checkpoint')
    monkeypatch.setattr(module,'predict',unavailable)
    csv=b'time_min,HR\n0,80\n1,81\n2,82\n3,83\n4,84'
    assert client.post('/api/predict',data={'file':(io.BytesIO(csv),'x.csv')}).status_code==503

def test_real_prediction_and_pdf(client):
    path=ROOT/'data/sample/real_recording.csv'
    if not path.exists() or not (ROOT/'models/checkpoints/transformer.pt').exists():
        pytest.skip('Requires real sample and trained checkpoint')
    response=client.post('/api/predict',data={'file':(io.BytesIO(path.read_bytes()),'real.csv')})
    assert response.status_code==200 and 0<=response.json['probability']<=1
    assert len(response.json['explanation']['feature_attribution'])==7
    pdf=client.post('/api/report',data={'file':(io.BytesIO(path.read_bytes()),'real.csv')})
    assert pdf.status_code==200 and pdf.data.startswith(b'%PDF')
