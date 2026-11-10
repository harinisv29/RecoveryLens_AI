import ast
import io
import time
import threading
import json
import pytest
from config import ROOT

@pytest.fixture
def application():
    import transformer_app
    return transformer_app

def submit(application,payload=None):
    client=application.app.test_client()
    payload=payload if payload is not None else (ROOT/'data/sample/real_recording.csv').read_bytes()
    response=client.post('/api/analysis-jobs',data={'file':(io.BytesIO(payload),'recording.csv')})
    assert response.status_code==202
    return response.json['job_id']

def wait(application,key):
    client=application.app.test_client(); deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        response=client.get('/api/analysis-jobs/'+key); assert response.headers['Cache-Control']=='no-store'
        if response.json['state'] in ['succeeded','failed','cancelled']: return response.json
        time.sleep(.01)
    pytest.fail('Job did not finish')

def test_genuine_stage_order_and_unchanged_prediction(application):
    key=submit(application); state=wait(application,key)
    assert state['state']=='succeeded'
    expected=[(key,status) for key in ['validation','preprocessing','inference','explainability','results'] for status in ['running','completed']]
    assert [(event['stage'],event['state']) for event in state['events']]==expected
    assert all(stage['started_at']<=stage['completed_at'] for stage in state['stages'])
    assert all('probability' not in json.dumps(event) for event in state['events'])
    result=application.app.test_client().get(f'/api/analysis-jobs/{key}/result').json['result']
    from utils.validation import validate_csv
    direct=application.engine.predict(validate_csv(ROOT/'data/sample/real_recording.csv'))
    assert result['probability']==direct['probability']
    assert result['explanation']==direct['explanation']
    assert result['quality']['minutes']==state['stages'][1]['details']['minute_bins']
    assert application.engine.load_count==1

@pytest.mark.parametrize('payload,failed_stage',[(b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80','validation'),(b'time_min,HR\n0,80\n1,81\n2,82\n3,83\n4,84','preprocessing')])
def test_invalid_and_insufficient_measurements(application,payload,failed_stage):
    key=submit(application,payload); state=wait(application,key)
    assert state['state']=='failed'
    assert next(stage for stage in state['stages'] if stage['id']==failed_stage)['state']=='failed'
    assert application.app.test_client().get(f'/api/analysis-jobs/{key}/result').status_code==422
    assert not state['result_ready']

@pytest.mark.parametrize('target,failed_stage',[('inference','inference'),('explanation','explainability')])
def test_processing_failures_withhold_result(application,monkeypatch,target,failed_stage):
    def fail(*args,**kwargs): raise RuntimeError('Injected '+target+' failure')
    if target=='inference': monkeypatch.setattr(application.engine.model,'forward',fail)
    else:
        import inference.live_transformer as service
        monkeypatch.setattr(service,'integrated_gradients',fail)
    key=submit(application); state=wait(application,key)
    assert state['state']=='failed' and state['error']=='Injected '+target+' failure'
    assert next(s for s in state['stages'] if s['id']==failed_stage)['state']=='failed'
    assert application.app.test_client().get(f'/api/analysis-jobs/{key}/result').status_code==422

def test_result_hidden_while_inference_is_running(application,monkeypatch):
    entered=threading.Event(); release=threading.Event(); original=application.engine.model.forward
    def blocked(*args,**kwargs):
        entered.set(); assert release.wait(10); return original(*args,**kwargs)
    monkeypatch.setattr(application.engine.model,'forward',blocked)
    key=submit(application)
    try:
        assert entered.wait(10)
        client=application.app.test_client(); state=client.get('/api/analysis-jobs/'+key).json
        assert next(s for s in state['stages'] if s['id']=='inference')['state']=='running'
        assert not state['result_ready'] and 'probability' not in json.dumps(state)
        assert client.get(f'/api/analysis-jobs/{key}/result').status_code==202
    finally: release.set()
    assert wait(application,key)['state']=='succeeded'

def test_sequential_uploads_and_job_isolation(application):
    first=submit(application); assert wait(application,first)['state']=='succeeded'
    second=submit(application,b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80')
    assert first!=second and wait(application,second)['state']=='failed'
    client=application.app.test_client()
    assert client.get(f'/api/analysis-jobs/{first}/result').json['job_id']==first
    assert client.get(f'/api/analysis-jobs/{second}/result').status_code==422
    assert client.delete('/api/analysis-jobs/'+first).status_code==200
    assert client.get(f'/api/analysis-jobs/{first}/result').status_code==410
    assert client.get('/api/analysis-jobs/missing').status_code==404

def test_supported_missing_measurements_are_preserved(application):
    import pandas as pd
    frame=pd.read_csv(ROOT/'data/sample/real_recording.csv')
    frame.loc[frame.index[::4],'HR']=float('nan')
    frame=frame.drop(columns=['RR'],errors='ignore')
    key=submit(application,frame.to_csv(index=False).encode())
    state=wait(application,key)
    assert state['state']=='succeeded'
    result=application.app.test_client().get(f'/api/analysis-jobs/{key}/result').json['result']
    assert result['signals']['HR'][0] is None
    assert 'RR' in result['quality']['missing_features']
    assert state['stages'][1]['details']['observed_fraction']==result['quality']['observed_fraction']

def test_progress_instrumentation_preserves_original_prediction_ast():
    original=ast.parse((ROOT/'results/workflow_before_live_transformer.py.bak').read_text(encoding='utf-8'))
    current=ast.parse((ROOT/'inference/live_transformer.py').read_text(encoding='utf-8'))
    old=next(node for node in ast.walk(original) if isinstance(node,ast.FunctionDef) and node.name=='predict')
    new=next(node for node in ast.walk(current) if isinstance(node,ast.FunctionDef) and node.name=='predict')
    new.args=old.args
    new.body=[node for node in new.body if not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='notify' for t in node.targets)) and not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='notify')]
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)

def test_separate_performance_page(application):
    client=application.app.test_client(); dashboard=client.get('/').get_data(as_text=True); performance=client.get('/performance').get_data(as_text=True)
    assert 'processing-accordion' in dashboard and 'AI Analysis in Progress' in dashboard
    assert 'model-metrics' not in dashboard and 'research-content' not in dashboard
    assert 'research-content' in performance and 'model-metrics' in performance
    assert 'Superiority over GRU was not statistically established' in performance
