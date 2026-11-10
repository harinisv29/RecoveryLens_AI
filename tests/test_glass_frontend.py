import hashlib
import json
from config import ROOT
import pytest

def test_glass_redesign_leaves_backend_and_research_unchanged():
    manifest=json.loads((ROOT/'results/glass_redesign_preservation.json').read_text())
    for path,expected in manifest['protected_files'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path

def test_glass_ui_routes_and_controls():
    from transformer_app import app
    client=app.test_client(); html=client.get('/').get_data(as_text=True)
    assert 'Postoperative recovery analysis' in html and 'single-workspace' in html
    assert 'id="sidebar"' not in html and 'sidebar-scrim' not in html
    assert 'medical-glass.css' in html and 'glass-dashboard.js' in html
    assert all(label in html for label in ['Upload Recording','AI Processing','Prediction Results','Physiological Monitoring','AI Explainability','Download Report'])
    assert all(name in html for name in ['Systolic pressure','Diastolic pressure','Mean arterial pressure','Respiratory rate'])
    assert 'chart-zoom-in' in html and 'heatmap-readout' in html and 'view-report' in html
    assert client.get('/performance').status_code==200

def test_independently_verified_synthetic_probability_unchanged():
    from pathlib import Path
    from utils.validation import validate_csv
    from transformer_app import engine
    csv=Path(r'C:\Users\Admin\Downloads\recoverylens_synthetic_test.csv')
    if not csv.exists(): pytest.skip('User-owned synthetic verification CSV not available')
    result=engine.predict(validate_csv(csv))
    assert result['probability']==0.1542341262102127
    assert abs(result['explanation']['prediction_logit']-(-1.7017707824707031))<1e-8
    assert result['binary_prediction']==0
    assert result['threshold']==pytest.approx(.370) and result['target_days']==pytest.approx(7.396099537037037)
