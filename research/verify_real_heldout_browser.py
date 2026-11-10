"""Use the actual live upload workflow for the preselected held-out patient CSV."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

OUT=ROOT/'results/real_patient_test'
def main():
    if (OUT/'browser_verification.json').exists():raise RuntimeError('Browser evidence exists; preserve it')
    direct=json.loads((OUT/'selection_and_inference.json').read_text())
    errors=[];base='http://127.0.0.1:5050'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(base,wait_until='networkidle');page.wait_for_function("document.querySelector('#status-label').textContent==='Transformer ready'")
        status=page.request.get(base+'/api/model-status').json()
        assert status['checkpoint_sha256']==direct['checkpoint_sha256'] and status['checkpoint_load_count']==1
        assert page.locator('#prediction-result').is_hidden()
        page.locator('#file').set_input_files(direct['csv_path'])
        assert page.locator('#filename').inner_text()=='vitaldb_heldout_test.csv'
        with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/api/analysis-jobs')) as posted:
            page.locator('#analyze').click()
        assert posted.value.status==202;job_info=posted.value.json()
        page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'",timeout=120000)
        assert page.locator('#processing-accordion').count()==1
        assert page.locator('.workflow-step.completed').count()==5
        job=page.request.get(base+job_info['status_url']).json();assert job['state']=='succeeded'
        response=page.request.get(base+job_info['result_url']).json()['result']
        assert response['input_sha256']==direct['input_sha256']
        assert response['checkpoint_sha256']==direct['checkpoint_sha256']
        assert abs(response['probability']-direct['sigmoid_probability'])<=1e-7
        assert response['binary_prediction']==direct['predicted_label']
        assert abs(response['explanation']['prediction_logit']-direct['raw_logit'])<=1e-6
        assert not {'caseid','subjectid','ground_truth_label','postoperative_stay_days'}&set(response)
        assert page.locator('#probability').inner_text()==f"{100*direct['sigmoid_probability']:.1f}%"
        assert 'output '+str(direct['predicted_label']) in page.locator('#model-version').inner_text()
        assert page.locator('#prediction-section').evaluate('e=>e.open')
        page.screenshot(path=str(OUT/'browser_prediction.png'),full_page=True)
        page.locator('#monitoring-section>summary').click();page.wait_for_function('Chart.getChart(document.querySelector("#hr")).width>100')
        for id,feature in [('hr','HR'),('spo2','SpO2'),('sbp','SBP'),('dbp','DBP'),('map','MAP'),('co2','EtCO2')]:
            displayed=page.evaluate('(id)=>Chart.getChart(document.querySelector("#"+id)).data.datasets[0].data',id)
            assert displayed==response['signals'][feature]
        assert page.locator('#rr-card').is_hidden()
        page.screenshot(path=str(OUT/'browser_signals.png'),full_page=True)
        page.locator('#explanation-section>summary').click();page.wait_for_function('Chart.getChart(document.querySelector("#features")).width>100')
        bars=page.evaluate('Chart.getChart(document.querySelector("#features")).data.datasets[0].data')
        assert bars==response['explanation']['feature_attribution']
        assert np.allclose(bars,direct['independent_feature_attribution'],atol=1e-6)
        heat=np.asarray(response['explanation']['temporal_heatmap']);assert np.allclose(heat,direct['independent_temporal_attribution'],atol=1e-6)
        assert page.locator('#heatmap').is_visible()
        page.screenshot(path=str(OUT/'browser_explainability.png'),full_page=True)
        page.locator('#report-section>summary').click();page.wait_for_function("document.querySelector('#report-section').open")
        with page.expect_download(timeout=120000) as download:page.locator('#report').click()
        pdf=OUT/'real_patient_prediction_report.pdf';download.value.save_as(str(pdf));reader=PdfReader(pdf)
        text='\n'.join(item.extract_text() for item in reader.pages)
        assert f"{100*direct['sigmoid_probability']:.2f}%" in text and '0.370' in text and '7.39610' in text
        assert 'Case ID' not in text and 'subjectid' not in text.lower()
        assert 'ground-truth' not in text.lower()
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
        assert not errors,errors
        evidence={'url':base,'checkpoint_load_count':status['checkpoint_load_count'],'validation_succeeded':True,'all_five_actual_stages_completed':True,'actual_stage_events':job['events'],'displayed_percentage':f"{100*direct['sigmoid_probability']:.1f}%",'displayed_binary_label':response['binary_prediction'],'api_probability':response['probability'],'independent_probability':direct['sigmoid_probability'],'absolute_probability_difference':abs(response['probability']-direct['sigmoid_probability']),'input_sha256_match':True,'checkpoint_sha256_match':True,'raw_signal_charts_exactly_match_response':True,'feature_attribution_matches_independent':True,'temporal_attribution_matches_independent':True,'pdf_downloaded':True,'pdf_pages':len(reader.pages),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'identifiers_and_ground_truth_not_in_public_input_or_response':True,'no_explicit_patient_id_in_pdf':True,'mobile_no_overflow':True,'javascript_errors':errors,'actual_prediction_response':response}
        (OUT/'browser_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print(json.dumps({key:evidence[key] for key in ['displayed_percentage','api_probability','displayed_binary_label','absolute_probability_difference','pdf_pages','javascript_errors']},indent=2))
        browser.close()
if __name__=='__main__':main()
