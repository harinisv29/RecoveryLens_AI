"""Live Transformer-only acceptance test with a genuine downloaded recording."""
import argparse
import io
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--url',required=True); parser.add_argument('--label',required=True); args=parser.parse_args()
    prefix=ROOT/'results'/args.label
    if prefix.with_suffix('.json').exists(): raise RuntimeError('Preserve previous verification evidence; choose a new label')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100}); page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle')
        page.wait_for_function("document.querySelectorAll('#model-metrics .metric').length===5")
        status=page.request.get(args.url+'/api/model-status').json()
        assert status['model_type']=='transformer' and status['checkpoint_load_count']==1 and status['input_dimension']==14
        assert status['version']=='full_20261110' and abs(status['decision_threshold']-.37)<1e-8
        sample=page.request.get(args.url+'/api/example-csv'); assert sample.status==200
        page.locator('#try-example').click(); page.wait_for_function("document.querySelector('#filename').textContent.includes('real-example.csv')")
        with page.expect_response(lambda response:response.url.endswith('/api/predict') and response.request.method=='POST') as prediction:
            page.locator('#analyze').click()
        result=prediction.value.json(); assert prediction.value.status==200
        assert result['model_type']=='transformer' and result['binary_prediction'] in [0,1]
        assert result['input_sha256']==result['explanation']['input_sha256']
        assert result['checkpoint_sha256']==status['checkpoint_sha256']==result['explanation']['checkpoint_sha256']
        page.wait_for_function("!document.querySelector('#report').disabled")
        assert page.locator('#error').inner_text()==''
        assert '7.39610' in page.locator('#target').inner_text() and '0.370' in page.locator('#target').inner_text()
        assert 'not clinically calibrated' in page.locator('.probability-note').inner_text().lower()
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).data.datasets[0].data.length')==result['quality']['minutes']
        assert page.evaluate('Chart.getChart(document.querySelector("#features")).data.datasets[0].data.length')==7
        page.wait_for_selector('#research-subgroup-attribute'); assert 'baseline' in page.locator('.final-research-note').inner_text()
        page.locator('#research-content img').evaluate_all("images=>images.forEach(image=>image.loading='eager')")
        page.wait_for_function("Array.from(document.querySelectorAll('#research-content img')).every(image=>image.complete&&image.naturalWidth>0)")
        page.screenshot(path=str(prefix)+'_desktop.png',full_page=True)
        with page.expect_download(timeout=120000) as download: page.locator('#report').click()
        pdf=Path(str(prefix)+'_sample_report.pdf'); download.value.save_as(str(pdf)); reader=PdfReader(pdf)
        text='\n'.join(item.extract_text() for item in reader.pages)
        assert '7.39610' in text and '0.370' in text and 'Estimated model probability' in text
        assert 'Temporal Transformer only' in text and 'Research validation and probability disclosure' in text
        # A new input must not leave a prior case's explanation or PDF enabled.
        page.locator('#file').set_input_files({'name':'bad.csv','mimeType':'text/csv','buffer':b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80'})
        assert page.locator('#report').is_disabled()
        with page.expect_response(lambda response:response.url.endswith('/api/predict')) as invalid: page.locator('#analyze').click()
        assert invalid.value.status==400
        page.wait_for_function("document.querySelector('#error').textContent.length>0")
        assert page.locator('#report').is_disabled()
        page.set_viewport_size({'width':390,'height':844}); page.wait_for_timeout(300)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(prefix)+'_mobile.png',full_page=True)
        final_status=page.request.get(args.url+'/api/model-status').json(); assert final_status['checkpoint_load_count']==1
        assert not errors,errors
        evidence={'url':args.url,'checkpoint_status':status,'actual_prediction':result,'pdf_pages':len(reader.pages),'example_download':True,'real_example_workflow':True,'correct_outcome_and_decision_threshold_labels':True,'chart_features_match':True,'invalid_input_rejected':True,'previous_case_display_cleared':True,'research_baselines_preserved':True,'mobile_no_overflow':True,'javascript_errors':errors}
        prefix.with_suffix('.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print({'model':result['model_type'],'version':result['version'],'probability':result['probability'],'binary':result['binary_prediction'],'checkpoint_load_count':final_status['checkpoint_load_count'],'pdf_pages':len(reader.pages),'javascript_errors':errors})
        browser.close()
if __name__=='__main__': main()
