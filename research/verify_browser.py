"""Version-asserted browser verification of the additive research dashboard."""
import argparse
import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.common import ROOT,OUT,RUN,SOURCE,save_json
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--url',default='http://127.0.0.1:5050'); parser.add_argument('--label',required=True); args=parser.parse_args()
    folder=OUT/'verification'/args.label
    if folder.exists(): raise RuntimeError('Verification label already exists; preserve evidence and use a new label')
    folder.mkdir(parents=True)
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100}); page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle')
        model=page.request.get(args.url+'/api/model-status').json(); assert model['version']==SOURCE and model['cohort_size']==6119
        summary=page.request.get(args.url+'/api/research/summary').json(); assert summary['run']==RUN
        page.wait_for_selector('#research-subgroup-attribute',timeout=30000)
        assert page.locator('#research-content table').count()>=4
        page.locator('#research-subgroup-attribute').select_option('surgery_type'); page.locator('#research-subgroup-model').select_option('gru')
        assert page.locator('#research-subgroup-results tbody tr').count()>=5
        assert page.request.get(args.url+'/research-artifacts/errors/transformer_false_negatives.csv').status==404
        page.locator('#file').set_input_files(str(ROOT/'data/sample/real_recording.csv'))
        with page.expect_response(lambda r:r.url.endswith('/api/predict') and r.request.method=='POST') as prediction:
            page.locator('#analyze').click()
        result=prediction.value.json(); assert result['version']==SOURCE
        assert result['research_calibration']['method']=='identity' and not result['research_calibration']['applied']
        page.wait_for_function("!document.querySelector('#report').disabled")
        page.wait_for_selector('#research-calibration-result')
        assert 'Raw model score retained' in page.locator('#research-calibration-result').inner_text()
        assert page.locator('#error').inner_text()==''
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).data.datasets[0].data.length')>0
        page.screenshot(path=str(folder/'dashboard_desktop.png'),full_page=True)
        with page.expect_download(timeout=120000) as download: page.locator('#report').click()
        pdf=folder/'sample_research_report.pdf'; download.value.save_as(str(pdf))
        reader=PdfReader(pdf); text='\n'.join(item.extract_text() for item in reader.pages)
        assert 'Research validation and probability disclosure' in text and 'identity' in text
        assert SOURCE in text
        save_json(folder/'actual_prediction.json',result)
        page.set_viewport_size({'width':390,'height':844}); page.wait_for_timeout(300)
        page.screenshot(path=str(folder/'dashboard_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors,errors
        report={'run':RUN,'url':args.url,'served_model':model,'csv_prediction':True,'calibration_disclosure':True,'interactive_signal_charts':True,'research_comparison_tables':True,'subgroup_controls':True,'attribution_visualizations':True,'pdf_pages':len(reader.pages),'pdf_research_appendix':True,'mobile_no_overflow':True,'javascript_errors':errors}
        save_json(folder/'browser_verification.json',report)
        print(report,flush=True); browser.close()
if __name__=='__main__': main()
