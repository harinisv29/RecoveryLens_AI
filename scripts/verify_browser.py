"""Exercise the running local dashboard with an actual browser and real sample CSV."""
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from playwright.sync_api import sync_playwright

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--url',default='http://127.0.0.1:5050'); parser.add_argument('--output'); parser.add_argument('--expected-run'); args=parser.parse_args()
    output=Path(args.output) if args.output else ROOT/'results'
    for directory in ['plots','explanations']: (output/directory).mkdir(parents=True,exist_ok=True)
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto(args.url,wait_until='networkidle')
        page.wait_for_function("document.querySelector('#status').textContent.includes('Model ready')")
        served_status=page.request.get(args.url+'/api/model-status').json()
        if args.expected_run: assert served_status['version']==args.expected_run,served_status
        page.locator('#file').set_input_files(str(ROOT/'data/sample/real_recording.csv'))
        page.locator('#analyze').click()
        page.wait_for_function("!document.querySelector('#report').disabled",timeout=120000)
        if args.expected_run: assert args.expected_run in page.locator('#status').inner_text()
        assert page.locator('#error').inner_text()==''
        assert page.evaluate('typeof Chart')=='function'
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).data.datasets[0].data.length')>0
        assert 'Most influential' in page.locator('#intervals').inner_text()
        page.screenshot(path=str(output/'plots/dashboard_desktop.png'),full_page=True)
        with page.expect_download(timeout=120000) as download:
            page.locator('#report').click()
        download.value.save_as(str(output/'explanations/sample_report.pdf'))
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(output/'plots/dashboard_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors,errors
        result={'browser':'Microsoft Edge, headless','csv':'real_recording.csv','prediction':page.locator('#probability').inner_text(),'charts_rendered':True,'attribution_rendered':True,'pdf_downloaded':True,'mobile_overflow':False,'javascript_errors':errors}
        result['url']=args.url
        result['served_model_status']=served_status
        (output/'browser_verification.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        browser.close()
if __name__=='__main__': main()
