"""Acceptance checks for the premium UI and genuine asynchronous workflow."""
import argparse
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
    if prefix.with_suffix('.json').exists(): raise RuntimeError('Preserve prior verification; use a new label')
    errors=[]; responses=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100}); page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until='networkidle'); page.wait_for_function("document.querySelector('#status-label').textContent==='Transformer ready'")
        assert page.locator('#prediction-result').is_hidden() and page.locator('#analysis-results').is_hidden()
        assert page.locator('#model-metrics').count()==0 and page.locator('#research-content').count()==0
        page.screenshot(path=str(prefix)+'_desktop_empty.png',full_page=True)
        page.locator('#try-example').click(); page.wait_for_function("document.querySelector('#filename').textContent.includes('real-example.csv')")
        assert page.locator('#file-ready-dot').is_visible() and page.locator('#prediction-result').is_hidden()
        with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/api/analysis-jobs')) as posted:
            page.locator('#analyze').click()
        created=posted.value.json(); assert posted.value.status==202
        page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'",timeout=120000)
        assert page.locator('#prediction-result').is_visible() and page.locator('#analysis-results').is_visible()
        assert page.locator('.workflow-step.completed').count()==5
        page.wait_for_function("!document.querySelector('#processing-accordion').open")
        job=page.request.get(args.url+created['status_url']).json(); assert job['state']=='succeeded'
        assert [(e['stage'],e['state']) for e in job['events']]==[(key,state) for key in ['validation','preprocessing','inference','explainability','results'] for state in ['running','completed']]
        assert 'probability' not in json.dumps(job)
        actual=page.request.get(args.url+created['result_url']).json()['result']
        assert actual['probability']==0.5672959685325623
        assert page.locator('#probability').inner_text()==f"{100*actual['probability']:.1f}%"
        assert actual['input_sha256']==actual['explanation']['input_sha256']
        assert page.locator('#outcome-value').inner_text()=='> 7.39610 days'
        assert page.locator('#cutoff-value').inner_text()=='≥ 0.370'
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).data.datasets[0].data.length')==actual['quality']['minutes']
        page.locator('#chart-window').select_option('30'); page.locator('#window-next').click()
        assert '30–59' in page.locator('#window-label').inner_text()
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).options.scales.x.min')==30
        page.locator('#chart-window').select_option('all')
        page.locator('#processing-accordion>summary').click(); page.wait_for_function("document.querySelector('#processing-accordion').open")
        page.screenshot(path=str(prefix)+'_desktop_result.png',full_page=True)
        page.locator('#processing-accordion>summary').click()
        with page.expect_download(timeout=120000) as download: page.locator('#report').click()
        pdf=Path(str(prefix)+'_report.pdf'); download.value.save_as(str(pdf)); reader=PdfReader(pdf)
        text='\n'.join(item.extract_text() for item in reader.pages)
        assert '7.39610' in text and '0.370' in text and 'Temporal Transformer only' in text
        # Next upload must clear all previously exposed results before processing.
        page.locator('#file').set_input_files({'name':'bad.csv','mimeType':'text/csv','buffer':b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80'})
        assert page.locator('#prediction-result').is_hidden() and page.locator('#report').is_disabled()
        page.locator('#analyze').click()
        page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis could not complete'")
        assert page.locator('.workflow-step.failed').count()==1
        assert page.locator('#prediction-result').is_hidden() and page.locator('#analysis-results').is_hidden()
        assert page.locator('#error').is_visible()
        page.set_viewport_size({'width':390,'height':844}); page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('#menu-toggle').click(); assert page.locator('#sidebar').evaluate("e=>e.classList.contains('open')")
        page.keyboard.press('Escape'); assert not page.locator('#sidebar').evaluate("e=>e.classList.contains('open')")
        page.screenshot(path=str(prefix)+'_mobile_failure.png',full_page=True)
        page.locator('#try-example').click(); page.wait_for_function("document.querySelector('#filename').textContent.includes('real-example.csv')"); page.locator('#analyze').click()
        page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'",timeout=120000)
        assert page.locator('#prediction-result').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(prefix)+'_mobile_result.png',full_page=True)
        page.emulate_media(reduced_motion='reduce'); assert page.locator('#prediction-result').evaluate("e=>getComputedStyle(e).animationName")=='none'
        page.goto(args.url+'/performance',wait_until='networkidle'); page.wait_for_selector('#research-subgroup-attribute'); page.wait_for_function("document.querySelectorAll('#model-metrics .performance-metric').length===5")
        assert page.locator('#research-content table').count()>=4
        page.locator('#research-subgroup-attribute').select_option('surgery_type'); page.locator('#research-subgroup-model').select_option('gru')
        assert page.locator('#research-subgroup-results tbody tr').count()>=5
        page.locator('#research-content img').evaluate_all("items=>items.forEach(e=>e.loading='eager')")
        page.wait_for_function("Array.from(document.querySelectorAll('#research-content img')).every(e=>e.complete&&e.naturalWidth>0)")
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(prefix)+'_mobile_performance.png',full_page=True)
        page.set_viewport_size({'width':1440,'height':1100}); page.screenshot(path=str(prefix)+'_desktop_performance.png',full_page=True)
        assert not errors,errors
        status=page.request.get(args.url+'/api/model-status').json(); assert status['checkpoint_load_count']==1
        result={'url':args.url,'model_status':status,'actual_result':actual,'genuine_job_events':job,'pdf_pages':len(reader.pages),'desktop_workflow':True,'mobile_workflow':True,'sequential_upload_isolation':True,'invalid_csv_failed_at_validation':True,'result_hidden_until_success':True,'chart_navigation_view_only':True,'separate_performance_page':True,'research_controls_work':True,'reduced_motion':True,'javascript_errors':errors}
        prefix.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print({'probability':actual['probability'],'completed_stages':len(job['stages']),'pdf_pages':len(reader.pages),'javascript_errors':errors,'checkpoint_loads':status['checkpoint_load_count']})
        browser.close()
if __name__=='__main__': main()
