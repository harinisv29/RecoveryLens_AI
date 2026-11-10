"""Actual browser acceptance of glass UI with the independently verified CSV."""
import argparse
import hashlib
import io
import json
import re
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--label',required=True);args=p.parse_args()
    prefix=ROOT/'results'/args.label
    if prefix.with_suffix('.json').exists(): raise RuntimeError('Preserve earlier evidence; choose a new label')
    csv=Path(r'C:\Users\Admin\Downloads\recoverylens_synthetic_test.csv'); errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until='networkidle');page.wait_for_function("document.querySelector('#status-label').textContent==='Transformer ready'")
        assert page.locator('.sidebar nav a').count()==6
        assert page.locator('h1').inner_text()=='RecoveryLens AI'
        assert page.locator('.hero-art').get_attribute('aria-hidden')=='true'
        assert page.locator('#prediction-result').is_hidden()
        page.screenshot(path=str(prefix)+'_empty.png',full_page=True)
        page.locator('#file').set_input_files(str(csv));assert page.locator('#file-ready-dot').is_visible()
        with page.expect_response(lambda r:r.url.endswith('/result') and r.status==200) as returned:
            page.locator('#analyze').click()
        data=returned.value.json()['result'];page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'")
        assert data['probability']==0.1542341262102127 and data['binary_prediction']==0
        assert data['explanation']['prediction_logit']==-1.7017707824707031
        assert page.locator('#probability').inner_text()=='15.4%'
        assert page.locator('.workflow-step.completed').count()==5
        assert '120 bins' in page.locator('#upload-data-summary').inner_text()
        assert '100.0%' in page.locator('#upload-data-summary').inner_text()
        assert page.locator('#rr-card').is_visible()
        charts=['hr','spo2','bp','sbp','dbp','map','co2','rr']
        for id in charts: assert page.evaluate('(id)=>Chart.getChart(document.querySelector("#"+id)).data.datasets[0].data.length',id)==120
        assert page.evaluate('Chart.getChart(document.querySelector("#sbp")).data.datasets[0].data')==data['signals']['SBP']
        assert page.evaluate('Chart.getChart(document.querySelector("#rr")).data.datasets[0].data')==data['signals']['RR']
        page.locator('#chart-zoom-in').click()
        assert page.locator('#chart-window').input_value()=='60'
        mins=[page.evaluate('(id)=>Chart.getChart(document.querySelector("#"+id)).options.scales.x.min',id) for id in charts]
        assert len(set(mins))==1
        page.locator('#chart-zoom-in').click();assert page.locator('#chart-window').input_value()=='30'
        page.locator('#window-next').click()
        mins=[page.evaluate('(id)=>Chart.getChart(document.querySelector("#"+id)).options.scales.x.min',id) for id in charts]
        assert len(set(mins))==1
        page.locator('#chart-reset').click();assert page.locator('#chart-window').input_value()=='all'
        page.locator('#heatmap').focus();page.keyboard.press('ArrowRight');page.keyboard.press('ArrowDown')
        readout=page.locator('#heatmap-readout').inner_text();assert 'SpO2' in readout and 'minute 1' in readout
        actual=data['explanation']['temporal_heatmap'][1][1]
        match=re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?) signed logit',readout);assert match and math.isclose(float(match.group(1)),actual,rel_tol=5e-5,abs_tol=1e-9)
        page.evaluate("()=>{window.__previewBlobs=[];const original=URL.createObjectURL.bind(URL);URL.createObjectURL=blob=>{window.__previewBlobs.push({size:blob.size,type:blob.type});return original(blob);};}")
        with page.expect_popup() as popup, page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/api/report')) as preview_response:
            page.locator('#view-report').click()
        preview=popup.value;page.wait_for_function("document.querySelector('#report-status').textContent.startsWith('Prediction report ready')");preview.wait_for_url('blob:**',timeout=30000)
        blobs=page.evaluate('window.__previewBlobs');assert blobs[-1]['size']>10000 and blobs[-1]['type']=='application/pdf'
        assert preview_response.value.status==200 and preview_response.value.headers['content-type']=='application/pdf';preview.close()
        with page.expect_download(timeout=120000) as download: page.locator('#report').click()
        pdf=Path(str(prefix)+'_report.pdf');download.value.save_as(str(pdf));reader=PdfReader(pdf)
        text='\n'.join(p.extract_text() for p in reader.pages)
        assert '15.42%' in text and '7.39610' in text and '0.370' in text
        layouts=[]
        for width,height,name in [(1920,1080,'desktop_1920'),(1440,1100,'desktop_1440'),(1366,900,'laptop'),(820,1180,'tablet'),(390,844,'mobile')]:
            page.set_viewport_size({'width':width,'height':height});page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(200)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),name
            page.screenshot(path=str(prefix)+'_'+name+'.png',full_page=True)
            layouts.append({'viewport':[width,height],'no_horizontal_overflow':True})
        page.locator('#menu-toggle').click();assert page.locator('#sidebar').evaluate("e=>e.classList.contains('open')")
        page.keyboard.press('Escape');assert not page.locator('#sidebar').evaluate("e=>e.classList.contains('open')")
        page.emulate_media(reduced_motion='reduce');assert page.locator('.hero-wave-line').evaluate('e=>getComputedStyle(e).animationName')=='none'
        page.goto(args.url+'/performance',wait_until='networkidle');page.wait_for_selector('#research-subgroup-attribute')
        assert page.locator('#model-metrics .performance-metric').count()==5
        assert '0.8639' in page.locator('#model-metrics').inner_text()
        page.locator('#research-subgroup-model').select_option('gru');page.locator('#research-subgroup-attribute').select_option('surgery_type')
        assert page.locator('#research-subgroup-results tbody tr').count()>=5
        page.locator('#research-content img').evaluate_all("items=>items.forEach(i=>i.loading='eager')")
        page.wait_for_function("Array.from(document.querySelectorAll('#research-content img')).every(i=>i.complete&&i.naturalWidth>0)")
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(prefix)+'_performance_mobile.png',full_page=True)
        page.set_viewport_size({'width':1440,'height':1100});page.screenshot(path=str(prefix)+'_performance_desktop.png',full_page=True)
        # Real sample workflow remains genuine; RR is hidden when that channel is absent.
        page.goto(args.url,wait_until='networkidle');page.locator('#try-example').click()
        page.wait_for_function("document.querySelector('#filename').textContent.includes('real-example.csv')")
        page.locator('#analyze').click();page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'")
        assert page.locator('#probability').inner_text()=='56.7%' and page.locator('#rr-card').is_hidden()
        page.locator('#file').set_input_files({'name':'bad.csv','mimeType':'text/csv','buffer':b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80'})
        page.locator('#analyze').click();page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis could not complete'")
        assert page.locator('#prediction-result').is_hidden() and page.locator('#report').is_disabled() and page.locator('#view-report').is_disabled()
        assert not errors,errors
        status=page.request.get(args.url+'/api/model-status').json();assert status['checkpoint_load_count']==1
        result={'url':args.url,'synthetic_csv_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'synthetic_prediction':data,'all_eight_signal_charts_from_actual_response':True,'synchronized_navigation_and_zoom':True,'keyboard_heatmap_readout':readout,'pdf_preview_and_download':True,'pdf_pages':len(reader.pages),'responsive_layouts':layouts,'real_sample_probability_display':'56.7%','missing_rr_hidden':True,'invalid_upload_withholds_results':True,'performance_page_and_controls':True,'reduced_motion':True,'checkpoint_load_count':1,'javascript_errors':errors}
        prefix.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print({'synthetic_probability':data['probability'],'raw_logit':data['explanation']['prediction_logit'],'binary':data['binary_prediction'],'layouts':len(layouts),'pdf_pages':len(reader.pages),'javascript_errors':errors})
        browser.close()
if __name__=='__main__': main()
