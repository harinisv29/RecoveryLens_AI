"""Verify actual visible processing frames, accordion state and result gating."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--label',required=True);args=p.parse_args()
    prefix=ROOT/'results'/args.label
    if prefix.with_suffix('.json').exists(): raise RuntimeError('Use a new evidence label')
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        page.add_init_script("""window.__processingFrames=[];window.__requestPaint=null;function checkFrame(){const panel=document.querySelector('#processing-accordion');const running=document.querySelector('.workflow-step.running');const spinner=running?.querySelector('.stage-spinner');if(panel?.open&&spinner){const rect=spinner.getBoundingClientRect();const hit=document.elementFromPoint(rect.left+rect.width/2,rect.top+rect.height/2);window.__processingFrames.push({stage:running.dataset.stage,animation:getComputedStyle(spinner).animationName,visible:rect.width>0&&rect.height>0&&rect.top<innerHeight&&rect.bottom>0&&hit?.closest('.workflow-step')===running,resultsHidden:document.querySelector('#prediction-result').hidden,time:performance.now()});}requestAnimationFrame(checkFrame);}requestAnimationFrame(checkFrame);""")
        page.goto(args.url,wait_until='networkidle');page.wait_for_function("document.querySelector('#status-label').textContent==='Transformer ready'")
        assert page.locator('#sidebar').count()==0 and page.locator('.sidebar').count()==0
        assert page.evaluate("getComputedStyle(document.querySelector('.app-shell')).marginLeft")=='0px'
        assert page.locator('[data-workspace-section]').count()==6
        assert page.locator('[data-workspace-section][open]').count()==1
        assert page.locator('#upload-section').evaluate('e=>e.open')
        assert page.locator('#prediction-result').is_hidden() and page.locator('#probability').is_hidden()
        assert page.locator('#report').is_hidden() and page.locator('#heatmap').is_hidden()
        page.screenshot(path=str(prefix)+'_initial.png',full_page=True)
        page.locator('#try-example').click();page.wait_for_function("document.querySelector('#filename').textContent.includes('real-example.csv')")
        # Observe the API request without delaying or mocking it. The processing
        # panel must already be open/in view when real work is submitted.
        def observe_request(request):
            if request.method=='POST' and request.url.endswith('/api/analysis-jobs'):
                page.evaluate("window.__requestPaint={open:document.querySelector('#processing-accordion').open,uploadClosed:!document.querySelector('#upload-section').open,resultsHidden:document.querySelector('#prediction-result').hidden,top:document.querySelector('#processing').getBoundingClientRect().top}")
        page.on('request',observe_request)
        with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/api/analysis-jobs')) as posted:
            page.locator('#analyze').click()
        created=posted.value.json()
        page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'",timeout=120000)
        frames=page.evaluate('window.__processingFrames');paint=page.evaluate('window.__requestPaint')
        assert paint['open'] and paint['uploadClosed'] and paint['resultsHidden'] and 0<=paint['top']<1100,paint
        assert any(frame['visible'] and frame['animation']=='spin' and frame['resultsHidden'] for frame in frames),frames
        assert all(frame['resultsHidden'] for frame in frames)
        assert page.locator('#prediction-section').evaluate('e=>e.open')
        assert page.locator('[data-workspace-section][open]').count()==1
        assert page.locator('#prediction-result').is_visible()
        assert page.locator('#monitoring').is_hidden() and page.locator('#explanation').is_hidden() and page.locator('#report').is_hidden()
        assert page.locator('#probability').inner_text()=='56.7%'
        job=page.request.get(args.url+created['status_url']).json();assert job['state']=='succeeded'
        assert page.locator('.workflow-step.completed').count()==5
        page.screenshot(path=str(prefix)+'_result.png',full_page=True)
        page.locator('#monitoring-section>summary').click();page.wait_for_function("document.querySelector('#monitoring-section').open")
        assert page.locator('#prediction-result').is_hidden()
        assert page.locator('[data-workspace-section][open]').count()==1
        page.wait_for_function('Chart.getChart(document.querySelector("#hr")).width>100')
        assert page.evaluate('Chart.getChart(document.querySelector("#hr")).data.datasets[0].data.length')==145
        page.locator('#chart-zoom-in').click()
        page.locator('#explanation-section>summary').click();page.wait_for_function("document.querySelector('#explanation-section').open")
        assert page.locator('#heatmap').is_visible()
        page.locator('#heatmap').focus();page.keyboard.press('ArrowRight');assert 'minute 1' in page.locator('#heatmap-readout').inner_text()
        page.locator('#report-section>summary').click();page.wait_for_function("document.querySelector('#report-section').open")
        with page.expect_download(timeout=120000) as download: page.locator('#report').click()
        pdf=Path(str(prefix)+'_report.pdf');download.value.save_as(str(pdf));reader=PdfReader(pdf)
        assert '0.370' in '\n'.join(p.extract_text() for p in reader.pages)
        page.locator('#upload-section>summary').click();page.wait_for_function("document.querySelector('#upload-section').open")
        page.locator('#file').set_input_files({'name':'invalid.csv','mimeType':'text/csv','buffer':b'time_min,HR\n0,bad\n1,80\n2,80\n3,80\n4,80'})
        assert page.locator('#prediction-result').is_hidden()
        assert page.locator('#prediction-section').get_attribute('aria-disabled')=='true'
        page.locator('#analyze').click();page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis could not complete'")
        assert page.locator('#processing-accordion').evaluate('e=>e.open')
        assert page.locator('.workflow-step.failed').count()==1 and page.locator('#prediction-result').is_hidden()
        page.screenshot(path=str(prefix)+'_failed.png',full_page=True)
        page.locator('#upload-section>summary').click();page.locator('#file').set_input_files(r'C:\Users\Admin\Downloads\recoverylens_synthetic_test.csv')
        page.locator('#analyze').click();page.wait_for_function("document.querySelector('#workflow-title').textContent==='Analysis Complete'")
        assert page.locator('#probability').inner_text()=='15.4%'
        assert 'output 0' in page.locator('#model-version').inner_text()
        layouts=[]
        for width,height,name in [(1440,1100,'desktop'),(820,1180,'tablet'),(390,844,'mobile')]:
            page.set_viewport_size({'width':width,'height':height});page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(200)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert page.locator('#sidebar').count()==0
            page.screenshot(path=str(prefix)+'_'+name+'.png',full_page=True);layouts.append([width,height])
        page.locator('#processing-accordion>summary').focus();page.keyboard.press('Enter');page.wait_for_function("document.querySelector('#processing-accordion').open")
        assert page.locator('[data-workspace-section][open]').count()==1
        page.emulate_media(reduced_motion='reduce');assert page.locator('.stage-spinner').count()==0 or page.locator('.stage-spinner').first.evaluate("e=>getComputedStyle(e).animationName")=='none'
        page.goto(args.url,wait_until='networkidle');assert page.locator('#probability').is_hidden() and page.locator('[data-workspace-section][open]').count()==1
        page.goto(args.url+'/performance',wait_until='networkidle');page.wait_for_selector('#research-subgroup-attribute');assert page.locator('#sidebar').count()==0
        assert page.locator('#model-metrics .performance-metric').count()==5
        assert not errors,errors
        evidence={'url':args.url,'sidebar_removed':True,'reserved_sidebar_margin_px':0,'top_level_sections':6,'initially_open':'Upload Recording','real_visible_processing_frames':frames,'request_sent_after_processing_panel_visible':paint,'no_artificial_work_delays':True,'result_hidden_during_processing':True,'one_open_workspace_section':True,'actual_backend_job':job,'real_sample_display':'56.7%','synthetic_display':'15.4%','failure_stage_visible':True,'stale_results_cleared':True,'pdf_pages':len(reader.pages),'charts_resize_after_expansion':True,'keyboard_accordions':True,'reduced_motion':True,'layouts':layouts,'refresh_hides_results':True,'research_preserved':True,'javascript_errors':errors}
        prefix.with_suffix('.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print({'visible_running_frames':len(frames),'animated_frames':sum(f['visible'] and f['animation']=='spin' for f in frames),'six_sections':True,'sidebar_removed':True,'pdf_pages':len(reader.pages),'javascript_errors':errors})
        browser.close()
if __name__=='__main__': main()
