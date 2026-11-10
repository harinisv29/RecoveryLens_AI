"""Install research routes and response additions without changing the original app files."""
import io
import json
from pathlib import Path
from flask import Blueprint,jsonify,send_file,request,g
from research.common import ROOT,OUT,CALIBRATORS,RUN,SOURCE,read_json,digest
from research.calibration import apply_calibrator

blueprint=Blueprint('research_validation',__name__)

@blueprint.get('/api/research/summary')
def summary():
    path=OUT/'dashboard_summary.json'
    if not path.exists(): return jsonify(error='Research evaluation in progress',run=RUN),503
    return jsonify(read_json(path))

@blueprint.get('/research-assets/<path:name>')
def assets(name):
    from flask import send_from_directory
    return send_from_directory(Path(__file__).parent/'static',name)

@blueprint.get('/research-artifacts/<path:name>')
def artifacts(name):
    from flask import send_from_directory,abort
    # Only rendered plots are exposed; clinical case-level CSVs are not public routes.
    if not name.endswith('.png'): abort(404)
    return send_from_directory(OUT,name)

def add_calibration(result):
    if result.get('version')!=SOURCE: return result
    name=result.get('model_type','transformer'); path=CALIBRATORS/f'{name}.json'
    if not path.exists(): return result
    config=read_json(path); active=read_json(ROOT/'models/active_model.json')
    if active.get('sha256')!=config['checkpoint_sha256']: return result
    value=float(apply_calibrator(config,[result['probability']])[0])
    result['research_calibration']={'method':config['method'],'raw_probability':result['probability'],'adjusted_probability':value,'applied':config['method']!='identity','selection':'Validation patient-group cross-fitted Brier score; test data never fitted','disclosure':'No recalibration selected; original uncalibrated model score retained' if config['method']=='identity' else 'Validation-fitted adjustment; not clinically validated','classification_policy':'Original raw-score validation cutoff is unchanged'}
    return result

def install(base):
    app=base.app; app.register_blueprint(blueprint)
    original=base.predict
    def predict(frame):
        result=add_calibration(original(frame)); g.research_prediction=result; return result
    base.predict=predict
    @app.after_request
    def enrich(response):
        if request.path=='/' and not app.config.get('SEPARATE_PERFORMANCE_PAGE',False) and response.status_code==200 and response.mimetype=='text/html':
            body=response.get_data(as_text=True)
            section='''<section id="research-validation" aria-label="Research validation"><div class="section-heading"><div><p class="eyebrow">RETROSPECTIVE RESEARCH EVALUATION</p><h2>Research validation</h2></div><span class="badge">Single-source · survivor-only</span></div><div class="card"><p id="research-loading">Loading audited research evidence…</p><div id="research-content"></div></div></section>'''
            body=body.replace('</main>',section+'</main>').replace('</head>','<link rel="stylesheet" href="/research-assets/validation.css"><script defer src="/research-assets/validation.js"></script></head>').replace('</nav>','<a href="#research-validation">Research validation</a></nav>')
            response.set_data(body)
        if request.path=='/api/report' and response.status_code==200 and response.mimetype=='application/pdf':
            result=getattr(g,'research_prediction',None)
            if result and result.get('research_calibration'):
                from pypdf import PdfReader,PdfWriter
                from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                response.direct_passthrough=False
                original_pdf=response.get_data(); appendix=io.BytesIO(); styles=getSampleStyleSheet(); c=result['research_calibration']
                text=['Research validation and probability disclosure',f"Frozen model version: {result['version']}",f"Raw model score: {result['probability']:.6f}",f"Calibration selection: {c['method']}; displayed adjusted value: {c['adjusted_probability']:.6f}",c['disclosure'],c['classification_policy'],'Primary workflow retains the original raw risk score. Validation-only cross-fitting chose between identity, Platt scaling and isotonic regression. No test labels were used to fit or choose these transformations.','Secondary retrospective analyses reuse a previously examined holdout. Patient bootstrap intervals condition on fitted models and do not establish external or prospective validity. Survivor-only selection excludes deaths; the endpoint is a training-derived statistical prolonged-stay threshold. Surgical complexity and discharge practices may confound associations.','Attribution describes model sensitivity and does not establish clinical causality. Not for clinical deployment.']
                story=[]
                for paragraph in text: story.extend([Paragraph(paragraph,styles['BodyText']),Spacer(1,12)])
                SimpleDocTemplate(appendix).build(story); appendix.seek(0); writer=PdfWriter()
                for source in [io.BytesIO(original_pdf),appendix]:
                    for page in PdfReader(source).pages: writer.add_page(page)
                merged=io.BytesIO(); writer.write(merged); response.set_data(merged.getvalue())
        return response
