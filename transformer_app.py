"""Final live entry point: cached Transformer plus preserved research comparisons."""
import importlib.util
import os
import sys
from flask import jsonify,render_template,send_file
from config import ROOT
from inference.live_transformer import LiveTransformer
from research.web import install

spec=importlib.util.spec_from_file_location('recoverylens_transformer_base',ROOT/'app.py')
base=importlib.util.module_from_spec(spec); sys.modules[spec.name]=base; spec.loader.exec_module(base)
app=base.app
try:
    engine=LiveTransformer(); startup_error=None
except (OSError,ValueError,KeyError,RuntimeError) as error:
    engine=None; startup_error=str(error)

def live_predict(frame):
    if engine is None: raise FileNotFoundError('Validated Transformer unavailable: '+startup_error)
    return engine.predict(frame)

base.predict=live_predict
install(base)
app.config['SEPARATE_PERFORMANCE_PAGE']=True
from inference.analysis_jobs import install_jobs
analysis_jobs=install_jobs(app,lambda:engine,base.inference_lock)

def model_status():
    if engine is None: return jsonify(available=False,model_type='transformer',message=startup_error),503
    return jsonify(engine.status())

def index(): return render_template('transformer.html')
app.view_functions['status']=model_status
app.view_functions['index']=index

@app.get('/performance')
def performance(): return render_template('performance.html')

@app.get('/api/example-csv')
def example():
    return send_file(ROOT/'data/sample/real_recording.csv',mimetype='text/csv',as_attachment=True,download_name='RecoveryLens-real-example.csv')

@app.after_request
def final_pdf_disclosure(response):
    from flask import request,g
    if request.path=='/api/report' and response.status_code==200:
        import io
        from pypdf import PdfReader,PdfWriter
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        result=getattr(g,'research_prediction',None)
        if result:
            response.direct_passthrough=False; original_pdf=response.get_data(); cover=io.BytesIO(); styles=getSampleStyleSheet()
            items=[Paragraph('RecoveryLens AI',styles['Title']),Paragraph('Temporal Transformer - postoperative risk estimate',styles['Heading2']),Spacer(1,20),Paragraph(f"Estimated model probability: {result['probability']:.2%}",styles['Heading1']),Paragraph('Not clinically calibrated. Research demonstration only.',styles['BodyText']),Spacer(1,18)]
            rows=[['Binary prediction',result['risk_category']],['Outcome definition',f"Postoperative stay > {result['target_days']:.5f} days"],['Probability decision cutoff',f"{result['threshold']:.3f} (separate from stay duration)"],['Model version',result['version']],['Live model','Temporal Transformer only'],['Observed input',f"{result['quality']['minutes']} minute bins; {result['quality']['observed_fraction']:.1%} observed"],['Missing channels',', '.join(result['quality']['missing_features']) or 'None'],['Input fingerprint',result['input_sha256'][:24]],['Checkpoint fingerprint',result['checkpoint_sha256'][:24]]]
            table=Table(rows,colWidths=[165,305],hAlign='LEFT'); table.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf5f8')),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#17334c')),('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor('#dde8ed'))])); items.extend([table,Spacer(1,20),Paragraph('Outcome is elapsed (dis - opend) / 86400; prolonged means strictly greater than the training cohort\'s 75th percentile. In-hospital deaths were excluded, so this model describes survivors only.',styles['BodyText']),Spacer(1,12),Paragraph('Physiological graphs and Integrated Gradients for this exact uploaded recording follow. Attributions explain model sensitivity, not clinical causation. Baseline models and early-window ablations are retrospective research comparisons only. No clinical deployment validation has been established.',styles['BodyText'])])
            SimpleDocTemplate(cover).build(items); cover.seek(0); writer=PdfWriter()
            for source in [cover,io.BytesIO(original_pdf)]:
                for page in PdfReader(source).pages: writer.add_page(page)
            output=io.BytesIO(); writer.write(output); response.set_data(output.getvalue())
        response.headers['X-Live-Model']='Temporal Transformer'
    return response

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.environ.get('RECOVERYLENS_PORT','5050')),debug=False)
