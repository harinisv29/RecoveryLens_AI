import io
import threading
from flask import Flask,render_template,request,jsonify,send_file
from werkzeug.exceptions import RequestEntityTooLarge
from config import ROOT,DISCLAIMER
from utils.validation import validate_csv
from inference.predict import predict,load_checkpoint

app=Flask(__name__)
app.json.sort_keys=False
app.config['MAX_CONTENT_LENGTH']=5*1024*1024
inference_lock=threading.Lock()

@app.get('/')
def index(): return render_template('index.html')

@app.get('/api/health')
def health(): return jsonify(status='ok',research_only=True)

@app.get('/api/model-status')
def status():
    try:
        _,checkpoint=load_checkpoint()
        return jsonify(available=True,version=checkpoint['version'],model_type=checkpoint.get('model','transformer'),cohort_size=checkpoint['cohort_size'],target_days=checkpoint['target_days'],limitations=DISCLAIMER)
    except FileNotFoundError as exc: return jsonify(available=False,message=str(exc))
    except Exception: return jsonify(available=False,message='Checkpoint could not be loaded'),503

def uploaded_prediction():
    file=request.files.get('file')
    if not file or not file.filename.lower().endswith('.csv'): raise ValueError('Upload a .csv file in the file field')
    frame=validate_csv(file.stream)
    if not inference_lock.acquire(blocking=False): return None
    try: return predict(frame)
    finally: inference_lock.release()

@app.post('/api/predict')
def api_predict():
    result=uploaded_prediction()
    if result is None: return jsonify(error='Inference busy; retry shortly'),429
    return jsonify(result)

@app.post('/api/report')
def report():
    # Recompute from CSV: never trust a client-supplied probability.
    result=uploaded_prediction()
    if result is None: return jsonify(error='Inference busy; retry shortly'),429
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image
    from reportlab.lib.styles import getSampleStyleSheet
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    output=io.BytesIO(); styles=getSampleStyleSheet(); story=[]
    for text in ['RecoveryLens AI - Research prediction report',f"Probability: {result['probability']:.1%} | {result['risk_category']}",f"Endpoint: postoperative stay greater than {result['target_days']:.2f} days",f"Model: {result.get('model_type','transformer')} / {result['version']} | cohort: {result['cohort_size']} cases",f"Input: {result['quality']['minutes']} minutes; observed fraction {result['quality']['observed_fraction']:.1%}",result['explanation']['method'],DISCLAIMER]:
        story.extend([Paragraph(text,styles['BodyText']),Spacer(1,12)])
    fig=Figure(figsize=(8,5)); axes=fig.subplots(2,2)
    for ax,features in zip(axes.flat,[['HR'],['SpO2'],['SBP','DBP','MAP'],['EtCO2']]):
        for f in features: ax.plot(result['signals'][f],label=f)
        ax.legend(); ax.set_xlabel('Minute')
    fig.tight_layout(); chart=io.BytesIO(); fig.savefig(chart,format='png'); chart.seek(0); story.append(Image(chart,width=470,height=294))
    fig2=Figure(figsize=(8,3)); ax=fig2.subplots(); ax.imshow(list(map(list,zip(*result['explanation']['temporal_heatmap']))),aspect='auto',cmap='coolwarm'); ax.set_yticks(range(7),list(result['signals'])); ax.set_xlabel('Minute; signed '+result['explanation'].get('units','attribution'))
    chart2=io.BytesIO(); fig2.savefig(chart2,format='png',bbox_inches='tight'); chart2.seek(0); story.append(Image(chart2,width=470,height=176))
    SimpleDocTemplate(output).build(story); output.seek(0)
    return send_file(output,mimetype='application/pdf',as_attachment=True,download_name='RecoveryLens-research-report.pdf')

@app.errorhandler(ValueError)
def bad_input(exc): return jsonify(error=str(exc)),400
@app.errorhandler(FileNotFoundError)
def no_model(exc): return jsonify(error=str(exc)),503
@app.errorhandler(RequestEntityTooLarge)
def too_large(exc): return jsonify(error='Upload exceeds 5 MB'),413

if __name__=='__main__': app.run(host='127.0.0.1',port=5050,debug=False)
