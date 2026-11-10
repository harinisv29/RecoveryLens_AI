"""Bounded in-memory local jobs with genuine stage events, no simulated delays."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,field
import copy
import io
import secrets
import threading
import time
from flask import jsonify,request
from utils.validation import validate_csv
from research.web import add_calibration

STAGES=[('validation','Data Validation'),('preprocessing','Signal Preprocessing'),('inference','Temporal Transformer Inference'),('explainability','Explainable AI'),('results','Results Preparation')]

class JobCancelled(Exception): pass

@dataclass
class Job:
    id:str
    filename:str
    created:float=field(default_factory=time.monotonic)
    state:str='queued'
    stages:list=field(default_factory=lambda:[{'id':key,'title':title,'state':'pending','details':{},'started_at':None,'completed_at':None} for key,title in STAGES])
    events:list=field(default_factory=list)
    result:dict|None=None
    error:str|None=None
    cancelled:bool=False

class AnalysisJobs:
    def __init__(self,get_engine,inference_lock,ttl=300,max_jobs=16):
        self.get_engine=get_engine; self.inference_lock=inference_lock; self.ttl=ttl; self.max_jobs=max_jobs
        self.lock=threading.RLock(); self.jobs={}; self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='recoverylens-analysis')

    def _purge(self):
        now=time.monotonic()
        for key in list(self.jobs):
            job=self.jobs[key]
            if job.state in ('succeeded','failed','cancelled') and now-job.created>self.ttl: del self.jobs[key]

    def submit(self,filename,payload):
        with self.lock:
            self._purge()
            if len(self.jobs)>=self.max_jobs:
                retired=[job for job in self.jobs.values() if job.state in ('succeeded','failed','cancelled')]
                if retired: del self.jobs[min(retired,key=lambda job:job.created).id]
                else: raise OverflowError('Analysis queue is full; retry shortly')
            job=Job(secrets.token_urlsafe(24),filename); self.jobs[job.id]=job
        self.pool.submit(self._run,job,payload)
        return job.id

    def _event(self,job,key,state,details):
        with self.lock:
            if job.cancelled: raise JobCancelled()
            index=next(i for i,stage in enumerate(job.stages) if stage['id']==key); stage=job.stages[index]
            if state=='running':
                if any(s['state']!='completed' for s in job.stages[:index]): raise RuntimeError('Invalid processing-stage order')
                stage['started_at']=time.monotonic(); job.state='running'
            elif stage['state']!='running': raise RuntimeError('Cannot complete a stage that did not start')
            stage['state']=state; stage['details']=details
            if state=='completed': stage['completed_at']=time.monotonic()
            job.events.append({'stage':key,'state':state,'elapsed_seconds':time.monotonic()-job.created,'details':copy.deepcopy(details)})

    def _run(self,job,payload):
        try:
            with self.inference_lock:
                self._event(job,'validation','running',{'work':'Reading CSV and checking timestamps, columns and numerical values'})
                frame=validate_csv(io.BytesIO(payload))
                self._event(job,'validation','completed',{'rows':len(frame),'columns':frame.columns.tolist(),'time_span_minutes':float(frame.time_min.iloc[-1]-frame.time_min.iloc[0]),'timestamps':'Finite, nonnegative and strictly increasing','numerical_values':'Validated; missing observations preserved'})
                engine=self.get_engine()
                if engine is None: raise RuntimeError('Validated Transformer unavailable; no fallback prediction is permitted')
                result=engine.predict(frame,progress=lambda key,state,details:self._event(job,key,state,details))
                result=add_calibration(result)
                with self.lock:
                    if job.cancelled: raise JobCancelled()
                    job.result=result
                    self._event(job,'results','completed',{'summary_ready':True,'physiological_series':len(result['signals']),'pdf':'Available on request using the existing PDF endpoint'})
                    job.state='succeeded'
        except JobCancelled:
            with self.lock: job.state='cancelled'; job.result=None
        except Exception as error:
            with self.lock:
                stage=next((s for s in job.stages if s['state']=='running'),None)
                if stage is None: stage=next((s for s in job.stages if s['state']=='pending'),job.stages[-1])
                stage['state']='failed'; stage['error']=str(error); stage['completed_at']=time.monotonic()
                job.state='failed'; job.error=str(error); job.result=None
                job.events.append({'stage':stage['id'],'state':'failed','elapsed_seconds':time.monotonic()-job.created,'error':str(error)})

    def snapshot(self,key):
        with self.lock:
            self._purge(); job=self.jobs.get(key)
            if job is None: return None
            return {'job_id':job.id,'filename':job.filename,'state':job.state,'stages':copy.deepcopy(job.stages),'events':copy.deepcopy(job.events),'error':job.error,'result_ready':job.state=='succeeded'}

    def result(self,key):
        with self.lock:
            self._purge(); job=self.jobs.get(key)
            return copy.deepcopy(job.result) if job and job.state=='succeeded' else None

    def cancel(self,key):
        with self.lock:
            job=self.jobs.get(key)
            if not job: return False
            job.cancelled=True; job.result=None
            if job.state!='running': job.state='cancelled'
            return True

def install_jobs(app,get_engine,inference_lock):
    store=AnalysisJobs(get_engine,inference_lock); app.extensions['analysis_jobs']=store
    @app.post('/api/analysis-jobs')
    def create_analysis_job():
        file=request.files.get('file')
        if not file or not file.filename.lower().endswith('.csv'): return jsonify(error='Upload a .csv file in the file field'),400
        try: key=store.submit(file.filename,file.read())
        except OverflowError as error: return jsonify(error=str(error)),429
        return jsonify(job_id=key,status_url='/api/analysis-jobs/'+key,result_url='/api/analysis-jobs/'+key+'/result'),202

    @app.get('/api/analysis-jobs/<key>')
    def analysis_job_status(key):
        state=store.snapshot(key)
        response=jsonify(state) if state else (jsonify(error='Analysis job not found or expired'),404)
        return response

    @app.get('/api/analysis-jobs/<key>/result')
    def analysis_job_result(key):
        state=store.snapshot(key)
        if state is None: return jsonify(error='Analysis job not found or expired'),404
        if state['state']=='failed': return jsonify(error=state['error'],job_id=key),422
        if state['state']=='cancelled': return jsonify(error='Analysis cancelled',job_id=key),410
        result=store.result(key)
        if result is None: return jsonify(job_id=key,state=state['state']),202
        return jsonify(job_id=key,result=result)

    @app.delete('/api/analysis-jobs/<key>')
    def discard_analysis_job(key):
        return (jsonify(discarded=True),200) if store.cancel(key) else (jsonify(error='Analysis job not found'),404)

    @app.after_request
    def private_job_responses(response):
        if request.path.startswith('/api/analysis-jobs'): response.headers['Cache-Control']='no-store'
        return response
    return store
