"""Additive research dashboard entry point. The original app.py remains unchanged."""
import os
import importlib.util
import sys
from config import ROOT
from research.web import install
spec=importlib.util.spec_from_file_location('recoverylens_research_base',ROOT/'app.py')
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)
install(original)
app=original.app

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.environ.get('RECOVERYLENS_RESEARCH_PORT','5050')),debug=False)
