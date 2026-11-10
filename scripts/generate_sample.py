"""Export a real, deidentified downloaded operation as an upload example."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from config import ROOT
from utils.preprocessing import resample

def main():
    path=next((ROOT/'data/raw').glob('case_*.json'))
    cid=int(path.stem.split('_')[1]); row=pd.read_csv(ROOT/'data/metadata/cases.csv').set_index('caseid').loc[cid]
    frame=pd.DataFrame({'time_min':np.arange(0,(row.opend-row.opstart)/60,1)})
    for feature,tid in json.loads(path.read_text()).items():
        track=pd.read_csv(ROOT/f'data/raw/{tid}.csv'); track.columns=['Time','Value']; track=track[(track.Time>=row.opstart)&(track.Time<row.opend)].copy(); track['time_min']=np.floor((track.Time-row.opstart)/60)
        frame=frame.merge(track.groupby('time_min').Value.median().rename(feature),on='time_min',how='left')
    values=resample(frame)
    from config import FEATURES
    out=pd.DataFrame(values,columns=FEATURES); out.insert(0,'time_min',np.arange(len(out))); out.to_csv(ROOT/'data/sample/real_recording.csv',index=False)
    (ROOT/'data/sample/provenance.json').write_text(json.dumps({'source':'VitalDB public dataset','caseid':cid,'window':'opstart <= Time < opend','synthetic':False},indent=2))
if __name__=='__main__': main()
