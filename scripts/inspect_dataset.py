import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import ROOT
from utils.outcomes import audit_outcomes

def main():
    df=pd.read_csv(ROOT/'data/metadata/cases.csv')
    source='los_postop' if 'los_postop' in df else '(dis - opend) / 86400: elapsed days from operation end to discharge; both times documented relative to recording start'
    y=pd.to_numeric(df.los_postop if 'los_postop' in df else (df.dis-df.opend)/86400,errors='coerce')
    report={'cases':len(df),'outcome':'los_postop: postoperative length of hospital stay (days), official VitalDB documentation','missing_labels':int(y.isna().sum()),'negative_labels':int((y<0).sum()),'distribution':y[y>=0].describe(percentiles=[.25,.5,.75,.9,.95]).to_dict(),'patient_identifier_available':'subjectid' in df,'threshold_policy':'Training-split 75th percentile; prolonged is strictly greater. Statistical upper-tail endpoint, not a clinical cutoff.','mortality_policy':'Exclude in-hospital deaths: death truncates stay; report cohort exclusions.'}
    report['outcome']=source
    audited=audit_outcomes(df)
    audited[['caseid','subjectid','dis','opstart','opend','death_inhosp','los_postop','outcome_eligible','exclusion_reasons']].to_csv(ROOT/'data/metadata/outcome_exclusions.csv',index=False)
    audited.to_csv(ROOT/'data/metadata/clinical_derived.csv',index=False)
    report['timestamp_reference_verification']={'source':'https://vitaldb.net/dataset/','casestart':'Recording start, set to zero for anonymization','opend':'Operation end time; from casestart, sec','dis':'Discharge time from casestart, sec','verified':'Same origin and units per official field dictionary; elapsed postoperative stay is (dis-opend)/86400. Cannot independently validate source EMR timestamps.'}
    report['exclusion_counts_overlapping']={reason:int(audited.exclusion_reasons.str.contains(reason,regex=False).sum()) for reason in ['missing_or_nonfinite_discharge','invalid_operation_timing','negative_postoperative_interval','missing_or_inconsistent_admission','in_hospital_death','unknown_mortality_status']}
    report['excluded_unique']=int((~audited.outcome_eligible).sum())
    report['eligible_nonfatal_with_valid_timing']=int(audited.outcome_eligible.sum())
    report['population']='Survivors only: known in-hospital survivors with valid timestamps. Excluding deaths changes the prediction population and prevents use as an all-patient outcome model.'
    (ROOT/'data/metadata/outcome_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
