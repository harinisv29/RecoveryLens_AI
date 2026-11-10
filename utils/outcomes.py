"""Elapsed postoperative stay derived from shared-reference timestamps."""
import numpy as np
import pandas as pd

def audit_outcomes(cases):
    df=cases.copy()
    for col in ['dis','opstart','opend','adm','death_inhosp']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df['los_postop']=(df.dis-df.opend)/86400
    reasons=[]
    for row in df.itertuples():
        why=[]
        if not np.isfinite(row.dis): why.append('missing_or_nonfinite_discharge')
        if not np.isfinite(row.opend) or not np.isfinite(row.opstart) or row.opend<=row.opstart: why.append('invalid_operation_timing')
        if row.los_postop<0: why.append('negative_postoperative_interval')
        if not np.isfinite(row.adm) or row.adm>row.opstart: why.append('missing_or_inconsistent_admission')
        if row.death_inhosp==1: why.append('in_hospital_death')
        elif row.death_inhosp!=0: why.append('unknown_mortality_status')
        reasons.append(';'.join(why))
    df['exclusion_reasons']=reasons
    df['outcome_eligible']=df.exclusion_reasons.eq('')
    return df
