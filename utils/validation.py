import numpy as np
import pandas as pd
from config import FEATURES

def validate_csv(source):
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise ValueError('Unreadable CSV') from exc
    if 'time_min' not in df or len(df) < 5:
        raise ValueError('Require time_min and at least five rows')
    if len(df) > 100000 or set(df) - set(FEATURES + ['time_min']):
        raise ValueError('Too many rows or unsupported columns')
    if not set(FEATURES).intersection(df):
        raise ValueError('No supported physiological features')
    for col in df:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except Exception as exc:
            raise ValueError(f'{col} must be numeric') from exc
        if np.isinf(df[col]).any():
            raise ValueError('Infinite values are unsupported')
    t = df.time_min.to_numpy()
    if not np.isfinite(t).all() or t[0] < 0 or (np.diff(t) <= 0).any() or t[-1]-t[0] > 720:
        raise ValueError('Timestamps must increase strictly and span no more than 720 minutes')
    return df
