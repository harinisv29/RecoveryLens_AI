import numpy as np
import pandas as pd
from config import FEATURES, LIMITS, MAX_STEPS

def resample(df):
    df = df.copy()
    df['time_min'] = df.time_min - df.time_min.iloc[0]
    df['bin'] = np.floor(df.time_min).astype(int)
    for f, (lo, hi) in zip(FEATURES, LIMITS):
        if f not in df:
            df[f] = np.nan
        df.loc[~df[f].between(lo, hi), f] = np.nan
    values = df.groupby('bin')[FEATURES].median().reindex(range(int(df.bin.max())+1)).to_numpy(dtype=np.float32)[:MAX_STEPS]
    if np.isfinite(values).sum() < 5:
        raise ValueError('Insufficient valid physiological observations')
    return values

def fit_stats(sequences):
    a = np.concatenate(sequences)
    mean = np.array([np.nanmean(a[:,i]) if np.isfinite(a[:,i]).any() else 0 for i in range(len(FEATURES))])
    std = np.array([np.nanstd(a[:,i]) if np.isfinite(a[:,i]).any() else 1 for i in range(len(FEATURES))])
    return {'mean': mean.tolist(), 'std': np.maximum(std, 1e-3).tolist(), 'features': FEATURES, 'interval_seconds': 60, 'max_steps': MAX_STEPS}

def transform(values, stats):
    observed = np.isfinite(values)
    z = (values - np.array(stats['mean'])) / np.array(stats['std'])
    return np.concatenate([np.nan_to_num(z), observed.astype(float)], axis=1).astype(np.float32)

def engineered(x):
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])
