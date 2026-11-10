from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
FEATURES = ['HR', 'SpO2', 'SBP', 'DBP', 'MAP', 'EtCO2', 'RR']
TRACKS = dict(zip(FEATURES, ['Solar8000/HR','Solar8000/PLETH_SPO2','Solar8000/ART_SBP','Solar8000/ART_DBP','Solar8000/ART_MBP','Solar8000/ETCO2','Solar8000/RR']))
LIMITS = [(20,250),(50,100),(30,280),(15,180),(20,220),(5,100),(2,80)]
SEED = 42
MAX_STEPS = 720
DISCLAIMER = 'Research prototype only. Not clinically validated. Do not use for patient care. Survivor-only prediction population: in-hospital deaths excluded. Attribution is not evidence of causation. Small single-center retrospective data may not generalize; predicted probabilities are not clinically calibrated.'
