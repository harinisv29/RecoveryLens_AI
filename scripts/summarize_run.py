import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT

def main():
    audit=json.loads((ROOT/'data/metadata/outcome_audit.json').read_text())
    cohort=json.loads((ROOT/'data/metadata/cohort.json').read_text())
    models={name:json.loads((ROOT/f'results/metrics/{name}.json').read_text()) for name in ['transformer','gru','xgboost']}
    summary={'timestamp_reference':audit['timestamp_reference_verification'],'metadata_cases':audit['cases'],'metadata_outcome_eligible':audit['eligible_nonfatal_with_valid_timing'],'metadata_exclusions':audit['exclusion_counts_overlapping'],'unique_metadata_exclusions':audit['excluded_unique'],'downloaded_cases':len(list((ROOT/'data/raw').glob('case_*.json'))),'final_eligible_cases':len(cohort['cohort']),'signal_exclusions':cohort['excluded'],'population':audit['population'],'split_sizes':{k:len(v) for k,v in cohort['splits'].items()},'positive_counts':cohort['positive_counts'],'prolonged_threshold_days_training_75th_percentile':cohort['threshold_days'],'models':models,'full_cohort_completed':False,'scope_limit':'128-case engineering feasibility subset, first eligible cases in public index; full 6306-case download and training not completed. Measured download throughput and per-track request volume make this a separate long-running job. Not representative or clinically validated.','cuda_runtime':'Training used existing read-only CUDA torch 2.5.1+cu121 through scripts/run_cuda.py; default isolated environment has torch 2.14.1 CPU for inference/tests.','checkpoint':'models/checkpoints/transformer.pt','preprocessing':'data/processed/preprocessing.json'}
    browser=ROOT/'results/browser_verification.json'
    if browser.exists(): summary['browser_verification']=json.loads(browser.read_text())
    (ROOT/'results/execution_summary.json').write_text(json.dumps(summary,indent=2))
    columns=['Model','AUROC','AUPRC','F1','Sensitivity','Specificity','Brier','GPU seconds']
    table=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for name,report in models.items():
        metrics=report['test']; values=[name]+[f'{metrics[k]:.4f}' for k in ['auroc','auprc','f1','sensitivity','specificity','brier_score']]+[f"{report['training_seconds']:.2f}" if 'training_seconds' in report else 'CPU']
        table.append('| '+' | '.join(values)+' |')
    text=f"# Executed research experiment\n\n127 eligible cases; subject splits 80/21/26; held-out test contains 6 positive and 20 negative cases. Prolonged stay > {cohort['threshold_days']:.6f} days (training 75th percentile only). Survivor-only population.\n\n"+'\n'.join(table)+'\n\nCalibration bin values and curves are saved for each model; probabilities have not undergone post-hoc calibration. The tiny test set and index-ordered subset do not support clinical or generalization claims. GPU seconds measure the synchronized training/validation epoch loop, excluding installation, downloads, preprocessing, evaluation, and browser verification.\n'
    (ROOT/'results/RESULTS.md').write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__': main()
