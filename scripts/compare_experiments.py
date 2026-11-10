import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from utils.experiments import run_paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args(); paths=run_paths(args.run)
    prior=ROOT/'results/runs/initial_127/results/metrics'
    cohort=json.loads((paths['metadata']/'cohort.json').read_text())
    download=json.loads((ROOT/'data/metadata/full_download_progress.json').read_text())
    selection=json.loads((paths['results']/'model_selection.json').read_text())
    inventory=json.loads((ROOT/'data/metadata/full_case_inventory.json').read_text())
    inaccessible=[row for row in inventory if not row['eligible_for_signal_screening']]
    summary={'run':args.run,'download':download,'eligible_signal_cases':len(cohort['cohort']),'signal_exclusions':cohort['excluded'],'split_sizes':{k:len(v) for k,v in cohort['splits'].items()},'positive_counts':cohort['positive_counts'],'target_days':cohort['threshold_days'],'population':'Survivors only. Deaths excluded.','split_policy':json.loads((paths['metadata']/'split_policy.json').read_text()),'selection':selection,'models':{},'comparison_limitation':'Descriptive comparison, not a paired statistical improvement test. Cohorts, holdout patients and training-derived endpoint cutoffs differ; all initial subjects are restricted to expanded training.'}
    summary['insufficient_downloaded_channels']=inaccessible
    summary['metadata_outcome_audit']=json.loads((ROOT/'data/metadata/outcome_audit.json').read_text())
    summary['track_failures']=json.loads((ROOT/'data/metadata/full_download_failures.json').read_text())
    assert len(cohort['cohort'])+len(cohort['excluded'])+len(inaccessible)==download['eligible_outcome_cases'],'Cohort accounting does not reconcile'
    lines=[f'# Expanded experiment: {args.run}','',f"Downloaded/verified {download['cases_with_at_least_three_tracks']} cases with >=3 tracks; {len(cohort['cohort'])} pass outcome and signal quality. Split sizes: {summary['split_sizes']}. Positive counts: {cohort['positive_counts']}.",'',f"Track requests: {download['requested_tracks']}; statuses: {download['counts']}; download wall time {download['elapsed_seconds']:.1f} seconds. Exclusions after outcome screening: {len(inaccessible)} cases have fewer than 3 retrieved channels, {len(cohort['excluded'])} fail signal quality.",'',f"Training-only upper-quartile outcome: postoperative stay > {cohort['threshold_days']:.8f} days. Survivor-only population.",'',f"Application checkpoint: `{selection['checkpoint']}`, selected exclusively by validation AUROC (Brier tie-break).",'', '| Model | Cohort | Test n | AUROC (95% CI) | AUPRC (95% CI) | F1 | Sensitivity | Specificity | Brier | Training seconds |','| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name in ['transformer','gru','xgboost']:
        initial=json.loads((prior/f'{name}.json').read_text()); expanded=json.loads((paths['results']/f'metrics/{name}.json').read_text())
        summary['models'][name]={'initial':initial,'expanded':expanded}
        for label,report in [('initial 127',initial),('expanded',expanded)]:
            m=report['test']; intervals=m.get('bootstrap_95_ci')
            def score(key):
                text=f'{m[key]:.4f}'
                if intervals: text+=f" ({intervals[key][0]:.4f}–{intervals[key][1]:.4f})"
                return text
            lines.append('| '+' | '.join([name,label,str(m['n']),score('auroc'),score('auprc')]+[f'{m[key]:.4f}' for key in ['f1','sensitivity','specificity','brier_score']]+[f"{report['training_seconds']:.2f}" if 'training_seconds' in report else 'not recorded'])+' |')
    lines+=['',summary['comparison_limitation'],'','Intervals use 2,000 patient-cluster bootstrap replicates (seed 42), conditional on the fitted model and fixed outcome. They do not include training uncertainty. Calibration bin values and plots, confusion matrices, case-level FP/FN records and descriptive duration/coverage subgroups are saved beside these results. No clinical validation or post-hoc probability calibration has been performed.']
    (paths['results']/'comparison.json').write_text(json.dumps(summary,indent=2))
    (paths['results']/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()
