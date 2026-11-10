"""Publish an evidence-backed run summary without modifying archived experiments."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT
from utils.experiments import run_paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args(); paths=run_paths(args.run)
    summary=json.loads((paths['results']/'comparison.json').read_text())
    browser=json.loads((paths['results']/'browser_verification.json').read_text())
    assert browser['charts_rendered'] and browser['attribution_rendered'] and browser['pdf_downloaded'] and not browser['javascript_errors']
    tests=(paths['results']/'tests.log').read_text()
    assert 'passed' in tests and 'failed' not in tests.lower() and 'skipped' not in tests.lower()
    integrity=json.loads((ROOT/'data/metadata/full_integrity_audit.json').read_text())
    assert not integrity['hash_mismatches'] and integrity['duplicate_track_ids']==0
    summary.update(browser_verification=browser,tests=tests.strip(),integrity=integrity)
    (paths['results']/'execution_summary.json').write_text(json.dumps(summary,indent=2))
    status={'stage':'complete','run':args.run,'tests':tests.strip(),'browser_verified':True,'selected_checkpoint':summary['selection']['checkpoint']}
    (paths['results']/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    lines=[f'## Current expanded experiment: {args.run}','',f"All {summary['download']['eligible_outcome_cases']:,} outcome-eligible survivors were examined for the configured channels. **{summary['download']['requested_tracks']:,} real numeric tracks** were verified ({summary['download']['counts']['downloaded']:,} new; {summary['download']['counts']['cached']:,} reused), with {summary['download']['counts']['failed']} download failures. The final cohort contains **{summary['eligible_signal_cases']:,} cases**; split sizes are {summary['split_sizes']} and positive counts are {summary['positive_counts']}. The training-only 75th percentile is **{summary['target_days']:.8f} days**. Survivor-only population.",'',f"Selected application model: **{summary['selection']['model']}**, using validation AUROC with Brier tie-break. Checkpoint: `{summary['selection']['checkpoint']}`. All original working checkpoints and the `initial_127` archive are retained.",'','| Model | AUROC (95% CI) | AUPRC (95% CI) | F1 | Sensitivity | Specificity | Brier | GPU seconds |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name in ['transformer','gru','xgboost']:
        report=summary['models'][name]['expanded']; m=report['test']; intervals=m['bootstrap_95_ci']
        values=[name]+[f"{m[k]:.4f} ({intervals[k][0]:.4f}–{intervals[k][1]:.4f})" for k in ['auroc','auprc']]+[f'{m[k]:.4f}' for k in ['f1','sensitivity','specificity','brier_score']]+[f"{report['training_seconds']:.2f}"]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['',f"Tests: {tests.strip()}. Edge browser verification passed CSV prediction, charts, explainability, mobile layout and PDF download. All {integrity['ledger_tracks']:,} ledger entries matched SHA-256, with zero duplicate IDs. Raw verified bytes: {integrity['verified_raw_bytes']:,}.",'',f"Evidence, confusion matrices, calibration curves, bootstrap details, case-level errors and initial-versus-expanded comparisons: `results/runs/{args.run}/RESULTS.md` and `execution_summary.json`. Expanded normalization: `data/processed/{args.run}/preprocessing.json`. Cohort/exclusions: `data/metadata/{args.run}/cohort.json`.",'',summary['comparison_limitation'],'','No clinical validation or post-hoc calibration has been performed. Confidence intervals are conditional patient-cluster bootstrap intervals. GPU times measure training loops (neural loops include validation), excluding downloads, preprocessing and evaluation.','']
    readme=ROOT/'README.md'; content=readme.read_text(encoding='utf-8')
    if '<!-- CURRENT_FULL_RUN_START -->' in content:
        start=content.index('<!-- CURRENT_FULL_RUN_START -->'); end=content.index('<!-- CURRENT_FULL_RUN_END -->')+len('<!-- CURRENT_FULL_RUN_END -->')
        content=content[:start]+content[end:]
    insertion='<!-- CURRENT_FULL_RUN_START -->\n'+'\n'.join(lines)+'\n<!-- CURRENT_FULL_RUN_END -->\n\n'
    content=content.replace('## Problem and endpoint',insertion+'## Problem and endpoint',1)
    content=content.replace('### Executed results','### Initial 127-case experiment (archived)')
    content=content.replace('Only the 128-case feasibility subset was downloaded; **the full 6,306-case eligible outcome cohort has not been downloaded or trained**.','At the time of the initial archived experiment, only the 128-case feasibility subset had been downloaded. The expanded experiment above supersedes that scope limit.')
    readme.write_text(content,encoding='utf-8')
    print(json.dumps(status,indent=2))
if __name__=='__main__': main()
