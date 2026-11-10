"""Audit completed download ledger against raw files without network access."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import ROOT

def main():
    ledger=ROOT/'data/metadata/full_download_tracks.jsonl'
    seen=set(); failures=[]; mismatches=[]; total_bytes=0
    for line in ledger.read_text().splitlines():
        record=json.loads(line)
        if record['tid'] in seen: raise RuntimeError('Duplicate track entry: '+record['tid'])
        seen.add(record['tid'])
        if record['status']=='failed': failures.append(record); continue
        path=ROOT/f"data/raw/{record['tid']}.csv"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: mismatches.append(record['tid'])
        total_bytes+=record['bytes']
    report={'ledger_tracks':len(seen),'request_failures':len(failures),'hash_mismatches':mismatches,'verified_raw_bytes':total_bytes,'duplicate_track_ids':0}
    (ROOT/'data/metadata/full_integrity_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if mismatches: raise RuntimeError('Raw integrity audit failed')
if __name__=='__main__': main()
