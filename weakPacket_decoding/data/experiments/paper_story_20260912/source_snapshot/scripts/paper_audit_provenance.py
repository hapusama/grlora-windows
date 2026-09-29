"""Record reproducibility hashes without copying private captures."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/experiments/paper_audit_20260911"
WORK=ROOT.parent.parent


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return dict(path=str(path.relative_to(WORK)),bytes=path.stat().st_size,sha256=h.hexdigest())


def main():
    files=list((ROOT/"scripts").glob("paper_*.py"))
    files+=list((ROOT/"weak_decoder").rglob("*.py"))
    files+=list((WORK/"lora-rfsr-savaux/weak_decoder/synchronization").glob("*.py"))
    for suffix in [8,16,32]:
        files.append(ROOT.parent/f"data/USRP_IQ/0_0_0_10_14_{suffix}.bin")
        files.append(ROOT/f"data/weak_sync_chain/header_first/0_0_0_10_14_{suffix}_header_first_symbols.csv")
    for audit in [OUT/"sfd/clean_sync_audit.csv",ROOT/"data/experiments/noisy_framesync_headroom_ota_awgn_20260821/clean_sync_audit.csv"]:
        files.append(audit)
        with audit.open(newline="") as f:rows=list(csv.DictReader(f))
        ota=WORK/"lora-rfsr-savaux/data/reference_phy/rfsr_db"
        for row in rows:
            metadata=ota/"metadata"/f"{row['packet_id']}.json"
            files.append(metadata)
            data=json.loads(metadata.read_text())
            files.append(ota/data["ota"]["relative_path"])
    files+=list(OUT.rglob("*.csv"))
    record=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
        note="Source hashes include existing uncommitted workspace state; no hardware or remote writes performed.",
        files=[digest(p) for p in sorted(set(files))])
    (OUT/"provenance.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(f"hashed {len(record['files'])} files")


if __name__=="__main__":main()
