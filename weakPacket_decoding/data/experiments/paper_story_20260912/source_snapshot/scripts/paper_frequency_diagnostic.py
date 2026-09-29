"""Explicitly selected failure cases: mechanism diagnostic, not held-out evidence."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from paper_receiver_frequency import ROOT, base, work

def main():
    out=ROOT/'data/experiments/paper_story_20260912/frequency_selected_diagnostic'
    out.mkdir(parents=True,exist_ok=True)
    paths=list(base.packet_metadata_paths(ROOT.parent.parent/'lora-rfsr-savaux/data/reference_phy/rfsr_db'))
    indices=[96,100,104]
    (out/'config.json').write_text(json.dumps(dict(metadata=[str(paths[i]) for i in indices],
        selection='Selected after inspecting failures: IDs 97,101,105; mechanism diagnostic ONLY, no generalization claim',
        snrs=[12,13,14],seeds=[91601,91602]),indent=2))
    audits=[];trials=[];attempts=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work,(str(paths[i]),[12.,13.,14.],[91601,91602])) for i in indices]
        for f in as_completed(futures):
            a,t,c=f.result();audits.append(a);trials.extend(t);attempts.extend(c)
            base.write_csv_rows(out/'clean_audit.csv',audits)
            base.write_csv_rows(out/'trials.csv',trials)
            base.write_csv_rows(out/'attempts.csv',attempts)
            print(f'completed {len(audits)}/3 selected diagnostic packets',flush=True)

if __name__=='__main__':main()
