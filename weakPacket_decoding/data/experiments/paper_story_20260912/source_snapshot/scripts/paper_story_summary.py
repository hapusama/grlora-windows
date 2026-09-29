"""Summarize paired receiver experiments; bootstrap physical packets, not noise rows."""
from pathlib import Path
import csv
import json
import hashlib
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/experiments/paper_story_20260912'

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def summarize(folder):
    rows=read(folder/'trials.csv')
    methods=sorted({r['method'] for r in rows})
    stats=[]
    for snr in ['all']+sorted({r['esn0_db'] for r in rows}):
        for method in methods:
            rr=[r for r in rows if r['method']==method and (snr=='all' or r['esn0_db']==snr)]
            stats.append(dict(esn0_db=snr,method=method,correct=sum(int(r['exact']) for r in rr),
                trials=len(rr),false_delivery=sum(int(r['false_delivery']) for r in rr),
                mean_attempts=float(np.mean([int(r['attempts']) for r in rr]))))
    comparisons=[]
    for method in methods:
        baseline='full_crc1_softfec'
        a={(r['packet_id'],r['seed'],r['esn0_db']):int(r['exact']) for r in rows if r['method']==method}
        b={(r['packet_id'],r['seed'],r['esn0_db']):int(r['exact']) for r in rows if r['method']==baseline}
        keys=sorted(a.keys()&b.keys())
        ids=sorted({k[0] for k in keys})
        delta=np.array([np.mean([a[k]-b[k] for k in keys if k[0]==pid]) for pid in ids])
        rng=np.random.default_rng(9122026)
        boot=delta[rng.integers(len(ids),size=(20000,len(ids)))].mean(axis=1)
        wins=sum(a[k]>b[k] for k in keys);losses=sum(a[k]<b[k] for k in keys)
        comparisons.append(dict(method=method,baseline=baseline,wins=wins,losses=losses,
            difference=float(delta.mean()),cluster_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
            descriptive_mcnemar_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1,
            note='McNemar ignores within-packet dependence; bootstrap resamples physical packets. Exploratory multiple comparisons.'))
    (folder/'summary.json').write_text(json.dumps(dict(aggregate=stats,paired=comparisons),indent=2))
    return stats

def main():
    mainstats=summarize(OUT)
    follow=OUT/'frequency_holdout'
    followstats=summarize(follow) if (follow/'trials.csv').exists() else []
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for ax,stats,title in [(axs[0],mainstats,'24 OTA packets + band-limited AWGN'),(axs[1],followstats,'12 different OTA packets: frequency control')]:
        for method,label in [('chip_strict','Strict / hard FEC'),('full_crc1_softfec','Full SFD / soft FEC / single'),('full_list4_softfec','Ridge list 4 / soft FEC'),('full_cfo3_softfec','Local CFO 3 / soft FEC'),('oracle_softfec','Clean-sync oracle / soft FEC')]:
            rr=[r for r in stats if r['method']==method and r['esn0_db']!='all']
            colors={'chip_strict':'#777777','full_crc1_softfec':'#e58d16','full_list4_softfec':'#2a9839','full_cfo3_softfec':'#276bd0','oracle_softfec':'#a246a3'}
            if rr:ax.plot([float(r['esn0_db'])-10*np.log10(4096) for r in rr],[r['correct']/r['trials'] for r in rr],'o-',color=colors[method],label=label)
        ax.set(title=title,xlabel='Channel SNR (dB)',ylabel='CRC-valid exact packet fraction',ylim=(0,1.04))
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    natural=read(OUT/'natural_weak/trials.csv')
    names=['chip_strict','full_crc1_softfec','full_cfo3_softfec','chip_list4_softfec']
    natural_cfo=read(OUT/'frequency_natural/trials.csv') if (OUT/'frequency_natural/trials.csv').exists() else natural
    counts=[sum(int(r['exact']) for r in (natural_cfo if m=='full_cfo3_softfec' else natural) if r['method']==m) for m in names]
    axs[2].bar(range(4),counts,color=['gray','orange','green','blue'])
    axs[2].set_xticks(range(4),['Chip strict','Full + soft','Local CFO 3','Chip list soft'],rotation=20,ha='right')
    axs[2].set(title='Natural weak SF10: no added noise',ylabel='Exact CRC packets in 68 detected events',ylim=(0,68))
    for i,n in enumerate(counts):axs[2].text(i,n+1,str(n),ha='center')
    axs[2].text(.02,.94,'7 recordings; last 3 yield no detector events\nNot end-to-end PDR',transform=axs[2].transAxes,fontsize=8,va='top')
    fig.tight_layout();fig.savefig(OUT/'story_evidence.png',dpi=180);fig.savefig(OUT/'story_evidence.svg');plt.close(fig)
    files=list(OUT.rglob('*.csv'))+list(OUT.rglob('config.json'))+list(ROOT.joinpath('scripts').glob('paper_*story.py'))+[Path(__file__),ROOT/'scripts/paper_receiver_frequency.py']
    sources=list(ROOT.joinpath('weak_decoder').rglob('*.py'))+list(ROOT.joinpath('scripts').glob('paper_*.py'))
    for src in sources:
        dst=OUT/'source_snapshot'/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    files+=sources
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    external=ROOT.parent.parent/'lora-rfsr-savaux/weak_decoder/synchronization/single_packet.py'
    shutil.copy2(external,OUT/'source_snapshot/external_single_packet.py')
    hashes[str(external)]=hashlib.sha256(external.read_bytes()).hexdigest()
    for cfg in [OUT/'config.json',follow/'config.json']:
        if not cfg.exists():continue
        for name in json.loads(cfg.read_text())['metadata']:
            path=Path(name)
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'main':[r for r in mainstats if r['esn0_db']=='all'],'frequency':[r for r in followstats if r['esn0_db']=='all']},indent=2))

if __name__=='__main__':main()
