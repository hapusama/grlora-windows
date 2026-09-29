"""Plot measured parameter changes for the six rescued validation trials."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'data/experiments/paper_story_20260912'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def key(r):return r['packet_id'],r['seed'],r['esn0_db']

trials=read(ROOT/'frequency_holdout/trials.csv')
attempts=read(ROOT/'frequency_holdout/attempts.csv')
baseline={key(r):r for r in trials if r['method']=='full_crc1_softfec'}
rescued=sorted([r for r in trials if r['method']=='full_cfo3_softfec' and
    int(r['exact'])>int(baseline[key(r)]['exact'])],key=key)
rows=[]
for r in rescued:
    aa=[a for a in attempts if key(a)==key(r)]
    before=next(a for a in aa if a['method']=='full_cfo3_softfec' and a['index']=='0')
    after=next(a for a in aa if a['method']=='full_cfo3_softfec' and a['exact']=='1')
    oracle=next(a for a in aa if a['method']=='oracle_softfec')
    rows.append(dict(packet_id=r['packet_id'],seed=r['seed'],esn0_db=r['esn0_db'],
        channel_snr_db=r['channel_snr_db'],
        cfo_error_before_bins=float(before['cfo_bins'])-float(oracle['cfo_bins']),
        cfo_error_after_bins=float(after['cfo_bins'])-float(oracle['cfo_bins']),
        timing_error_samples=int(before['payload_start'])-int(oracle['payload_start']),
        timing_change_samples=int(after['payload_start'])-int(before['payload_start']),
        before_header_valid=int(before['header_valid']),before_crc=int(before['crc_accept']),
        after_crc=int(after['crc_accept']),after_exact=int(after['exact'])))
assert len(rows)==6 and all(r['timing_change_samples']==0 for r in rows)
with (ROOT/'frequency_mechanism.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
x=np.arange(len(rows))
ax[0].scatter(x,[r['cfo_error_before_bins'] for r in rows],color='#db7434',label='Original estimate: decode fails',s=55)
ax[0].scatter(x,[r['cfo_error_after_bins'] for r in rows],color='#276bd0',label='CFO -1, same timing: exact packet',s=55)
for i,r in enumerate(rows):
    ax[0].plot([i,i],[r['cfo_error_after_bins'],r['cfo_error_before_bins']],color='#aaaaaa',zorder=0)
ax[0].axhline(0,color='black',lw=.8,ls='--')
ax[0].set_xticks(x,[f"{r['packet_id'].split('_')[1]}\n{r['esn0_db']} dB" for r in rows],fontsize=8)
ax[0].set(xlabel='Packet ID / Es/N0 (two rows can share one physical packet)',ylabel='CFO error relative to clean-sync reference (bins)',title='Mechanism: all six rescues correct a +1-bin error',ylim=(-.12,1.18))
ax[0].legend(fontsize=8,loc='center right');ax[0].grid(alpha=.2)
methods=['full_crc1_softfec','full_list4_softfec','full_cfo3_softfec']
counts=[sum(int(r['exact']) for r in trials if r['method']==m) for m in methods]
costs=[np.mean([int(r['attempts']) for r in trials if r['method']==m]) for m in methods]
ax[1].bar(range(3),counts,color=['#e58d16','#2a9839','#276bd0'])
for i,(n,c) in enumerate(zip(counts,costs)):
    ax[1].text(i,n+1,f'{n}/72\n{c:.3f} calls/trial',ha='center',fontsize=10)
ax[1].set_xticks(range(3),['Strong single','Coupled ridge 4','Local CFO 3'])
ax[1].set(ylabel='CRC-valid, byte-exact packets',title='All 72 validation trials, including failures',ylim=(0,60))
fig.suptitle('Same-session OTA + band-limited AWGN; 12 different physical packets',fontsize=12)
fig.tight_layout();fig.savefig(ROOT/'frequency_mechanism.png',dpi=180);fig.savefig(ROOT/'frequency_mechanism.svg')
print(rows)
