"""Existing-IQ validation of residue trellis, header decoder and metric.

All methods use clean timing/CFO/LDRO/frame extents, but decode their headers
and payloads from retained samples. No CRC-based payload list selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import probe_coded_undersampling as g
import probe_coded_undersampling_noise as old
import coded_sampling_trellis as t

METHODS={
 'u1024_soft_legacy':('uniform1024','soft','legacy','soft'),
 'u1024_soft_bessel':('uniform1024','soft','bessel','soft'),
 'r512_soft_legacy':('random512','soft','legacy','soft'),
 'r512_soft_bessel':('random512','soft','bessel','soft'),
 'u512_trellis_top2head':('uniform512','trellis','power','top2'),
 'u512_trellis_softhead':('uniform512','trellis','power','soft'),
 'u512_trellis_bessel':('uniform512','trellis','bessel','soft'),
 'u512_trellis_headerlist':('uniform512','trellis','bessel','list'),
}


def receive(raw,count,downclock,decoder,metric,head_method):
    header_scores=t.symbol_scores(raw[:8],4,'bessel' if metric=='bessel' else 'legacy',count)
    candidates=[]
    if head_method in ('soft','list'):
        candidates.append(t.soft_decode_scores(np.tile(header_scores,(1,downclock)),8,4))
    if head_method=='list':
        candidates.append(t.soft_decode_scores(np.tile(t.symbol_scores(raw[:8],4,'legacy',count),(1,downclock)),8,4))
    if head_method in ('top2','list'):
        scores=t.symbol_scores(raw[:8],4,'power',count)
        h=old.list_block(scores,8,4,downclock)
        if h is not None:candidates.append(h)
    eligible=[]
    for h in candidates:
        fields=g.decode_explicit_header(h,10,125000,0)
        if fields.header_valid and 1<=fields.cr<=4:
            score=float(np.sum(header_scores[np.arange(8),np.array(h)%header_scores.shape[1]]))
            eligible.append((score,h,fields))
    if not eligible:return None,'header_invalid'
    _,head,fields=max(eligible,key=lambda x:x[0])
    powers=t.symbol_scores(raw[8:],1,metric,count)
    body=[]
    for offset in range(0,len(powers),fields.cr+4):
        block=powers[offset:offset+fields.cr+4]
        if len(block)!=fields.cr+4:return None,'incomplete_block'
        if decoder=='trellis' and fields.cr==1:
            symbols,_,_=t.decode_block(block,10,1,downclock)
        else:
            symbols=t.soft_decode_scores(np.tile(block,(1,downclock)),10,fields.cr)
        body.extend(symbols)
    return g.decode_explicit_frame_symbols(head,body,10,125000,0,'grlora'),'decoded'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('fresh output required')
    args.output.mkdir(parents=True)
    permutation=np.random.default_rng(914101).permutation(1024)
    patterns={'uniform1024':np.arange(1024),'uniform512':np.arange(0,1024,2),'random512':np.sort(permutation[:512])}
    bank=np.exp(-2j*np.pi*np.arange(1024)[:,None]*patterns['random512'][None,:]/1024)
    seeds=(914111,914112,914113,914311,914312,914313)
    rows=[];start=time.perf_counter()
    for di,dataset in enumerate(g.common.DEFAULT_DATASETS):
        path,meta=g.common.dataset_paths(dataset);iq=np.memmap(path,dtype=np.complex64,mode='r')
        for p in g.common.load_packets(meta):
            rate=p['os_factor'];idx=np.arange(1024)*rate+rate//2
            clean=np.stack([np.asarray(iq[int(s['start_sample'])+idx]) for s in p['header_symbols']+p['payload_symbols']])
            ref=g._oversampled_downchirp(10,rate,p['cfo_int'],p['cfo_frac'])[::rate]
            target=g.decode_explicit_frame_symbols([s['symbol_value'] for s in p['header_symbols']],
                [s['symbol_value'] for s in p['payload_symbols']],10,125000,0,'grlora').payload.payload_bytes
            power=float(np.mean(np.abs(clean.astype(np.complex128))**2))
            for seed in seeds:
                rng=np.random.default_rng(np.random.SeedSequence([seed,di,p['packet_index']]))
                noise=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                for rgr in (None,-10,-12,-14,-16,-18):
                    if rgr is None and seed!=seeds[0]:continue
                    samples=clean if rgr is None else clean+noise*np.sqrt(power*10**(-rgr/10))
                    spectra={};demod={}
                    for pattern,positions in patterns.items():
                        begin=time.perf_counter();retained=samples[:,positions]*ref[None,positions]
                        spectrum=retained@bank.T if pattern=='random512' else np.fft.fft(retained,axis=1)
                        spectra[pattern]=np.abs(spectrum)**2;demod[pattern]=time.perf_counter()-begin
                    for method,(pattern,decoder,metric,head_method) in METHODS.items():
                        count=len(patterns[pattern]);d=2 if pattern=='uniform512' else 1
                        begin=time.perf_counter();frame,status=receive(spectra[pattern],count,d,decoder,metric,head_method)
                        seconds=time.perf_counter()-begin+demod[pattern]
                        accept=frame is not None and frame.header.header_valid and frame.header.has_crc and frame.payload.crc_valid
                        exact=accept and frame.payload.payload_bytes==target
                        rows.append(dict(dataset=dataset,packet_index=p['packet_index'],seed=seed,
                            split='replay' if seed<914300 else 'new_noise',rgr_db='clean' if rgr is None else rgr,
                            method=method,status=status,exact=int(exact),false_accept=int(accept and not exact),seconds=seconds))
        print(dataset,round(time.perf_counter()-start,1),flush=True)
    summary=[]
    for split in ('replay','new_noise'):
        for rgr in ('clean',-10,-12,-14,-16,-18):
            for method in METHODS:
                group=[r for r in rows if (r['split'],r['rgr_db'],r['method'])==(split,rgr,method)]
                if not group:continue
                good=[r['seconds']*1000 for r in group if r['exact']]
                summary.append(dict(split=split,rgr_db=rgr,method=method,trials=len(group),
                    exact=sum(r['exact'] for r in group),false_accept=sum(r['false_accept'] for r in group),
                    median_success_ms=float(np.median(good)) if good else ''))
    g.common.write_csv(args.output/'packet_trials.csv',rows)
    g.common.write_csv(args.output/'summary.csv',summary)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),Path(t.__file__),Path(g.__file__),Path(old.__file__)]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    (args.output/'config.json').write_text(json.dumps(dict(scope=__doc__,methods=METHODS,seeds=seeds,
        patterns={k:v.tolist() for k,v in patterns.items()},timing='selected-sample dechirp, demod and full decoder; cached setup, no IO/acquisition',
        metric='Bessel likelihood is a known noncoherent AWGN model with blind pooled moment estimates; nonpositive amplitude estimate falls back to legacy metric',
        baseline='vectorized existing independent-bit soft Hamming; 120 random-block exact-output equivalence checks passed across CR1..4 and alias factors1,2,4',
        header='soft, Top-2 or finite union; checksum-valid header selected by spectral score; no payload-CRC list search',
        limits='new noise uses same 28 captured packets, not a capture holdout; one random pattern; no SOTA or ADC-power claim'),indent=2),encoding='utf-8')
    print(json.dumps([r for r in summary if r['split']=='new_noise' and r['rgr_db'] in (-14,-16)],indent=2))

if __name__=='__main__':main()
