"""Separate candidate truncation, metric errors and induced-code distance.

Only existing complete SF10/CR1 33-byte OTA packets are used. Clean symbol
windows and CFO remain supplied. Earlier noise seeds are replay diagnostics;
fresh seeds are noise validation, not independent capture validation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import probe_coded_undersampling as g
import probe_coded_undersampling_noise as old
import coded_sampling_trellis as trellis
from probe_coded_frame_identifiability import binary_rank


def distance_audit():
    rows=[]
    for sf_app in (8,10,12):
        for cr in (1,2,3,4):
            for d in (1,2,4,8):
                a,_,pivots,kept=g.observation_system(sf_app,cr,d)
                minimum=0
                if len(pivots)==4*sf_app:
                    for weight in range(1,cr+5):
                        found=False
                        for support in combinations(range(cr+4),weight):
                            remaining=[i for s in range(cr+4) if s not in support for i in range(s*kept,(s+1)*kept)]
                            if binary_rank(a[remaining]) < 4*sf_app:
                                found=True;break
                        if found:
                            minimum=weight;break
                rows.append(dict(sf_app=sf_app,cr=cr,downclock=d,rank=len(pivots),
                    parity_bits=len(a)-len(pivots),ambiguity_bits=4*sf_app-len(pivots),
                    min_residue_symbol_distance=minimum))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('fresh output required')
    args.output.mkdir(parents=True)
    checks=trellis.self_check()
    g.common.write_csv(args.output/'distance_audit.csv',distance_audit())
    rows=[];blocks=[]
    seeds=(914111,914112,914113,914211,914212,914213)
    start=time.perf_counter()
    for di,dataset in enumerate(g.common.DEFAULT_DATASETS):
        iq_path,meta=g.common.dataset_paths(dataset)
        iq=np.memmap(iq_path,dtype=np.complex64,mode='r')
        for p in g.common.load_packets(meta):
            r=p['os_factor'];idx=np.arange(1024)*r+r//2
            clean=np.stack([np.asarray(iq[int(s['start_sample'])+idx]) for s in p['header_symbols']+p['payload_symbols']])
            reference=g._oversampled_downchirp(10,r,p['cfo_int'],p['cfo_frac'])[::r]
            truth=g.decode_explicit_frame_symbols([s['symbol_value'] for s in p['header_symbols']],
                [s['symbol_value'] for s in p['payload_symbols']],10,125000,0,'grlora').payload.payload_bytes
            power=float(np.mean(np.abs(clean.astype(np.complex128))**2))
            for seed in seeds:
                rng=np.random.default_rng(np.random.SeedSequence([seed,di,p['packet_index']]))
                noise=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                for rgr in (-10,-12,-14,-16):
                    samples=clean+noise*np.sqrt(power*10**(-rgr/10))
                    begin=time.perf_counter()
                    raw=np.abs(np.fft.fft(samples[:,::2]*reference[None,::2],axis=1))**2
                    demod=time.perf_counter()-begin
                    header_powers=np.max(np.roll(raw[:8],-1,axis=1).reshape(8,-1,4),axis=2)
                    head=old.list_block(header_powers,8,4,2)
                    valid=False
                    if head is not None:
                        fields=g.decode_explicit_header(head,10,125000,0)
                        valid=fields.header_valid and fields.cr==1
                    powers=np.roll(raw[8:],-1,axis=1)
                    bodies={'top2':[],'trellis':[]};elapsed={'top2':demod,'trellis':demod}
                    covered=True; forced_error=False
                    for offset in range(0,len(powers),5):
                        scores=powers[offset:offset+5]
                        target=np.array([s['symbol_value'] for s in p['payload_symbols'][offset:offset+5]])
                        residues=target%512
                        ranks=np.sum(scores > scores[np.arange(5),residues][:,None],axis=1)+1
                        covered=covered and bool(np.all(ranks<=2))
                        groups=trellis.kernel(10,1,2)[4]
                        local_wrong=0
                        for j,(indices,mask) in enumerate(groups):
                            state,position=np.argwhere((indices==residues[j]) & mask)[0]
                            if np.max(np.where(mask[state],scores[j,indices[state]],-np.inf)) > scores[j,residues[j]]+1e-7:
                                local_wrong+=1
                        forced_error=forced_error or local_wrong>0
                        begin=time.perf_counter();first=old.list_block(scores,10,1,2);elapsed['top2']+=time.perf_counter()-begin
                        begin=time.perf_counter();second,metric,chosen=trellis.decode_block(scores);elapsed['trellis']+=time.perf_counter()-begin
                        for method,values in [('top2',first),('trellis',second)]:
                            if values is None:bodies[method]=None
                            elif bodies[method] is not None:bodies[method].extend(values)
                        blocks.append(dict(dataset=dataset,packet_index=p['packet_index'],seed=seed,rgr_db=rgr,block=offset//5,
                            truth_in_top2=int(np.all(ranks<=2)),max_truth_rank=int(np.max(ranks)),
                            local_intra_syndrome_wrong=local_wrong,top2_exact=int(first is not None and np.array_equal(first,target)),
                            trellis_exact=int(np.array_equal(second,target)),
                            trellis_score=metric,truth_score=float(np.sum(scores[np.arange(5),residues]))))
                    for method,body in bodies.items():
                        accept=exact=False
                        if valid and body is not None:
                            frame=g.decode_explicit_frame_symbols(head,body,10,125000,0,'grlora')
                            accept=frame.header.header_valid and frame.header.has_crc and frame.payload.crc_valid
                            exact=accept and frame.payload.payload_bytes==truth
                        rows.append(dict(dataset=dataset,packet_index=p['packet_index'],seed=seed,
                            split='replay' if seed<914200 else 'new_noise',rgr_db=rgr,method=method,
                            header_valid=int(valid),all_payload_truth_in_top2=int(covered),
                            any_intra_syndrome_error=int(forced_error),exact=int(exact),false_accept=int(accept and not exact),
                            payload_demod_decode_seconds=elapsed[method]))
        print(dataset,round(time.perf_counter()-start,1),flush=True)
    summary=[]
    for split in ('replay','new_noise'):
        for rgr in (-10,-12,-14,-16):
            for method in ('top2','trellis'):
                selected=[r for r in rows if (r['split'],r['rgr_db'],r['method'])==(split,rgr,method)]
                summary.append(dict(split=split,rgr_db=rgr,method=method,trials=len(selected),
                    exact=sum(r['exact'] for r in selected),header_valid=sum(r['header_valid'] for r in selected),
                    top2_payload_coverage=sum(r['all_payload_truth_in_top2'] for r in selected),
                    intra_syndrome_clean_packets=sum(not r['any_intra_syndrome_error'] for r in selected),
                    false_accept=sum(r['false_accept'] for r in selected)))
    g.common.write_csv(args.output/'packet_trials.csv',rows)
    g.common.write_csv(args.output/'block_diagnostics.csv',blocks)
    g.common.write_csv(args.output/'summary.csv',summary)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for module in [Path(__file__),Path(trellis.__file__),Path(g.__file__),Path(old.__file__)]:
        (snapshot/module.name).write_bytes(module.read_bytes())
    (args.output/'config.json').write_text(json.dumps(dict(scope=__doc__,checks=checks,seeds=seeds,
        decoder='exact maximum sum of symbol FFT powers for identifiable half-rate payload code; header remains Top-2',
        diagnostics='truth only used to score and attribute errors; does not participate in either decoder',
        distance='exact minimum number of differing residue symbols in the induced linear code; zero denotes noninjective input mapping',
        noise='paired B-grid iid complex Gaussian; original captured noise remains; RGR not RF SNR',
        timing='includes half-rate FFT and payload block solver, excludes header/CRC/IO and acquisition'),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
