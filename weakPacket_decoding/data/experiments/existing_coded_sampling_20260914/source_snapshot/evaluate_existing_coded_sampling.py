"""Paired complete-frame sampling/decoder ablation on existing SF10 captures.

Clean timing, CFO, LDRO and frame extent are supplied. Header and payload use
the stated sampling pattern. CRC only checks the final selection. No waveform
is truncated into a short packet. This is conditional decoding, not acquisition.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import probe_coded_undersampling as gate
import probe_coded_undersampling_noise as joint
from weak_decoder.os_lora.system.soft_hamming_crc import soft_repair_interleaver_block


def decode(raw, method, downclock):
    def block(start, width, header, cr):
        powers = raw[start:start+width]
        if len(powers) != width:
            return None
        divisor = 4 if header else 1
        if method == "soft":
            expanded = np.tile(powers, (1, downclock))
            return list(soft_repair_interleaver_block(expanded, sf=10, is_header=header, cr=cr, ldro=False).symbol_values)
        semantic = np.max(np.roll(powers, -1, axis=1).reshape(width, -1, divisor), axis=2)
        if method == "hard":
            return np.argmax(semantic, axis=1).tolist()
        return joint.list_block(semantic, 8 if header else 10, cr, downclock)
    head = block(0, 8, True, 4)
    if head is None:
        return None, "header_no_candidate"
    fields = gate.decode_explicit_header(head, 10, 125000, 0)
    if not fields.header_valid or not 1 <= fields.cr <= 4:
        return None, "header_invalid"
    body = []
    for offset in range(8, len(raw), fields.cr+4):
        value = block(offset, fields.cr+4, False, fields.cr)
        if value is None:
            return None, "payload_no_unique_candidate"
        body.extend(value)
    frame = gate.decode_explicit_frame_symbols(head, body, 10, 125000, 0, "grlora")
    return frame, "decoded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(914101)
    permutation = rng.permutation(1024)
    patterns = {f"uniform{m}": np.arange(0,1024,1024//m) for m in (1024,512,256)}
    patterns.update({f"random{m}": np.sort(permutation[:m]) for m in (512,256)})
    banks = {name: np.exp(-2j*np.pi*np.arange(1024)[:,None]*positions[None,:]/1024)
             for name,positions in patterns.items() if name.startswith("random")}
    seeds = (914111,914112,914113)
    levels = (None,-10,-12,-14,-16)
    rows=[]
    began=time.perf_counter()
    for di,dataset in enumerate(gate.common.DEFAULT_DATASETS):
        iq_path,csv_path=gate.common.dataset_paths(dataset)
        iq=np.memmap(iq_path,dtype=np.complex64,mode="r")
        for packet in gate.common.load_packets(csv_path):
            assert (packet['sf'],packet['cr'],packet['ldro']) == (10,1,False)
            items=packet['header_symbols']+packet['payload_symbols']
            rate=packet['os_factor']
            index=np.arange(1024)*rate+rate//2
            clean=np.stack([np.asarray(iq[int(s['start_sample'])+index]) for s in items])
            reference=gate._oversampled_downchirp(10,rate,packet['cfo_int'],packet['cfo_frac'])[::rate]
            truth=gate.decode_explicit_frame_symbols([s['symbol_value'] for s in packet['header_symbols']],
                [s['symbol_value'] for s in packet['payload_symbols']],10,125000,0,'grlora')
            assert truth.payload.crc_valid
            target=truth.payload.payload_bytes
            payload_hash=hashlib.sha256(target).hexdigest()
            power=float(np.mean(np.abs(clean.astype(np.complex128))**2))
            for seed in seeds:
                rng=np.random.default_rng(np.random.SeedSequence([seed,di,packet['packet_index']]))
                noise=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                for level in levels:
                    if level is None and seed != seeds[0]: continue
                    samples=clean if level is None else clean+noise*np.sqrt(power*10**(-level/10))
                    for name,positions in patterns.items():
                        start=time.perf_counter()
                        retained=samples[:,positions]*reference[None,positions]
                        downclock=1024//len(positions) if name.startswith('uniform') else 1
                        transformed=np.fft.fft(retained,axis=1) if name.startswith('uniform') else retained@banks[name].T
                        raw=np.abs(transformed)**2
                        demod_seconds=time.perf_counter()-start
                        for method in ('hard','soft','joint'):
                            start=time.perf_counter()
                            frame,status=decode(raw,method,downclock)
                            decode_seconds=time.perf_counter()-start
                            accepted=frame is not None and frame.header.header_valid and frame.header.has_crc and frame.payload.crc_valid
                            exact=accepted and frame.payload.payload_bytes==target
                            rows.append(dict(dataset=dataset,packet_index=packet['packet_index'],payload_sha256=payload_hash,
                                seed=seed,rgr_db='clean' if level is None else level,pattern=name,decoder=method,
                                samples_per_symbol=len(positions),frame_observations=len(items)*len(positions),
                                status=status,crc_accept=int(accepted),exact=int(exact),false_accept=int(accepted and not exact),
                                demod_seconds=demod_seconds,decode_seconds=decode_seconds,
                                dictionary_bytes=banks[name].nbytes if name in banks else 0))
        print(f'{dataset}: {time.perf_counter()-began:.1f}s',flush=True)
    summary=[]
    for level in ('clean',-10,-12,-14,-16):
        for name in patterns:
            for method in ('hard','soft','joint'):
                group=[r for r in rows if (r['rgr_db'],r['pattern'],r['decoder'])==(level,name,method)]
                successful=[r for r in group if r['exact']]
                summary.append(dict(rgr_db=level,pattern=name,decoder=method,trials=len(group),exact=sum(r['exact'] for r in group),
                    false_accept=sum(r['false_accept'] for r in group),
                    median_success_ms=float(np.median([1000*(r['demod_seconds']+r['decode_seconds']) for r in successful])) if successful else '',
                    dictionary_bytes=group[0]['dictionary_bytes']))
    gate.common.write_csv(args.output/'packet_trials.csv',rows)
    gate.common.write_csv(args.output/'summary.csv',summary)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    import inspect
    paths=[Path(__file__),Path(gate.__file__),Path(joint.__file__),Path(inspect.getfile(soft_repair_interleaver_block))]
    for path in paths: (snapshot/path.name).write_bytes(path.read_bytes())
    (args.output/'config.json').write_text(json.dumps(dict(scope=__doc__,seeds=seeds,rgr_db=levels,
        noise='paired iid complex Gaussian on B-rate grid; RGR includes existing noise; no RF SNR calibration',
        patterns={k:v.tolist() for k,v in patterns.items()},pattern_seed=914101,
        soft='existing bitwise soft Hamming; exact uniform alias likelihoods tiled, not broken by roundoff',
        joint='top 2 residues per symbol; full-rank GF(2) block constraint; rank-deficient blocks rejected',
        truth='clean CRC-valid decoded full payload used only for scoring; CRC not used to search',
        timing='single-run Python diagnostic; cached dictionary; selected-sample dechirp, demodulation and codec included; excludes IO/acquisition/setup',
        limitations='three captures, 28 packets, 24 distinct payloads; repeated noise not independent OTA; one frozen random pattern per budget; no claim of SOTA reproduction',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')
    print(json.dumps([r for r in summary if r['rgr_db']==-12],indent=2))

if __name__=='__main__': main()
