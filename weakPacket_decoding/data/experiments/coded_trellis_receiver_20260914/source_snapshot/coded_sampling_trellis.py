"""Exact additive-metric decoding of identifiable LoRa residue codes.

Classical syndrome-trellis dynamic programming is applied to the induced
Hamming/interleaver/inverse-Gray observation code. The optimizer is exact for
the supplied additive symbol scores, not a claim of an exact RF channel model.
CRC and known payloads are never used by this decoder.
"""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp, i0e
import probe_coded_undersampling as gate


@lru_cache(maxsize=None)
def kernel(sf_app, cr, downclock):
    matrix, transform, pivots, kept = gate.observation_system(sf_app, cr, downclock)
    if len(pivots) != 4*sf_app:
        raise ValueError('residue code is not injective')
    check = transform[len(pivots):]
    checks = len(check)
    if checks > 10:
        raise ValueError('reference trellis limited to 1024 states')
    states = 1 << checks
    alphabet = 1 << kept
    residue_bits = ((np.arange(alphabet)[:,None] >> np.arange(kept)) & 1).astype(np.uint8)
    groups = []
    for column in range(cr+4):
        syndrome = (residue_bits @ check[:,column*kept:(column+1)*kept].T) & 1
        labels = np.sum(syndrome * (1 << np.arange(checks)),axis=1)
        members = [np.flatnonzero(labels == state) for state in range(states)]
        width = max(len(m) for m in members)
        indices = np.zeros((states,width),dtype=np.int64)
        valid = np.zeros_like(indices,dtype=bool)
        for state, values in enumerate(members):
            indices[state,:len(values)] = values
            valid[state,:len(values)] = True
        groups.append((indices,valid))
    xor = np.arange(states)[:,None] ^ np.arange(states)[None,:]
    return matrix,transform,pivots,kept,groups,xor


def decode_block(scores, sf_app=10, cr=1, downclock=2):
    scores=np.asarray(scores,dtype=np.float64)
    matrix,transform,pivots,kept,groups,xor=kernel(sf_app,cr,downclock)
    if scores.shape != (cr+4,1<<kept):
        raise ValueError('wrong symbol-score shape')
    states=len(xor)
    previous=np.full(states,-np.inf); previous[0]=0
    parents=[]; choices=[]
    for column,(indices,valid) in enumerate(groups):
        grouped=np.where(valid,scores[column,indices],-np.inf)
        best=np.argmax(grouped,axis=1)
        best_score=grouped[np.arange(states),best]
        best_residue=indices[np.arange(states),best]
        transitions=previous[None,:]+best_score[xor]
        parent=np.argmax(transitions,axis=1)
        previous=transitions[np.arange(states),parent]
        parents.append(parent); choices.append(best_residue)
    state=0; residues=np.zeros(cr+4,dtype=np.int64)
    for column in range(cr+3,-1,-1):
        parent=int(parents[column][state])
        residues[column]=choices[column][state ^ parent]
        state=parent
    observed=((residues[:,None] >> np.arange(kept)) & 1).reshape(-1).astype(np.uint8)
    rhs=(transform @ observed) & 1
    if np.any(rhs[len(pivots):]):
        raise AssertionError('trellis output violates parity constraints')
    bits=np.zeros(4*sf_app,dtype=np.uint8); bits[list(pivots)]=rhs[:len(pivots)]
    selected=gate.encode_block(bits,sf_app,cr)
    return selected,float(previous[0]),residues


@lru_cache(maxsize=None)
def soft_kernel(sf_app,cr):
    symbols=np.arange(1<<sf_app)
    gray=symbols ^ (symbols>>1)
    masks=((gray[None,:] >> np.arange(sf_app-1,-1,-1)[:,None]) & 1).astype(bool)
    words=np.array([gate.encode_hamming_nibble(n,cr) for n in range(16)])
    code_bits=((words[:,None] >> np.arange(cr+3,-1,-1)[None,:]) & 1).astype(int)
    return masks,code_bits


def soft_decode_scores(scores,sf_app,cr):
    """Vectorized independent-bit soft Hamming baseline, no alias inference."""
    scores=np.asarray(scores,dtype=np.float64)
    masks,code_bits=soft_kernel(sf_app,cr)
    bit_logs=np.stack([logsumexp(np.where(masks[None,:,:]==bit,scores[:,None,:],-np.inf),axis=2)
                       for bit in (0,1)],axis=-1)
    bit_logs-=logsumexp(bit_logs,axis=-1)[...,None]
    words=[]
    for row in range(sf_app):
        indices=(np.arange(cr+4)-row-1)%sf_app
        metrics=np.sum(bit_logs[np.arange(cr+4)[None,:],indices[None,:],code_bits],axis=1)
        words.append(int(np.argsort(metrics)[-1]))
    bits=[(n >> shift)&1 for n in words for shift in (3,2,1,0)]
    return gate.encode_block(bits,sf_app,cr)


def symbol_scores(raw,divisor=1,kind='legacy',samples_per_symbol=None):
    """Construct scores using observed spectra only; no clean signal power.

    Bessel scores use a plug-in constant-amplitude, unknown-phase AWGN model.
    Noise and amplitude are pooled across the supplied symbol spectra.
    """
    raw=np.asarray(raw,dtype=np.float64)
    if kind=='legacy':
        noise=np.maximum(np.median(raw,axis=1)/np.log(2),1e-30)
        scores=raw/noise[:,None]
    elif kind=='bessel':
        if samples_per_symbol is None:raise ValueError('sample count required')
        noise=max(float(np.median(raw)/np.log(2)),1e-30)
        signal=float((np.mean(raw)-noise)/samples_per_symbol)
        if signal <= 0:
            return symbol_scores(raw,divisor,'legacy')
        argument=2*np.sqrt(signal*raw)/(noise/samples_per_symbol)
        scores=np.log(i0e(argument))+argument
    elif kind=='power':scores=raw
    else:raise ValueError('unknown metric')
    grouped=np.max(np.roll(scores,-1,axis=1).reshape(len(raw),-1,divisor),axis=2)
    grouped-=np.max(grouped,axis=1)[:,None]
    return np.maximum(grouped,-80) if kind=='legacy' else grouped


def self_check():
    rng=np.random.default_rng(914201)
    # An independently enumerated 4096-message small code validates optimality.
    candidates=np.array([gate.encode_block([(m >> b)&1 for b in range(12)],3,1) for m in range(1<<12)])
    for _ in range(24):
        scores=rng.standard_normal((5,8))
        selected,score,residues=decode_block(scores,3,1,1)
        brute=np.sum(scores[np.arange(5)[None,:],candidates],axis=1)
        if not np.isclose(score,np.max(brute),rtol=1e-12,atol=1e-12):
            raise AssertionError('trellis disagrees with exhaustive codebook')
    for _ in range(64):
        bits=rng.integers(0,2,40,dtype=np.uint8)
        symbols=gate.encode_block(bits,10,1)
        scores=np.zeros((5,512));scores[np.arange(5),np.array(symbols)%512]=100
        recovered,_,_=decode_block(scores)
        if recovered != symbols:
            raise AssertionError('noiseless half-rate block failed')
    return dict(exhaustive_optimality_checks=24,codebook_size=4096,noiseless_sf10_checks=64,half_rate_states=32)

if __name__=='__main__': print(self_check())
