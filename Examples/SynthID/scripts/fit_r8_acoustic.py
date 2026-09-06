#!/usr/bin/env python3
"""R8 acoustic kick: jointly fit bounded modal scalars, never waveform data.

The v6 coordinate search could not move coupled mode frequency/phase/glide
parameters together. This renderer uses freely phased membrane/shell modes,
a two-rate tension relaxation, multiband beater noise, and the ROM's finite
release. Training is phase-blind: multiresolution log/linear magnitudes and
pooled band power. No time-domain loss, residual/table parameters or EQ FIR.
Torch supplies CPU derivatives only; scipy L-BFGS-B enforces scalar bounds.
NumPy is the independent shipping reference. Dependencies: numpy scipy torch.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import signal, optimize
import torch

import compare
import render_reference as reference
from fit_r8_kick import biquad_coeffs

torch.set_num_threads(1)
DTYPE = torch.float64
# Measured low resonances and struck shell/beater resonances, not a harmonic bank.
LOW = (45., 80., 93., 128., 170., 195., 250.)
HIGH = (365., 396., 623., 1010., 1120., 1953., 2420., 2800., 2920., 5557.)
NOISE = (("bp", 350., .9), ("bp", 700., .9), ("bp", 1400., 1.2),
         ("bp", 2600., 2.), ("bp", 4200., 1.5), ("hp", 6500., .707))
BOUNDS = {}
SEED = {}


def scalar(name, default, lo, hi, log=False):
    BOUNDS[name] = (lo, hi, "log" if log else "linear")
    SEED[name] = default


scalar("fastTime", 1/481., .0005, .01, True)
scalar("slowTime", 1/35., .012, .10, True)
scalar("attackTime", .00015, .00004, .002, True)
scalar("fadeStart", .148, .12, .17)
scalar("fadeEnd", 8032/44100., .180, .185)
scalar("pressure", .5, 0., 4.)
scalar("outGain", 1., 1., 1.)  # fixed; no gain/amp degeneracy or hidden normalizer
for i, f in enumerate(LOW + HIGH, 1):
    low = i <= len(LOW)
    scalar(f"f{i}", f, f*.80, f*1.20, True)
    scalar(f"a{i}", .18 if low else .018, .00001, .9, True)
    scalar(f"d{i}", 20. if low else 100., 4. if low else 15., 180. if low else 2200., True)
    scalar(f"p{i}", .3 if low else 0., -2., 2.)
    if low:
        scalar(f"r{i}", .012 if i==1 else .0005, .00004, .04, True)
        scalar(f"g{i}", .6, 0., 1.8)
        scalar(f"s{i}", .6, 0., 1.8)
for i in range(1, len(NOISE)+1):
    scalar(f"n{i}", .25, .000001, 3., True)
    scalar(f"nd{i}", 160., 35., 2500., True)
    scalar(f"nr{i}", .0003, .00004, .008, True)
    scalar(f"nt{i}", .005, .000001, .3, True)
    scalar(f"ntd{i}", 35., 8., 150., True)

ORDER = [k for k in BOUNDS if k != "outGain"]


def encoded(p):
    return np.array([math.log(p[k]) if BOUNDS[k][2] == "log" else p[k] for k in ORDER])


def decoded(z):
    p = {k: math.exp(float(v)) if BOUNDS[k][2] == "log" else float(v) for k, v in zip(ORDER, z)}
    p["outGain"] = 1.
    return p


def noise_bands(frames, sr):
    n = reference.dgen_noise(frames).astype(np.float64)*2-1
    result = []
    for mode, fc, q in NOISE:
        b0,b1,b2,a1,a2 = biquad_coeffs(mode, fc, q, sr)
        result.append(signal.lfilter([b0,b1,b2], [1,a1,a2], n))
    return np.array(result)


def render(p, frames, sr):
    t = np.arange(frames, dtype=np.float64)/sr
    fast = p["fastTime"] * -np.expm1(-t/p["fastTime"])
    slow = p["slowTime"] * -np.expm1(-t/p["slowTime"])
    attack = -np.expm1(-t/p["attackTime"])
    mix = np.zeros(frames)
    for i in range(1, len(LOW)+len(HIGH)+1):
        phase = p[f"f{i}"] * (t + (p[f"g{i}"]*fast + p[f"s{i}"]*slow if i<=len(LOW) else 0)) + p[f"p{i}"]
        rise = -np.expm1(-t/p[f'r{i}']) if i<=len(LOW) else attack
        mix += p[f"a{i}"]*np.exp(-p[f"d{i}"]*t)*np.sin(2*np.pi*(phase-np.floor(phase)))*rise
    pressure = 1 + p['pressure']*4*np.maximum(-mix, 0)
    for i, n in enumerate(noise_bands(frames, sr), 1):
        mix += n*(p[f"n{i}"]*np.exp(-p[f"nd{i}"]*t)*pressure + p[f"nt{i}"]*np.exp(-p[f"ntd{i}"]*t)) * -np.expm1(-t/p[f"nr{i}"])
    u = np.clip((t-p["fadeStart"])/(p["fadeEnd"]-p["fadeStart"]),0,1)
    return (mix*(1-u*u*(3-2*u))*p["outGain"]).astype(np.float32)


class Fit:
    def __init__(self, target, sr):
        self.sr = sr
        self.t = torch.arange(len(target), dtype=DTYPE)/sr
        self.noise = torch.tensor(noise_bands(len(target), sr), dtype=DTYPE)
        self.specs = []
        # Centered frames include the onset and the ROM fade, unlike v6's
        # unpadded long windows. Equal logarithmic frequency-band weighting
        # prevents empty high-frequency bins overwhelming the membrane.
        for w in (256, 512, 1024, 2048, 4096, 8192):
            win = torch.hann_window(w, periodic=False, dtype=DTYPE)
            hz = np.fft.rfftfreq(w, 1/sr)
            edges = np.geomspace(30., min(sr*.49, 16000.), 37)
            pool = np.zeros((len(hz), len(edges)-1))
            for j, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
                pool[(hz>=a)&(hz<b),j] = 1
            pool = pool[:,pool.sum(axis=0)>0]
            pool /= pool.sum(axis=0, keepdims=True)
            pool = torch.tensor(pool, dtype=DTYPE)
            fine = torch.tensor((hz>=30)&(hz<350))
            self.specs.append((w,win,pool,fine))
        with torch.no_grad():
            self.target = self.features(torch.tensor(target, dtype=DTYPE))
        self.calls = 0
        self.start = time.monotonic()

    def voice(self, z):
        p = {k: torch.exp(z[j]) if BOUNDS[k][2] == 'log' else z[j] for j,k in enumerate(ORDER)}
        t = self.t
        fast = p['fastTime'] * -torch.expm1(-t/p['fastTime'])
        slow = p['slowTime'] * -torch.expm1(-t/p['slowTime'])
        attack = -torch.expm1(-t/p['attackTime'])
        mix = torch.zeros_like(t)
        for i in range(1, len(LOW)+len(HIGH)+1):
            phase = p[f'f{i}']*(t+(p[f'g{i}']*fast+p[f's{i}']*slow if i<=len(LOW) else 0))+p[f'p{i}']
            rise = -torch.expm1(-t/p[f'r{i}']) if i<=len(LOW) else attack
            mix = mix+p[f'a{i}']*torch.exp(-p[f'd{i}']*t)*torch.sin(2*torch.pi*phase)*rise
        pressure = 1+p['pressure']*4*torch.clamp(-mix,min=0)
        for i,n in enumerate(self.noise,1):
            mix = mix+n*(p[f'n{i}']*torch.exp(-p[f'nd{i}']*t)*pressure+p[f'nt{i}']*torch.exp(-p[f'ntd{i}']*t)) * -torch.expm1(-t/p[f'nr{i}'])
        u = torch.clamp((t-p['fadeStart'])/(p['fadeEnd']-p['fadeStart']),0,1)
        return mix*(1-u*u*(3-2*u))

    def features(self,y):
        features = []
        for w,win,pool,fine in self.specs:
            x = torch.nn.functional.pad(y,(w//2,w//2)).unfold(0,w,w//4)*win
            mag = torch.sqrt(torch.abs(torch.fft.rfft(x))**2+1e-16)/(win.sum()/2)
            pooled = torch.sqrt(mag.square()@pool+1e-14)
            # Within-band spectral contrast distinguishes a diffuse shell
            # response from a single sine with the same pooled power.
            contrast = torch.log(mag+.00001)@pool-torch.log(pooled+.00001)
            features.append((torch.log(mag[:,fine]+.0005), torch.log(pooled+.00001), mag[:,fine], contrast))
        return features

    def __call__(self,z):
        v = torch.tensor(z, dtype=DTYPE, requires_grad=True)
        feats = self.features(self.voice(v))
        loss = torch.zeros((), dtype=DTYPE)
        for (f,p,m,c),(tf,tp,tm,tc) in zip(feats,self.target):
            # Smooth L1 of log magnitude: no sample-wise phase matching.
            loss = loss+.35*torch.sqrt((f-tf).square()+.01**2).mean()
            loss = loss+.65*torch.sqrt((p-tp).square()+.01**2).mean()
            loss = loss+.5*torch.sqrt((c-tc).square()+.01**2).mean()
            loss = loss+.15*torch.abs(m-tm).sum()/tm.sum()
        loss = loss/len(feats)
        loss.backward()
        self.calls += 1
        if self.calls%50 == 0:
            print(f'eval {self.calls}: {float(loss.detach()):.6f} ({time.monotonic()-self.start:.1f}s)', flush=True)
        return float(loss.detach()), v.grad.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target',default='Assets/r8-kick03.wav')
    ap.add_argument('--out',required=True)
    ap.add_argument('--start')
    ap.add_argument('--iterations',type=int,default=500)
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    x,sr0=compare.read_wav(args.target)
    sr=48000
    target=signal.resample(x,round(len(x)*sr/sr0)).astype(np.float32)
    # Real silence after the ROM hit is part of the objective and audition.
    target=np.pad(target,(0,round(.24*sr)-len(target)))
    p=dict(SEED)
    if args.start:
        prior=json.loads(Path(args.start).read_text())
        p.update(prior.get('params',prior))
    if args.seed:
        rng=np.random.default_rng(args.seed)
        for i in range(1,len(LOW)+len(HIGH)+1):
            p[f'p{i}']=rng.uniform(-.5,.5)
            p[f'f{i}']*=math.exp(rng.normal(0,.02))
    bounds=np.array([(math.log(lo),math.log(hi)) if kind=='log' else (lo,hi) for k,(lo,hi,kind) in BOUNDS.items() if k!='outGain'])
    z=np.clip(encoded(p),bounds[:,0]+1e-8,bounds[:,1]-1e-8)
    baseline=decoded(bounds.mean(axis=1))
    obj=Fit(target,sr)
    # Confirm the independent reference before optimizing.
    delta=np.max(np.abs(obj.voice(torch.tensor(z,dtype=DTYPE)).detach().numpy()-render(decoded(z),len(target),sr)))
    assert delta<1e-6,delta
    print(f'Reference parity {delta:.3g}; {len(z)} bounded scalars',flush=True)
    result=optimize.minimize(obj,z,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':args.iterations,'ftol':1e-11,'gtol':1e-6,'maxcor':30,'maxls':30})
    p=decoded(result.x)
    p['lengthMs']=p['fadeEnd']*1000.
    learned=render(p,len(target),sr)
    initial=render(baseline,len(target),sr)
    hp=lambda y:compare.capture_highpass(y,sr,30.)
    gate=compare.mrstft(hp(target),hp(learned))
    gate0=compare.mrstft(hp(target),hp(initial))
    pinned=[k for k,v,(lo,hi) in zip(ORDER,result.x,bounds) if min(v-lo,hi-v)<1e-5]
    report={'params':p,'sampleRate':sr,'frames':len(target),'target':args.target,'loss':float(result.fun),'gate':{'learned':gate,'baseline':gate0,'improvement':1-gate/gate0},'pinned':pinned,'bounds':BOUNDS,'optimizer':{'iterations':result.nit,'evaluations':result.nfev,'message':str(result.message)},'referenceParity':float(delta)}
    (out/'recovered_params.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'fit_source.py').write_text(Path(__file__).read_text())
    for name,y in [('target',target),('learned',learned),('initial',initial)]:
        # Float WAV, deliberately no clipping or per-file normalization.
        from scipy.io.wavfile import write
        write(str(out/f'{name}.wav'),sr,y.astype(np.float32))
    gap=np.zeros(int(.35*sr),dtype=np.float32)
    ab = np.concatenate([target,gap,learned,gap]*3)
    write(str(out/'ab.wav'),sr,(ab*min(1.,.95/np.max(np.abs(ab)))).astype(np.float32))
    compare.write_comparison_png(str(out/'compare.png'),target,initial,learned,sr)
    print(json.dumps({k:v for k,v in report.items() if k not in ('params','bounds')},indent=2),flush=True)


if __name__=='__main__':
    main()
