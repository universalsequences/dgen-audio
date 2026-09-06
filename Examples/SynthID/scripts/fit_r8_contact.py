#!/usr/bin/env python3
"""R8 contact revision: membrane + finite pressure pulse + diffuse shell.

No stationary shell sine bank. Broad, low-Q noise bands have independent
struck/decay envelopes; a finite half-sine force through HP/LP filters provides
a solid impact rather than a sustained resonant substitute. The v8 membrane
is retained as a measured starting point. All fitted values are bounded scalars.
The inherited objective is phase-blind; no waveform/residual arrays are fitted.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, signal
from scipy.io.wavfile import write
import torch

import compare
import render_reference as reference
from fit_r8_acoustic import Fit as SpectralFit, BOUNDS as OLD_BOUNDS
from fit_r8_kick import biquad_coeffs

DTYPE = torch.float64
BODY = (
    (42.3541,.604850,16.0268,.0113388,1.8,.901470,.0226853),
    (83.6367,.693599,29.7337,.060479,1.8,.710404,.000938021),
    (95.7017,.067445,11.9265,.379009,.853351,1.11357,.000509956),
    (117.642,.198733,11.5722,-.341254,1.04069,.758852,.000363006),
    (164.348,.049701,10.9308,-.493401,.8172,.633368,.000481222),
    (200.308,.0915199,24.5897,.813543,.821733,.0509336,.000348785),
    (232.699,.0256056,11.2013,-1.48097,.131917,.899677,.000510371),
)
BANDS = ((350.,.65,2),(700.,.75,2),(1400.,.85,2),
         (2600.,1.,2),(4200.,.8,2),(6500.,.707,1))
SEED = {'fastTime':.0026288,'slowTime':.0391563,
        'fadeStart':.17,'fadeEnd':.180514,'outGain':1.}
BOUNDS = {k:OLD_BOUNDS[k] for k in ('fastTime','slowTime')}
for i,row in enumerate(BODY,1):
    for name,value in zip(('f','a','d','p','g','s','r'),row):
        key=f'{name}{i}'
        SEED[key]=value
        BOUNDS[key]=OLD_BOUNDS[key]

CONTACT_BOUNDS = {'forceAmp':(.00001,2.,'log'), 'forceWidth':(.0003,.006,'log')}
SEED.update(forceAmp=.4,forceWidth=.0012)
for i in range(1,7):
    # The 1.4/2.6 kHz struck bands need high pre-envelope amplitude for their
    # measured delayed rise. Widen these two, not every layer or output drive.
    CONTACT_BOUNDS.update({f'n{i}':(.000001,6. if i in (3,4) else 3.,'log'),f'nd{i}':(35.,2500.,'log'),
                          f'nr{i}':(.00004,.008,'log'),f'nt{i}':(.000001,.5,'log'),
                          f'ntd{i}':(5.,350.,'log')})
    SEED.update({f'n{i}':(.2,1.,.8,1.,.06,.05)[i-1],f'nd{i}':180.,
                 f'nr{i}':.0005,f'nt{i}':(.035,.08,.06,.025,.01,.001)[i-1],
                 f'ntd{i}':(30.,35.,40.,60.,80.,15.)[i-1]})
BOUNDS.update(CONTACT_BOUNDS)


def filtered(x, fc, q, mode, sr):
    kind = {0:'lp',1:'hp',2:'bp'}[mode]
    b0,b1,b2,a1,a2=biquad_coeffs(kind,min(fc,sr*.45),q,sr)
    return signal.lfilter([b0,b1,b2],[1,a1,a2],x)


def noise_bands(frames,sr):
    # Constant spectral density over host rates; finite exciter bandwidth.
    n=(reference.dgen_noise(frames).astype(np.float64)*2-1)*math.sqrt(sr/48000.)
    n=filtered(n,14000.,.707,0,sr)
    return np.array([filtered(n,f,q,m,sr) for f,q,m in BANDS])


def force_filter(x,sr):
    return filtered(filtered(x,180.,.707,1,sr),3500.,.707,0,sr)


def membrane(p,t):
    fast=p['fastTime']*-np.expm1(-t/p['fastTime'])
    slow=p['slowTime']*-np.expm1(-t/p['slowTime'])
    y=np.zeros(len(t))
    for i in range(1,8):
        ph=p[f'f{i}']*(t+p[f'g{i}']*fast+p[f's{i}']*slow)+p[f'p{i}']
        y+=p[f'a{i}']*np.exp(-p[f'd{i}']*t)*-np.expm1(-t/p[f'r{i}'])*np.sin(2*np.pi*(ph-np.floor(ph)))
    return y


def render(p,frames,sr):
    t=np.arange(frames)/sr
    phase=np.clip(t/p['forceWidth'],0,1)
    force=np.sin(np.pi*phase)*(t<p['forceWidth'])*math.sqrt(.0012/p['forceWidth'])
    y=membrane(p,t)+p['forceAmp']*force_filter(force,sr)
    for i,n in enumerate(noise_bands(frames,sr),1):
        y+=n*-np.expm1(-t/p[f'nr{i}'])*(p[f'n{i}']*np.exp(-p[f'nd{i}']*t)+p[f'nt{i}']*np.exp(-p[f'ntd{i}']*t))
    u=np.clip((t-p['fadeStart'])/(p['fadeEnd']-p['fadeStart']),0,1)
    return (y*(1-u*u*(3-2*u))*p['outGain']).astype(np.float32)


class Fit(SpectralFit):
    def __init__(self,target,sr,seed,order):
        super().__init__(target,sr)
        self.seed,self.order=seed,order
        self.noise=torch.tensor(noise_bands(len(target),sr),dtype=DTYPE)
        # Generated IIR response, NOT target data or fitted FIR coefficients.
        # Zero-padded convolution differentiates the force pulse through the
        # exact fixed biquads. The shipping implementation uses those biquads.
        impulse=np.zeros(len(target));impulse[0]=1
        self.fft_size=2*len(target)
        self.force_response=torch.fft.rfft(torch.tensor(force_filter(impulse,sr),dtype=DTYPE),n=self.fft_size)

    def voice(self,z):
        p=dict(self.seed)
        for k,v in zip(self.order,z):
            p[k]=torch.exp(v) if BOUNDS[k][2]=='log' else v
        t=self.t
        fast=p['fastTime']*-torch.expm1(-t/p['fastTime'])
        slow=p['slowTime']*-torch.expm1(-t/p['slowTime'])
        y=torch.zeros_like(t)
        for i in range(1,8):
            ph=p[f'f{i}']*(t+p[f'g{i}']*fast+p[f's{i}']*slow)+p[f'p{i}']
            y=y+p[f'a{i}']*torch.exp(-p[f'd{i}']*t)*-torch.expm1(-t/p[f'r{i}'])*torch.sin(2*torch.pi*ph)
        phase=torch.clamp(t/p['forceWidth'],0,1)
        pulse=torch.sin(torch.pi*phase)*(t<p['forceWidth'])*torch.sqrt(.0012/p['forceWidth'])
        force=torch.fft.irfft(torch.fft.rfft(pulse,n=self.fft_size)*self.force_response,n=self.fft_size)[:len(t)]
        y=y+p['forceAmp']*force
        for i,n in enumerate(self.noise,1):
            y=y+n*-torch.expm1(-t/p[f'nr{i}'])*(p[f'n{i}']*torch.exp(-p[f'nd{i}']*t)+p[f'nt{i}']*torch.exp(-p[f'ntd{i}']*t))
        u=torch.clamp((t-p['fadeStart'])/(p['fadeEnd']-p['fadeStart']),0,1)
        return y*(1-u*u*(3-2*u))


def transform(k,v):
    return math.log(v) if BOUNDS[k][2]=='log' else v


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start',type=Path)
    ap.add_argument('--target',default='Assets/r8-kick03.wav')
    ap.add_argument('--iterations',type=int,default=1500)
    ap.add_argument('--joint',action='store_true',help='also refine membrane scalars')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    x,sr0=compare.read_wav(args.target);sr=48000
    target=signal.resample(x,round(len(x)*sr/sr0)).astype(np.float32)
    target=np.pad(target,(0,round(.24*sr)-len(target)))
    seed=dict(SEED)
    if args.start:
        prior=json.loads(args.start.read_text());seed.update(prior.get('params',prior))
    # Validate/clamp frozen scalars too, not just the subset being optimized.
    for key,(lo,hi,_) in BOUNDS.items():
        value=float(seed[key])
        if not math.isfinite(value):
            ap.error(f'non-finite seed parameter {key}')
        seed[key]=min(hi,max(lo,value))
    if not 0 < seed['fadeStart'] < seed['fadeEnd'] < .24:
        ap.error('seed fade must end within the 240 ms comparison window')
    seed['outGain']=1.  # fixed raw calibration; headroom is applied at port time
    order=list(BOUNDS if args.joint else CONTACT_BOUNDS)
    bounds=np.array([(transform(k,lo),transform(k,hi)) for k in order for lo,hi,_ in [BOUNDS[k]]])
    z=np.clip([transform(k,seed[k]) for k in order],bounds[:,0]+1e-8,bounds[:,1]-1e-8)
    obj=Fit(target,sr,seed,order)
    def decode(v):
        return dict(seed,**{k:math.exp(float(a)) if BOUNDS[k][2]=='log' else float(a) for k,a in zip(order,v)})
    parity=np.max(np.abs(obj.voice(torch.tensor(z,dtype=DTYPE)).detach().numpy()-render(decode(z),len(target),sr)))
    assert parity<1e-6,parity
    print(f'Reference parity {parity:.3g}; optimizing {len(z)} scalars',flush=True)
    result=optimize.minimize(obj,z,jac=True,method='L-BFGS-B',bounds=bounds,
                             options={'maxiter':args.iterations,'maxcor':30,'ftol':1e-11,'gtol':1e-6,'maxls':30})
    p=decode(result.x);p['lengthMs']=p['fadeEnd']*1000
    learned=render(p,len(target),sr)
    baseline=dict(SEED)
    for k,(lo,hi,mode) in BOUNDS.items():
        baseline[k]=math.sqrt(lo*hi) if mode=='log' else (lo+hi)/2
    initial=render(baseline,len(target),sr)
    hp=lambda a:compare.capture_highpass(a,sr,30.)
    gate0=compare.mrstft(hp(target),hp(initial));gate=compare.mrstft(hp(target),hp(learned))
    report={'params':p,'sampleRate':sr,'frames':len(target),'target':args.target,'bounds':BOUNDS,
            'train':float(result.fun),'gate':{'learned':gate,'baseline':gate0,'improvement':1-gate/gate0},
            'optimized':order,'iterations':result.nit,'referenceParity':float(parity),
            'pinned':[k for k,v,(lo,hi) in zip(order,result.x,bounds) if min(v-lo,hi-v)<1e-5]}
    (args.out/'recovered_params.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'fit_source.py').write_text(Path(__file__).read_text())
    for name,y in [('target',target),('learned',learned),('initial',initial)]:
        write(str(args.out/f'{name}.wav'),sr,y.astype(np.float32))
    gap=np.zeros(sr//3,dtype=np.float32)
    ab=np.concatenate([target,gap,learned,gap]*3)
    write(str(args.out/'ab.wav'),sr,(ab*min(1.,.95/np.max(np.abs(ab)))).astype(np.float32))
    compare.write_comparison_png(str(args.out/'compare.png'),target,initial,learned,sr)
    print(json.dumps({k:v for k,v in report.items() if k not in ('params','bounds','optimized')},indent=2),flush=True)


if __name__=='__main__':
    main()
