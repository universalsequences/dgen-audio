#!/usr/bin/env python3
"""R8 diagnostics: shared-scale waveform/spectra and windowed band deficits."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compare

ap=argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--old', default='output/r8kick_v6/learned.wav', help='explicit prior-render WAV for this comparison')
a=ap.parse_args()
r=Path(a.run)
x,sr=compare.read_wav(r/'target.wav')
y,_=compare.read_wav(r/'learned.wav')
old,osr=compare.read_wav(a.old)
if osr != sr:
    raise ValueError(f'old render is {osr} Hz, target is {sr} Hz; supply a matching render')
old=np.pad(old,(0,max(0,len(x)-len(old))))[:len(x)]
print(f'Old reference: {a.old}')
bands=[(30,100),(100,200),(200,400),(400,800),(800,1500),(1500,3000),(3000,6000),(6000,12000)]
windows=[(0,3),(3,8),(8,15),(15,30),(30,60),(60,100),(100,150),(150,182),(182,240)]
filtered=[]
for lo,hi in bands:
    sos=signal.butter(3,[lo,hi],btype='band',fs=sr,output='sos')
    filtered.append([signal.sosfilt(sos,z) for z in (x,old,y)])
print('Band RMS target / (old-target) / (new-target), dB. Causal filters; final window includes their ringdown.')
print('ms              '+' '.join(f'{lo}-{hi}'.rjust(20) for lo,hi in bands))
for l,h in windows:
    vals=[]
    for signals in filtered:
        db=[20*np.log10(np.sqrt(np.mean(z[round(l*sr/1000):round(h*sr/1000)]**2))+1e-12) for z in signals]
        vals.append(f'{db[0]:5.1f}/{db[1]-db[0]:+5.1f}/{db[2]-db[0]:+5.1f}')
    print(f'{l:3}-{h:3}         '+' '.join(vals))
print('Peak/RMS:',[(float(np.abs(z).max()),float(np.sqrt(np.mean(z*z)))) for z in (x,old,y)])
fig,ax=plt.subplots(4,2,figsize=(16,11))
t=np.arange(len(x))/sr*1000
for z,label in ((x,'target'),(old,'old'),(y,'new')):
    ax[0,0].plot(t,z,label=label,alpha=.7,lw=.7)
    ax[0,1].plot(t,z,label=label,alpha=.7,lw=.7)
ax[0,0].set_xlim(0,190);ax[0,1].set_xlim(0,15)
ax[0,0].legend();ax[0,1].legend()
for row,(z,label) in enumerate(((x,'target'),(old,'old'),(y,'new')),1):
    for col,(w,ylim) in enumerate(((4096,(25,600)),(512,(300,12000)))):
        f,tt,S=signal.stft(z,sr,nperseg=w,noverlap=w-w//8)
        ax[row,col].pcolormesh(tt*1000,f,20*np.log10(np.abs(S)+1e-8),vmin=-75,vmax=-8,cmap='magma',shading='auto')
        ax[row,col].set(xlim=(0,190),ylim=ylim,title=label,yscale='log')
fig.tight_layout();fig.savefig(r/'diagnostics.png',dpi=130)
