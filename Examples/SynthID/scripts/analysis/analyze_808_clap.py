#!/usr/bin/env python3
"""Phase-1 measurements for Assets/808-clap-r8.wav (Roland R-8 MkII '808Clap')."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compare
y, sr = compare.read_wav(sys.argv[1] if len(sys.argv) > 1 else "Assets/808-clap-r8.wav")
y = y.astype(np.float64); pk = np.abs(y).max()
print(f"sr={sr} frames={len(y)} len={1000*len(y)/sr:.1f}ms peak={20*np.log10(pk):.1f}dBFS dc={y.mean():.2e}")
# fine envelope: 0.25 ms hop, 0.5 ms rms
h = max(1, int(0.00025*sr)); w = max(2, int(0.0005*sr))
env = np.array([np.sqrt(np.mean(y[i:i+w]**2)) for i in range(0, len(y)-w, h)])
t = np.arange(len(env))*h/sr*1000
db = 20*np.log10(env/env.max()+1e-9)
# burst onsets = local maxima in first 40 ms above -12 dB with 3ms refractory
peaks=[]; last=-10
for i in range(1,len(db)-1):
    if t[i] > 45: break
    if db[i]>db[i-1] and db[i]>=db[i+1] and db[i]>-14 and t[i]-last>3:
        peaks.append((round(t[i],2), round(db[i],1))); last=t[i]
print("burst peaks (ms, dB):", peaks)
if len(peaks)>1: print("spacings ms:", [round(peaks[i+1][0]-peaks[i][0],2) for i in range(len(peaks)-1)])
# per-burst decay: dB drop between peak and the trough before next burst
for i,(pt,pd) in enumerate(peaks[:-1]):
    nt=peaks[i+1][0]; m=(t>=pt)&(t<nt); j=np.argmin(db[m]); print(f"  burst@{pt}ms: trough {db[m][j]:.1f} dB at +{t[m][j]-pt:.2f} ms")
# tail decay fit (log rms vs t) over windows
def fit(a,b):
    m=(t>=a)&(t<b); p=np.polyfit(t[m]/1000, np.log(env[m]+1e-9), 1); return p[0]
for a,b in [(35,80),(80,160),(160,300),(35,300)]:
    k=fit(a,b); print(f"tail decay {a}-{b}ms: {k:.1f}/s  (T60 {6.9/-k*1000:.0f} ms)")
# spectra: bursts (0-32ms) vs tail (40-200ms), 1/3-octave-ish bands
def bands(seg, label):
    Y=np.abs(np.fft.rfft(seg*np.hanning(len(seg))))**2; f=np.fft.rfftfreq(len(seg),1/sr)
    edges=[200,400,600,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,13000]
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(f>=lo)&(f<hi); out.append(10*np.log10(Y[m].sum()/(hi-lo)+1e-15))
    out=np.array(out); out-=out.max()
    print(label, " ".join(f"{lo}:{v:.0f}" for lo,v in zip(edges[:-1],out)))
    m=np.argsort(Y)[::-1][:5]; print("   top bins Hz:", sorted(np.round(f[m]).astype(int).tolist()))
bands(y[:int(0.032*sr)], "burst   dB/Hz rel:")
bands(y[int(0.04*sr):int(0.2*sr)], "tail    dB/Hz rel:")
bands(y[int(0.2*sr):], "late    dB/Hz rel:")
# noise floor after -60dB
thr=pk*1e-3; last=np.where(np.abs(y)>thr)[0][-1]; print(f"-60dB end {1000*last/sr:.0f} ms; rms of final 30ms {20*np.log10(np.sqrt(np.mean(y[-int(0.03*sr):]**2))/pk):.1f} dB")
