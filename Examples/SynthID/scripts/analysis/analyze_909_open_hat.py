#!/usr/bin/env python3
"""Phase-1 measurements for Assets/909-open-hat.wav (TR-909 HHOD0, open hat, decay 0)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compare
y, sr = compare.read_wav(sys.argv[1] if len(sys.argv) > 1 else "Assets/909-open-hat.wav")
y = y.astype(np.float64); pk = np.abs(y).max()
print(f"sr={sr} frames={len(y)} len={1000*len(y)/sr:.1f}ms peak={20*np.log10(pk):.1f}dBFS dc={y.mean():.2e}")
print("first 24 samples:", np.round(y[:24]/pk, 3).tolist())
# distinct sample values (6-bit source?)
vals = np.unique(np.round(y*32768)); print(f"distinct 16-bit values: {len(vals)}")
h = max(1, int(0.00025*sr)); w = max(2, int(0.001*sr))
env = np.array([np.sqrt(np.mean(y[i:i+w]**2)) for i in range(0, len(y)-w, h)])
t = np.arange(len(env))*h/sr*1000
db = 20*np.log10(env/env.max()+1e-9)
print("envelope dB @ms:", " ".join(f"{int(a)}:{db[np.argmin(abs(t-a))]:.0f}" for a in [0,1,2,4,8,12,16,24,32,48,64,96,128,160,192,224,248]))
print("peak env at ms:", round(t[np.argmax(db)],2))
def fit(a,b):
    m=(t>=a)&(t<b); p=np.polyfit(t[m]/1000, np.log(env[m]+1e-9), 1); return p[0]
for a,b in [(5,30),(30,80),(80,160),(160,240),(30,240)]:
    k=fit(a,b); print(f"decay {a}-{b}ms: {k:.1f}/s  (T60 {6.9/-k*1000:.0f} ms)")
# band spectra including HF
edges=[200,400,600,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000,22050]
def bands(seg, label):
    Y=np.abs(np.fft.rfft(seg*np.hanning(len(seg))))**2; f=np.fft.rfftfreq(len(seg),1/sr)
    out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(f>=lo)&(f<hi); out.append(10*np.log10(Y[m].sum()/(hi-lo)+1e-15))
    out=np.array(out); out-=out.max()
    print(label, " ".join(f"{lo}:{v:.0f}" for lo,v in zip(edges[:-1],out)))
    m=np.argsort(Y)[::-1][:8]; print("   top bins Hz:", sorted(np.round(f[m]).astype(int).tolist()))
bands(y[:int(0.02*sr)], "attack 0-20   dB/Hz:")
bands(y[int(0.02*sr):int(0.08*sr)], "early 20-80  dB/Hz:")
bands(y[int(0.08*sr):int(0.16*sr)], "mid 80-160   dB/Hz:")
bands(y[int(0.16*sr):], "late 160+    dB/Hz:")
# per-band decay: is the spectrum tilting over time?
def banddecay(lo,hi):
    b0,b1,b2,a1,a2 = None,None,None,None,None
    Y=np.fft.rfft(y); f=np.fft.rfftfreq(len(y),1/sr); m=(f>=lo)&(f<hi); Z=Y*m
    z=np.fft.irfft(Z,len(y)); e=np.array([np.sqrt(np.mean(z[i:i+w]**2)) for i in range(0,len(z)-w,h)])
    def k(a,b):
        mm=(t>=a)&(t<b); return np.polyfit(t[mm]/1000,np.log(e[mm]+1e-9),1)[0]
    return k(20,120), k(120,240)
print("band decay /s (20-120ms, 120-240ms):")
for lo,hi in [(300,1000),(1000,2500),(2500,5000),(5000,8000),(8000,12000),(12000,16000),(16000,22000)]:
    a,b=banddecay(lo,hi); print(f"  {lo}-{hi}: {a:.0f} {b:.0f}")
# persistent narrow peaks in the tail (modal ringing?)
seg=y[int(0.05*sr):int(0.25*sr)]; N=len(seg)
Y=np.abs(np.fft.rfft(seg*np.hanning(N)))**2; f=np.fft.rfftfreq(N,1/sr)
S=10*np.log10(Y+1e-15); Sm=np.convolve(S,np.ones(41)/41,'same'); pr=S-Sm
idx=[i for i in range(2,len(S)-2) if pr[i]>9 and S[i]>=S[i-2:i+3].max() and f[i]>500]
print("tail narrow peaks >9dB over local mean (Hz, prominence):", [(int(f[i]),round(pr[i],1)) for i in sorted(idx,key=lambda i:-pr[i])[:16]])
thr=pk*1e-3; last=np.where(np.abs(y)>thr)[0][-1]; print(f"-60dB end {1000*last/sr:.0f} ms; rms of final 10ms {20*np.log10(np.sqrt(np.mean(y[-int(0.01*sr):]**2))/pk):.1f} dB")
