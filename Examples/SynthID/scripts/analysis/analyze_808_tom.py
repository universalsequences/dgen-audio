import numpy as np, wave, sys
def load(p):
    w=wave.open(p); sr=w.getframerate(); n=w.getnframes(); ch=w.getnchannels(); sw=w.getsampwidth()
    raw=w.readframes(n)
    if sw==2: x=np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
    elif sw==3:
        b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(np.int32)
        v=(b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)); v=np.where(v&0x800000, v-(1<<24), v); x=v.astype(np.float64)/(1<<23)
    elif sw==4: x=np.frombuffer(raw,dtype='<i4').astype(np.float64)/2**31
    x=x.reshape(-1,ch); return sr,x
sr,d=load(sys.argv[1]); print('sr',sr,'shape',d.shape)
if d.shape[1]==2: print('L-R max diff',np.abs(d[:,0]-d[:,1]).max())
x=d.mean(1)
pk=np.abs(x).max(); print('len s %.3f peak dBFS %.2f dc %.5f'%(len(x)/sr,20*np.log10(pk),x.mean()))
idx=np.where(np.abs(x)>pk*1e-3)[0]; on=idx[0]; print('onset ms %.1f, -60dB end ms %.1f'%(on/sr*1e3, idx[-1]/sr*1e3))
idx40=np.where(np.abs(x)>pk*1e-2)[0]; print('-40dB end ms %.1f'%(idx40[-1]/sr*1e3))
y=x[on:]
w=int(0.01*sr); print('RMS envelope (10ms windows):')
ts=[];db=[]
for t0 in np.arange(0,min(len(y)/sr,1.6),0.025):
    i=int(t0*sr); seg=y[i:i+w]
    if len(seg)<w: break
    r=np.sqrt((seg**2).mean()); ts.append(t0); db.append(20*np.log10(r+1e-12))
    print(f'{t0*1000:5.0f}ms {db[-1]:6.1f}dB',end=' | ')
print()
ts=np.array(ts);db=np.array(db)
def slope(a,b):
    m=(ts>=a)&(ts<=b); p=np.polyfit(ts[m],db[m]/8.686,1); return p[0]
for a,b in [(0.02,0.1),(0.1,0.3),(0.3,0.6),(0.6,1.0)]:
    try: print('amp decay 1/s over %.2f-%.2f: %.2f'%(a,b,slope(a,b)))
    except Exception as e: pass
zc=np.where((y[:-1]<0)&(y[1:]>=0))[0]; per=np.diff(zc)/sr
print('zero-crossing f(t):')
f=1/per; tz=zc[:-1]/sr
for t0 in np.arange(0,0.8,0.02):
    m=(tz>=t0)&(tz<t0+0.02)
    if m.any(): print(f'{t0*1000:4.0f}ms {np.median(f[m]):6.1f}Hz',end=' | ')
print()
# fit f(t)=fEnd+(fStart-fEnd)exp(k t) on 10..600ms via grid on k
m=(tz>0.002)&(tz<0.35)&(f<400); T=tz[m]; F=f[m]
best=None
for k in np.linspace(-120,-2,300):
    A=np.stack([np.ones_like(T),np.exp(k*T)],1); c,res,_,_=np.linalg.lstsq(A,F,rcond=None)
    r=np.sqrt(((A@c-F)**2).mean())
    if best is None or r<best[0]: best=(r,k,c)
print('pitch fit: fEnd=%.2f fStart=%.2f pitchDecay=%.1f rmsHz=%.2f'%(best[2][0],best[2][0]+best[2][1],best[1],best[0]))
# harmonics: STFT at few times
def harm(t0,win=4096):
    i=int(t0*sr); seg=y[i:i+win]*np.hanning(win); S=np.abs(np.fft.rfft(seg)); fr=np.fft.rfftfreq(win,1/sr)
    f0=fr[np.argmax(S[(fr>30)&(fr<300)])+np.searchsorted(fr,30)]
    out=[]
    for h in range(1,9):
        band=(fr>f0*h*0.9)&(fr<f0*h*1.1); out.append(20*np.log10(S[band].max()/S[(fr>30)&(fr<300)].max()+1e-12))
    return f0,out
for t0 in [0.01,0.05,0.1,0.2,0.3]:
    f0,o=harm(t0); print('t=%.2f f0=%.1f H1..H8 dB:'%(t0,f0),' '.join('%.0f'%v for v in o))
# attack / high band energy over time
def band_db(t0,lo,hi,win=1024):
    i=int(t0*sr); seg=y[i:i+win]*np.hanning(win); S=np.abs(np.fft.rfft(seg))**2; fr=np.fft.rfftfreq(win,1/sr)
    return 10*np.log10(S[(fr>lo)&(fr<hi)].sum()/S.sum()+1e-12), 10*np.log10(S[(fr>lo)&(fr<hi)].sum()+1e-12)
print('high band (1k-15k) share of energy, abs:')
for t0 in [0,0.005,0.01,0.02,0.04,0.08,0.15,0.3,0.5]:
    print(f'{t0*1000:4.0f}ms rel {band_db(t0,1000,15000)[0]:6.1f}dB abs {band_db(t0,1000,15000)[1]:6.1f}',end=' | ')
print()
# noise spectral centroid at 5-30ms in >800Hz
i=int(0.005*sr); seg=y[i:i+2048]*np.hanning(2048); S=np.abs(np.fft.rfft(seg))**2; fr=np.fft.rfftfreq(2048,1/sr)
m=fr>800; print('noise centroid >800Hz at 5ms: %.0f Hz'%((fr[m]*S[m]).sum()/S[m].sum()))
for lo,hi in [(800,2000),(2000,4000),(4000,8000),(8000,16000)]:
    print(f'  {lo}-{hi}: {10*np.log10(S[(fr>lo)&(fr<hi)].sum()):.1f} dB',end='')
print()
# tail noise floor
tail=x[int(idx[-1]*0.98):]; print('tail rms dB', 20*np.log10(np.sqrt((x[-int(0.05*sr):]**2).mean())+1e-12))
