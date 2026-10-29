"""Feature tracks of the preset demo renders: START | END | morph (preset time).
low = strongest peak 20-200 Hz (4096 window), cen = spectral centroid (2048 window), dB = RMS."""
import json, os, sys
from lib import *
MAD.use("kick")
from kick_common import read_wav_f32
TS = (0.03, 0.08, 0.15, 0.25, 0.35)
def low(x, t, n=4096):
    i = int(t * SR); seg = np.pad(x, (n, n))[i + n - n // 2:i + n + n // 2] * np.hanning(n)
    S = np.abs(np.fft.rfft(seg, 8 * n)); f = np.fft.rfftfreq(8 * n, 1 / SR); lo = (f > 20) & (f < 200)
    return f[lo][np.argmax(S[lo])]
def cen(x, t, n=2048):
    i = int(t * SR); seg = x[i:i + n] * np.hanning(n); S = np.abs(np.fft.rfft(seg)); f = np.fft.rfftfreq(n, 1 / SR)
    return (S * f).sum() / (S.sum() + 1e-12)
def db(x, t): i = int(t * SR); return 20 * np.log10(np.sqrt(np.mean(x[i:i + 960] ** 2)) + 1e-9)
summ = json.load(open("renders/summary.json"))
seg = int(1.0 * SR); g = int(0.25 * SR)
out = {}
for name, s in summ.items():
    y = read_wav_f32(f"renders/{name.lower().replace(' ', '-')}.wav").astype(float)
    parts = [y[k * (seg + g):k * (seg + g) + seg] for k in range(3)]
    print(f"== {name}  {s['A']} -> {s['B']}  morph {s['morph']} ms")
    for lab, x in zip(("START", "END", "morph"), parts):
        print(f"  {lab:5s} low " + " ".join(f"{low(x, t):5.0f}" for t in TS) + " | cen " + " ".join(f"{cen(x, t):5.0f}" for t in TS)
              + " | dB " + " ".join(f"{db(x, t):6.1f}" for t in TS))
