"""Per-kick features + morph quality over candidate pairs (compiled morph build, START = END for singles)."""
import json, itertools
from lib import *
m = None
def env_db(y, hop=480):
    n = len(y) // hop; return 20 * np.log10(np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-7)
def feats(y):
    e = env_db(y); pk = e.max()
    t20 = np.argmax(e < pk - 20) * 0.01 if (e < pk - 20).any() else 1.0
    S = np.abs(np.fft.rfft(y[480:4800] * np.hanning(4320), 32768)); f = np.fft.rfftfreq(32768, 1 / SR)
    lo = (f > 20) & (f < 200); sub = f[lo][np.argmax(S[lo])]
    S2 = np.abs(np.fft.rfft(y[:2400] * np.hanning(2400))) ; f2 = np.fft.rfftfreq(2400, 1 / SR)
    cen = float((S2 * f2).sum() / S2.sum())
    return dict(t20=float(t20), sub=float(sub), cen=cen, peak=float(np.abs(y).max()), env=e)
if __name__ == "__main__":
  m = load("Latent Kick Morph")
  F = {}
  for n in NAMES:
      y = render(m, 1.0, params={**zp(CODE[n]), **zp(CODE[n], "b")})[:, 0]
      F[n] = feats(y)
  for n in NAMES:
      f = F[n]; print(f"{n} t20 {f['t20']:.2f}s sub {f['sub']:5.1f} Hz click-centroid {f['cen']:6.0f} pk {f['peak']:.2f}")
  json.dump({n: {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f.items()} for n, f in F.items()}, open("survey_singles.json", "w"))
