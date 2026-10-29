"""Mid-hit control: (a) END knob b1 stepped at 100 ms with morph 30 ms (z(t) = END after 30 ms);
(b) mod1 -> b2 (slot 1, depth 1) ramped 0 -> 1 over 80..200 ms. Reports the output difference before /
after the change vs the unchanged render and the boundary 2nd-difference excess around the change."""
from lib import *
m = load("Latent Kick Morph")
A = CODE["K55"]; B = CODE["K62"]
base = {**zp(A), **zp(B, "b"), "morph": 30.0}
def rel_win(a, b, t0, t1):
    i, j = int(t0 * SR), int(t1 * SR); return float(np.sqrt(np.sum((a[i:j] - b[i:j]) ** 2) / max(np.sum(b[i:j] ** 2), 1e-30)))
y0 = render(m, 0.6, params=base)[:, 0]
ya, _ = m.render(0.6, pitch=C4, vel=1.0, params=base, ramps={"b1": [(0.0, B[0]), (0.0999, B[0]), (0.1, B[0] + 1.5), (0.6, B[0] + 1.5)]})
ya = np.asarray(ya, float)[:, 0]
print("(a) b1 +1.5 at 100 ms: diff before %.2e, after %.3f" % (rel_win(ya, y0, 0.0, 0.098), rel_win(ya, y0, 0.11, 0.4)))
pm = {**base, "__mod__b2__active": 1.0, "__mod__b2__depth__slot1": 1.0}
y1 = render(m, 0.6, params=pm)[:, 0]
yb, _ = m.render(0.6, pitch=C4, vel=1.0, params=pm, ramps={"mod1": [(0.0, 0.0), (0.08, 0.0), (0.2, 1.0), (0.6, 1.0)]})
yb = np.asarray(yb, float)[:, 0]
print("(b) mod1 -> b2 ramp 80..200 ms: diff before %.2e, after %.3f" % (rel_win(yb, y1, 0.0, 0.078), rel_win(yb, y1, 0.12, 0.4)))
for lab, y in (("a", ya), ("b", yb)):
    d = np.abs(np.diff(y, 2)); i = int(0.1 * SR)
    print(f"  ({lab}) max |d2| 95..130 ms {d[i - 240:i + 1440].max():.2e} vs 20..95 ms {d[960:i - 240].max():.2e}")
