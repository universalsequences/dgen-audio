#!/usr/bin/env python3
"""E5 snare-wire layer as a splice for scripts/e3proto/gen_dsp.py output. E7
(scripts/e7/build_instrument.py) imports wire_block / SVF_MACROS and applies steps 3-4 to the
exported instrument; the steps below describe the full in-generator version.

How to apply in gen_dsp.py:
  1. `from wire_splice import WIRE_SEGS, WIRE_FIXED, wire_block, SVF_MACROS` (or paste).
  2. SEGS = MODE_SEGS + NOISE_SEGS + GLOBAL_SEGS + WIRE_SEGS      (P grows by len(WIRE_SEGS))
  3. emit SVF_MACROS once at top level (before `(def hit ...)`).
  4. replace `(def dry (+ head noise-out))` by wire_block(NSTR) + `(def dry (+ head noise-out wire-out))`.
  5. random_weights: setv("wire_thr", 0.042); setv("wire_level", 2e-3) (~off); setv("wire_lo", 1300);
     setv("wire_hi", 7000); setv("wire_k", 1.3)  (+ drive 1.2 / hz 620 / t60 1.0 if decoded).
  6. reference_render.py: scripts/e5/wires.c `wires()` + `svf()` are the exact numpy/C twin (float32),
     knob mapping in WIRE_TWIN below. The layer is chaotic: compare twin vs dylib by band /
     MR-STFT metrics (bit-exact only for the first ~300-1100 samples).
Decoded knobs are made from the decoder's raw-<name> outputs exactly like the other GLOBAL
segments (sigmoid squash on the event clock); fixed knobs are constants.
    python3 wire_splice.py [NSTR]    # prints the dgenlisp block
"""
import sys
from gen_test import wire_snippet, SVF_MACROS  # noqa: F401  (single source of the snippet)

NSTR = 2  # strands: 1 / 2 / 6 cost +1.9 / +2.1 / +3.0 us per 128-frame block (19.6 base)

# (name, width, squash, lo, hi) -- gen_dsp.py segment format. Ranges cover the 156 VILLAIN
# renders with margin (table: thr 0.0087-0.148, rattle 0-12.9, buzz_lo 200-3707,
# buzz_hi 1789-15470, buzz_k 0.17-2) and the per-snare E5 fits.
WIRE_SEGS = [
    ("wire_thr",   1, "log", 0.004, 0.3),     # contact threshold in head units (= gap/drive)
    ("wire_level", 1, "log", 1e-3, 40.0),     # strand output gain (VILLAIN c_rattle); 1e-3 = off
    ("wire_lo",    1, "log", 150.0, 5000.0),  # wire band-pass SVF high-pass, Hz
    ("wire_hi",    1, "log", 1200.0, 16000.0),# wire band-pass SVF low-pass, Hz
    ("wire_k",     1, "log", 0.12, 2.0),      # LP damping (1.414 = Butterworth, small = resonant)
]
# Second tier: barely vary in VILLAIN (hz 620-801, t60 clamped to 1 ms, drive 0.6-5.8 mostly a
# saturation knob once thr is free). Keep fixed, or append to WIRE_SEGS to decode them.
WIRE_OPTIONAL_SEGS = [
    ("wire_drive", 1, "log", 0.4, 8.0),       # head -> wire coupling (sets ceiling/stiffness saturation)
    ("wire_hz",    1, "log", 300.0, 1500.0),  # wire mode-1 tuning (tension), mode 2 = 2.41x
    ("wire_t60",   1, "log", 0.4, 4.0),       # wire mode-1 T60 in ms (damping); mode 2 = 0.4x
]
WIRE_FIXED = {"wire_drive": 1.2, "wire_hz": 620.0, "wire_t60": 1.0}

def wire_block(nstr=NSTR, decoded=None):
    """dgenlisp for the template: event-clock knob defs + the wire layer. `decoded` = names in
    SEGS (default: WIRE_SEGS names); the rest of drive/hz/t60 come from WIRE_FIXED."""
    names = set(decoded or [s[0] for s in WIRE_SEGS])
    v = lambda n: f"raw-{n}" if n in names else repr(WIRE_FIXED[n])
    knobs = f""";; wire knobs on the hit's event clock (decoder outputs / fixed family constants)
(def ev-w-drive {v('wire_drive')})
(def ev-w-gap (* raw-wire_thr ev-w-drive))
(def ev-w-hz {v('wire_hz')})
(def ev-w-t60 {v('wire_t60')})
(def ev-w-level raw-wire_level)
(def ev-w-lo raw-wire_lo)
(def ev-w-hi raw-wire_hi)
(def ev-w-k raw-wire_k)
"""
    return knobs + wire_snippet(nstr, reset=False)

# twin mapping (e5lib.wires / wires.c): zs=drive, gap=thr*drive, wire_hz=hz, t60_ms=t60,
# level=60*level, then svf(hp, lo, 1.414) -> svf(lp, hi, k)
WIRE_TWIN = "zs=drive gap=thr*drive wire_hz=hz t60_ms=t60 level=60*level; buzz_bp(x, lo, hi, k)"

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NSTR
    print(SVF_MACROS); print(wire_block(n)); print("(def dry (+ head noise-out wire-out))")
