; Representative wavetable instrument with a two-wave bank and bilinear lookup.
(param freq @default 110 @min 20 @max 20000 @unit Hz)
(param wave @default 0.35 @min 0 @max 1)
(param gain @default 0.2 @min 0 @max 1)

(def waves (wavetable @shape [16 2] @file "waves/tiny-bank.json"))
(def phase (phasor freq))
(def sample-index (* (wrap phase 0 1) 16))
(def sample (peek waves sample-index wave))
(out (* gain sample) 1 @name audio)
