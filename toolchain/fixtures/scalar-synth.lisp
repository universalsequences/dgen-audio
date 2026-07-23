; Representative scalar instrument: stateful oscillator, nonlinear waveshaping,
; and parameter-dependent math. The phasor feedback keeps the oscillator path
; scalar in today's C lowering.
(param freq @default 220 @min 20 @max 20000 @unit Hz)
(param drive @default 1.75 @min 0.1 @max 8)
(param gain @default 0.2 @min 0 @max 1)

(def phase (phasor freq))
(def fundamental (sin (* phase twopi)))
(def harmonic (cos (* phase (* twopi 2))))
(def shaped (tanh (* drive (+ fundamental (* harmonic 0.125)))))
(out (* gain shaped) 1 @name audio)
