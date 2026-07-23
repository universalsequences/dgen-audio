; Representative FFT/spectral effect. Partitioned convolution exercises the
; complete current vDSP surface: setup, forward/inverse FFT, and complex MAC.
; Phase 1 inventories this Accelerate-backed output; hermetic non-SDK linking
; begins with the other fixtures as required by the accepted spec.
(def dry (in 1 @name audio-in))
(def impulse (tensor @shape [32] @data [
  1 0 0 0 0 0 0 0
  0.35 0 0 0 0 0 0 0
  0.15 0 0 0 0 0 0 0
  0.05 0 0 0 0 0 0 0]))
(out (partitioned-convolve dry impulse @N 16 @hop 8 @gain 0.5) 1 @name audio-out)
