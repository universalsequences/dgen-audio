; Filter Table — a wavetable frame reinterpreted as a filter's magnitude
; response, applied by FFT convolution with overlap-add.
;
; This is the reference example of assembling a Kilohearts-Filter-Table-style
; effect from dgenlisp primitives. There is no monolithic `filter-table` op:
; everything below is `gather` / `iota` / `fft` / `ifft` / `buffer` /
; `overlap-add`. Every transform runs `@backend accelerated` (vDSP), the
; forward-only path a compiled C plugin uses. Nothing here is differentiable,
; and nothing needs to be.
;
; Backends
; --------
; `@backend accelerated` routes the transforms through the host's vDSP services
; and is C-only by construction — a Metal compile fails with an explicit error
; telling you to use the tensor FFT. Dropping every `@backend accelerated` here
; gives the composed tensor FFT, which compiles and runs on both backends and
; (verified) produces the same output to the last bit.
;
; Signal flow
; -----------
;   table row  --frame-->  curve      (half-spectrum magnitudes, NBINS long)
;              --cutoff--> slid       (curve resampled along the frequency axis)
;              --res----->  shaped    (peaks/troughs exaggerated about the mean)
;              --mirror-->  full      (real symmetric spectrum, N long)
;              --ifft---->  ir        (zero-phase impulse response)
;              --window-->  bounded   (IR truncated to IRLEN taps)
;              --fft----->  (hRe,hIm) (the filter's complex spectrum)
;
;   input --buffer--> frame --*hann--> fft --*(hRe,hIm)--> ifft --*hann-->
;         --overlap-add--> wet
;
; Why the IR window is not decoration
; -----------------------------------
; Multiplying a frame's spectrum by an arbitrary magnitude response is
; *circular* convolution. A magnitude curve with sharp features has an impulse
; response longer than the frame, and everything past the frame boundary wraps
; around to the front as time-aliased garbage. Windowing the IR to IRLEN
; samples bounds the smear so overlap-add stays linear. Tests measure this: an
; unwindowed brickwall response leaks ~10x more energy into the stopband.
;
; Control rate
; ------------
; `filter-table` is written so the *same* body works whether its controls are
; lisp constants or live signals:
;
;   * constant controls  -> `peek-vec` and everything after it are pure tensor
;     arithmetic, so the compiler folds the whole response chain into a static
;     block. The per-sample cost is the input STFT and nothing else. This is the
;     "frozen patch" case, and what the numerical tests pin.
;   * signal controls    -> the same expressions lift to frame-varying tensors.
;     Wrap the controls in `hop-hold` (as the patch at the bottom does) so the
;     response is rebuilt once per hop rather than once per sample; the table
;     read then lands in a hop-gated block.

; ---------------------------------------------------------------- constants
(def N 64)        ; FFT size
(def HOP 16)      ; 4x overlap
(def NBINS 33)    ; N/2 + 1 — the half spectrum the table stores
(def LASTBIN 32)  ; NBINS - 1
(def HALF 12)     ; half the impulse-response budget (IR is ~24 taps)
(def PI 3.14159265358979)

; ------------------------------------------------------------------ macros

; Vectorized interpolated table read — `peek` reads one position, this reads a
; whole tensor of positions at once. Two gathers plus a lerp: `gather` reads the
; FLATTENED source, so row `i` column `c` of an `[R, cols]` table lives at
; `i*cols + c`, and stepping one row is `+ cols`.
;
; `gather` clamps out-of-range indices, so a position of `R-1` reads row R-1
; twice and lands exactly on the last row instead of running off the end. That
; is what makes the `min(..., LASTBIN)` clamp below safe with no epsilon fudge.
(defmacro peek-vec (table pos cols col)
  (def i0 (floor pos))
  (def frac (- pos i0))
  (def base (+ (* i0 cols) col))
  (def a (gather table base))
  (def b (gather table (+ base cols)))
  (+ (* a (- 1 frac)) (* b frac)))

; Mirror a half spectrum into the full real symmetric one. This is a
; permutation, not arithmetic: the ramp `min(k, N-k)` is [0 1 2 ... N/2 ... 2 1],
; so one gather does what an [NBINS, N] mirror matmul would, with N reads and
; zero multiply-adds.
(defmacro mirror-spectrum (half-spectrum)
  (gather half-spectrum (min (iota 64) (- 64 (iota 64)))))

; Zero-phase Hann taper centered on sample 0 with circular wraparound — that is
; where a real symmetric spectrum puts the IR's peak, so the window must be
; symmetric under n -> N-n too. Composed from `iota` rather than baked in as
; data; it matches `spectralFilterIRWindowData(fftSize:irLength:)` exactly.
(defmacro ir-window (n half)
  (def dist (min (iota 64) (- 64 (iota 64))))
  (* (* 0.5 (+ 1 (cos (* PI (/ dist half))))) (lte dist half)))

; The effect itself.
;   table  [FRAMES, NBINS] magnitudes in 0..1
;   input  signal to filter
;   frame  fractional row index — scans/morphs between filter shapes
;   cutoff frequency-axis scale — >1 opens the filter, <1 closes it
;   res    magnitude contrast about the curve's mean — 1 is the identity
(defmacro filter-table (table input frame cutoff res)
  (def cols (iota 33))

  ; 1. scan: one interpolated row read. Broadcasting the scalar frame position
  ;    across all NBINS columns is what lets `peek-vec` do the whole row at once.
  (def curve (peek-vec table (* (ones 33) frame) 33 cols))

  ; 2. cutoff: slide the curve along the frequency axis by resampling it — bin k
  ;    reads position k/cutoff. Past the top bin the shape's tail simply holds.
  (def bin-pos (min (* cols (/ 1 cutoff)) LASTBIN))
  (def slid (peek-vec curve bin-pos 1 0))

  ; 3. resonance: raise the curve to a power *about its own mean*, which deepens
  ;    peaks and troughs while leaving overall level roughly fixed.
  (def avg (+ (mean slid) 0.0001))
  (def shaped (* avg (pow (/ slid avg) res)))

  ; 4. magnitudes -> zero-phase impulse response
  (def full (mirror-spectrum shaped))
  (def ir (ifft full (* full 0) 64 @backend accelerated))

  ; 5. bound the IR, then back to a spectrum
  (def bounded (* ir (ir-window 64 HALF)))
  (def (h-re h-im) (fft bounded 64 @backend accelerated))

  ; 6. STFT the input
  (def win (hann 64))
  (def in-frame (* (reshape (buffer input 64 16) @shape [64]) win))
  (def (x-re x-im) (fft in-frame 64 @backend accelerated))

  ; 7. convolve in the frequency domain, window again, overlap-add
  (def (y-re y-im) (complex-mul x-re x-im h-re h-im))
  (overlap-add (* (ifft y-re y-im 64 @backend accelerated) win) 16))

; ------------------------------------------------------------------- table
; Four filter shapes across the half spectrum. Rows are magnitudes in 0..1 at
; bins 0..32; scanning `frame` morphs between them.
(def table (tensor @shape [4 33] @data [
  ; frame 0 — lowpass, flat to bin 8, cosine taper to zero by bin 14
  1 1 1 1 1 1 1 1 1 0.933 0.75
  0.5 0.25 0.067 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0
  ; frame 1 — highpass, zero to bin 12, cosine rise to flat by bin 18
  0 0 0 0 0 0 0 0 0 0 0
  0 0 0.067 0.25 0.5 0.75 0.933 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1
  ; frame 2 — bandpass centered on bin 16
  0 0 0 0 0 0 0 0 0 0.0381 0.1464
  0.3087 0.5 0.6913 0.8536 0.9619 1 0.9619 0.8536 0.6913 0.5 0.3087
  0.1464 0.0381 0 0 0 0 0 0 0 0 0
  ; frame 3 — flat, the identity filter
  1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1]))

; %%% PATCH %%%
; Everything above this marker is the reusable library; the tests splice their
; own instantiation on after it. Everything below is the plugin patch.

(param frame @default 0 @min 0 @max 3)
(param cutoff @default 1 @min 0.25 @max 4)
(param resonance @default 1 @min 1 @max 8)
(param mix @default 1 @min 0 @max 1)

(def dry (in 1 @name audio-in))

; Hold the controls across each hop: the response only needs rebuilding once per
; STFT frame, and hop-holding is what puts the table read in a hop-gated block
; instead of a per-sample one.
(def wet (filter-table table dry
  (hop-hold frame HOP) (hop-hold cutoff HOP) (hop-hold resonance HOP)))

; The analysis buffer only reports a frame once it is full, so the wet path lags
; the input by one window. Delay the dry path to match before crossfading.
(out (+ (* wet mix) (* (delay dry N) (- 1 mix))) 1 @name audio-out)
