; Shared instrument helpers injected at compile time.
; `samplerate` is provided by DGenLisp as runtime host sample-rate context.

(defmacro mod_unipolar (m)
  (* (+ m 1.0) 0.5))

(defmacro apply_pitch_mod_semi (base_hz mod amt_semi)
  (def ln2 (log 2))
  (* base_hz (exp (* ln2 (/ (* mod amt_semi) 12)))))

(defmacro apply_cutoff_mod_safe (base mod amt)
  (min 11000 (max 60 (+ base (* mod amt)))))

(defmacro apply_pw_mod_safe (base mod amt)
  (clip (+ base (* mod amt)) 0.03 0.97))

; PolyBLEP transition correction for anti-aliased hard edges.
; Kept with a polypleb alias because that typo is memorable and fun.
(defmacro polyblep (phase freq)
  (def dt (clip (/ freq samplerate) 0.000001 0.5))
  (def left_x (/ phase dt))
  (def left (+ (- (* 2.0 left_x) (* left_x left_x)) -1.0))
  (def right_x (/ (- phase 1.0) dt))
  (def right (+ (* right_x right_x) (* 2.0 right_x) 1.0))
  (+ (* (lt phase dt) left)
     (* (gt phase (- 1.0 dt)) right)))

(defmacro polypleb (phase freq)
  (polyblep phase freq))

(defmacro polyblep_saw (phase freq)
  (- (scale phase 0 1 -1 1)
     (polyblep phase freq)))

(defmacro polyblep_pulse (phase width freq)
  (def w (clip width 0.01 0.99))
  (def falling_phase (wrap (- phase w) 0 1))
  (+ (scale (lt phase w) 0 1 -1 1)
     (polyblep phase freq)
     (* -1.0 (polyblep falling_phase freq))))

; Wavetable helpers assume tensor shape [samples, waves]. sample is shape-aware
; (normalized phase scaled by the table's row count at compile time).
(defmacro wavetable-read (table wave phase)
  (sample table phase wave))

(defmacro wavetable-morph (table wave_a wave_b phase morph)
  (wavetable-read table (+ wave_a (* (clip morph 0 1) (- wave_b wave_a))) phase))

; Deprecated aliases (baked 512-row assumption no longer needed):
(defmacro wavetable-read-512 (table wave phase)
  (wavetable-read table wave phase))

(defmacro wavetable-morph-512 (table wave_a wave_b phase morph)
  (wavetable-morph table wave_a wave_b phase morph))

; Cytomic-style ZDF state variable filter.
; cutoff in Hz, q is resonance (0.5 = no resonance, higher = more).
; mode: 0=LP, 1=BP, 2=HP, 3=notch, 4=peak, 5=allpass.
(defmacro svf (input cutoff q mode)
  (def safe_cutoff (clip cutoff 1.0 (* samplerate 0.49)))
  (def safe_q (max q 0.001))
  (def g (tan (* pi (/ safe_cutoff samplerate))))
  (def k (/ 1.0 safe_q))
  (def a1 (/ 1.0 (+ 1.0 (* g (+ g k)))))
  (def a2 (* g a1))
  (def a3 (* g a2))

  (make-history ic1eq)
  (make-history ic2eq)

  (def ic1 (read-history ic1eq))
  (def ic2 (read-history ic2eq))
  (def v3 (- input ic2))
  (def v1 (+ (* a1 ic1) (* a2 v3)))
  (def v2 (+ ic2 (* a2 ic1) (* a3 v3)))

  (write-history ic1eq (- (* 2.0 v1) ic1))
  (write-history ic2eq (- (* 2.0 v2) ic2))

  (def lp v2)
  (def bp v1)
  (def hp (- input (* k v1) v2))
  (def notch (+ hp lp))
  (def peak (- lp hp))
  (def ap (- notch (* k v1)))

  (+ (* (eq mode 0) lp)
     (* (eq mode 1) bp)
     (* (eq mode 2) hp)
     (* (eq mode 3) notch)
     (* (eq mode 4) peak)
     (* (eq mode 5) ap)))

; ZDF Moog ladder filter, 4-pole, with input drive, tanh feedback saturation,
; and resonance-proportional passband gain compensation.
; cutoff in Hz, res is 0..1, drive pre-saturates the input.
(defmacro ladder (input cutoff res drive)
  (def wd (* twopi cutoff))
  (def T (/ 1 samplerate))
  (def wa (* (/ 2.0 T) (tan (* wd T 0.5))))
  (def g (* wa T 0.5))
  (def G (/ g (+ 1 g)))
  (def G4 (* G G G G))
  (def k (* res 4))

  (def fb_trim 0.5)

  (make-history z1)
  (make-history z2)
  (make-history z3)
  (make-history z4)

  (def hz1 (read-history z1))
  (def hz2 (read-history z2))
  (def hz3 (read-history z3))
  (def hz4 (read-history z4))
  (def inv_1pg (/ 1 (+ 1 g)))
  (def S (+ (* hz1 G G G inv_1pg)
            (* hz2 G G inv_1pg)
            (* hz3 G inv_1pg)
            (* hz4 inv_1pg)))

  (def driven_input (tanh (* drive input)))
  (def u (/ (- driven_input (* k fb_trim S))
            (+ 1 (* k fb_trim G4))))
  (def x1 (- u (* k (tanh (* fb_trim (+ (* G4 u) S))))))

  (def v1 (* (- x1 hz1) G))
  (def y1 (+ v1 hz1))
  (write-history z1 (+ y1 v1))

  (def v2 (* (- y1 hz2) G))
  (def y2 (+ v2 hz2))
  (write-history z2 (+ y2 v2))

  (def v3 (* (- y2 hz3) G))
  (def y3 (+ v3 hz3))
  (write-history z3 (+ y3 v3))

  (def v4 (* (- y3 hz4) G))
  (def y4 (+ v4 hz4))
  (write-history z4 (+ y4 v4))

  (+ y4 (* res 0.0013 input)))

(defmacro adsr (gate_sig trigger_sig attack_ms decay_ms sustain release_ms)
  (make-history env)
  (make-history gate_hist)
  (make-history stage_hist)

  ; Retriggers first fade any leftover voice history to silence over a
  ; short de-click window, then start a linear attack from near zero.
  ; Decay/release are one-pole curves scaled to settle near the target
  ; over the requested number of milliseconds.
  (def sr samplerate)
  (def env_time_scale 6.907755)
  (def reset_samples (* 0.003 sr))
  (def attack_samples (max 1.0 (* attack_ms 0.001 sr)))
  (def decay_samples (max 1.0 (* decay_ms 0.001 sr)))
  (def release_samples (max 1.0 (* release_ms 0.001 sr)))
  (def reset_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) reset_samples))))
  (def decay_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) decay_samples))))
  (def release_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) release_samples))))

  (def prev_env (read-history env))
  (def prev_gate (read-history gate_hist))
  (def prev_stage (read-history stage_hist))

  (def gate_on (gt gate_sig 0.5))
  (def gate_rising (* gate_on (lte prev_gate 0.5)))
  (def retrigger (max gate_rising trigger_sig))
  (def attack_stage 1.0)
  (def decay_stage 2.0)
  (def reset_stage 3.0)
  (def attack_done (gte prev_env 0.999))
  (def reset_done (lte prev_env 0.0001))

  (def stage_from_gate
    (gswitch gate_on
      (gswitch retrigger
        (gswitch (gt prev_env 0.0001) reset_stage attack_stage)
        prev_stage)
      0.0))

  (def stage
    (gswitch (eq stage_from_gate reset_stage)
      (gswitch reset_done attack_stage reset_stage)
      (gswitch attack_done
        (gswitch (eq stage_from_gate attack_stage) decay_stage stage_from_gate)
        stage_from_gate)))

  (def target
    (gswitch gate_on
      (gswitch (eq stage reset_stage)
        0.0
        (gswitch (eq stage attack_stage) 1.0 sustain))
      0.0))

  (def rate
    (gswitch gate_on
      (gswitch (eq stage reset_stage) reset_coeff decay_coeff)
      release_coeff))

  (def one_pole_level (+ prev_env (* rate (- target prev_env))))
  (def attack_level (+ prev_env (/ 1.0 attack_samples)))
  (def level_raw
    (gswitch (eq stage attack_stage)
      attack_level
      one_pole_level))
  (def level (clip level_raw 0 1))
  (write-history env level)
  (write-history gate_hist gate_sig)
  (write-history stage_hist stage)
  level)

; A finite-duration, power-curved ADSR. Both curve arguments are positive
; exponents: 1 is linear, values above 1 are convex, and values below 1 are
; concave. Separate attack/fall curves can model the concave attack and convex
; decay/release typical of analog RC envelopes. Sustain remains literal.
(defmacro adsrexp
  (gate_sig trigger_sig attack_ms decay_ms sustain release_ms attack_curve fall_curve)
  (make-history env)
  (make-history gate_hist)
  (make-history stage_hist)
  (make-history phase_hist)
  (make-history release_start_hist)

  (def sr samplerate)
  (def reset_samples (* 0.003 sr))
  (def reset_coeff (- 1.0 (exp (/ -6.907755 reset_samples))))
  ; The differentiable one-sample floor also matches the train-time analytic
  ; lowering exactly, including at a zero-millisecond duration.
  (def attack_samples (+ 1.0 (* attack_ms 0.001 sr)))
  (def decay_samples (+ 1.0 (* decay_ms 0.001 sr)))
  (def release_samples (+ 1.0 (* release_ms 0.001 sr)))
  (def attack_shape (max 0.01 attack_curve))
  (def fall_shape (max 0.01 fall_curve))
  ; Keep power bases strictly positive so learning either curve never
  ; encounters log(0), then normalize both shaped ranges to exact endpoints.
  (def curve_epsilon 0.000001)
  (def curve_domain (- 1.0 curve_epsilon))
  (def attack_curve_floor (pow curve_epsilon attack_shape))
  (def attack_curve_scale (/ 1.0 (- 1.0 attack_curve_floor)))
  (def fall_curve_floor (pow curve_epsilon fall_shape))
  (def fall_curve_scale (/ 1.0 (- 1.0 fall_curve_floor)))

  (def prev_env (read-history env))
  (def prev_gate (read-history gate_hist))
  (def prev_stage (read-history stage_hist))
  (def prev_phase (read-history phase_hist))
  (def prev_release_start (read-history release_start_hist))

  (def gate_on (gt gate_sig 0.5))
  (def gate_rising (* gate_on (lte prev_gate 0.5)))
  (def gate_falling (* (lte gate_sig 0.5) (gt prev_gate 0.5)))
  (def retrigger (max gate_rising trigger_sig))
  (def attack_stage 1.0)
  (def decay_stage 2.0)
  (def reset_stage 3.0)
  (def reset_done (lte prev_env 0.0001))
  ; A completed release also leaves phase at 1. Only treat that phase as an
  ; attack completion when this is a continuation of the previous attack,
  ; never on a fresh gate or trigger.
  (def attack_done
    (* (eq prev_stage attack_stage)
       (lte retrigger 0.5)
       (gte prev_phase 1.0)))

  (def stage_from_gate
    (gswitch gate_on
      (gswitch retrigger
        (gswitch (gt prev_env 0.0001) reset_stage attack_stage)
        prev_stage)
      0.0))
  (def stage
    (gswitch (eq stage_from_gate reset_stage)
      (gswitch reset_done attack_stage reset_stage)
      (gswitch (eq stage_from_gate attack_stage)
        (gswitch attack_done decay_stage attack_stage)
        stage_from_gate)))

  (def phase_start
    (gswitch (eq stage prev_stage) prev_phase 0.0))
  (def phase_step
    (gswitch (eq stage attack_stage)
      (/ 1.0 attack_samples)
      (gswitch (eq stage decay_stage)
        (/ 1.0 decay_samples)
        (gswitch gate_on 0.0 (/ 1.0 release_samples)))))
  (def phase
    (gswitch (eq stage reset_stage)
      0.0
      (clip (+ phase_start phase_step) 0.0 1.0)))

  (def release_start
    (gswitch gate_falling prev_env prev_release_start))
  (def attack_level
    (* (- (pow (+ curve_epsilon (* curve_domain phase)) attack_shape)
          attack_curve_floor)
       attack_curve_scale))
  (def remaining (- 1.0 phase))
  (def shaped_remaining
    (* (- (pow (+ curve_epsilon (* curve_domain remaining)) fall_shape)
          fall_curve_floor)
       fall_curve_scale))
  (def decay_level
    (+ sustain (* (- 1.0 sustain) shaped_remaining)))
  (def release_level (* release_start shaped_remaining))
  (def reset_level (+ prev_env (* reset_coeff (- 0.0 prev_env))))
  (def level_raw
    (gswitch gate_on
      (gswitch (eq stage reset_stage)
        reset_level
        (gswitch (eq stage attack_stage) attack_level decay_level))
      release_level))
  (def level (clip level_raw 0.0 1.0))

  (write-history env level)
  (write-history gate_hist gate_sig)
  (write-history stage_hist stage)
  (write-history phase_hist phase)
  (write-history release_start_hist release_start)
  level)

; Independent finite-stage contour used by Heat's filter and amp envelopes.
; Times are milliseconds except sustain_seconds; negative sustain_seconds
; selects Analog's displayed infinite-sustain setting: linear holds, while
; exponential still traverses its 1000-second contour (measured in the
; envelope-times corpus). restart is an event pulse already filtered by the
; caller's per-envelope legato policy. gate remains the physical held gate.
; Modes: 0 ADSR, 1 AD-R, 2 ADR-R, 3 ADS-AR. Free ignores note-off:
; mode 0 runs ADR once, modes 1/2 loop, mode 3 runs ADAR once.
(defmacro heat-envelope
  (gate restart attack_ms decay_ms sustain sustain_seconds release_ms exponential loop_mode free_run)
  (make-history stage_hist)
  (make-history phase_hist)
  (make-history start_hist)
  (make-history value_hist)
  (make-history gate_hist)
  (def previous_stage (read-history stage_hist))
  (def previous_phase (read-history phase_hist))
  (def previous_start (read-history start_hist))
  (def previous_value (read-history value_hist))
  (def previous_gate (read-history gate_hist))
  (def held (gt gate 0.5))
  (def free (gt free_run 0.5))
  (def mode (clip (round loop_mode) 0 3))
  (def level (clip sustain 0 1))
  ; States: idle 0, attack 1, decay 2, sustain 3, release 4,
  ; note-off/Free return attack 5. All transitions use the same history set.
  (def completed (gt (* (gt previous_stage 0) (gte previous_phase 1)) 0.5))
  ; selector accepts both static settings and signal-rate controls. Keep the
  ; state policy numeric so callers can use constants or live parameters.
  (def free_end (selector (+ (eq mode 3) 1) 4 5))
  (def sustain_or_free (selector (+ free 1) 3 free_end))
  (def after_decay (selector (+ mode 1) sustain_or_free 1 4 sustain_or_free))
  (def after_release (gswitch (gt (* (eq mode 2) (max held free)) 0.5) 1 0))
  (def next_stage
    (selector (+ previous_stage 1) 0 2 after_decay 0 after_release 4))
  (def note_off (gt (* previous_gate (- 1 held) (- 1 free) (gt previous_stage 0)) 0.5))
  (def begin (gt restart 0.5))
  (def transition (gt (max begin note_off completed) 0.5))
  (def stage
    (gswitch begin 1
      (gswitch note_off free_end
        (gswitch completed next_stage previous_stage))))
  (def endpoint (selector (+ previous_stage 1) 0 1 level 0 0 1))
  (def start
    (gswitch (gt (max begin note_off) 0.5) previous_value
      (gswitch completed endpoint previous_start)))
  (def phase (gswitch transition 0 previous_phase))
  (def infinite_setting (lt sustain_seconds 0))
  (def infinite_hold (gt (* (eq stage 3) infinite_setting (lte exponential 0.5)) 0.5))
  (def sustain_duration (selector (+ infinite_setting 1) (max 0 sustain_seconds) 1000))
  (def duration_ms
    (selector (+ stage 1) 1 attack_ms decay_ms
      (* 1000 sustain_duration) release_ms attack_ms))
  (def increment (gswitch infinite_hold 0 (/ 1000 (* samplerate (max 0.001 duration_ms)))))
  (def progress (clip phase 0 1))
  (def curve (selector (+ (gt exponential 0.5) 1) progress
    (/ (- 1 (exp (* -3.5 progress))) 0.9698026166)))
  (def target (selector (+ stage 1) 0 1 level 0 0 1))
  (def value (gswitch (eq stage 0) 0 (+ start (* (- target start) curve))))
  (write-history stage_hist stage)
  (write-history phase_hist (gswitch (eq stage 0) 0 (+ phase increment)))
  (write-history start_hist start)
  (write-history value_hist value)
  (write-history gate_hist held)
  value)

; Finite exponential pitch decay. Initial is semitones, time is milliseconds.
; Development law (k=5); measured Analog timing-map calibration is pending.
(defmacro heat-pitch-envelope (note_on initial time_ms)
  (make-history elapsed_hist)
  (make-history initialized_hist)
  (def restart (max (eq (read-history initialized_hist) 0) (gt note_on 0.5)))
  ; Quantize duration to whole samples so the endpoint is exactly zero.
  (def frames (max 0 (round (* time_ms (/ samplerate 1000)))))
  (def elapsed (gswitch restart 0 (read-history elapsed_hist)))
  (def t (clip (/ elapsed (max 1 frames)) 0 1))
  (def curve (/ (- (exp (* -5 t)) 0.006737946999) 0.993262053001))
  (write-history elapsed_hist (min frames (+ elapsed 1)))
  (write-history initialized_hist 1)
  (* initial curve (lt elapsed frames)))

; Per-voice modulation oscillator. Physical units: Hz, cycles, milliseconds.
; Runs at audio rate: reproducing Analog's internal control grid is outside
; Heat's scope. Shapes: sine, skew triangle, pulse, random step, random ramp.
(defmacro heat-lfo (rate_hz width shape note_on retrigger phase_offset delay_ms fade_ms)
  (make-history initialized_hist)
  (make-history phase_hist)
  (make-history phase_fraction_hist)
  (make-history random_hist)
  (make-history random_start_hist)
  (make-history elapsed_hist)
  (def first (eq (read-history initialized_hist) 0))
  (def note_start (max first (gt note_on 0.5)))
  (def restart (max first (gt (* note_on retrigger) 0.5)))
  ; Keep the accumulator's fractional remainder small. Adding tiny LFO
  ; increments directly to a single float phase drifts at high sample rates.
  ; Coarse phase steps are exact binary fractions; this also supports changes
  ; of rate without deriving phase from an ever-growing sample counter.
  (def previous_coarse (read-history phase_hist))
  (def fraction (+ (read-history phase_fraction_hist) (/ (max 0 rate_hz) samplerate)))
  (def carry (/ (floor (* fraction 1024)) 1024))
  (def offset (wrap phase_offset 0 1))
  (def offset_coarse (/ (floor (* offset 1024)) 1024))
  (def coarse (gswitch restart offset_coarse (wrap (+ previous_coarse carry) 0 1)))
  (def remainder (gswitch restart (- offset offset_coarse) (- fraction carry)))
  (def phase (+ coarse remainder))
  (def cycle (max restart (gte (+ previous_coarse carry) 1)))
  (def previous_random (read-history random_hist))
  (def random_value (gswitch cycle (noise) previous_random))
  (def random_start (gswitch cycle previous_random (read-history random_start_hist)))
  (def duty (clip width 0 1))
  (def skew (+ 0.05 (* 0.9 duty)))
  (def triangle (min 1 (gswitch (lt phase duty)
    (- 1 (/ (* 2 phase) skew))
    (- 1 (/ (* 2 (- 1 phase)) (- 1 skew))))))
  (def pulse (selector (+ (lt phase duty) 1) -1 1))
  (def random_ramp (+ random_start
    (* (- random_value random_start) (clip (/ phase 0.4) 0 1))))
  (def raw (selector (+ 1 (clip (round shape) 0 4))
    (sin (* 6.28318530718 phase)) triangle pulse random_value random_ramp))
  ; Delay/fade restart for each note even when oscillator retrigger is off.
  ; Count samples exactly over the bounded delay/fade interval instead of
  ; accumulating rounded millisecond increments on every sample.
  (def elapsed_frames (gswitch note_start 0 (read-history elapsed_hist)))
  (def elapsed (* elapsed_frames (/ 1000 samplerate)))
  (def delay (max 0 delay_ms))
  (def fade (max 0 fade_ms))
  (def gain (gswitch (gt fade 0) (clip (/ (- elapsed delay) (max 0.001 fade)) 0 1)
    (gte elapsed delay)))
  (write-history phase_hist coarse)
  (write-history phase_fraction_hist remainder)
  (write-history initialized_hist 1)
  (write-history random_hist random_value)
  (write-history random_start_hist random_start)
  (write-history elapsed_hist (min (ceil (* (+ delay fade) (/ samplerate 1000))) (+ elapsed_frames 1)))
  (* raw gain))

; Heat's linear filter family, identified from isolated Analog captures.
; Mode: LP12, LP24, BP6, BP12, Notch2, Notch4, HP12, HP24.
; cutoff is the physical corner in Hz; q is the dimensionless resonance,
; not a normalized knob value. LP24/HP24 split Q across their two stages;
; BP12/Notch4 retain Q in each stage. No gain compensation or saturation.
; The shared svf is a trapezoidal-integrator state-variable filter.
(defmacro heat-linear-filter (input cutoff q mode)
  (def kind (clip (round mode) 0 7))
  (def double-stage (eq (% kind 2) 1))
  (def split-q (* double-stage (+ (lt kind 2) (gte kind 6))))
  (def stage-q (selector (+ (gt split-q 0.5) 1) (clip q 0.1 100) (sqrt (clip q 0.1 100))))
  (def family (floor (/ kind 2)))
  ; svf's HP and notch enum order differs from Heat's menu order.
  (def svf-mode (selector (+ family 1) 0 1 3 2))
  (def first (svf input cutoff stage-q svf-mode))
  (def second (svf first cutoff stage-q svf-mode))
  (selector (+ double-stage 1) first second))

; Symmetric soft knee with unit slope below knee and a finite asymptote.
; The rational tail matches the isolated Analog drive measurements. Explicit
; multiplication keeps the cubic cheap and avoids a general power operation.
(defmacro heat-soft-clip (input knee ceiling)
  (def threshold (max 0 knee))
  (def span (max 0.000001 (- ceiling threshold)))
  (def magnitude (abs input))
  (def excess (max 0 (- magnitude threshold)))
  (def reciprocal (/ 1 (+ 1 (/ excess (* 3 span)))))
  (def tail (* span (- 1 (* reciprocal reciprocal reciprocal))))
  (* (selector (+ (lt input 0) 1) 1 -1)
    (+ (min magnitude threshold) tail)))



; Off, Sym 1/2/3, Asym 1/2/3. Signal units put the drive knee at 1.
; Asymmetric modes retain the broad negative tail while lowering the positive
; ceiling. Amplifier gain, pan and the final output limiter belong downstream.
(defmacro heat-drive (input mode)
  (def choice (clip (round mode) 0 6))
  (def gain (selector (+ choice 1) 1 1 1.41421356237 2 1 1.41421356237 2))
  (def positive_limit (selector (+ choice 1) 8 8 4 2 4 2 1.3))
  (def negative_limit (selector (+ choice 1) 8 8 4 2 8 8 8))
  (def ceiling (selector (+ (lt input 0) 1) positive_limit negative_limit))
  (def driven (heat-soft-clip (* input gain) 1 ceiling))
  (selector (+ (gt choice 0) 1) input driven))

; Heat development voice. Not yet a factory-accepted Analog match.
; Uses physical units throughout. Source waveform/level calibration, hard
; sync, formant control mapping, unison and vibrato remain release work.
; Named host signals distinguish a physical note-on from an envelope trigger.






(def gate (in 1 @name gate))
(def pitch (in 2 @name pitch))
(def velocity (in 3 @name velocity))
(def trigger (in 4 @name trigger))
(def note_on (in 5 @name note_on))
(def legato (in 6 @name legato))
(def pressure (in 7 @name pressure))
(def mod1 (in 8 @name mod1 @modulator 1))
(def mod2 (in 9 @name mod2 @modulator 2))
(def mod3 (in 10 @name mod3 @modulator 3))
(def mod4 (in 11 @name mod4 @modulator 4))

(defmacro heat-db (db) (exp (* 0.11512925465 db)))
(defmacro heat-octaves (octaves) (exp (* 0.69314718056 octaves)))

; First-order target slew, initialized directly to the first target.
; Smooth continuous level/pan controls without smearing note/event signals.
(defmacro heat-control (target)
  (make-history ready_hist)
  (make-history value_hist)
  (def old (read-history value_hist))
  (def coefficient 0.010362)
  (def value (gswitch (eq (read-history ready_hist) 0) target
    (+ old (* coefficient (- target old)))))
  (write-history ready_hist 1)
  (write-history value_hist value)
  value)

; Development source family, normalized peak amplitude. The two oscillators
; own separate phases and sub phases. These analytical waveforms still need
; complete comparison against Analog; no reference waveform samples are used.
(defmacro heat-source (frequency wave duty sub_level)
  (def hz (clip frequency 0.01 (* 0.45 samplerate)))
  (def phase (phasor hz))
  (def sub_hz (* 0.5 hz))
  (def sub_phase (phasor sub_hz))
  (def main (selector (+ 1 (clip (round wave) 0 3))
    (sin (* 6.28318530718 phase))
    (polyblep_saw phase hz)
    (polyblep_pulse phase (clip duty 0.01 0.99) hz)
    (noise)))
  (+ main (* (clip sub_level 0 1) (polyblep_pulse sub_phase 0.5 sub_hz))))

(param volume_db @default -18 @min -60 @max 6 @unit dB @mod true @mod-mode additive)
(param tune_semitones @default 0 @min -48 @max 48 @unit st @mod true @mod-mode additive)
(param pressure_pitch_semitones @default 0 @min -24 @max 24 @unit st)
(param pressure_filter_octaves @default 0 @min -8 @max 8)
(param pressure_amp_db @default 0 @min -36 @max 12 @unit dB)

; Lane 1: independently stored source, modulation and articulation.
(param osc1_enabled @default 1 @min 0 @max 1)
(param osc1_wave @default 1 @min 0 @max 3)
(param osc1_level_db @default -6 @min -60 @max 12 @unit dB @mod true @mod-mode additive)
(param osc1_to_filter1 @default 1 @min 0 @max 1 @mod true @mod-mode additive)
(param osc1_pitch_env_initial @default 0 @min -48 @max 48 @unit st)
(param osc1_pitch_env_time_ms @default 500 @min 0 @max 15000 @unit ms)
(param osc1_semitones @default 0 @min -48 @max 48 @unit st @mod true @mod-mode additive)
(param osc1_cents @default 0 @min -300 @max 300 @unit ct @mod true @mod-mode additive)
(param osc1_keytrack @default 1 @min -2 @max 2)
(param osc1_pulse_duty @default 0.5 @min 0.01 @max 0.99 @mod true @mod-mode additive)
(param osc1_sub_level @default 0 @min 0 @max 1 @mod true @mod-mode additive)
(param osc1_lfo_pitch_semitones @default 0 @min -24 @max 24 @unit st)
(param osc1_lfo_pw @default 0 @min -0.49 @max 0.49)
(param lfo1_enabled @default 0 @min 0 @max 1)
(param lfo1_rate_hz @default 1 @min 0.01 @max 100)
(param lfo1_shape @default 0 @min 0 @max 4)
(param lfo1_width @default 0.5 @min 0 @max 1)
(param lfo1_retrigger @default 1 @min 0 @max 1)
(param lfo1_phase @default 0 @min 0 @max 1)
(param lfo1_delay_ms @default 0 @min 0 @max 10000)
(param lfo1_fade_ms @default 0 @min 0 @max 10000)
(param filter1_enabled @default 1 @min 0 @max 1)
(param filter1_mode @default 0 @min 0 @max 7)
(param filter1_cutoff_hz @default 1800 @min 30 @max 22000 @unit Hz @mod true @mod-mode additive)
(param filter1_q @default 0.707 @min 0.1 @max 100 @mod true @mod-mode additive)
(param filter1_drive @default 0 @min 0 @max 6)
(param filter1_keytrack @default 0 @min -2 @max 2)
(param filter1_env_octaves @default 2 @min -8 @max 8 @mod true @mod-mode additive)
(param filter1_lfo_octaves @default 0 @min -8 @max 8)
(param filter1_env_q @default 0 @min -50 @max 50)
(param filter1_lfo_q @default 0 @min -50 @max 50)
(param amp1_enabled @default 1 @min 0 @max 1)
(param amp1_level_db @default 0 @min -60 @max 12 @unit dB @mod true @mod-mode additive)
(param amp1_pan @default 0 @min -1 @max 1 @mod true @mod-mode additive)
(param amp1_lfo_level @default 0 @min -1 @max 1)
(param amp1_lfo_pan @default 0 @min -1 @max 1)
(param amp1_env_pan @default 0 @min -1 @max 1)
(param amp1_key_pan @default 0 @min -1 @max 1)
(param amp1_key_level_db @default 0 @min -24 @max 24 @unit dB)
(param filter1_env_attack_ms @default 5 @min 0.01 @max 15000)
(param filter1_env_decay_ms @default 350 @min 0.01 @max 15000)
(param filter1_env_sustain @default 0.25 @min 0 @max 1)
(param filter1_env_sustain_seconds @default -1 @min -1 @max 1000)
(param filter1_env_release_ms @default 250 @min 0.01 @max 15000)
(param filter1_env_exponential @default 1 @min 0 @max 1)
(param filter1_env_loop @default 0 @min 0 @max 3)
(param filter1_env_free @default 0 @min 0 @max 1)
(param filter1_env_legato @default 1 @min 0 @max 1)
(param filter1_env_velocity @default 0 @min 0 @max 1)
(param amp1_env_attack_ms @default 5 @min 0.01 @max 15000)
(param amp1_env_decay_ms @default 150 @min 0.01 @max 15000)
(param amp1_env_sustain @default 0.8 @min 0 @max 1)
(param amp1_env_sustain_seconds @default -1 @min -1 @max 1000)
(param amp1_env_release_ms @default 250 @min 0.01 @max 15000)
(param amp1_env_exponential @default 1 @min 0 @max 1)
(param amp1_env_loop @default 0 @min 0 @max 3)
(param amp1_env_free @default 0 @min 0 @max 1)
(param amp1_env_legato @default 1 @min 0 @max 1)
(param amp1_env_velocity @default 0.5 @min 0 @max 1)

; Lane 2: independently stored source, modulation and articulation.
(param osc2_enabled @default 0 @min 0 @max 1)
(param osc2_wave @default 1 @min 0 @max 3)
(param osc2_level_db @default -6 @min -60 @max 12 @unit dB @mod true @mod-mode additive)
(param osc2_to_filter1 @default 0 @min 0 @max 1 @mod true @mod-mode additive)
(param osc2_pitch_env_initial @default 0 @min -48 @max 48 @unit st)
(param osc2_pitch_env_time_ms @default 500 @min 0 @max 15000 @unit ms)
(param osc2_semitones @default 0 @min -48 @max 48 @unit st @mod true @mod-mode additive)
(param osc2_cents @default 0 @min -300 @max 300 @unit ct @mod true @mod-mode additive)
(param osc2_keytrack @default 1 @min -2 @max 2)
(param osc2_pulse_duty @default 0.5 @min 0.01 @max 0.99 @mod true @mod-mode additive)
(param osc2_sub_level @default 0 @min 0 @max 1 @mod true @mod-mode additive)
(param osc2_lfo_pitch_semitones @default 0 @min -24 @max 24 @unit st)
(param osc2_lfo_pw @default 0 @min -0.49 @max 0.49)
(param lfo2_enabled @default 0 @min 0 @max 1)
(param lfo2_rate_hz @default 1 @min 0.01 @max 100)
(param lfo2_shape @default 0 @min 0 @max 4)
(param lfo2_width @default 0.5 @min 0 @max 1)
(param lfo2_retrigger @default 1 @min 0 @max 1)
(param lfo2_phase @default 0 @min 0 @max 1)
(param lfo2_delay_ms @default 0 @min 0 @max 10000)
(param lfo2_fade_ms @default 0 @min 0 @max 10000)
(param filter2_enabled @default 1 @min 0 @max 1)
(param filter2_mode @default 0 @min 0 @max 7)
(param filter2_cutoff_hz @default 1800 @min 30 @max 22000 @unit Hz @mod true @mod-mode additive)
(param filter2_q @default 0.707 @min 0.1 @max 100 @mod true @mod-mode additive)
(param filter2_drive @default 0 @min 0 @max 6)
(param filter2_keytrack @default 0 @min -2 @max 2)
(param filter2_env_octaves @default 2 @min -8 @max 8 @mod true @mod-mode additive)
(param filter2_lfo_octaves @default 0 @min -8 @max 8)
(param filter2_env_q @default 0 @min -50 @max 50)
(param filter2_lfo_q @default 0 @min -50 @max 50)
(param amp2_enabled @default 1 @min 0 @max 1)
(param amp2_level_db @default 0 @min -60 @max 12 @unit dB @mod true @mod-mode additive)
(param amp2_pan @default 0 @min -1 @max 1 @mod true @mod-mode additive)
(param amp2_lfo_level @default 0 @min -1 @max 1)
(param amp2_lfo_pan @default 0 @min -1 @max 1)
(param amp2_env_pan @default 0 @min -1 @max 1)
(param amp2_key_pan @default 0 @min -1 @max 1)
(param amp2_key_level_db @default 0 @min -24 @max 24 @unit dB)
(param filter2_env_attack_ms @default 5 @min 0.01 @max 15000)
(param filter2_env_decay_ms @default 350 @min 0.01 @max 15000)
(param filter2_env_sustain @default 0.25 @min 0 @max 1)
(param filter2_env_sustain_seconds @default -1 @min -1 @max 1000)
(param filter2_env_release_ms @default 250 @min 0.01 @max 15000)
(param filter2_env_exponential @default 1 @min 0 @max 1)
(param filter2_env_loop @default 0 @min 0 @max 3)
(param filter2_env_free @default 0 @min 0 @max 1)
(param filter2_env_legato @default 1 @min 0 @max 1)
(param filter2_env_velocity @default 0 @min 0 @max 1)
(param amp2_env_attack_ms @default 5 @min 0.01 @max 15000)
(param amp2_env_decay_ms @default 150 @min 0.01 @max 15000)
(param amp2_env_sustain @default 0.8 @min 0 @max 1)
(param amp2_env_sustain_seconds @default -1 @min -1 @max 1000)
(param amp2_env_release_ms @default 250 @min 0.01 @max 15000)
(param amp2_env_exponential @default 1 @min 0 @max 1)
(param amp2_env_loop @default 0 @min 0 @max 3)
(param amp2_env_free @default 0 @min 0 @max 1)
(param amp2_env_legato @default 1 @min 0 @max 1)
(param amp2_env_velocity @default 0.5 @min 0 @max 1)
(param filter1_to_filter2 @default 0 @min 0 @max 1 @mod true @mod-mode additive)
(param filter2_follow @default 0 @min 0 @max 1)
(param filter2_offset_octaves @default 0 @min -8 @max 8 @mod true @mod-mode additive)
(param noise_enabled @default 0 @min 0 @max 1)
(param noise_level_db @default -24 @min -60 @max 12 @unit dB @mod true @mod-mode additive)
(param noise_color_hz @default 8000 @min 30 @max 22000 @unit Hz @mod true @mod-mode additive)
(param noise_to_filter1 @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)

(out (+ (heat-control osc1_level_db) (heat-control amp1_pan) (heat-control osc1_to_filter1) (heat-control amp1_level_db) (heat-control volume_db)) 1)
(out 0 2)
