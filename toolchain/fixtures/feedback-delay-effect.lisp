; Representative stateful effect: audio input, variable delay, and explicit
; one-sample feedback. This intentionally exercises both delay storage and
; history feedback in today's code generator.
(param delay-samples @default 37 @min 1 @max 256)
(param feedback @default 0.35 @min 0 @max 0.95)
(param wet @default 0.4 @min 0 @max 1)

(def dry (in 1 @name audio-in))
(make-history fb)
(def delayed (delay (+ dry (* feedback (read-history fb))) delay-samples))
(write-history fb delayed)
(out (mix dry delayed wet) 1 @name audio-out)
