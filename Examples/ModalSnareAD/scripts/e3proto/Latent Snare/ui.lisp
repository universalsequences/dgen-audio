;; Latent Modal Snare (E3 prototype): four latent knobs feed the per-hit
;; decoder; tune / decay / level act after it.
(def latent-block ()
  (eseq.effects.custom-ui-lego/ui-control-block-medium-s "LATENT" (eseq.effects.custom-ui-lego/ui-accent-cyan) 0
    (h-stack :gap 0.32 :align :start
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "z1" "z1" 4.8 (eseq.effects.custom-ui-lego/ui-accent-cyan) 2)
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "z2" "z2" 4.8 (eseq.effects.custom-ui-lego/ui-accent-cyan) 2)
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "z3" "z3" 4.8 (eseq.effects.custom-ui-lego/ui-accent-violet) 2)
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "z4" "z4" 4.8 (eseq.effects.custom-ui-lego/ui-accent-violet) 2))))

(def shape-block ()
  (eseq.effects.custom-ui-lego/ui-control-block-medium-s "SHAPE" (eseq.effects.custom-ui-lego/ui-accent-orange) 0
    (h-stack :gap 0.32 :align :start
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "tune" "tune" 4.8 (eseq.effects.custom-ui-lego/ui-accent-orange) 1)
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "decay" "decay" 4.8 (eseq.effects.custom-ui-lego/ui-accent-orange) 2)
      (eseq.effects.custom-ui-lego/ui-lego-knob-s 0 "level" "level" 4.8 (eseq.effects.custom-ui-lego/ui-accent-green) 2))))

(def info-block ()
  (eseq.effects.custom-ui-lego/ui-readout-block-small-s "DECODER" (eseq.effects.custom-ui-lego/ui-accent-blue) 0
    (eseq.effects.custom-ui-lego/ui-lego-text-row-3
      (label "5-64-64-282" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-blue) :bg :transparent)
      (label "24 modes" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-cyan) :bg :transparent)
      (label "16 bands" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-violet) :bg :transparent))))

(def rate-block ()
  (eseq.effects.custom-ui-lego/ui-readout-block-small-s "RATE" (eseq.effects.custom-ui-lego/ui-accent-orange) 0
    (eseq.effects.custom-ui-lego/ui-lego-text-row-3
      (label "decoder" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-orange) :bg :transparent)
      (label "per hit" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-orange) :bg :transparent)
      (label "bank/sample" :font-size 9.0 :color (eseq.effects.custom-ui-lego/ui-accent-green) :bg :transparent))))

(defsynth-ui
  (h-stack :width :fill :gap 0.35 :align :stretch
    (eseq.effects.custom-ui-lego/ui-lego-column-2
      (latent-block)
      (info-block))
    (eseq.effects.custom-ui-lego/ui-lego-column-2
      (shape-block)
      (rate-block))))
