;; encoder.lisp — audio → latent code for a trained latent instrument
;; (docs/LATENT_ENCODER_SPEC.md §2). Train with:
;;
;;   dgenlisp train-dataset --stage encode --patch voice.lisp --encoder encoder.lisp \
;;     --dataset KICKS[,SYNTH/renders] --codes RUN/export/codes.json[,SYNTH/renders/labels.json] \
;;     --init-from RUN --job-dir ENC --batch 32 --val 10 --steps 2000 --lr 1e-3 --clip 1
;;
;; Input: normalised host log-mel, 64 bands × 90 frames (48 kHz, 0.5 s, window 1024, hop 256),
;; channels first, time last. Set D (the latent-table @dim of the voice) in the last two tensors.

(def x (reshape (row-data @name feats @shape [64 90] (zeros [64 90])) @shape [-1 64 90]))

;; conv1: 64 → 96 channels, k 5, stride 1 — [B,64,86,5] → [B·86, 64·5]
(def c1 (reshape (transpose (windows1d x @size 5) @axes [0 2 1 3]) @shape [-1 320]))
(def h1 (relu (+ (matmul c1 (tensor @shape [320 96] @learnable true @name enc_c1 @file "enc_c1.json"))
                 (tensor @shape [1 96] @learnable true @name enc_c1b @file "enc_c1b.json" @init zeros))))

;; back to channels first: [B, 96, 86]
(def h1t (transpose (reshape h1 @shape [-1 86 96]) @axes [0 2 1]))

;; conv2: 96 → 128 channels, k 5, stride 2 — T' = (86 − 5)/2 + 1 = 41
(def c2 (reshape (transpose (windows1d h1t @size 5 @stride 2) @axes [0 2 1 3]) @shape [-1 480]))
(def h2 (relu (+ (matmul c2 (tensor @shape [480 128] @learnable true @name enc_c2 @file "enc_c2.json"))
                 (tensor @shape [1 128] @learnable true @name enc_c2b @file "enc_c2b.json" @init zeros))))

;; time average, MLP head
(def g (mean-axis (reshape h2 @shape [-1 41 128]) @axis 1))
(def h3 (tanh (+ (matmul g (tensor @shape [128 128] @learnable true @name enc_w1 @file "enc_w1.json"))
                 (tensor @shape [1 128] @learnable true @name enc_b1 @file "enc_b1.json" @init zeros))))
(latent-out (+ (matmul h3 (tensor @shape [128 8] @learnable true @name enc_w2 @file "enc_w2.json"))
               (tensor @shape [1 8] @learnable true @name enc_b2 @file "enc_b2.json" @init zeros)))
