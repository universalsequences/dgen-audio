;; Standalone compiler/release smoke: outputs 1, 2, 3, ... across host blocks.
;; The fresh second peek must observe the write in the same sample.
(def counter (tensor @shape [8]))
(def result (seq (poke counter 0 (+ (peek counter 0) 1)) (peek counter 0)))
(out result 1 @name counter)
