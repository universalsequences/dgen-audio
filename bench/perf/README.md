# C backend performance bench

`bench.py` compiles every `*.lisp` in this directory with the local
`.build/release/DGenLisp`, drives `dgen_process_v1` the way a host does
(48 kHz, 128-frame blocks, one held note), and prints microseconds per block,
percent of one core per voice, and the number of frame loops and scratch
stores in the emitted C.

```sh
swift build -c release --product DGenLisp
python3 bench/perf/bench.py --toolchain-root <staged dgen-toolchain> \
    --save bench/perf/ref-baseline          # record traces once
python3 bench/perf/bench.py --toolchain-root <staged dgen-toolchain> \
    --ref bench/perf/ref-baseline           # later: time + bit-exactness check
```

`--keep DIR` retains each build (`patch.c`, `patch.dylib`, `patch.json`);
`time_dylib.py DIR a.dylib b.dylib` times hand-edited variants of one build.

Sources:

- `five-smoothers.lisp` — a Heat-sized parameter block with five `heat-control`
  one-pole smoothers. Isolates dead modulation work and loop-boundary cost.
- `heat-head.lisp` — the eseq Heat development voice as of eseq `b24bc1f0`,
  macros inlined. Two oscillators, two filters, four envelopes, two LFOs.

Pass toggles for A/B runs: `DGEN_NO_DCE=1`, `DGEN_NO_STATIC_HOIST=1`,
`DGEN_NO_COALESCE=1`, `DGEN_COALESCE_MAX_PARALLEL=<n>` (default 16),
`DGEN_AFFINE_SORT=1` (opt-in), `DGEN_FORCE_SCALAR=1`.

Measured on an M1 Max (2026-09-06), baseline → all passes:

| patch | before | after |
| --- | --- | --- |
| five-smoothers | 12.2 µs/block | 1.8 µs/block |
| heat-head | 78.5 µs/block | 49.2 µs/block |
