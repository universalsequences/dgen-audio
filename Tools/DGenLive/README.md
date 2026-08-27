# DGen Live

A deliberately small Linux live-coding host: Emacs compiles one DGenLisp buffer
to a `.so` plus manifest, then tells a C/ALSA server to replace its one running
DSP node.

## Build and run

Requirements: ALSA, json-c, pkg-config, a C compiler, and Emacs.

```sh
cd Tools/DGenLive
make
./dgen-live
```

The default socket is `/tmp/dgen-live-$UID.sock`; audio is 48 kHz stereo on
the ALSA `default` device with 512-frame blocks and 100 ms of buffering. These
defaults match typical PipeWire settings and avoid needless resampling. The block
size must be divisible by four because generated DGen kernels operate on four
SIMD lanes. Options:

```text
--socket PATH --device ALSA_DEVICE --sample-rate HZ --block-size N
--latency-ms N --no-audio
```

`--no-audio` is useful for protocol tests. The line protocol supports `PING`,
`STATUS`, `LOAD /absolute/path/to/patch.json`, `STOP`, and `QUIT`. `STATUS`
reports recovered xruns and partial ALSA writes. `RENDER N` is available only
with `--no-audio`, and `N` must also be divisible by four.

The server validates manifest v3 / `dgen-host-abi-v1`, allocates at least
`max(totalMemorySlots, 1024)` floats, applies `tensorInitData`, fills each
parameter's physical `cellId..<(cellId + cellSpan)` with its default, and calls
`dgen_process_v1`. A new library and state are prepared before an atomic pointer
swap. Retired libraries remain loaded until shutdown, so the audio thread can
never execute unmapped code.

## Emacs

Make `dgenlisp` available on `PATH`, or point `dgenlisp-compiler` at the built
executable. Add this directory to `load-path` and load the mode:

```elisp
(add-to-list 'load-path "/path/to/dgen-audio/Tools/DGenLive")
(require 'dgenlisp-mode)
(setq dgenlisp-compiler "/path/to/dgen-audio/.build/debug/DGenLisp")
```

Open a `.lisp` file and use:

- `C-c C-c` — compile the entire buffer and hot-swap it
- `C-c C-s` — stop the current patch (silence)
- `M-TAB` / completion-at-point — complete DGenLisp forms

Set `dgenlisp-max-frames` to a multiple of four at least as large as the server
block size, and keep
`dgenlisp-sample-rate` equal to the server sample rate. Relative wavetable and
tensor assets resolve from the source buffer's directory.

## Tests

```sh
make test
```

The C integration test loads a fixture through the Unix socket and checks that
parameter defaults, parameter spans, tensor initialization, sample-rate context,
hot swapping, rendering, stopping, and shutdown work. ERT tests cover mode
activation/completion and the Emacs Unix-socket client.
