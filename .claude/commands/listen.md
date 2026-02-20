Analyze the audio output of the DGen code we're currently working on.

You have access to the full conversation context. Follow these steps in order:

## Step 1: Identify the goal

Look at the current conversation and the code we're working on. Determine:
- What signal/audio are we trying to produce?
- What would "correct" output sound like? (e.g., a 440Hz sine, an 808 kick, a filtered noise burst, silence except for a click, etc.)
- What sample rate and frame count are in use?

State your understanding in 1-2 sentences before proceeding.

## Step 2: Find or create the export point

Look at the test or code we're editing. Identify the best Signal or Tensor to export — this should be the **forward pass output**, NOT the loss or backward result.

Common patterns:
- A `Signal` variable like `synth`, `output`, `wave`, `sig` before it hits `spectralLossFFT` or `backward()`
- A `Tensor` that represents audio samples
- The return value of a synth-building function

If no clean export point exists, add a temporary WAV export to the test. Use this pattern:

```swift
// Temporary debug export — remove after debugging
let debugSamples = try <SIGNAL>.realize(frames: <FRAME_COUNT>)
try AudioFile.save(url: URL(fileURLWithPath: "/tmp/debug_listen.wav"), samples: debugSamples, sampleRate: <SAMPLE_RATE>)
```

If the code already exports a WAV (e.g., to `/tmp/something.wav`), just use that.

**Do NOT export from:**
- `loss.backward()` return values (these are per-frame loss scalars, not audio)
- Gradient tensors
- Intermediate spectral representations (FFT magnitude bins)

## Step 3: Run the test to generate the WAV

Build and run only the specific test that produces the WAV:

```
swift test --filter <TestClass>/<testMethod>
```

If the test fails, report the error — don't try to fix it in this skill. The point is to hear what the current code produces, even if the test assertions fail. Consider using `2>&1 | tail -20` to keep output manageable.

## Step 4: Analyze the WAV

Run the analysis script on the exported WAV:

```
python3 Assets/analyze_wav.py /tmp/<filename>.wav
```

## Step 5: Interpret results

Compare the analysis against what you identified in Step 1. Report:

1. **Expected vs actual**: Does the output match the goal?
2. **Red flags**: Any of these indicate bugs:
   - `ALL ZEROS` — signal chain is broken, output not connected, or memory not written
   - `HAS NaN` / `HAS Inf` — gradient explosion, division by zero, or uninitialized memory read
   - `CLIPPING` with high sample count — gain staging issue or feedback runaway
   - `near-silent` when it shouldn't be — signal is being attenuated to nothing
   - Wrong fundamental frequency — parameter not updating, wrong cell being read, transpose bug
   - `DC-offset` — missing window function, or accumulator not resetting
   - `noise-like` when expecting tonal — race condition (multiple threads writing same memory), or reading uninitialized memory
3. **Diagnosis hints**: If something looks wrong, suggest what to investigate based on the specific symptoms and the code we're working on.

Keep the interpretation concise — focus on what's actionable.
