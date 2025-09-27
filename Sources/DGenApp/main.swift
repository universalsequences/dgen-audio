import AVFoundation
import DGen

// Build a simple test tone through a biquad (offline render to WAV)
let g = Graph()

// Source: 220 Hz sine
let freq = g.n(.constant(50.0))
let reset = g.n(.constant(0.0))
let phase = g.n(.phasor(g.alloc()), freq, reset)

let phase2 = g.triangle(g.n(.phasor(g.alloc()), g.n(.constant(3.0)), reset), g.n(.constant(0.4)))

// Biquad lowpass: cutoff 1000 Hz, resonance 0.7, unity gain, lowpass mode 0
let cutoff = g.n(.add, g.n(.constant(50)), g.n(.mul, phase2, g.n(.constant(100.0))))
let resonance = g.n(.constant(7))
let gain = g.n(.constant(1.0))
let mode = g.n(.constant(1.0))
let filtered = g.biquad(phase, cutoff, resonance, gain, mode)
_ = g.n(.output(0), filtered)

let seconds: Double = 10.0
let sampleRate: Double = 44100

// Compile and render with C backend to WAV
let cResult = try CompilationPipeline.compile(
  graph: g, backend: .c, options: .init(frameCount: 128, debug: true))
print(cResult.source)
let cRuntime = CCompiledKernel(
  source: cResult.source,
  cellAllocations: cResult.cellAllocations,
  memorySize: cResult.totalMemorySlots
)
let cURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  .appendingPathComponent("biquad_c.wav")
try cRuntime.writeWAV(to: cURL, seconds: seconds, sampleRate: sampleRate, volumeScale: 1.0)
print("Wrote C WAV to \(cURL.path)")

// Compile and render with Metal backend to WAV
let mResult = try CompilationPipeline.compile(
  graph: g, backend: .metal, options: .init(frameCount: 128, debug: false))
for kernel in mResult.kernels {
  print(kernel.source)

}
let mRuntime = try MetalCompiledKernel(
  kernels: mResult.kernels, cellAllocations: mResult.cellAllocations)
let mURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
  .appendingPathComponent("biquad_metal.wav")
try mRuntime.writeWAV(to: mURL, seconds: seconds, sampleRate: sampleRate, volumeScale: 1.0)
print("Wrote Metal WAV to \(mURL.path)")
