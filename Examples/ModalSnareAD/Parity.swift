import DGen
import DGenLazy
import Foundation

func configureRuntime(frames: Int, sampleRate: Float = VillainData.sampleRate) {
  DGenConfig.backend = .metal
  DGenConfig.sampleRate = sampleRate
  DGenConfig.maxFrameCount = frames
  DGenConfig.debug = false
  DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
}

func relativeRMS(_ a: [Float], _ ref: [Float]) -> Double {
  let n = min(a.count, ref.count)
  var e = 0.0, r = 0.0
  for i in 0..<n {
    let d = Double(a[i]) - Double(ref[i])
    e += d * d
    r += Double(ref[i]) * Double(ref[i])
  }
  return (e / max(r, 1e-30)).squareRoot()
}

enum HeadParity {
  static func run(ids: [String], frames: Int, outDir: URL?) throws -> [(String, Double)] {
    configureRuntime(frames: frames)
    let renders = try VillainData.loadRenders(ids: ids)
    let arrays = HeadArrays(renders: renders)
    LazyGraphContext.reset()
    let head = arrays.tensors()
    let time = Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate)
    let y = HeadRenderer.render(head, time: time, sampleRate: VillainData.sampleRate)
    let t0 = Date()
    let flat = try y.realize(frames: frames)
    let dt = Date().timeIntervalSince(t0)
    let per = deinterleave(flat, batch: ids.count)
    var results: [(String, Double)] = []
    for (b, id) in ids.enumerated() {
      let ref = try WAV.read(Paths.stem(id, "head"))
      let err = relativeRMS(per[b], Array(ref.prefix(frames)))
      results.append((id, err))
      if let outDir {
        try WAV.write(outDir.appendingPathComponent("\(id).dgen_head.wav"), per[b])
      }
    }
    print(String(format: "realize B=%d K=%d frames=%d: %.3f s (incl. compile)", ids.count, arrays.K, frames, dt))
    return results
  }
}
