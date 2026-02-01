import DGen

public func writeKernelsToDisk2(_ result: CompilationResult, _ filename: String) {
  // Write kernels to disk for inspection
  let allKernels = result.kernels.enumerated().map {
    "// KERNEL \($0.offset)\n\n// ThreadCountScale \($0.element.threadCountScale)\n ThreadGroupSize \($0.element.threadGroupSize)\n// ThreadCount \($0.element.threadCount)\n\($0.element.source)"
  }.joined(separator: "\n\n")
  try! allKernels.write(
    toFile: filename, atomically: true,
    encoding: .utf8)
  print("!!!Wrote kernels to \(filename)")
}
