import DGen

public func writeKernelsToDisk(_ result: CompilationResult, _ filename: String) {
  // Write kernels to disk for inspection
  let allKernels = result.kernels.enumerated().map {
    """
    // KERNEL \($0.offset)
    // FrameOrder: \($0.element.frameOrder)
    // ThreadCountScale \($0.element.threadCountScale)
    // ThreadGroupSize \($0.element.threadGroupSize)
    // ThreadCount \($0.element.threadCount)
    \($0.element.source)
    """
  }.joined(separator: "\n\n")
  try! allKernels.write(
    toFile: filename, atomically: true,
    encoding: .utf8)
  print("!!!Wrote kernels to \(filename)")
}
