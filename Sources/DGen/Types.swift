public typealias TensorID = Int
public typealias Shape = [Int]

public typealias CProcessFunction = @convention(c) (
  UnsafePointer<UnsafeMutablePointer<Float>?>?,
  UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?,
  UnsafeMutableRawPointer?
) -> Void

public enum ValueShape: Equatable {
  case scalar
  case tensor(Shape)
}

public enum Temporality: Equatable {
  case frameBased  // runs every frame (phasor, input, audio processing)
  case static_     // runs once (wavetable generation, constants)
}
