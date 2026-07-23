public typealias TensorID = Int
public typealias Shape = [Int]

@frozen
public struct DGenProcessContextV1 {
  public var abiVersion: UInt32
  public var structSize: UInt32
  public var sampleRate: Float
  public var reserved: UInt32

  public init(sampleRate: Float) {
    abiVersion = 1
    structSize = UInt32(MemoryLayout<Self>.size)
    self.sampleRate = sampleRate
    reserved = 0
  }
}

public typealias CProcessFunction = @convention(c) (
  UnsafePointer<UnsafePointer<Float>?>?,
  UnsafePointer<UnsafeMutablePointer<Float>?>?,
  UInt32,
  UnsafeMutableRawPointer?,
  UnsafeRawPointer?,
  UnsafeRawPointer?
) -> Void

public enum ValueShape: Equatable {
  case scalar
  case tensor(Shape)
}

public enum Temporality: Equatable {
  case frameBased                              // runs every frame (phasor, input, audio processing)
  case hopBased(hopSize: Int, counterNode: NodeID) // runs every N frames (FFT output, spectral ops)
  case static_                                 // runs once (wavetable generation, constants)
}
