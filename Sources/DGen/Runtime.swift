import Foundation
import AVFoundation
import Metal


public protocol CompiledKernelRuntime {
    func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int, volumeScale: Float)
    func runAndPlay(durationSeconds: Double , sampleRate: Double , channels: Int, volumeScale: Float ) throws
}

public class CCompiledKernel: CompiledKernelRuntime {
    public let source: String
    public let symbolName: String = "process"

    private var dylibHandle: UnsafeMutableRawPointer? = nil
    private var processFn: (@convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, Int32) -> Void)?

    public init(source: String) {
        self.source = source
    }

    public func compileAndLoad() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let cFile = tmpDir.appendingPathComponent("kernel.c")
        let dylibFile = tmpDir.appendingPathComponent("libkernel.dylib")

        try source.write(to: cFile, atomically: true, encoding: .utf8)

        let compile = Process()
        compile.launchPath = "/usr/bin/clang"
        compile.arguments = [
            "-O3", "-march=armv8-a", "-fPIC", "-shared",
            "-o", dylibFile.path, cFile.path
        ]
        compile.launch()
        compile.waitUntilExit()

        guard compile.terminationStatus == 0 else {
            throw NSError(domain: "CompileError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to compile kernel"])
        }

        dylibHandle = dlopen(dylibFile.path, RTLD_NOW)
        guard let handle = dylibHandle else {
            throw NSError(domain: "DLError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load .dylib"])
        }

        guard let sym = dlsym(handle, symbolName) else {
            throw NSError(domain: "DLError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Symbol \(symbolName) not found"])
        }

        processFn = unsafeBitCast(sym, to: (@convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, Int32) -> Void).self)
    }

    public func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int, volumeScale: Float = 1.0) {
        guard let fn = processFn else {
            fatalError("Kernel not compiled/loaded")
        }
        fn(outputs, inputs, Int32(frameCount))
        
        // Apply volume scaling
        if volumeScale != 1.0 {
            for i in 0..<frameCount {
                outputs[i] *= volumeScale
            }
        }
    }

    public func runAndPlay(durationSeconds: Double, sampleRate: Double, channels: Int, volumeScale: Float = 1.0) throws {
        guard let fn = processFn else {
            fatalError("Kernel not compiled/loaded")
        }

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(channels))!
        let engine = AVAudioEngine()

        let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList -> OSStatus in
            let bufferList = UnsafeMutableAudioBufferListPointer(audioBufferList)

            if channels == 1 {
                let outBuffer = bufferList[0].mData!.assumingMemoryBound(to: Float.self)
                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: outBuffer, inputs: input.baseAddress!, frameCount: Int(frameCount), volumeScale: volumeScale)
                }
            } else {
                let interleaved = UnsafeMutablePointer<Float>.allocate(capacity: Int(frameCount) * channels)
                defer { interleaved.deallocate() }

                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: interleaved, inputs: input.baseAddress!, frameCount: Int(frameCount), volumeScale: volumeScale)
                }

                for ch in 0..<channels {
                    let chBuf = bufferList[ch].mData!.assumingMemoryBound(to: Float.self)
                    for frame in 0..<Int(frameCount) {
                        chBuf[frame] = interleaved[frame * channels + ch]
                    }
                }
            }

            return noErr
        }

        engine.attach(sourceNode)
        engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
        try engine.start()

        print("🟢 Playing for \(durationSeconds) seconds...")
        Thread.sleep(forTimeInterval: durationSeconds)
        engine.stop()
    }
}

public class MetalCompiledKernel: CompiledKernelRuntime {
    public let kernels: [CompiledKernel]
    private let device: MTLDevice
    private let library: MTLLibrary
    private let commandQueue: MTLCommandQueue
    private var bufferPool: [String: MTLBuffer] = [:]
    private var functions: [MTLFunction] = []
    
    public init(kernels: [CompiledKernel]) throws {
        self.kernels = kernels
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal device"])
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = commandQueue
        
        // Combine all kernel sources and compile
        let combinedSource = kernels.map { $0.source }.joined(separator: "\n\n")
        do {
            self.library = try device.makeLibrary(source: combinedSource, options: nil)
        } catch {
            print("Metal compilation error:")
            print("=== Combined Metal Source ===")
            print(combinedSource)
            print("==============================")
            throw error
        }
        
        // Get function references
        for kernel in kernels {
            guard let function = library.makeFunction(name: kernel.name) else {
                throw NSError(domain: "MetalError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Function \(kernel.name) not found"])
            }
            functions.append(function)
        }
        
        try initializeBuffers()
    }
    
    private func initializeBuffers() throws {
        // Collect all unique buffer names across kernels
        let allBufferNames = Set(kernels.flatMap { $0.buffers })
        
        // Create MTLBuffers for each unique buffer
        for bufferName in allBufferNames {
            // Memory buffer needs to be larger to match C implementation (512 floats)
            let bufferSize = bufferName == "memory" ? 512 * MemoryLayout<Float>.size : 128 * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw NSError(domain: "MetalError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer \(bufferName)"])
            }
            
            // Initialize buffer contents to zero
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
                bufferContents[i] = 0.0
            }
            
            bufferPool[bufferName] = buffer
        }
        
        // Create frameCount buffer for scalar kernels
        guard let frameCountBuffer = device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared) else {
            throw NSError(domain: "MetalError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create frameCount buffer"])
        }
        bufferPool["frameCount"] = frameCountBuffer
    }
    
    public func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int, volumeScale: Float = 1.0) {
        // Update frameCount buffer with current frameCount value
        if let frameCountBuffer = bufferPool["frameCount"] {
            let frameCountPtr = frameCountBuffer.contents().assumingMemoryBound(to: Int32.self)
            frameCountPtr[0] = Int32(frameCount)
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // Execute kernels in sequence
        for (index, kernel) in kernels.enumerated() {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { continue }
            
            let function = functions[index]
            guard let pipelineState = try? device.makeComputePipelineState(function: function) else { continue }
            
            computeEncoder.setComputePipelineState(pipelineState)
            
            // Bind buffers based on kernel.buffers array
            for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
                if let buffer = bufferPool[bufferName] {
                    computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)
                }
            }
            
            // Configure thread groups
            let threadsPerGroup = MTLSize(width: kernel.threadGroupSize, height: 1, depth: 1)
            let numThreadGroups = MTLSize(width: (frameCount + kernel.threadGroupSize - 1) / kernel.threadGroupSize, height: 1, depth: 1)
            
            computeEncoder.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy results back to outputs
        copyResultsToOutputs(outputs: outputs, frameCount: frameCount, volumeScale: volumeScale)
    }
    
    private func copyResultsToOutputs(outputs: UnsafeMutablePointer<Float>, frameCount: Int, volumeScale: Float) {
        // Find output buffers and copy their contents
        
        // First check for dedicated "outputs" buffer
        if let outputBuffer = bufferPool["outputs"] {
            let bufferContents = outputBuffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<min(frameCount, 128) {
                outputs[i] = bufferContents[i] * volumeScale
            }
            return
        }
        
        // Fallback to other buffers if no dedicated outputs buffer
        for kernel in kernels {
            for bufferName in kernel.buffers {
                // Check if this buffer contains output data
                if bufferName.hasPrefix("t") || bufferName == "memory" {
                    if let buffer = bufferPool[bufferName] {
                        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
                        // Copy the relevant portion to outputs with volume scaling
                        for i in 0..<min(frameCount, 128) {
                            outputs[i] = bufferContents[i] * volumeScale
                        }
                        break // Take first matching buffer as output
                    }
                }
            }
        }
    }
    
    public func runAndPlay(durationSeconds: Double, sampleRate: Double, channels: Int, volumeScale: Float = 1.0) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(channels))!
        let engine = AVAudioEngine()
        
        let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList -> OSStatus in
            let bufferList = UnsafeMutableAudioBufferListPointer(audioBufferList)
            
            // Debug: Check if frameCount matches what we compiled for
            if frameCount != 128 {
                print("⚠️  Metal runAndPlay: AVAudioEngine requested frameCount=\(frameCount), but kernels compiled for 128!")
            }
            
            if channels == 1 {
                let outBuffer = bufferList[0].mData!.assumingMemoryBound(to: Float.self)
                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: outBuffer, inputs: input.baseAddress!, frameCount: Int(frameCount), volumeScale: volumeScale)
                }
            } else {
                let interleaved = UnsafeMutablePointer<Float>.allocate(capacity: Int(frameCount) * channels)
                defer { interleaved.deallocate() }
                
                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: interleaved, inputs: input.baseAddress!, frameCount: Int(frameCount), volumeScale: volumeScale)
                }
                
                for ch in 0..<channels {
                    let chBuf = bufferList[ch].mData!.assumingMemoryBound(to: Float.self)
                    for frame in 0..<Int(frameCount) {
                        chBuf[frame] = interleaved[frame * channels + ch]
                    }
                }
            }
            
            return noErr
        }
        
        engine.attach(sourceNode)
        engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
        try engine.start()
        
        print("🟢 Playing Metal kernels for \(durationSeconds) seconds...")
        Thread.sleep(forTimeInterval: durationSeconds)
        engine.stop()
    }
    
    // Helper method to read buffer contents for backpropagation
    public func readBuffer(named: String) -> [Float]? {
        guard let buffer = bufferPool[named] else { return nil }
        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
        let count = named == "memory" ? 512 : 128
        return Array(UnsafeBufferPointer(start: bufferContents, count: count))
    }
    
    // Debug method to print intermediate buffer states
    public func debugBufferStates() {
        print("=== METAL BUFFER DEBUG ===")
        for (name, buffer) in bufferPool.sorted(by: { $0.key < $1.key }) {
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            let count = name == "memory" ? 512 : 128
            let values = Array(UnsafeBufferPointer(start: bufferContents, count: count))
            
            // Show first 10 and last 10 values for large buffers
            if count > 20 {
                let first10 = Array(values.prefix(10))
                let last10 = Array(values.suffix(10))
                print("\(name): [\(first10.map { String(format: "%.3f", $0) }.joined(separator: ", "))...] (length: \(count))")
                if name == "memory" {
                    // Show specific memory indices used in kernels
                    print("  memory[0]: \(String(format: "%.6f", values[0]))")
                    print("  memory[1]: \(String(format: "%.6f", values[1]))")  
                    print("  memory[2]: \(String(format: "%.6f", values[2]))")
                }
            } else {
                print("\(name): [\(values.map { String(format: "%.3f", $0) }.joined(separator: ", "))]")
            }
        }
        print("=========================")
    }
}
