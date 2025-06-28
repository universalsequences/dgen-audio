import Foundation
import AVFoundation
import Metal


public protocol CompiledKernelRuntime {
    func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int)
    func runAndPlay(durationSeconds: Double , sampleRate: Double , channels: Int ) throws
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

    public func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int) {
        guard let fn = processFn else {
            fatalError("Kernel not compiled/loaded")
        }
        fn(outputs, inputs, Int32(frameCount))
    }

    public func runAndPlay(durationSeconds: Double, sampleRate: Double, channels: Int) throws {
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
                    fn(outBuffer, input.baseAddress!, Int32(frameCount))
                }
            } else {
                let interleaved = UnsafeMutablePointer<Float>.allocate(capacity: Int(frameCount) * channels)
                defer { interleaved.deallocate() }

                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    fn(interleaved, input.baseAddress!, Int32(frameCount))
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
            let bufferSize = 128 * MemoryLayout<Float>.size // Match C implementation
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw NSError(domain: "MetalError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer \(bufferName)"])
            }
            bufferPool[bufferName] = buffer
        }
    }
    
    public func run(outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int) {
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
        copyResultsToOutputs(outputs: outputs, frameCount: frameCount)
    }
    
    private func copyResultsToOutputs(outputs: UnsafeMutablePointer<Float>, frameCount: Int) {
        // Find output buffers and copy their contents
        for kernel in kernels {
            for bufferName in kernel.buffers {
                // Check if this buffer contains output data
                if bufferName.hasPrefix("t") || bufferName == "memory" {
                    if let buffer = bufferPool[bufferName] {
                        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
                        // Copy the relevant portion to outputs
                        for i in 0..<min(frameCount, 128) {
                            outputs[i] = bufferContents[i]
                        }
                        break // Take first matching buffer as output
                    }
                }
            }
        }
    }
    
    public func runAndPlay(durationSeconds: Double, sampleRate: Double, channels: Int) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(channels))!
        let engine = AVAudioEngine()
        
        let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList -> OSStatus in
            let bufferList = UnsafeMutableAudioBufferListPointer(audioBufferList)
            
            if channels == 1 {
                let outBuffer = bufferList[0].mData!.assumingMemoryBound(to: Float.self)
                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: outBuffer, inputs: input.baseAddress!, frameCount: Int(frameCount))
                }
            } else {
                let interleaved = UnsafeMutablePointer<Float>.allocate(capacity: Int(frameCount) * channels)
                defer { interleaved.deallocate() }
                
                let silentInput = [Float](repeating: 0, count: Int(frameCount))
                silentInput.withUnsafeBufferPointer { input in
                    self.run(outputs: interleaved, inputs: input.baseAddress!, frameCount: Int(frameCount))
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
        return Array(UnsafeBufferPointer(start: bufferContents, count: 128))
    }
}
