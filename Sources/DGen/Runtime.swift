import AVFoundation
import AudioToolbox  // kAudioUnitProperty_MaximumFramesPerSlice
import Foundation
import Metal

public typealias AudioLevelCallback = (UnsafePointer<Float>, Int) -> Void

let MIN_FRAME_COUNT = 128

public protocol CompiledKernelRuntime {
    var cellAllocations: CellAllocations { get }
    func run(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
        volumeScale: Float)
    func runAndPlay(engine: AVAudioEngine, sampleRate: Double, channels: Int, volumeScale: Float)
        throws -> AVAudioNode
    func runAndPlay(
        engine: AVAudioEngine, sampleRate: Double, channels: Int, volumeScale: Float,
        levelCallback: AudioLevelCallback?
    ) throws -> AVAudioNode
    func setParamValue(cellId: CellID, value: Float)
    func cleanup()

    // New memory management methods for per-node memory
    func getMemorySize() -> Int
    func allocateNodeMemory() -> UnsafeMutableRawPointer?
    func deallocateNodeMemory(_ memory: UnsafeMutableRawPointer)
    func runWithMemory(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, 
        memory: UnsafeMutableRawPointer, frameCount: Int
    )
}

// MARK: - Offline rendering to WAV (shared helper)

extension CompiledKernelRuntime {
    // Render 'seconds' of audio to a mono 32-bit float WAV file using 128-frame blocks
    public func writeWAV(to url: URL, seconds: Double, sampleRate: Double, volumeScale: Float = 1.0) throws {
        let totalFrames = Int(seconds * sampleRate)
        let block = MIN_FRAME_COUNT
        var pcm = [Float](repeating: 0, count: totalFrames)
        var zeroIn = [Float](repeating: 0, count: block)

        var rendered = 0
        var blockIndex = 0
        while rendered < totalFrames {
            let n = min(block, totalFrames - rendered)
            pcm.withUnsafeMutableBufferPointer { outPtr in
                zeroIn.withUnsafeBufferPointer { inPtr in
                    self.run(
                        outputs: outPtr.baseAddress!.advanced(by: rendered),
                        inputs: inPtr.baseAddress!,
                        frameCount: n,
                        volumeScale: volumeScale
                    )
                }
            }
            // Debug: dump memory after each run call to verify accumulation
            if blockIndex < 5 {
            if let c = self as? CCompiledKernel {
                c.debugPrintMemory(tag: "block_\(blockIndex)")
            } else if let m = self as? MetalCompiledKernel {
                m.debugPrintMemory(tag: "block_\(blockIndex)")
            }
            }
            rendered += n
            blockIndex += 1
        }

        // Write 32-bit float WAV (mono)
        let bytesPerSample = 4
        let numChannels = 1
        let byteRate = Int(sampleRate) * numChannels * bytesPerSample
        let blockAlign = numChannels * bytesPerSample
        let dataBytes = totalFrames * numChannels * bytesPerSample
        let riffSize = 36 + dataBytes

        var data = Data()
        func append(_ s: String) { data.append(s.data(using: .ascii)!) }
        func appendUInt32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        func appendUInt16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        append("RIFF")
        appendUInt32(UInt32(riffSize))
        append("WAVE")
        append("fmt ")
        appendUInt32(16)                     // PCM fmt chunk size
        appendUInt16(3)                      // AudioFormat 3 = IEEE float
        appendUInt16(UInt16(numChannels))
        appendUInt32(UInt32(sampleRate))
        appendUInt32(UInt32(byteRate))
        appendUInt16(UInt16(blockAlign))
        appendUInt16(32)                     // bits per sample
        append("data")
        appendUInt32(UInt32(dataBytes))

        // Samples
        pcm.withUnsafeBytes { raw in
            data.append(contentsOf: raw.bindMemory(to: UInt8.self))
        }

        try data.write(to: url, options: .atomic)
    }
}

// MARK: - Debug helpers to inspect runtime memory after each block

extension CCompiledKernel {
    func debugPrintMemory(tag: String = "") {
        guard let mem = nodeMemory else { print("[C] 🧠 no nodeMemory"); return }
        let p = mem.assumingMemoryBound(to: Float.self)
        let total = max(0, cellAllocations.totalMemorySlots)
        let limit = min(total, 32)
        var parts: [String] = []
        parts.reserveCapacity(limit)
        for i in 0..<limit { parts.append(String(format: "%.6f", p[i])) }
        print("[C] 🧠 memory after \(tag): [\(parts.joined(separator: ", "))]")
    }
}

extension MetalCompiledKernel {
    func debugPrintMemory(tag: String = "") {
        guard let buf = bufferPool["memory"] else { print("[Metal] 🧠 no memory buffer"); return }
        let total = max(kernels.first?.memorySize ?? 0, cellAllocations.totalMemorySlots)
        let limit = min(total, 32)
        let ptr = buf.contents().assumingMemoryBound(to: Float.self)
        var parts: [String] = []
        parts.reserveCapacity(limit)
        for i in 0..<limit { parts.append(String(format: "%.6f", ptr[i])) }
        print("[Metal] 🧠 memory after \(tag): [\(parts.joined(separator: ", "))]")
    }
}

public class CCompiledKernel: CompiledKernelRuntime {
    public let source: String
    public let symbolName: String = "process"
    public let setParamValueSymbolName: String = "setParamValue"

    var outputBuffer: [Float] = []
    var scratchBuffer: [Float] = []
    var inputAccumulator: [Float] = []
    public var ringRenderer: AudioRingRenderer?

    public let cellAllocations: CellAllocations
    private let memorySize: Int
    private var dylibHandle: UnsafeMutableRawPointer? = nil
    private var nodeMemory: UnsafeMutableRawPointer? = nil
    // Match audiograph's actual KernelFn signature (based on Swift binding)
    private var processFn:
        (
            @convention(c) (
                UnsafePointer<UnsafeMutablePointer<Float>?>?,
                UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
            ) -> Void
        )?

    private var setParamValueFn: (@convention(c) (Int32, Float) -> Void)?

    public init(source: String, cellAllocations: CellAllocations, memorySize: Int = 1024) {
        self.source = source
        self.cellAllocations = cellAllocations
        self.memorySize = memorySize
    }

    public func compileAndLoad() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        // Use timestamp to force fresh compilation
        let timestamp = String(Int(Date().timeIntervalSince1970 * 1000))
        let cFile = tmpDir.appendingPathComponent("kernel_\(timestamp).c")
        let dylibFile = tmpDir.appendingPathComponent("libkernel_\(timestamp).dylib")

        try source.write(to: cFile, atomically: true, encoding: .utf8)

        let compile = Process()
        compile.launchPath = "/usr/bin/clang"
        let arguments = [
            "-O3", "-march=armv8-a", "-fPIC", "-shared",
            "-framework", "Accelerate",
            "-std=c11",  // Ensure C11 standard, not C++
            "-x", "c",  // Explicitly treat as C source, not C++
            "-o", dylibFile.path, cFile.path,
        ]

        print("clang \(arguments.joined(separator: " "))")

        // Capture stderr to see any compilation errors or warnings
        let errorPipe = Pipe()
        compile.standardError = errorPipe

        compile.arguments = arguments
        compile.launch()
        compile.waitUntilExit()

        // Read any error output
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let errorOutput = String(data: errorData, encoding: .utf8) ?? ""

        if !errorOutput.isEmpty {
            print("⚠️ CLANG STDERR OUTPUT:")
            print(errorOutput)
        }

        guard compile.terminationStatus == 0 else {
            print("❌ CLANG COMPILATION FAILED with status \(compile.terminationStatus)")
            if !errorOutput.isEmpty {
                print("Error details: \(errorOutput)")
            }
            throw NSError(
                domain: "CompileError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to compile kernel: \(errorOutput)"])
        }

        dylibHandle = dlopen(dylibFile.path, RTLD_NOW)
        guard let handle = dylibHandle else {
            let error = String(cString: dlerror())
            print("❌ DYLIB LOADING FAILED: \(error)")
            throw NSError(
                domain: "DLError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load .dylib: \(error)"])
        }
        print("🔍 Looking for symbols in dylib...")
        print("🔍 Available symbols:")
        // List some symbols to debug what's actually in the dylib
        let processSymAddr = dlsym(handle, symbolName)
        let constantKernelSymAddr = dlsym(handle, "constant_kernel")
        let setParamSymAddr = dlsym(handle, setParamValueSymbolName)
        print("🔍   process symbol: \(processSymAddr == nil ? "NOT FOUND" : "\(processSymAddr!)")")
        print(
            "🔍   constant_kernel symbol: \(constantKernelSymAddr == nil ? "NOT FOUND" : "\(constantKernelSymAddr!)")"
        )
        print(
            "🔍   setParamValue symbol: \(setParamSymAddr == nil ? "NOT FOUND" : "\(setParamSymAddr!)")"
        )

        // PRODUCTION: Use the real DGen-generated process function
        print("✅ Using real DGen-generated process function")

        // CRITICAL: Log the raw symbol address before casting
        print("🔍 RAW SYMBOL ADDRESS: \(processSymAddr!)")

        processFn = unsafeBitCast(
            processSymAddr!,
            to: (@convention(c) (
                UnsafePointer<UnsafeMutablePointer<Float>?>?,
                UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
            ) -> Void)
            .self)

        // CRITICAL: Log the function pointer after casting
        let functionPointer = unsafeBitCast(processFn!, to: UnsafeRawPointer.self)
        print("🔍 FUNCTION POINTER AFTER CAST: \(functionPointer)")

        print("✅ Real DGen process function loaded successfully")

        guard let paramSym = dlsym(handle, setParamValueSymbolName) else {
            let error = String(cString: dlerror())
            print("❌ SYMBOL NOT FOUND: \(setParamValueSymbolName) - \(error)")
            throw NSError(
                domain: "DLError", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Symbol \(setParamValueSymbolName) not found: \(error)"
                ])

        }

        setParamValueFn = unsafeBitCast(
            paramSym,
            to: (@convention(c) (Int32, Float) -> Void)
                .self)

        // Allocate persistent node memory once here based on effective size
        if nodeMemory == nil {
            let slots = max(self.memorySize, self.cellAllocations.totalMemorySlots)
            let byteSize = max(1, slots) * MemoryLayout<Float>.size
            nodeMemory = malloc(byteSize)
            if let mem = nodeMemory { memset(mem, 0, byteSize) }
            print("🧠 Allocated C node memory: \(slots) floats")
        }
    }

    public func run(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
        volumeScale: Float = 1.0
    ) {
        if processFn == nil {
            // Attempt to compile and load on-demand (allocates node memory once)
            do { try compileAndLoad() } catch {
                print("❌ Failed to compile/load kernel on-demand: \(error)")
                return
            }
        }
        guard let process = processFn else {
            print("⚠️ Process function not loaded")
            return
        }
        guard let statePtr = nodeMemory else {
            print("⚠️ No node memory allocated")
            return
        }

        // Create channel arrays (mono) for C signature
        let outputChannels: [UnsafeMutablePointer<Float>?] = [outputs]
        let inputChannels: [UnsafeMutablePointer<Float>?] = [UnsafeMutablePointer(mutating: inputs)]

        outputChannels.withUnsafeBufferPointer { outPtr in
            inputChannels.withUnsafeBufferPointer { inPtr in
                process(inPtr.baseAddress, outPtr.baseAddress, Int32(frameCount), statePtr)
            }
        }
    }

    public func setParamValue(cellId: CellID, value: Float) {
        guard let fn = setParamValueFn else {
            fatalError("Kernel does not have setParamValue compiled")
        }

        fn(Int32(cellId), value)
    }

    public func runAndPlay(
        engine: AVAudioEngine,
        sampleRate: Double,
        channels: Int,
        volumeScale: Float = 1.0,
        levelCallback: AudioLevelCallback? = nil
    ) throws -> AVAudioNode {
        if processFn == nil { try compileAndLoad() }

        // Shared ring renderer that always requests 128-frame blocks
        let zero = [Float](repeating: 0, count: MIN_FRAME_COUNT)
        let ring = AudioRingRenderer(channels: channels, levelCallback: levelCallback) {
            out, frames in
            zero.withUnsafeBufferPointer { inPtr in
                self.run(
                    outputs: out, inputs: inPtr.baseAddress!, frameCount: frames,
                    volumeScale: volumeScale)
            }
            // Debug: print memory head every 100 blocks
            if let mem = self.nodeMemory {
                let p = mem.assumingMemoryBound(to: Float.self)
                let head = (0..<min(8, self.cellAllocations.totalMemorySlots)).map { String(format: "%.6f", p[$0]) }.joined(separator: ", ")
                print("🧠 C state head: [\(head)]")
            }
        }
        // Retain ring renderer for the lifetime of playback
        self.ringRenderer = ring
        let sourceNode = ring.makeSourceNode(sampleRate: sampleRate)
        engine.attach(sourceNode)
        engine.connect(
            sourceNode, to: engine.mainMixerNode, format: sourceNode.outputFormat(forBus: 0))
        try engine.start()
        return sourceNode
    }

    // Default implementation for backwards compatibility
    public func runAndPlay(
        engine: AVAudioEngine, sampleRate: Double, channels: Int, volumeScale: Float
    ) throws -> AVAudioNode {
        return try runAndPlay(
            engine: engine, sampleRate: sampleRate, channels: channels, volumeScale: volumeScale,
            levelCallback: nil)
    }

    public func cleanup() {
        if let handle = dylibHandle {
            dlclose(handle)
            dylibHandle = nil
            processFn = nil
        }

        if let mem = nodeMemory {
            free(mem)
            nodeMemory = nil
        }

        // Clear ring buffers to prevent audio artifacts when loading new graphs
        outputBuffer.removeAll()
        scratchBuffer.removeAll()
        inputAccumulator.removeAll()
    }

    // MARK: - New Memory Management Methods

    public func getMemorySize() -> Int {
        return memorySize
    }

    public func allocateNodeMemory() -> UnsafeMutableRawPointer? {
        let byteSize = memorySize * MemoryLayout<Float>.size
        let ptr = malloc(byteSize)
        if let ptr = ptr {
            // Zero initialize the memory
            memset(ptr, 0, byteSize)
        }
        return ptr
    }

    public func deallocateNodeMemory(_ memory: UnsafeMutableRawPointer) {
        free(memory)
    }

    public func runWithMemory(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>,
        memory: UnsafeMutableRawPointer, frameCount: Int
    ) {
        guard let process = processFn else {
            print("⚠️ Process function not loaded")
            return
        }

        // Create channel arrays for audiograph signature compatibility
        let outputChannels: [UnsafeMutablePointer<Float>?] = [outputs]
        let inputChannels: [UnsafeMutablePointer<Float>?] = [UnsafeMutablePointer(mutating: inputs)]

        outputChannels.withUnsafeBufferPointer { outPtr in
            inputChannels.withUnsafeBufferPointer { inPtr in
                process(inPtr.baseAddress, outPtr.baseAddress, Int32(frameCount), memory)
            }
        }
    }

    public func getProcessFunction() -> (
        @convention(c) (
            UnsafePointer<UnsafeMutablePointer<Float>?>?,
            UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
        ) -> Void
    )? {
        return processFn
    }

    // Get the typed function directly (preferred method, like working example)
    public func getKernelFunction() -> (
        @convention(c) (
            UnsafePointer<UnsafeMutablePointer<Float>?>?,
            UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
        ) -> Void
    )? {
        return processFn
    }

    // Get the already-loaded function pointer as raw symbol for audiograph
    public func getRawSymbolPointer() -> UnsafeRawPointer? {
        guard let fn = processFn else {
            print("❌ No process function loaded")
            return nil
        }

        // Return the already-cast function as raw pointer to avoid double-casting
        let rawPointer = unsafeBitCast(fn, to: UnsafeRawPointer.self)
        print("🔧 Returning already-loaded function as raw symbol: \(rawPointer)")
        return rawPointer
    }

    deinit {
        cleanup()
    }
}

public class MetalCompiledKernel: CompiledKernelRuntime {
    public let kernels: [CompiledKernel]
    public let cellAllocations: CellAllocations
    private let device: MTLDevice
    private let library: MTLLibrary
    public var ringRenderer: AudioRingRenderer?
    private let commandQueue: MTLCommandQueue
    private var bufferPool: [String: MTLBuffer] = [:]  // shared source of buffers used by all kernels
    private var functions: [MTLFunction] = []  // one per kernel

    public init(kernels: [CompiledKernel], cellAllocations: CellAllocations) throws {
        self.kernels = kernels
        self.cellAllocations = cellAllocations

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(
                domain: "MetalError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal device"])
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(
                domain: "MetalError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
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
                throw NSError(
                    domain: "MetalError", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Function \(kernel.name) not found"])
            }
            functions.append(function)
        }

        try initializeBuffers()
    }

    private func initializeBuffers() throws {
        // Collect all unique buffer names across kernels
        let allBufferNames = Set(kernels.flatMap { $0.buffers })

        // Create MTLBuffers for each unique buffer
        // Use larger buffer size to handle varying frameCount from AVAudioEngine
        let maxFrameCount = 2048  // Handle up to 2048 frames per callback
        for bufferName in allBufferNames {
            // Memory buffer uses effective size, others use maxFrameCount
            let effectiveMem = max(kernels.first?.memorySize ?? 1024, cellAllocations.totalMemorySlots)
            let elementCount = bufferName == "memory" ? effectiveMem : maxFrameCount
            let bufferSize = elementCount * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
            else {
                throw NSError(
                    domain: "MetalError", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer \(bufferName)"])
            }

            // Initialize buffer contents to zero
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
                bufferContents[i] = 0.0
            }

            bufferPool[bufferName] = buffer
        }

        // Create frameCount buffer for kernels
        guard
            let frameCountBuffer = device.makeBuffer(
                length: MemoryLayout<Int32>.size, options: .storageModeShared)
        else {
            throw NSError(
                domain: "MetalError", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create frameCount buffer"])
        }
        bufferPool["frameCount"] = frameCountBuffer
    }

    public func setParamValue(cellId: CellID, value: Float) {
        // TODO - implement
    }

    public func run(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
        volumeScale: Float = 1.0
    ) {
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
            guard let pipelineState = try? device.makeComputePipelineState(function: function)
            else { continue }

            computeEncoder.setComputePipelineState(pipelineState)

            // Detect segmented kernel by presence of segmentLen buffer
            let isSegmented = kernel.buffers.contains("segmentLen")
            let segmentCapacity = MIN_FRAME_COUNT

            if isSegmented {
                // We'll reuse the same encoder for multiple segment dispatches
                var base = 0
                while base < frameCount {
                    let thisLen = min(segmentCapacity, frameCount - base)

                    // Update segmentLen
                    if let segBuf = bufferPool["segmentLen"] {
                        let segPtr = segBuf.contents().assumingMemoryBound(to: Int32.self)
                        segPtr[0] = Int32(thisLen)
                    }
                    // Update segmentBase (in frames)
                    if let segBaseBuf = bufferPool["segmentBase"] {
                        let basePtr = segBaseBuf.contents().assumingMemoryBound(to: Int32.self)
                        basePtr[0] = Int32(base)
                    }

                    // Bind buffers with per-segment offsets
                    for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
                        guard let buffer = bufferPool[bufferName] else { continue }
                        // Offset per-frame buffers (t*, outputs); never offset memory or frameCount/segmentLen
                        let needsOffset = (bufferName.hasPrefix("t") || bufferName == "outputs")
                        let byteOffset = needsOffset ? base * MemoryLayout<Float>.size : 0
                        computeEncoder.setBuffer(buffer, offset: byteOffset, index: bufferIndex)
                    }

                    let threadsPerGroup = MTLSize(width: thisLen, height: 1, depth: 1)
                    let numThreadGroups = MTLSize(width: 1, height: 1, depth: 1)
                    computeEncoder.dispatchThreadgroups(
                        numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
                    base += thisLen
                }
                computeEncoder.endEncoding()
            } else {
                // Non-segmented kernel: single dispatch as before
                for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
                    if let buffer = bufferPool[bufferName] {
                        computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)
                    }
                }

                let threadGroupSize = kernel.threadGroupSize ?? min(frameCount, MIN_FRAME_COUNT)
                let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
                let numThreadGroups = MTLSize(
                    width: (frameCount + threadGroupSize - 1) / threadGroupSize, height: 1, depth: 1
                )

                computeEncoder.dispatchThreadgroups(
                    numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
                computeEncoder.endEncoding()
            }
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Copy results back to outputs
        copyResultsToOutputs(outputs: outputs, frameCount: frameCount, volumeScale: volumeScale)
    }

    private func copyResultsToOutputs(
        outputs: UnsafeMutablePointer<Float>, frameCount: Int, volumeScale: Float
    ) {
        // Find output buffers and copy their contents

        // First check for dedicated "outputs" buffer
        if let outputBuffer = bufferPool["outputs"] {
            let bufferContents = outputBuffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<frameCount {
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
                        for i in 0..<frameCount {
                            outputs[i] = bufferContents[i] * volumeScale
                        }
                        break  // Take first matching buffer as output
                    }
                }
            }
        }
    }

    public func runAndPlay(
        engine: AVAudioEngine,
        sampleRate: Double,
        channels: Int,
        volumeScale: Float = 1.0,
        levelCallback: AudioLevelCallback? = nil
    ) throws -> AVAudioNode {
        // Ring-buffered audio callback using fixed-size 128-frame blocks
        let zero = [Float](repeating: 0, count: MIN_FRAME_COUNT)
        let ring = AudioRingRenderer(channels: channels, levelCallback: levelCallback) {
            out, frames in
            zero.withUnsafeBufferPointer { inPtr in
                self.run(
                    outputs: out, inputs: inPtr.baseAddress!, frameCount: frames,
                    volumeScale: volumeScale)
            }
        }
        self.ringRenderer = ring
        let sourceNode = ring.makeSourceNode(sampleRate: sampleRate)
        print("attaching source node to engine")
        engine.attach(sourceNode)
        engine.connect(
            sourceNode, to: engine.mainMixerNode, format: sourceNode.outputFormat(forBus: 0))
        print("connecting to engine main mixer node... ()")
        print("engine starting...")
        try engine.start()
        return sourceNode

        //  print("🟢 Playing Metal kernels for \(durationSeconds) seconds …")
        //Thread.sleep(forTimeInterval: durationSeconds)
        //engine.stop()

        //     interleavedScratch?.deallocate()
    }

    // Default implementation for backwards compatibility
    public func runAndPlay(
        engine: AVAudioEngine, sampleRate: Double, channels: Int, volumeScale: Float
    ) throws -> AVAudioNode {
        return try runAndPlay(
            engine: engine, sampleRate: sampleRate, channels: channels, volumeScale: volumeScale,
            levelCallback: nil)
    }

    // Helper method to read buffer contents for backpropagation
    public func readBuffer(named: String) -> [Float]? {
        guard let buffer = bufferPool[named] else { return nil }

        // Special handling for frameCount buffer which stores Int32
        if named == "frameCount" {
            let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
            let intValue = intContents[0]
            return [Float(intValue)]  // Return frameCount as a single-element Float array
        }

        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
        let count = named == "memory" ? 512 : 2048  // Match the allocated size
        return Array(UnsafeBufferPointer(start: bufferContents, count: count))
    }

    // Debug method to print intermediate buffer states
    public func debugBufferStates() {
        print("=== METAL BUFFER DEBUG ===")
        for (name, buffer) in bufferPool.sorted(by: { $0.key < $1.key }) {
            let values: [Float]
            let count: Int

            // Special handling for frameCount buffer which stores Int32
            if name == "frameCount" {
                let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
                let intValue = intContents[0]
                values = [Float(intValue)]
                count = 1
            } else {
                let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
                count = name == "memory" ? 512 : 2048  // Match the allocated size
                values = Array(UnsafeBufferPointer(start: bufferContents, count: count))
            }

            // Show first 10 and last 10 values for large buffers
            if count > 20 {
                let first10 = Array(values.prefix(10))
                let last10 = Array(values.suffix(10))
                print(
                    "\(name): [\(first10.map { String(format: "%.3f", $0) }.joined(separator: ", "))...] (length: \(count))"
                )
                if name == "memory" {
                    // Show specific memory indices used in kernels
                    print("  memory[0]: \(String(format: "%.6f", values[0]))")
                    print("  memory[1]: \(String(format: "%.6f", values[1]))")
                    print("  memory[2]: \(String(format: "%.6f", values[2]))")
                }
            } else {
                print(
                    "\(name): [\(values.map { String(format: "%.3f", $0) }.joined(separator: ", "))]"
                )
            }
        }
        print("=========================")
    }

    public func cleanup() {
        // Reset all buffers to zero
        for (name, buffer) in bufferPool {
            if name == "frameCount" {
                // Special handling for frameCount buffer
                let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
                intContents[0] = 0
            } else {
                // Zero out float buffers
                let bufferSize = name == "memory" ? 512 : 2048
                memset(buffer.contents(), 0, bufferSize * MemoryLayout<Float>.size)
            }
        }

        // Clear function references
        functions.removeAll()

        // Ensure all GPU work is complete by creating and waiting for a command buffer
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }

    // MARK: - New Memory Management Methods

    public func getMemorySize() -> Int {
        // For Metal, memory size is determined by the first kernel
        return kernels.first?.memorySize ?? 1024
    }

    public func allocateNodeMemory() -> UnsafeMutableRawPointer? {
        let byteSize = getMemorySize() * MemoryLayout<Float>.size
        let ptr = malloc(byteSize)
        if let ptr = ptr {
            // Zero initialize the memory
            memset(ptr, 0, byteSize)
        }
        return ptr
    }

    public func deallocateNodeMemory(_ memory: UnsafeMutableRawPointer) {
        free(memory)
    }

    public func runWithMemory(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>,
        memory: UnsafeMutableRawPointer, frameCount: Int
    ) {
        // Metal implementation would need significant changes to support external memory
        // For now, fall back to the existing run method which uses internal buffers
        run(outputs: outputs, inputs: inputs, frameCount: frameCount, volumeScale: 1.0)
        print(
            "⚠️ MetalCompiledKernel.runWithMemory not fully implemented - falling back to internal buffers"
        )
    }

    deinit {
        cleanup()
    }
}
