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

public class CCompiledKernel: CompiledKernelRuntime {
    public let source: String
    public let symbolName: String = "process"
    public let setParamValueSymbolName: String = "setParamValue"

    var outputBuffer: [Float] = []
    var scratchBuffer: [Float] = []
    var inputAccumulator: [Float] = []

    public let cellAllocations: CellAllocations
    private let memorySize: Int
    private var dylibHandle: UnsafeMutableRawPointer? = nil
    private var dylibFileURL: URL? = nil
    private var duplicateHandles: [UnsafeMutableRawPointer] = []
    private var duplicateProcessFns:
        [(
            @convention(c) (
                UnsafePointer<UnsafeMutablePointer<Float>?>?,
                UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
            ) -> Void
        )] = []
    // Match audiograph's actual KernelFn signature (based on Swift binding)
    private var processFn:
        (
            @convention(c) (
                UnsafePointer<UnsafeMutablePointer<Float>?>?,
                UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
            ) -> Void
        )?

    private var setParamValueFn: (@convention(c) (Int32, Float) -> Void)?

    public var voiceCellId: Int? = nil

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
            // Optimization and CPU tuning
            "-Ofast",
            "-mcpu=native",
            "-flto=thin",
            // Floating point fast math (ok for DSP if acceptable)
            "-ffast-math",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-ffp-contract=fast",
            // Vectorizer hints (mostly enabled by -Ofast, but explicit here)
            "-fvectorize",
            "-fslp-vectorize",
            "-funroll-loops",
            // Dylib and platform flags
            "-fPIC", "-shared",
            "-framework", "Accelerate",
            // Language & input
            "-std=c11",  // Ensure C11 standard
            "-x", "c",  // Explicitly treat as C source
            // Output
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
        dylibFileURL = dylibFile
        guard let handle = dylibHandle else {
            let error = String(cString: dlerror())
            print("❌ DYLIB LOADING FAILED: \(error)")
            throw NSError(
                domain: "DLError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load .dylib: \(error)"])
        }
        // List some symbols to debug what's actually in the dylib
        let processSymAddr = dlsym(handle, symbolName)
        let constantKernelSymAddr = dlsym(handle, "constant_kernel")
        let setParamSymAddr = dlsym(handle, setParamValueSymbolName)

        processFn = unsafeBitCast(
            processSymAddr!,
            to: (@convention(c) (
                UnsafePointer<UnsafeMutablePointer<Float>?>?,
                UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
            ) -> Void)
            .self)

        // CRITICAL: Log the function pointer after casting
        let functionPointer = unsafeBitCast(processFn!, to: UnsafeRawPointer.self)

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
    }

    public func run(
        outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
        volumeScale: Float = 1.0
    ) {
        fatalError("run() deprecated - use runWithMemory() instead for per-node memory management")
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
        fatalError(
            "runAndPlay() is deprecated - use AudioGraphManager multi-graph processing instead")

        guard processFn != nil else { fatalError("Kernel not compiled/loaded") }

        //------------------------------------------------------------------
        // 1 · Ask the OS for 128-frame I/O buffers (best-effort request)
        //------------------------------------------------------------------

        //------------------------------------------------------------------
        // 2 · Prepare one 128-frame scratch buffer we’ll reuse every callback
        //------------------------------------------------------------------
        let blockSize = MIN_FRAME_COUNT
        let silentInput = [Float](repeating: 0, count: blockSize)

        var interleavedScratch: UnsafeMutablePointer<Float>? = nil
        if channels > 1 {
            interleavedScratch = UnsafeMutablePointer<Float>
                .allocate(capacity: blockSize * channels)
        }

        //------------------------------------------------------------------
        // 3 · Render callback — handles ANY frameCount Core Audio gives us
        //------------------------------------------------------------------
        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: AVAudioChannelCount(channels))!

        let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, abl -> OSStatus in
            let buffers = UnsafeMutableAudioBufferListPointer(abl)
            let intFrameCount = Int(frameCount)

            // Ensure scratch/output buffers exist
            if self.scratchBuffer.count < blockSize * channels {
                self.scratchBuffer = [Float](repeating: 0, count: blockSize * channels)
            }
            if self.outputBuffer.count < intFrameCount * channels {
                self.outputBuffer.reserveCapacity(4096 * channels)
            }

            //------------------------------------------------------------------
            // 1. Generate as many 256-frame blocks as needed to fill demand
            //------------------------------------------------------------------
            while self.outputBuffer.count < intFrameCount * channels {
                self.scratchBuffer.withUnsafeMutableBufferPointer { scratch in
                    silentInput.withUnsafeBufferPointer { zero in
                        self.run(
                            outputs: scratch.baseAddress!,
                            inputs: zero.baseAddress!,
                            frameCount: blockSize,
                            volumeScale: volumeScale
                        )
                    }
                }

                self.outputBuffer.append(contentsOf: self.scratchBuffer[0..<blockSize * channels])
            }

            //------------------------------------------------------------------
            // 2. Copy outputBuffer → Core Audio non-interleaved output
            //------------------------------------------------------------------
            for ch in 0..<channels {
                let dst = buffers[ch].mData!.assumingMemoryBound(to: Float.self)
                for frame in 0..<intFrameCount {
                    dst[frame] = self.outputBuffer[frame * channels + ch]
                }
            }

            //------------------------------------------------------------------
            // 3. Remove used frames
            //------------------------------------------------------------------
            self.outputBuffer.removeFirst(intFrameCount * channels)

            //------------------------------------------------------------------
            // 4. Level callback
            //------------------------------------------------------------------
            if let cb = levelCallback {
                let mono = buffers[0].mData!.assumingMemoryBound(to: Float.self)
                cb(mono, intFrameCount)
            }

            return noErr
        }

        //------------------------------------------------------------------
        // 4 · Spin the engine
        //------------------------------------------------------------------
        engine.attach(sourceNode)
        engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
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
        for h in duplicateHandles { dlclose(h) }
        duplicateHandles.removeAll()
        duplicateProcessFns.removeAll()

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

    /// Return per-voice process functions by duplicating the .dylib to unique paths and dlopen'ing
    public func getProcessFunctions(count: Int) throws -> [(
        @convention(c) (
            UnsafePointer<UnsafeMutablePointer<Float>?>?,
            UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
        ) -> Void
    )] {
        let t0 = CFAbsoluteTimeGetCurrent()
        guard count > 0 else { return [] }
        guard let primary = processFn else {
            throw NSError(
                domain: "DLError", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Kernel not compiled/loaded"])
        }
        if count == 1 {
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            print(
                "getProcessFunctions(count=1): total=\(String(format: "%.3f", elapsed))ms (no copies)"
            )
            return [primary]
        }

        // If we already have enough duplicates, just return cached array
        if duplicateProcessFns.count >= count - 1 {
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            print(
                "getProcessFunctions(count=\(count)): using cached copies, total=\(String(format: "%.3f", elapsed))ms"
            )
            return [primary] + Array(duplicateProcessFns.prefix(count - 1))
        }

        // Create additional copies
        guard let baseURL = dylibFileURL else {
            throw NSError(
                domain: "DLError", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "No dylib path recorded"])
        }
        let fm = FileManager.default
        let tmpDir = FileManager.default.temporaryDirectory
        let baseName = baseURL.deletingPathExtension().lastPathComponent
        let startIndex = duplicateProcessFns.count
        let needed = (count - 1) - startIndex

        var copyMs: Double = 0
        var dlopenMs: Double = 0
        var dlsymMs: Double = 0

        for i in startIndex..<(count - 1) {
            let copyURL = tmpDir.appendingPathComponent("\(baseName)_copy_\(i).dylib")

            // Copy original dylib
            let c0 = CFAbsoluteTimeGetCurrent()
            if fm.fileExists(atPath: copyURL.path) {
                try? fm.removeItem(at: copyURL)
            }
            try fm.copyItem(at: baseURL, to: copyURL)
            copyMs += (CFAbsoluteTimeGetCurrent() - c0) * 1000.0

            // Load
            let l0 = CFAbsoluteTimeGetCurrent()
            guard let handle = dlopen(copyURL.path, RTLD_NOW) else {
                let err = String(cString: dlerror())
                throw NSError(
                    domain: "DLError", code: 7,
                    userInfo: [NSLocalizedDescriptionKey: "dlopen failed: \(err)"])
            }
            dlopenMs += (CFAbsoluteTimeGetCurrent() - l0) * 1000.0

            // Symbol
            let s0 = CFAbsoluteTimeGetCurrent()
            guard let sym = dlsym(handle, symbolName) else {
                let err = String(cString: dlerror())
                dlclose(handle)
                throw NSError(
                    domain: "DLError", code: 8,
                    userInfo: [NSLocalizedDescriptionKey: "dlsym failed: \(err)"])
            }
            dlsymMs += (CFAbsoluteTimeGetCurrent() - s0) * 1000.0

            let fn = unsafeBitCast(
                sym,
                to: (@convention(c) (
                    UnsafePointer<UnsafeMutablePointer<Float>?>?,
                    UnsafePointer<UnsafeMutablePointer<Float>?>?, Int32, UnsafeMutableRawPointer?
                ) -> Void).self)
            duplicateHandles.append(handle)
            duplicateProcessFns.append(fn)
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print(
            "getProcessFunctions(count=\(count)): newCopies=\(needed), copy=\(String(format: "%.3f", copyMs))ms, dlopen=\(String(format: "%.3f", dlopenMs))ms, dlsym=\(String(format: "%.3f", dlsymMs))ms, total=\(String(format: "%.3f", totalMs))ms, dylib=\(baseURL.lastPathComponent)"
        )

        return [primary] + Array(duplicateProcessFns.prefix(count - 1))
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
    private let commandQueue: MTLCommandQueue
    private var bufferPool: [String: MTLBuffer] = [:]  // shared source of buffers used by all kernels
    private var functions: [MTLFunction] = []  // one per kernel
    private var context: IRContext
    private let maxFrameCount = 2048  // Handle up to 2048 frames per callback
    private var firstDebug = true

    public init(kernels: [CompiledKernel], cellAllocations: CellAllocations, context: IRContext)
        throws
    {
        self.kernels = kernels
        self.cellAllocations = cellAllocations
        self.context = context

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

    private func getElementCount(_ bufferName: String) -> Int {
        return bufferName == "frameCount"
            ? 1
            : bufferName == "memory" || bufferName == "grad_memory"
                ? getMemorySize()
                : bufferName == "t"
                    ? maxFrameCount * context.globals.count
                    : bufferName == "gradient"
                        ? maxFrameCount * context.gradients.count : maxFrameCount

    }

    private func initializeBuffers() throws {
        // Collect all unique buffer names across kernels
        let allBufferNames = Set(kernels.flatMap { $0.buffers })

        // Create MTLBuffers for each unique buffer
        // Use larger buffer size to handle varying frameCount from AVAudioEngine
        for bufferName in allBufferNames {
            // Memory buffer size comes from getMemorySize(), others need maxFrameCount
            let elementCount = getElementCount(bufferName)

            let bufferSize = elementCount * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
            else {
                throw NSError(
                    domain: "MetalError", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer \(bufferName)"])
            }
            print("name=\(bufferName) buffer size = \(bufferSize)")

            // Initialize buffer contents to zero
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
                bufferContents[i] = 0.0
            }
            bufferPool[bufferName] = buffer
        }

        // Create frameCount buffer for scalar kernels
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

    public func resetGradientBuffers(numFrames: Int) {
        guard let buffer = bufferPool["gradients"] else { return }

        let elementCount = getElementCount("gradients")
        let bufferSize = elementCount * MemoryLayout<Float>.size
        guard context.seedGradients.count > 0 else { return }

        let seedId = context.seedGradients[0]
        // Initialize buffer contents to zero
        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
            bufferContents[i] = 0.0
        }

        for i in (numFrames * seedId)..<(numFrames * (seedId + 1)) {
            bufferContents[i] = 1.0
        }
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

        //resetGradientBuffers(numFrames: frameCount)

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

            // isSegments is true when we use delay1 which requires dividing up a simd kernel into batches of 4
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
                        if firstDebug {
                            print(
                                "kernel[\(index) setting buffer=\(bufferName) index=\(bufferIndex)]"
                            )
                        }

                        guard let buffer = bufferPool[bufferName] else { continue }
                        computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)

                        /*
                        // Offset per-frame buffers (t*, outputs); never offset memory or frameCount/segmentLen
                        let needsOffset = (bufferName.hasPrefix("t") || bufferName == "outputs")
                        let byteOffset = needsOffset ? base * MemoryLayout<Float>.size : 0
                        computeEncoder.setBuffer(buffer, offset: byteOffset, index: bufferIndex)

                         */
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
                        if firstDebug {
                            print(
                                "kernel[\(index) setting buffer=\(bufferName) index=\(bufferIndex)]"
                            )
                        }
                        computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)
                    }
                }

                let threadGroupSize =
                    kernel.kind == .scalar
                    ? 1 : kernel.threadGroupSize ?? min(frameCount, MIN_FRAME_COUNT)
                let threadsPerGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
                let numThreadGroups = MTLSize(
                    width: (frameCount + threadGroupSize - 1) / threadGroupSize, height: 1, depth: 1
                )

                computeEncoder.dispatchThreadgroups(
                    numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
                computeEncoder.endEncoding()
            }
        }

        firstDebug = false
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
            for i in 0..<min(frameCount, MIN_FRAME_COUNT) {
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
                        for i in 0..<min(frameCount, MIN_FRAME_COUNT) {
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
        //------------------------------------------------------------------
        // 1 ‧ Audio engine / format
        //------------------------------------------------------------------
        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: AVAudioChannelCount(channels))!

        //------------------------------------------------------------------
        // 2 ‧ One reusable 128-sample scratch buffer
        //------------------------------------------------------------------
        let blockSize = 128  // what the Metal kernel expects
        let silentInput = [Float](repeating: 0, count: blockSize)

        var interleavedScratch: UnsafeMutablePointer<Float>? = nil
        if channels > 1 {
            interleavedScratch = UnsafeMutablePointer<Float>
                .allocate(capacity: blockSize * channels)
        }

        //------------------------------------------------------------------
        // 3 ‧ Source node — handles ANY host frameCount, feeds kernel in 128s
        //------------------------------------------------------------------
        let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, abl -> OSStatus in
            let buffers = UnsafeMutableAudioBufferListPointer(abl)

            var done = 0
            while done < Int(frameCount) {

                let n = min(blockSize, Int(frameCount) - done)  // ≤ 128

                // ----------------------------------------------------------
                // MONO
                // ----------------------------------------------------------
                if channels == 1 {
                    let dst = buffers[0].mData!
                        .assumingMemoryBound(to: Float.self)
                        .advanced(by: done)

                    silentInput.withUnsafeBufferPointer { zero in
                        self.run(
                            outputs: dst,
                            inputs: zero.baseAddress!,
                            frameCount: n,
                            volumeScale: volumeScale)
                    }

                    // Call level callback after volume scaling if provided
                    levelCallback?(dst, n)

                    // ----------------------------------------------------------
                    // MULTICHANNEL (Metal kernel returns interleaved)
                    // ----------------------------------------------------------
                } else if let scratch = interleavedScratch {
                    silentInput.withUnsafeBufferPointer { zero in
                        self.run(
                            outputs: scratch,
                            inputs: zero.baseAddress!,
                            frameCount: n,
                            volumeScale: volumeScale)
                    }

                    // Call level callback with first channel after volume scaling if provided
                    levelCallback?(scratch, n)

                    // De-interleave into Core Audio’s non-interleaved buffers
                    for ch in 0..<channels {
                        let dst = buffers[ch].mData!
                            .assumingMemoryBound(to: Float.self)
                            .advanced(by: done)
                        for i in 0..<n {
                            dst[i] = scratch[i * channels + ch]
                        }
                    }
                }

                done += n
            }
            return noErr
        }

        //------------------------------------------------------------------
        // 4 ‧ Spin the engine
        //------------------------------------------------------------------
        engine.attach(sourceNode)
        engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
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
        let count = buffer.length / MemoryLayout<Float>.size  // Use actual buffer size
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
                count = buffer.length / MemoryLayout<Float>.size  // Use actual buffer size
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
        // Metal uses its own internal persistent memory buffer in bufferPool["memory"]
        // No need to copy external memory - just use the existing run() implementation
        run(outputs: outputs, inputs: inputs, frameCount: frameCount, volumeScale: 1.0)
    }

    deinit {
        cleanup()
    }
}

extension CompiledKernelRuntime {
    // Render 'seconds' of audio to a mono 32-bit float WAV file using 128-frame blocks
    public func writeWAV(to url: URL, seconds: Double, sampleRate: Double, volumeScale: Float = 1.0)
        throws
    {
        let totalFrames = Int(seconds * sampleRate)
        let block = MIN_FRAME_COUNT
        var pcm = [Float](repeating: 0, count: totalFrames)
        var zeroIn = [Float](repeating: 0, count: block)

        // Allocate memory for the kernel
        guard let memory = allocateNodeMemory() else {
            throw NSError(
                domain: "RuntimeError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to allocate node memory"])
        }
        defer { deallocateNodeMemory(memory) }

        print("Allocated node memory")

        var rendered = 0
        var blockIndex = 0
        while rendered < totalFrames {
            let n = min(block, totalFrames - rendered)
            pcm.withUnsafeMutableBufferPointer { outPtr in
                zeroIn.withUnsafeBufferPointer { inPtr in
                    self.runWithMemory(
                        outputs: outPtr.baseAddress!.advanced(by: rendered),
                        inputs: inPtr.baseAddress!,
                        memory: memory,
                        frameCount: n
                    )
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
        appendUInt32(16)  // PCM fmt chunk size
        appendUInt16(3)  // AudioFormat 3 = IEEE float
        appendUInt16(UInt16(numChannels))
        appendUInt32(UInt32(sampleRate))
        appendUInt32(UInt32(byteRate))
        appendUInt16(UInt16(blockAlign))
        appendUInt16(32)  // bits per sample
        append("data")
        appendUInt32(UInt32(dataBytes))

        // Samples
        pcm.withUnsafeBytes { raw in
            data.append(contentsOf: raw.bindMemory(to: UInt8.self))
        }

        try data.write(to: url, options: .atomic)
    }
}
