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
                    self.run(
                        outputs: outPtr.baseAddress!.advanced(by: rendered),
                        inputs: inPtr.baseAddress!,
                        frameCount: n,
                        volumeScale: 1.0
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
