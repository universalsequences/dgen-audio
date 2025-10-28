import AVFoundation
import Foundation

/// Manages the lifecycle of DGen graph compilation and playback
/// TODO - make this a true graph allowing multiple nodes to be connected and played
public class GraphManager {
    private var currentRuntime: CompiledKernelRuntime?
    public var cellAllocations: CellAllocations?
    private var isPlaying: Bool = false
    private var targetDevice: Device = .C
    private var engine: AVAudioEngine
    private var node: AVAudioNode?

    // Parameter update ring buffer for real-time param changes
    public let paramUpdateBuffer = ParamUpdateRingBuffer(capacity: 256)
    private var updateTimer: Timer?

    // Audio level tracking for meter
    private var _currentLevel: Float = 0.0
    private var _peakLevel: Float = 0.0
    private var _peakHoldTime: TimeInterval = 0.0
    private var _lastUpdateTime: TimeInterval = 0.0
    private let peakHoldDuration: TimeInterval = 1.0  // Hold peak for 1 second
    private let levelSmoothingFactor: Float = 0.3  // Lower = more smoothing

    // Performance optimization: accumulate samples between updates
    private var _sampleAccumulator: Float = 0.0
    private var _peakAccumulator: Float = 0.0
    private var _sampleCount: Int = 0
    private var _lastMeterUpdateTime: TimeInterval = 0.0
    private let meterUpdateInterval: TimeInterval = 0.05  // Update meter 20 times per second

    public init(targetDevice: Device = .C) {
        self.targetDevice = targetDevice
        self.engine = AVAudioEngine()

        // Start parameter update timer (processes ring buffer at 60Hz)
        startUpdateTimer()
    }

    deinit {
        stopUpdateTimer()
    }
    
    public var compiledRuntime: CompiledKernelRuntime? {
        return currentRuntime
    }
    
    // MARK: - Audio Engine Access (Phase 5)
    
    /// Access to the AVAudioEngine for external coordination
    public var audioEngine: AVAudioEngine {
        return engine
    }
    
    /// Access to the current audio node for external management
    public var audioNode: AVAudioNode? {
        get { return node }
        set { node = newValue }
    }

    /// Compiles and plays a new graph, stopping any currently playing graph
    public func compile(graph: Graph) throws {
        print("🎛️ GraphManager.compile called:")
        print("   - GraphManager instance: \(ObjectIdentifier(self))")
        print("   - Engine before stop: \(self.engine.isRunning)")

        // Stop and dispose current runtime if playing
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
        print("🚨 GRAPHMANAGER.COMPILE() IS STOPPING THE AVAUDIOENGINE! 🚨")
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
        self.engine.stop()

        print("   - Engine after stop: \(self.engine.isRunning)")

        // Explicitly cleanup old runtime if it exists
        currentRuntime?.cleanup()
        currentRuntime = nil

        // Compile the graph
        let compiledRuntime = try compileGraph(graph)

        // Store the new runtime and start playing
        currentRuntime = compiledRuntime

        play()
    }

    /// Stops the current playback and disposes resources
    public func stop() {
        if isPlaying {
            // Note: The current runtime doesn't have explicit stop/dispose methods
            // but when we replace it with a new one, the old one should be deallocated
            currentRuntime = nil
            isPlaying = false

            // Clear any pending parameter updates
            paramUpdateBuffer.clear()

            print("🛑 Stopped current playback")
        }
    }

    /// Compilation completed - audio playback is handled by Project's AudioGraphManager
    private func play() {
        print("🎛️ GraphManager compilation complete - audio playback handled by AudioGraphManager")
        // No longer starts playback directly - Project's startAudioProcessing() handles this
        isPlaying = false
    }

    /// Compiles a graph into a runtime without starting playback
    private func compileGraph(_ graph: Graph) throws -> CompiledKernelRuntime {
        let backend: Backend = (targetDevice == .Metal) ? .metal : .c
        let options = CompilationPipeline.Options(
            frameCount: 128,
            debug: true,
            printBlockStructure: false,
            forceScalar: false
        )

        let result = try CompilationPipeline.compile(
            graph: graph,
            backend: backend,
            options: options
        )

        self.cellAllocations = result.cellAllocations

        let kernels = result.kernels
        for kernel in kernels {
            print("\nKernel \(kernel.name):")
            kernel.buffers.forEach { print("- buffer: \($0)") }
            print("📄 KERNEL SOURCE TO COMPILE:")
            print(String(repeating: "=", count: 80))
            print(kernel.source)
            print(String(repeating: "=", count: 80))
        }

        // Build runtime
        let runtime: CompiledKernelRuntime
        if targetDevice == .Metal {
            runtime = try MetalCompiledKernel(
                kernels: kernels, cellAllocations: result.cellAllocations)
        } else {
            let source = kernels.first?.source ?? ""
            let memorySize = kernels.first?.memorySize ?? 1024
            let compiled = CCompiledKernel(source: source, cellAllocations: result.cellAllocations, memorySize: memorySize)
            try compiled.compileAndLoad()
            runtime = compiled
        }

        // Optional Metal buffer read-back
        if let metalRuntime = runtime as? MetalCompiledKernel {
            for name in Set(kernels.flatMap(\.buffers)) {
                if let data = metalRuntime.readBuffer(named: name) {
                    let preview = data.prefix(5)
                        .map { String(format: "%.3f", $0) }
                        .joined(separator: ", ")
                }
            }
        }

        return runtime
    }

    /// Returns whether a graph is currently playing
    public var isCurrentlyPlaying: Bool {
        return isPlaying
    }

    /// Changes the target device for future compilations
    public func setTargetDevice(_ device: Device) {
        targetDevice = device
    }

    /// Returns the current audio level (0.0 to 1.0+)
    public var currentLevel: Float {
        return _currentLevel
    }

    /// Returns the peak audio level (0.0 to 1.0+)
    public var peakLevel: Float {
        let currentTime = CACurrentMediaTime()

        // Reset peak if hold time has expired
        if currentTime - _peakHoldTime > peakHoldDuration {
            _peakLevel = _currentLevel
            _peakHoldTime = currentTime
        }

        return _peakLevel
    }

    /// Returns whether the meter should be visible (has recent audio activity)
    public var shouldShowMeter: Bool {
        let currentTime = CACurrentMediaTime()
        // Show meter if playing and had audio in the last 2 seconds
        return isPlaying && (currentTime - _lastUpdateTime < 2.0)
    }

    /// Updates audio levels based on audio buffer data
    private func updateAudioLevels(from buffer: UnsafePointer<Float>, frameCount: Int) {
        guard frameCount > 0 else { return }

        // Accumulate samples for RMS calculation
        var sumSquares: Float = 0.0
        var bufferPeak: Float = 0.0

        for i in 0..<frameCount {
            let sample = buffer[i]
            sumSquares += sample * sample
            bufferPeak = max(bufferPeak, abs(sample))
        }

        _sampleAccumulator += sumSquares
        _peakAccumulator = max(_peakAccumulator, bufferPeak)
        _sampleCount += frameCount

        let currentTime = CACurrentMediaTime()

        // Only update the meter at reasonable intervals (20Hz)
        if currentTime - _lastMeterUpdateTime >= meterUpdateInterval && _sampleCount > 0 {
            // Calculate RMS from accumulated samples
            let rms = sqrt(_sampleAccumulator / Float(_sampleCount))

            // Smooth the current level
            _currentLevel =
                _currentLevel * (1.0 - levelSmoothingFactor) + rms * levelSmoothingFactor

            // Update peak if new peak is higher or if it's time to refresh
            if _peakAccumulator > _peakLevel || currentTime - _peakHoldTime > peakHoldDuration {
                _peakLevel = _peakAccumulator
                _peakHoldTime = currentTime
            }

            // Reset accumulators
            _sampleAccumulator = 0.0
            _peakAccumulator = 0.0
            _sampleCount = 0
            _lastMeterUpdateTime = currentTime
            _lastUpdateTime = currentTime
        }
    }

    // MARK: - Parameter Update Timer

    /// Start the timer that processes parameter updates from the ring buffer
    private func startUpdateTimer() {
        stopUpdateTimer()  // Ensure no duplicate timers

        updateTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] _ in
            self?.processParameterUpdates()
        }
    }

    /// Stop the parameter update timer
    private func stopUpdateTimer() {
        updateTimer?.invalidate()
        updateTimer = nil
    }

    /// Process all pending parameter updates from the ring buffer
    private func processParameterUpdates() {
        paramUpdateBuffer.processUpdates(runtime: currentRuntime)
    }

    /// Queue a parameter update to be applied to the runtime
    /// This is thread-safe and can be called from any thread
    public func queueParamUpdate(cellId: CellID, value: Float) {
        if !paramUpdateBuffer.enqueue(cellId: cellId, value: value) {
            print("⚠️ Parameter update buffer full, dropping update for cell \(cellId)")
        }
    }
}
