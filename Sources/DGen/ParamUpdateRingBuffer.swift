import Foundation
import QuartzCore  // For CACurrentMediaTime

/// A lock-free ring buffer for queuing parameter updates from the patch to the runtime
/// This allows the param operator to queue value changes that will be applied to the runtime
public class ParamUpdateRingBuffer {
    /// A single parameter update message
    public struct ParamUpdate {
        let cellId: CellID
        let value: Float
        let timestamp: TimeInterval
    }

    // Ring buffer storage
    private var buffer: [ParamUpdate?]
    private let capacity: Int

    // Atomic indices for lock-free operation
    private var writeIndex = AtomicInt(0)
    private var readIndex = AtomicInt(0)

    /// Initialize with specified capacity (must be power of 2 for efficient modulo)
    public init(capacity: Int = 256) {
        // Ensure capacity is power of 2
        var adjustedCapacity = 1
        while adjustedCapacity < capacity {
            adjustedCapacity *= 2
        }
        self.capacity = adjustedCapacity
        self.buffer = Array(repeating: nil, count: adjustedCapacity)
    }

    /// Enqueue a parameter update (called from patch/UI thread)
    /// Returns false if buffer is full
    @discardableResult
    public func enqueue(cellId: CellID, value: Float) -> Bool {
        let currentWrite = writeIndex.load()
        let nextWrite = (currentWrite + 1) & (capacity - 1)

        // Check if buffer is full
        if nextWrite == readIndex.load() {
            return false  // Buffer full, drop the update
        }

        // Write the update
        buffer[currentWrite] = ParamUpdate(
            cellId: cellId,
            value: value,
            timestamp: CACurrentMediaTime()
        )

        // Update write index
        writeIndex.store(nextWrite)
        return true
    }

    /// Dequeue a parameter update (called from audio thread)
    /// Returns nil if buffer is empty
    public func dequeue() -> ParamUpdate? {
        let currentRead = readIndex.load()

        // Check if buffer is empty
        if currentRead == writeIndex.load() {
            return nil
        }

        // Read the update
        guard let update = buffer[currentRead] else {
            return nil
        }

        // Clear the slot
        buffer[currentRead] = nil

        // Update read index
        let nextRead = (currentRead + 1) & (capacity - 1)
        readIndex.store(nextRead)

        return update
    }

    /// Process all pending updates by applying them to the runtime
    /// Called periodically from the GraphManager
    public func processUpdates(runtime: CompiledKernelRuntime?) {
        guard let runtime = runtime else { return }

        let cellAllocations = runtime.cellAllocations

        // Process all available updates
        while let update = dequeue() {
            // Get the true mapped cell ID from allocations
            guard let mappedCellId = cellAllocations.cellMappings[update.cellId] else {
                print("⚠️ No allocation found for cellId \(update.cellId)")
                continue
            }
            guard let kind = cellAllocations.cellKinds[update.cellId] else {
                print("⚠️ No allocation found for cellId \(update.cellId)")
                continue
            }

            // Determine vector width based on kernel kind
            let vectorWidth: Int
            switch kind {
            case .scalar:
                vectorWidth = 1
            case .simd:
                vectorWidth = 4
            }

            // Set param value for each vector element
            for i in 0..<vectorWidth {
                let finalCellId = mappedCellId + CellID(i)
                runtime.setParamValue(cellId: finalCellId, value: update.value)
            }
        }
    }

    /// Get count of pending updates (approximate due to concurrent access)
    public var pendingCount: Int {
        let write = writeIndex.load()
        let read = readIndex.load()
        return (write - read + capacity) & (capacity - 1)
    }

    /// Clear all pending updates
    public func clear() {
        readIndex.store(writeIndex.load())
    }
}

/// Simple atomic integer wrapper for thread-safe operations
private class AtomicInt {
    private var value: Int
    private let lock = NSLock()

    init(_ value: Int) {
        self.value = value
    }

    func load() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func store(_ newValue: Int) {
        lock.lock()
        defer { lock.unlock() }
        value = newValue
    }
}
