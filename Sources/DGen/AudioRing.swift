import AVFoundation
import Foundation
import os

public final class AudioRingRenderer {
    public typealias ProduceBlock = (_ out: UnsafeMutablePointer<Float>, _ frames: Int) -> Void

    private let blockSize: Int
    private let channels: Int
    private let produce: ProduceBlock
    private let levelCallback: AudioLevelCallback?

    // Ring buffer
    private let maxOutputBufferSize: Int
    private var outputBuffer: [Float]
    private var outputBufferWriteIndex: Int = 0
    private var outputBufferReadIndex: Int = 0
    private var outputBufferCount: Int = 0

    // Scratch
    private var scratchBuffer: [Float]
    private var zeroInput: [Float]

    // Lock and counters
    private var renderLock = os_unfair_lock()
    private var renderAudioCounter: Int = 0
    private var producedBlocks: Int = 0

    public init(
        blockSize: Int = 128,
        channels: Int,
        bufferSeconds: Double = 1.0,
        levelCallback: AudioLevelCallback? = nil,
        produce: @escaping ProduceBlock
    ) {
        self.blockSize = blockSize
        self.channels = channels
        self.levelCallback = levelCallback
        self.produce = produce

        // 1 second of ring buffer at 44.1kHz mono by default, but cap minimal size
        let minCapacity = blockSize * 4
        let capacity = max(minCapacity, Int(44100 * bufferSeconds))
        self.maxOutputBufferSize = capacity
        self.outputBuffer = [Float](repeating: 0, count: capacity)
        self.scratchBuffer = [Float](repeating: 0, count: blockSize)
        self.zeroInput = [Float](repeating: 0, count: blockSize)
    }

    public func makeSourceNode(sampleRate: Double) -> AVAudioSourceNode {
        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: AVAudioChannelCount(max(1, channels)))!
        let node = AVAudioSourceNode(format: format) { [unowned self] _, _, frameCount, abl -> OSStatus in
            return self.renderAudio(frameCount: frameCount, audioBufferList: abl)
        }
        return node
    }

    public func renderAudio(
        frameCount: UInt32, audioBufferList: UnsafeMutablePointer<AudioBufferList>
    ) -> OSStatus {
        renderAudioCounter &+= 1
        let intFrameCount = Int(frameCount)

        let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
        guard ablPointer.count >= 1,
            let leftBuffer = ablPointer[0].mData?.assumingMemoryBound(to: Float.self)
        else {
            return kAudioUnitErr_InvalidParameter
        }
        let rightBuffer =
            ablPointer.count > 1 ? ablPointer[1].mData?.assumingMemoryBound(to: Float.self) : nil

        os_unfair_lock_lock(&renderLock)

        // Fill ring buffer with fixed-size blocks until we have enough samples
        while outputBufferCount < intFrameCount {
            if outputBufferCount >= maxOutputBufferSize - blockSize {
                // Not expected with reasonable buffer sizing
                break
            }

            // Release lock while doing the heavy work
            os_unfair_lock_unlock(&renderLock)

            // Produce exactly blockSize frames into scratch
            scratchBuffer.withUnsafeMutableBufferPointer { scratch in
                // Produce mono; if multi-channel, app could duplicate
                produce(scratch.baseAddress!, blockSize)
            }

            producedBlocks &+= 1
            if producedBlocks % 100 == 0 {
                let preview = scratchBuffer.prefix(min(8, scratchBuffer.count))
                let s = preview.map { String(format: "%.6f", $0) }.joined(separator: ", ")
                print("🔊 Ring produced block #\(producedBlocks). First 8: [\(s)]")
            }

            os_unfair_lock_lock(&renderLock)
            // Write scratch to ring buffer
            var w = outputBufferWriteIndex
            for i in 0..<blockSize {
                outputBuffer[w] = scratchBuffer[i]
                w = (w + 1) % maxOutputBufferSize
            }
            outputBufferWriteIndex = w
            outputBufferCount = min(maxOutputBufferSize, outputBufferCount + blockSize)
        }

        // Read from ring to ABL
        let samplesToRead = min(intFrameCount, outputBufferCount)
        var readIndex = outputBufferReadIndex
        outputBufferReadIndex = (outputBufferReadIndex + samplesToRead) % maxOutputBufferSize
        outputBufferCount -= samplesToRead

        os_unfair_lock_unlock(&renderLock)

        for frame in 0..<samplesToRead {
            let sample = outputBuffer[readIndex]
            leftBuffer[frame] = sample
            if let rightBuffer = rightBuffer { rightBuffer[frame] = sample }
            readIndex = (readIndex + 1) % maxOutputBufferSize
        }
        // Zero remainder
        if samplesToRead < intFrameCount {
            for frame in samplesToRead..<intFrameCount {
                leftBuffer[frame] = 0.0
                if let rightBuffer = rightBuffer { rightBuffer[frame] = 0.0 }
            }
        }

        // Level callback for metering
        if let cb = levelCallback {
            cb(leftBuffer, intFrameCount)
        }
        return noErr
    }
}
