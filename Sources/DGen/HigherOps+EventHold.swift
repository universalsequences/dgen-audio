import Foundation

extension Graph {
  /// The shared event clock for `trigger`: 0 on frames where `trigger > 0`,
  /// 1 otherwise. Uses with the same trigger expression share one clock.
  func eventClock(for trigger: NodeID) -> NodeID {
    if let existing = eventHoldClocks[trigger] { return existing }
    let zero = n(.constant(0), [])
    let one = n(.constant(1), [])
    let clock = n(.gswitch, n(.gt, trigger, zero), zero, one)
    eventHoldClocks[trigger] = clock
    eventClockNodes.insert(clock)
    return clock
  }

  /// Sample a value on arbitrary positive trigger frames and schedule its pure
  /// consumers on that same clock. Stateful/audio consumers must explicitly
  /// latch the final coefficients back to frame rate. Unlike a periodic hop,
  /// events can occur on adjacent frames, so outbound storage is never sliced.
  public func eventHold(_ input: NodeID, when trigger: NodeID) -> NodeID {
    let clock = eventClock(for: trigger)
    // Use a private tagged predicate. Tagging the caller's trigger would
    // incorrectly turn its other, ordinary latch consumers into rate boundaries.
    let predicate = n(.eq, clock, n(.constant(0), []))
    nodeHopRate[predicate] = (1, clock)
    return latch(input, predicate)
  }

  /// Bring an event-rate scalar back to frame rate: `latch(value, trigger)`
  /// for a `value` computed on `trigger`'s event clock. Unlike a plain latch
  /// it is not frame-serial: between events every lane reads the same held
  /// cell, so the C backend keeps the consumer loop SIMD and pays one event
  /// test plus one broadcast load per lane group.
  public func eventLatch(_ value: NodeID, when trigger: NodeID) -> NodeID {
    let clock = eventClock(for: trigger)
    let cell = alloc()
    persistentCells.insert(cell)
    return n(.eventLatch(cell), value, clock)
  }
}
