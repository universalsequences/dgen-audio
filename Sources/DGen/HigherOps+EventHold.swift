import Foundation

extension Graph {
  /// Sample a value on arbitrary positive trigger frames and schedule its pure
  /// consumers on that same clock. Stateful/audio consumers must explicitly
  /// latch the final coefficients back to frame rate. Unlike a periodic hop,
  /// events can occur on adjacent frames, so outbound storage is never sliced.
  public func eventHold(_ input: NodeID, when trigger: NodeID) -> NodeID {
    let clock: NodeID
    if let existing = eventHoldClocks[trigger] {
      clock = existing
    } else {
      let zero = n(.constant(0), [])
      let one = n(.constant(1), [])
      clock = n(.gswitch, n(.gt, trigger, zero), zero, one)
      eventHoldClocks[trigger] = clock
      eventClockNodes.insert(clock)
    }
    // Use a private tagged predicate. Tagging the caller's trigger would
    // incorrectly turn its other, ordinary latch consumers into rate boundaries.
    let predicate = n(.eq, clock, n(.constant(0), []))
    nodeHopRate[predicate] = (1, clock)
    return latch(input, predicate)
  }
}
