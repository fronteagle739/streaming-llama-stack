import Foundation
import Metal

/// Pins cache slots until the GPU command buffer that consumes them finishes.
/// An unbound lease releases on deinit, covering abandoned/error plans.
public final class ExpertCacheLease: @unchecked Sendable {
    private let lock = NSLock()
    private var releaseAction: (() -> Void)?
    private var ownerReleased = false
    private var pendingCompletions = 0
    private var executionStarted = false

    public init(release: @escaping @Sendable () -> Void) {
        self.releaseAction = release
    }

    public func retainUntilCompletion(
        register: (@escaping @Sendable () -> Void) -> Void
    ) {
        lock.lock()
        guard releaseAction != nil, !ownerReleased else {
            lock.unlock()
            return
        }
        pendingCompletions += 1
        lock.unlock()
        register { [self] in completeOne() }
    }

    public func retain(until commandBuffer: MTLCommandBuffer) {
        retainUntilCompletion { completion in
            commandBuffer.addCompletedHandler { _ in completion() }
        }
    }

    func beginExecution() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !executionStarted, !ownerReleased, releaseAction != nil else { return false }
        executionStarted = true
        return true
    }

    public func release() {
        let action: (() -> Void)?
        lock.lock()
        ownerReleased = true
        if pendingCompletions == 0 {
            action = releaseAction
            releaseAction = nil
        } else {
            action = nil
        }
        lock.unlock()
        action?()
    }

    private func completeOne() {
        let action: (() -> Void)?
        lock.lock()
        if pendingCompletions > 0 { pendingCompletions -= 1 }
        if ownerReleased && pendingCompletions == 0 {
            action = releaseAction
            releaseAction = nil
        } else {
            action = nil
        }
        lock.unlock()
        action?()
    }

    deinit { release() }
}
