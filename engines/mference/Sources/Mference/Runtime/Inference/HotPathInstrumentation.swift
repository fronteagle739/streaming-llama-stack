import Foundation

public struct HotPathMetrics: Sendable, Equatable {
    public let commandBufferCommits: UInt64
    public let completionWaits: UInt64
    public let completionWaitNanos: UInt64
    public let routerWaits: UInt64
    public let routerWaitNanos: UInt64
    public let indexerWaits: UInt64
    public let indexerWaitNanos: UInt64
    public let cachePlanCalls: UInt64
    public let cachePlanNanos: UInt64
    public let cachePlanHits: UInt64
    public let cachePlanMisses: UInt64

    public init(commandBufferCommits: UInt64,
                completionWaits: UInt64,
                completionWaitNanos: UInt64,
                routerWaits: UInt64,
                routerWaitNanos: UInt64,
                indexerWaits: UInt64,
                indexerWaitNanos: UInt64,
                cachePlanCalls: UInt64,
                cachePlanNanos: UInt64,
                cachePlanHits: UInt64,
                cachePlanMisses: UInt64) {
        self.commandBufferCommits = commandBufferCommits
        self.completionWaits = completionWaits
        self.completionWaitNanos = completionWaitNanos
        self.routerWaits = routerWaits
        self.routerWaitNanos = routerWaitNanos
        self.indexerWaits = indexerWaits
        self.indexerWaitNanos = indexerWaitNanos
        self.cachePlanCalls = cachePlanCalls
        self.cachePlanNanos = cachePlanNanos
        self.cachePlanHits = cachePlanHits
        self.cachePlanMisses = cachePlanMisses
    }

    public func delta(from start: Self) -> Self {
        func difference(_ end: UInt64, _ beginning: UInt64) -> UInt64 {
            end >= beginning ? end - beginning : 0
        }
        return Self(
            commandBufferCommits: difference(commandBufferCommits, start.commandBufferCommits),
            completionWaits: difference(completionWaits, start.completionWaits),
            completionWaitNanos: difference(completionWaitNanos, start.completionWaitNanos),
            routerWaits: difference(routerWaits, start.routerWaits),
            routerWaitNanos: difference(routerWaitNanos, start.routerWaitNanos),
            indexerWaits: difference(indexerWaits, start.indexerWaits),
            indexerWaitNanos: difference(indexerWaitNanos, start.indexerWaitNanos),
            cachePlanCalls: difference(cachePlanCalls, start.cachePlanCalls),
            cachePlanNanos: difference(cachePlanNanos, start.cachePlanNanos),
            cachePlanHits: difference(cachePlanHits, start.cachePlanHits),
            cachePlanMisses: difference(cachePlanMisses, start.cachePlanMisses))
    }

    public var logDescription: String {
        "hot_path command_buffers=\(commandBufferCommits) waits=\(completionWaits) "
            + "wait_ns=\(completionWaitNanos) router_waits=\(routerWaits) "
            + "router_wait_ns=\(routerWaitNanos) indexer_waits=\(indexerWaits) "
            + "indexer_wait_ns=\(indexerWaitNanos) cache_plans=\(cachePlanCalls) "
            + "cache_plan_ns=\(cachePlanNanos) cache_hits=\(cachePlanHits) "
            + "cache_misses=\(cachePlanMisses)"
    }
}

enum HotPathWaitKind {
    case other
    case router
    case indexer
}

/// Locking is skipped entirely unless `MFERENCE_HOTPATH_METRICS=1` (or the
/// recorder is explicitly enabled by a test). This keeps attribution opt-in
/// and permits completion waits from nonisolated helper contexts.
final class HotPathInstrumentation: @unchecked Sendable {
    let enabled: Bool
    private let lock = NSLock()
    private var commandBufferCommits: UInt64 = 0
    private var completionWaits: UInt64 = 0
    private var completionWaitNanos: UInt64 = 0
    private var routerWaits: UInt64 = 0
    private var routerWaitNanos: UInt64 = 0
    private var indexerWaits: UInt64 = 0
    private var indexerWaitNanos: UInt64 = 0
    private var cachePlanCalls: UInt64 = 0
    private var cachePlanNanos: UInt64 = 0
    private var cachePlanHits: UInt64 = 0
    private var cachePlanMisses: UInt64 = 0

    init(enabled: Bool = ProcessInfo.processInfo.environment["MFERENCE_HOTPATH_METRICS"] == "1") {
        self.enabled = enabled
    }

    func recordCommandBufferCommit() {
        guard enabled else { return }
        lock.lock(); commandBufferCommits &+= 1; lock.unlock()
    }

    func recordWait(nanos: UInt64, kind: HotPathWaitKind) {
        guard enabled else { return }
        lock.lock()
        completionWaits &+= 1
        completionWaitNanos &+= nanos
        switch kind {
        case .other: break
        case .router:
            routerWaits &+= 1
            routerWaitNanos &+= nanos
        case .indexer:
            indexerWaits &+= 1
            indexerWaitNanos &+= nanos
        }
        lock.unlock()
    }

    func recordCachePlan(nanos: UInt64, hits: Int, misses: Int) {
        guard enabled else { return }
        lock.lock()
        cachePlanCalls &+= 1
        cachePlanNanos &+= nanos
        cachePlanHits &+= UInt64(max(0, hits))
        cachePlanMisses &+= UInt64(max(0, misses))
        lock.unlock()
    }

    func snapshot() -> HotPathMetrics {
        lock.lock(); defer { lock.unlock() }
        return HotPathMetrics(
            commandBufferCommits: commandBufferCommits,
            completionWaits: completionWaits,
            completionWaitNanos: completionWaitNanos,
            routerWaits: routerWaits,
            routerWaitNanos: routerWaitNanos,
            indexerWaits: indexerWaits,
            indexerWaitNanos: indexerWaitNanos,
            cachePlanCalls: cachePlanCalls,
            cachePlanNanos: cachePlanNanos,
            cachePlanHits: cachePlanHits,
            cachePlanMisses: cachePlanMisses)
    }
}
