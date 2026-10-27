import Foundation

private final class CoalescingPersistenceQueue: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mference.global-expert-heatmap.persistence",
                                      qos: .utility)
    private let queueKey = DispatchSpecificKey<UInt8>()
    private let lock = NSLock()
    private let group = DispatchGroup()
    private var latest: (@Sendable () -> Void)?
    private var draining = false

    init() { queue.setSpecific(key: queueKey, value: 1) }

    func schedule(_ operation: @escaping @Sendable () -> Void) {
        lock.lock()
        latest = operation
        guard !draining else { lock.unlock(); return }
        draining = true
        group.enter()
        lock.unlock()
        queue.async { [self] in drain() }
    }

    func wait() {
        // Releasing the final owner from a queued persistence operation may
        // run deinit on this queue. Waiting for our own drain group there
        // would deadlock; the just-finished operation already has newest state.
        guard DispatchQueue.getSpecific(key: queueKey) == nil else { return }
        group.wait()
    }

    private func drain() {
        while true {
            lock.lock()
            guard let operation = latest else {
                draining = false
                lock.unlock()
                group.leave()
                return
            }
            latest = nil
            lock.unlock()
            operation()
        }
    }
}

/// Stable identity for one routed expert in a model-wide cache.
public struct GlobalExpertKey: Codable, Hashable, Sendable, Comparable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.layer == rhs.layer ? lhs.expert < rhs.expert : lhs.layer < rhs.layer
    }
}

public struct GlobalExpertCacheMetrics: Sendable, Equatable {
    public let hits: UInt64
    public let misses: UInt64
    public let admissions: UInt64
    public let rejectedAdmissions: UInt64
    public let evictions: UInt64
    public let bytesRead: UInt64
    public let residentBytes: UInt64
    public let residentExperts: Int
    public let budgetBytes: UInt64

    public var hitRate: Double {
        let total = hits &+ misses
        return total == 0 ? 0 : Double(hits) / Double(total)
    }

    public var logDescription: String {
        let rate = String(format: "%.4f", hitRate)
        return "global_expert_cache hits=\(hits) misses=\(misses) hit_rate=\(rate) "
            + "admissions=\(admissions) rejected=\(rejectedAdmissions) "
            + "evictions=\(evictions) bytes_read=\(bytesRead) "
            + "resident_bytes=\(residentBytes) resident_experts=\(residentExperts) "
            + "budget_bytes=\(budgetBytes)"
    }
}

/// Fingerprints every cache-relevant manifest fact, including all weight file
/// hashes. Heat from a repack or partially replaced model is therefore never
/// applied to a different expert layout.
public enum GlobalExpertModelFingerprint {
    public static func make(manifest: Manifest) -> String {
        var components = [
            "global-expert-heatmap-v\(GlobalExpertHeatmap.persistenceVersion)",
            "manifest=\(manifest.versionMajor).\(manifest.versionMinor)",
            "model=\(manifest.modelID)",
            "snapshot=\(manifest.sourceSnapshotHash ?? "-")",
            "layers=\(manifest.numLayers)",
            "experts=\(manifest.expertsPerLayer)",
            "stride=\(manifest.expertStride)",
        ]
        for name in manifest.files.keys.sorted() {
            guard let file = manifest.files[name] else { continue }
            components.append("file=\(name):\(file.size):\(file.sha256.lowercased())")
        }
        return Sha256Verifier.hashData(Data(components.joined(separator: "\n").utf8))
    }
}

public struct GlobalExpertAdmission: Sendable, Equatable {
    public let key: GlobalExpertKey
    public let wasHit: Bool
    public let isResident: Bool
    public let evicted: [GlobalExpertKey]
    public let residentBytes: UInt64
}

public struct GlobalExpertBudget: Sendable, Equatable {
    public let requestedBytes: UInt64
    public let effectiveBytes: UInt64
    public let availableMemoryBytes: UInt64
    public let wasClamped: Bool

    /// Applies both the model-size ceiling and the physical-memory/headroom
    /// ceiling. Arithmetic is saturating so malformed metadata cannot wrap a
    /// very large request into a small, apparently safe allocation.
    public static func resolve(requestedBytes: UInt64,
                               totalExpertBytes: UInt64,
                               physicalMemoryBytes: UInt64,
                               residentModelBytes: UInt64,
                               headroomBytes: UInt64) -> Self {
        let reserved: UInt64
        let (sum, overflow) = residentModelBytes.addingReportingOverflow(headroomBytes)
        reserved = overflow ? UInt64.max : sum
        let available = physicalMemoryBytes > reserved ? physicalMemoryBytes - reserved : 0
        let effective = min(requestedBytes, min(totalExpertBytes, available))
        return Self(requestedBytes: requestedBytes,
                    effectiveBytes: effective,
                    availableMemoryBytes: available,
                    wasClamped: effective < requestedBytes)
    }
}

/// Thread-safe, model-global heatmap and byte-aware admission controller.
///
/// This class deliberately owns policy rather than Metal buffers. A miss that
/// is not admitted should be served from a transient streaming buffer; admitted
/// `(layer, expert)` pairs are the set a shared Metal cache must retain.
public final class GlobalExpertHeatmap: @unchecked Sendable {
    public static let persistenceVersion = 1

    private struct Record: Codable, Sendable {
        var frequency: UInt64
        var lastAccess: UInt64
        var byteCount: UInt64
    }

    private struct PersistentEntry: Codable {
        let key: GlobalExpertKey
        let record: Record
        let resident: Bool
    }

    private struct PersistentState: Codable {
        let version: Int
        let modelFingerprint: String
        let clock: UInt64
        let entries: [PersistentEntry]
    }

    public let modelFingerprint: String
    public let budgetBytes: UInt64
    public let policy: ExpertCachePolicy

    private let persistenceURL: URL?
    private let persistEveryAccesses: UInt64
    private let lock = NSLock()
    private var records: [GlobalExpertKey: Record] = [:]
    private var resident: Set<GlobalExpertKey> = []
    private var residentByteCount: UInt64 = 0
    private var coldStartCandidates: [GlobalExpertKey] = []
    private var clock: UInt64 = 0
    private var hits: UInt64 = 0
    private var misses: UInt64 = 0
    private var admissions: UInt64 = 0
    private var rejectedAdmissions: UInt64 = 0
    private var evictions: UInt64 = 0
    private var bytesRead: UInt64 = 0
    private var accessesSincePersistence: UInt64 = 0
    private let persistenceQueue = CoalescingPersistenceQueue()

    public init(modelFingerprint: String,
                budgetBytes: UInt64,
                policy: ExpertCachePolicy = .lfu,
                persistenceURL: URL? = nil,
                persistEveryAccesses: UInt64 = 1_024) {
        precondition(!modelFingerprint.isEmpty, "model fingerprint must not be empty")
        self.modelFingerprint = modelFingerprint
        self.budgetBytes = budgetBytes
        self.policy = policy
        self.persistenceURL = persistenceURL
        self.persistEveryAccesses = persistEveryAccesses
        restoreIfCompatible()
    }

    deinit {
        persistenceQueue.wait()
        try? persist()
    }

    /// Records an actual expert request and returns the resulting desired
    /// resident set transition. Frequency is retained even when admission is
    /// rejected, allowing a repeatedly streamed expert to become hot enough to
    /// displace a colder resident later.
    @discardableResult
    public func recordAccess(key: GlobalExpertKey,
                             byteCount: UInt64,
                             bytesStreamedOnMiss: UInt64? = nil,
                             protectedKeys: Set<GlobalExpertKey> = []) -> GlobalExpertAdmission {
        precondition(byteCount > 0, "expert byte count must be positive")
        var shouldPersist = false
        let result: GlobalExpertAdmission
        lock.lock()

        clock = clock == UInt64.max ? renormalizeClockLocked() : clock + 1
        var record = records[key] ?? Record(frequency: 0, lastAccess: 0, byteCount: byteCount)
        if record.frequency == UInt64.max { renormalizeFrequenciesLocked() }
        record = records[key] ?? record
        record.frequency &+= 1
        record.lastAccess = clock
        // Expert geometry is immutable for a fingerprint. Retaining the first
        // observed size prevents a bad concurrent caller from inflating an
        // already-resident entry past the hard budget.
        records[key] = record
        let effectiveByteCount = record.byteCount

        let hit = resident.contains(key)
        var removed: [GlobalExpertKey] = []
        if hit {
            hits &+= 1
        } else {
            misses &+= 1
            bytesRead &+= bytesStreamedOnMiss ?? byteCount
            if effectiveByteCount <= budgetBytes {
                var freeBytes = budgetBytes - min(budgetBytes, residentByteCount)
                let coldest = resident.filter { !protectedKeys.contains($0) }
                    .sorted { isColderLocked($0, than: $1) }
                var proposed: [GlobalExpertKey] = []
                for victim in coldest where freeBytes < effectiveByteCount {
                    // Admission is LFU/LRU, not unconditional scan pollution:
                    // the candidate must be hotter than every displaced item.
                    guard isColderLocked(victim, than: key) else {
                        proposed.removeAll()
                        break
                    }
                    proposed.append(victim)
                    freeBytes &+= records[victim]?.byteCount ?? 0
                }
                if freeBytes >= effectiveByteCount {
                    for victim in proposed {
                        if resident.remove(victim) != nil {
                            residentByteCount &-= records[victim]?.byteCount ?? 0
                        }
                    }
                    resident.insert(key)
                    residentByteCount &+= effectiveByteCount
                    removed = proposed
                    admissions &+= 1
                    evictions &+= UInt64(proposed.count)
                } else {
                    rejectedAdmissions &+= 1
                }
            } else {
                rejectedAdmissions &+= 1
            }
        }

        accessesSincePersistence &+= 1
        if persistenceURL != nil,
           persistEveryAccesses > 0,
           accessesSincePersistence >= persistEveryAccesses {
            accessesSincePersistence = 0
            shouldPersist = true
        }
        result = GlobalExpertAdmission(key: key,
                                       wasHit: hit,
                                       isResident: resident.contains(key),
                                       evicted: removed,
                                       residentBytes: residentByteCount)
        lock.unlock()
        if shouldPersist {
            persistenceQueue.schedule { [weak self] in try? self?.persist() }
        }
        return result
    }

    public func contains(_ key: GlobalExpertKey) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return resident.contains(key)
    }

    public func residentKeys() -> [GlobalExpertKey] {
        lock.lock()
        defer { lock.unlock() }
        return resident.sorted()
    }

    /// Returns persisted/observed heat in deterministic hottest-first order.
    /// This does not mutate counters or residency and is safe to use for a
    /// bounded startup preload after the model fingerprint has been checked.
    public func hottestKeys(limit: Int? = nil) -> [GlobalExpertKey] {
        lock.lock()
        defer { lock.unlock() }
        let ordered = records.keys.sorted(by: { isHotterLocked($0, than: $1) })
        guard let limit else { return ordered }
        return Array(ordered.prefix(max(0, limit)))
    }

    /// Retains persisted frequencies/recency while marking all buffer contents
    /// absent. A process restart cannot treat last run's Metal residency as a
    /// hit until bytes have been streamed into this run's pool.
    public func beginColdBufferSession() {
        lock.lock()
        coldStartCandidates = resident.sorted(by: { isHotterLocked($0, than: $1) })
        resident.removeAll(keepingCapacity: true)
        residentByteCount = 0
        lock.unlock()
    }

    public func coldStartPreloadKeys(limit: Int? = nil) -> [GlobalExpertKey] {
        lock.lock()
        defer { lock.unlock() }
        guard let limit else { return coldStartCandidates }
        return Array(coldStartCandidates.prefix(max(0, limit)))
    }

    /// Repairs logical residency after a multi-expert plan is rolled back.
    /// Heat remains intact; only entries that physically remain ready are kept.
    func reconcileResidency(_ keys: Set<GlobalExpertKey>) {
        lock.lock()
        resident.removeAll(keepingCapacity: true)
        var used: UInt64 = 0
        for key in keys.sorted(by: { isHotterLocked($0, than: $1) }) {
            guard let size = records[key]?.byteCount,
                  size <= budgetBytes - min(used, budgetBytes) else { continue }
            resident.insert(key)
            used &+= size
        }
        residentByteCount = used
        lock.unlock()
    }

    public func metrics() -> GlobalExpertCacheMetrics {
        lock.lock()
        defer { lock.unlock() }
        return GlobalExpertCacheMetrics(hits: hits,
                                        misses: misses,
                                        admissions: admissions,
                                        rejectedAdmissions: rejectedAdmissions,
                                        evictions: evictions,
                                        bytesRead: bytesRead,
                                        residentBytes: residentByteCount,
                                        residentExperts: resident.count,
                                        budgetBytes: budgetBytes)
    }

    /// Atomically replaces a compatible heatmap file. Runtime counters are
    /// intentionally session-local; only heat and desired residency persist.
    public func persist() throws {
        guard let persistenceURL else { return }
        let state: PersistentState
        lock.lock()
        state = PersistentState(
            version: Self.persistenceVersion,
            modelFingerprint: modelFingerprint,
            clock: clock,
            entries: records.keys.sorted().compactMap { key in
                records[key].map { PersistentEntry(key: key,
                                                   record: $0,
                                                   resident: resident.contains(key)) }
            })
        lock.unlock()

        let data = try JSONEncoder().encode(state)
        try FileManager.default.createDirectory(
            at: persistenceURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: persistenceURL, options: .atomic)
    }

    /// Waits for coalesced background writes and synchronously persists the
    /// newest state. Call from explicit model shutdown paths when available;
    /// deinit performs the same final flush as a safety net.
    public func flushPersistence() throws {
        persistenceQueue.wait()
        try persist()
    }

    /// Test/debug invariant for the cached byte total. Kept out of the hot
    /// path so release decode never walks the resident set on a cache hit.
    func residentByteInvariantHolds() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return residentByteCount == residentBytesRecomputedLocked()
    }

    private func restoreIfCompatible() {
        guard let persistenceURL,
              let data = try? Data(contentsOf: persistenceURL),
              let state = try? JSONDecoder().decode(PersistentState.self, from: data),
              state.version == Self.persistenceVersion,
              state.modelFingerprint == modelFingerprint else { return }

        lock.lock()
        clock = state.clock
        records.removeAll(keepingCapacity: true)
        resident.removeAll(keepingCapacity: true)
        residentByteCount = 0
        for entry in state.entries {
            // Last duplicate wins instead of trusting persisted JSON enough to
            // use Dictionary(uniqueKeysWithValues:), which traps on duplicates.
            records[entry.key] = entry.record
        }
        // A changed byte budget is safe: restore hottest entries greedily.
        let requested = state.entries.filter(\.resident).map(\.key).sorted {
            isHotterLocked($0, than: $1)
        }
        var used: UInt64 = 0
        for key in requested {
            guard let size = records[key]?.byteCount, size <= budgetBytes - min(used, budgetBytes) else {
                continue
            }
            resident.insert(key)
            used &+= size
        }
        residentByteCount = used
        lock.unlock()
    }

    private func residentBytesRecomputedLocked() -> UInt64 {
        resident.reduce(0) { partial, key in
            let size = records[key]?.byteCount ?? 0
            let (sum, overflow) = partial.addingReportingOverflow(size)
            return overflow ? UInt64.max : sum
        }
    }

    private func isColderLocked(_ lhs: GlobalExpertKey, than rhs: GlobalExpertKey) -> Bool {
        let left = records[lhs] ?? Record(frequency: 0, lastAccess: 0, byteCount: 0)
        let right = records[rhs] ?? Record(frequency: 0, lastAccess: 0, byteCount: 0)
        if policy == .lfu, left.frequency != right.frequency {
            return left.frequency < right.frequency
        }
        if left.lastAccess != right.lastAccess { return left.lastAccess < right.lastAccess }
        return lhs < rhs
    }

    private func isHotterLocked(_ lhs: GlobalExpertKey, than rhs: GlobalExpertKey) -> Bool {
        isColderLocked(rhs, than: lhs)
    }

    @discardableResult
    private func renormalizeClockLocked() -> UInt64 {
        let ordered = records.keys.sorted {
            let left = records[$0]?.lastAccess ?? 0
            let right = records[$1]?.lastAccess ?? 0
            return left == right ? $0 < $1 : left < right
        }
        for (index, key) in ordered.enumerated() {
            guard var record = records[key] else { continue }
            record.lastAccess = UInt64(index + 1)
            records[key] = record
        }
        return UInt64(ordered.count + 1)
    }

    private func renormalizeFrequenciesLocked() {
        for key in Array(records.keys) {
            guard var record = records[key] else { continue }
            record.frequency = max(1, record.frequency / 2)
            records[key] = record
        }
    }
}
