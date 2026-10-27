import Darwin
import Foundation
import Metal

/// One bounded Metal allocation shared by every routed layer in a model.
/// Resident slots follow the global heatmap; scratch slots guarantee that a
/// cold miss can stream without destroying a hotter resident.
public final class GlobalExpertBufferPool: @unchecked Sendable {
    private struct Slot {
        var key: GlobalExpertKey?
        var ready = false
        var leases = 0
        let resident: Bool
    }

    public let slotCount: Int
    public let residentSlotCount: Int
    public let allocatedBytes: UInt64
    public let expertStride: UInt64
    public let heatmap: GlobalExpertHeatmap

    private let pointers: [UnsafeMutableRawPointer]
    private let buffers: [MTLBuffer]
    private let lock = NSLock()
    private var slots: [Slot]
    /// Ready resident slots only. Scratch and in-flight loads deliberately do
    /// not appear here, so a lookup can never expose incomplete bytes.
    private var readySlotByKey: [GlobalExpertKey: Int] = [:]
    /// Includes in-flight resident admissions. This prevents concurrent plans
    /// for the same cold key from creating two resident copies; the later plan
    /// streams through scratch until the canonical resident is ready.
    private var residentSlotByKey: [GlobalExpertKey: Int] = [:]

    public init(device: MTLDevice,
                expertStride: UInt64,
                budgetBytes: UInt64,
                scratchSlots: Int,
                heatmap: GlobalExpertHeatmap) throws {
        precondition(expertStride > 0)
        precondition(scratchSlots > 0)
        let pageSize = UInt64(getpagesize())
        let aligned = ((expertStride + pageSize - 1) / pageSize) * pageSize
        let count64 = budgetBytes / aligned
        guard count64 <= UInt64(Int.max), Int(count64) > scratchSlots else {
            throw StreamerError.allocFailed(errno: ENOMEM)
        }
        let count = Int(count64)
        self.slotCount = count
        self.residentSlotCount = count - scratchSlots
        self.allocatedBytes = UInt64(count) * aligned
        self.expertStride = expertStride
        self.heatmap = heatmap
        heatmap.beginColdBufferSession()

        var madePointers: [UnsafeMutableRawPointer] = []
        var madeBuffers: [MTLBuffer] = []
        madePointers.reserveCapacity(count)
        madeBuffers.reserveCapacity(count)
        func unwind() {
            for index in madeBuffers.count..<madePointers.count { free(madePointers[index]) }
        }
        for _ in 0..<count {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, PreadExpertStreamer.scratchAlignment, Int(aligned))
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            madePointers.append(pointer)
            nonisolated(unsafe) let captured = pointer
            guard let buffer = device.makeBuffer(bytesNoCopy: pointer,
                                                 length: Int(aligned),
                                                 options: .storageModeShared,
                                                 deallocator: { _, _ in free(captured) }) else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            madeBuffers.append(buffer)
        }
        self.pointers = madePointers
        self.buffers = madeBuffers
        self.slots = (0..<count).map {
            Slot(key: nil, resident: $0 < count - scratchSlots)
        }
    }

    deinit { try? heatmap.flushPersistence() }

    public func plan(layer: Int,
                     experts: [Int],
                     avoidingSlots _: Set<Int> = []) -> ExpertCachePlan? {
        guard experts.count <= slotCount else { return nil }
        lock.lock()
        var assigned = [Int](repeating: -1, count: experts.count)
        var misses: [Int] = []
        var leased: Set<Int> = []
        var newlyAssigned: Set<Int> = []

        for index in experts.indices {
            let key = GlobalExpertKey(layer: layer, expert: experts[index])
            if let prior = experts[..<index].firstIndex(of: experts[index]) {
                let priorSlot = assigned[prior]
                assigned[index] = priorSlot
                continue
            }
            if let hit = readySlotByKey[key] {
                assert(slots[hit].resident && slots[hit].ready && slots[hit].key == key)
                _ = heatmap.recordAccess(key: key, byteCount: expertStride)
                assigned[index] = hit
                if leased.insert(hit).inserted { slots[hit].leases += 1 }
                continue
            }
            let residentLoadInFlight = residentSlotByKey[key].map {
                !slots[$0].ready
            } ?? false

            let protected = Set(slots.indices.compactMap {
                slots[$0].resident && slots[$0].leases > 0 ? slots[$0].key : nil
            })
            let admission = heatmap.recordAccess(key: key,
                                                 byteCount: expertStride,
                                                 protectedKeys: protected)
            for victim in admission.evicted {
                if let slot = readySlotByKey[victim], slots[slot].leases == 0 {
                    readySlotByKey.removeValue(forKey: victim)
                    if residentSlotByKey[victim] == slot {
                        residentSlotByKey.removeValue(forKey: victim)
                    }
                    slots[slot].key = nil
                    slots[slot].ready = false
                }
            }

            let range = admission.isResident && !residentLoadInFlight
                ? 0..<residentSlotCount
                : residentSlotCount..<slotCount
            guard let selected = range.first(where: {
                slots[$0].leases == 0 && slots[$0].key == nil
            }) ?? range.first(where: {
                !slots[$0].resident && slots[$0].leases == 0
            }) else {
                for slot in leased { slots[slot].leases -= 1 }
                for slot in newlyAssigned {
                    if let key = slots[slot].key, readySlotByKey[key] == slot {
                        readySlotByKey.removeValue(forKey: key)
                    }
                    if let key = slots[slot].key, residentSlotByKey[key] == slot {
                        residentSlotByKey.removeValue(forKey: key)
                    }
                    slots[slot].key = nil
                    slots[slot].ready = false
                }
                let actualResidents = Set(readySlotByKey.keys)
                heatmap.reconcileResidency(actualResidents)
                lock.unlock()
                return nil
            }
            slots[selected].key = key
            slots[selected].ready = false
            if slots[selected].resident { residentSlotByKey[key] = selected }
            assigned[index] = selected
            misses.append(index)
            newlyAssigned.insert(selected)
            if leased.insert(selected).inserted { slots[selected].leases += 1 }
        }

        let leasedSlots = leased
        let lease = ExpertCacheLease { [weak self] in self?.release(slots: leasedSlots) }
        lock.unlock()
        return ExpertCachePlan(experts: experts,
                               assignedSlots: assigned,
                               misses: misses,
                               hits: experts.count - misses.count,
                               cacheLease: lease)
    }

    public func execute(_ plan: ExpertCachePlan,
                        load: @escaping @Sendable (_ destination: UnsafeMutableRawPointer,
                                                  _ expert: Int) throws -> Void) throws {
        guard plan.cacheLease?.beginExecution() == true else {
            throw StreamerError.cachePlanAlreadyExecuted
        }
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { offset in
            let index = plan.misses[offset]
            do { try load(self.pointers[plan.assignedSlots[index]], plan.experts[index]) }
            catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        lock.lock()
        if firstError == nil {
            for index in plan.misses {
                let slot = plan.assignedSlots[index]
                slots[slot].ready = true
                if slots[slot].resident, let key = slots[slot].key {
                    readySlotByKey[key] = slot
                }
            }
        } else {
            for index in plan.misses {
                let slot = plan.assignedSlots[index]
                if let key = slots[slot].key, readySlotByKey[key] == slot {
                    readySlotByKey.removeValue(forKey: key)
                }
                if let key = slots[slot].key, residentSlotByKey[key] == slot {
                    residentSlotByKey.removeValue(forKey: key)
                }
                slots[slot].key = nil
                slots[slot].ready = false
            }
        }
        lock.unlock()
        if let firstError { plan.cacheLease?.release(); throw firstError }
    }

    public func buffers(for plan: ExpertCachePlan) -> [(MTLBuffer, UInt64, UInt64)] {
        plan.assignedSlots.map { (buffers[$0], 0, expertStride) }
    }

    public func contains(layer: Int, expert: Int) -> Bool {
        let key = GlobalExpertKey(layer: layer, expert: expert)
        lock.lock(); defer { lock.unlock() }
        return readySlotByKey[key] != nil
    }

    func readyIndexInvariantHolds() -> Bool {
        lock.lock(); defer { lock.unlock() }
        let pairs: [(GlobalExpertKey, Int)] = slots.indices.compactMap { index in
            let slot = slots[index]
            guard slot.resident, slot.ready, let key = slot.key else { return nil }
            return (key, index)
        }
        let recomputed = Dictionary(uniqueKeysWithValues: pairs)
        let residentPairs: [(GlobalExpertKey, Int)] = slots.indices.compactMap { index in
            let slot = slots[index]
            guard slot.resident, let key = slot.key else { return nil }
            return (key, index)
        }
        let residentRecomputed = Dictionary(uniqueKeysWithValues: residentPairs)
        return readySlotByKey == recomputed && residentSlotByKey == residentRecomputed
    }

    public func preloadCandidates() -> [GlobalExpertKey] {
        heatmap.coldStartPreloadKeys(limit: residentSlotCount)
    }

    private func release(slots released: Set<Int>) {
        lock.lock()
        for index in released where slots[index].leases > 0 {
            slots[index].leases -= 1
            if slots[index].leases == 0 && !slots[index].resident {
                slots[index].key = nil
                slots[index].ready = false
            }
        }
        lock.unlock()
    }
}
