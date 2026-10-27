import Foundation
import Metal
import Darwin

public struct GlobalExpertPreloadStats: Sendable, Equatable {
    public let candidates: Int
    public let loadedExperts: Int
    public let loadedBytes: UInt64
    public let elapsedNanos: UInt64

    public var logDescription: String {
        let seconds = Double(elapsedNanos) / 1_000_000_000
        let elapsed = String(format: "%.3f", seconds)
        return "global_expert_preload candidates=\(candidates) loaded_experts=\(loadedExperts) "
            + "loaded_bytes=\(loadedBytes) elapsed_seconds=\(elapsed)"
    }
}

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }

    public func retainCacheSlots(until commandBuffer: MTLCommandBuffer) {
        cachePlan.retainCacheSlots(until: commandBuffer)
    }

    public func finishCacheSlotRetention() { cachePlan.finishCacheSlotRetention() }
}

extension Model {
    /// Materializes fingerprint-compatible persisted heat before server
    /// readiness. Batches stay small enough to bound concurrent SSD reads;
    /// leases release immediately after pread because no GPU command consumes
    /// these preload buffers.
    public func preloadGlobalExpertCache(batchSize: Int = 16) throws -> GlobalExpertPreloadStats? {
        guard let pool = streamersBox.globalPool else { return nil }
        precondition(batchSize > 0)
        let candidates = pool.preloadCandidates().filter {
            $0.layer >= 0 && $0.layer < packedExpertsLayout.numLayers
                && $0.expert >= 0 && $0.expert < packedExpertsLayout.expertsPerLayer
        }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var loaded = 0
        var loadedBytes: UInt64 = 0
        for layer in 0..<packedExpertsLayout.numLayers {
            let experts = candidates.filter { $0.layer == layer }.map(\.expert)
            var offset = 0
            while offset < experts.count {
                let end = min(experts.count, offset + batchSize)
                try ensureLayerOpened(layer)
                let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
                guard let plan = streamer.planExpertsCachedIfPossible(
                    experts: Array(experts[offset..<end])) else {
                    throw ModelError.routedExpertPlanUnavailable(layer: layer)
                }
                defer { plan.finishCacheSlotRetention() }
                _ = try streamer.executeExpertCachePlan(plan)
                loaded += plan.misses.count
                loadedBytes &+= UInt64(plan.misses.count) * streamer.layout.expertStride
                offset = end
            }
        }
        try pool.heatmap.persist()
        return GlobalExpertPreloadStats(candidates: candidates.count,
                                        loadedExperts: loaded,
                                        loadedBytes: loadedBytes,
                                        elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
    }

    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts, avoidingSlots: validSlots) else { return nil }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    /// Per-layer gate_proj quant group size, derived from the manifest's
    /// scale-slice byte size. DeepSeek V4's conversion ships gate_proj at
    /// group 32 on most layers and 64 on the last; up/down stay at the
    /// base group size.
    public func routedGateGroupSize(layer: Int) -> Int {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        guard let scales = expert.subTensors["gate_scales"] else {
            return Quantization.groupSize
        }
        let rows = UInt64(config.moeIntermediateSize)
        let cols = UInt64(config.hiddenSize)
        guard rows > 0 else { return Quantization.groupSize }
        let groups = scales.size / (2 * rows)
        guard groups > 0, cols.isMultiple(of: groups) else {
            return Quantization.groupSize
        }
        return Int(cols / groups)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        switch streamingMode {
        case .pread(let slotCount): return slotCount
        case .globalPread: return streamersBox.globalPool?.slotCount
        }
    }


    public func globalExpertCacheMetrics() -> GlobalExpertCacheMetrics? {
        streamersBox.globalPool?.heatmap.metrics()
    }

    public func flushGlobalExpertCachePersistence() throws {
        try streamersBox.globalPool?.heatmap.flushPersistence()
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: 0)
        }
    }
}
