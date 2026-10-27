import Foundation
import Mference

/// Model facts needed to size the routed-expert cache without loading weights.
public struct ServerExpertCacheGeometry: Equatable, Sendable {
    public let expertsPerLayer: Int
    public let layerCount: Int
    public let expertStride: UInt64
    public let minimumSlots: Int
    /// Full resident weights file, including its index. Reserving the whole
    /// file is conservative relative to the mapped resident payload.
    public let residentBytes: UInt64

    public init(expertsPerLayer: Int,
                layerCount: Int,
                expertStride: UInt64,
                minimumSlots: Int,
                residentBytes: UInt64 = 0) {
        self.expertsPerLayer = expertsPerLayer
        self.layerCount = layerCount
        self.expertStride = expertStride
        self.minimumSlots = minimumSlots
        self.residentBytes = residentBytes
    }

    public static func load(modelDirectory: URL) throws -> Self {
        let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        guard let baseline = ArchConfig.knownArchitectures[family] else {
            throw ServerArgumentError.invalid("unsupported model family \(family.rawValue)")
        }
        let manifest = try ManifestReader.load(directoryURL: modelDirectory, expecting: baseline)
        return Self(expertsPerLayer: manifest.expertsPerLayer,
                    layerCount: manifest.numLayers,
                    expertStride: manifest.expertStride,
                    minimumSlots: manifest.arch.topKExperts,
                    residentBytes: manifest.files["model_weights.bin"]?.size ?? 0)
    }

    var bytesPerSlot: UInt64? {
        expertStride.multipliedReportingOverflow(by: UInt64(layerCount)).overflow
            ? nil
            : expertStride * UInt64(layerCount)
    }
}

public struct ResolvedServerExpertCache: Equatable, Sendable {
    public let slotCount: Int
    public let cacheBytes: UInt64
    public let wasHeadroomClamped: Bool

    /// Slot-only callers retain the legacy replacement policy. The command
    /// selects the global byte policy when `--expert-cache-gb` was supplied.
    public var policyDescription: String { "per-layer LFU" }
}

/// Selects one uniform cache capacity for each routed layer. The existing
/// streamer continues to keep independent LFU/LRU state per layer; this type
/// only controls that fixed capacity and does not introduce a global heatmap.
public struct ServerExpertCacheConfiguration: Equatable, Sendable {
    public let requestedSlots: Int?
    public let targetGigabytes: Double?
    public let headroomGigabytes: Double

    public init(requestedSlots: Int?,
                targetGigabytes: Double?,
                headroomGigabytes: Double) {
        self.requestedSlots = requestedSlots
        self.targetGigabytes = targetGigabytes
        self.headroomGigabytes = headroomGigabytes
    }

    public func resolve(
        geometry: ServerExpertCacheGeometry,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) throws -> ResolvedServerExpertCache {
        guard geometry.expertsPerLayer >= geometry.minimumSlots,
              geometry.minimumSlots > 0,
              geometry.layerCount > 0,
              geometry.expertStride > 0,
              let bytesPerSlot = geometry.bytesPerSlot,
              bytesPerSlot > 0 else {
            throw ServerArgumentError.invalid("model has invalid expert-cache geometry")
        }
        guard (requestedSlots == nil) != (targetGigabytes == nil) else {
            throw ServerArgumentError.invalid(
                "choose exactly one of --expert-cache-slots or --expert-cache-gb")
        }
        guard headroomGigabytes.isFinite, headroomGigabytes >= 0,
              let headroomBytes = Self.gibibytesToBytes(headroomGigabytes) else {
            throw ServerArgumentError.invalid("--expert-cache-headroom-gb must be finite and non-negative")
        }
        let (reservedBytes, reservedOverflow) = headroomBytes.addingReportingOverflow(
            geometry.residentBytes)
        guard !reservedOverflow, physicalMemoryBytes > reservedBytes else {
            throw ServerArgumentError.invalid(
                "resident model and expert-cache headroom leave no memory available for the cache")
        }

        let requestedCount: Int
        if let requestedSlots {
            guard requestedSlots > 0 else {
                throw ServerArgumentError.invalid("--expert-cache-slots must be positive")
            }
            requestedCount = requestedSlots
        } else if let targetGigabytes,
                  targetGigabytes.isFinite,
                  targetGigabytes > 0,
                  let targetBytes = Self.gibibytesToBytes(targetGigabytes) {
            requestedCount = Self.slotCount(forBytes: targetBytes, bytesPerSlot: bytesPerSlot)
        } else {
            throw ServerArgumentError.invalid("--expert-cache-gb must be finite and positive")
        }

        let memoryAvailableForCache = physicalMemoryBytes - reservedBytes
        let headroomSlotLimit = Self.slotCount(
            forBytes: memoryAvailableForCache,
            bytesPerSlot: bytesPerSlot)
        let slotCount = min(requestedCount, geometry.expertsPerLayer, headroomSlotLimit)
        guard slotCount >= geometry.minimumSlots else {
            throw ServerArgumentError.invalid(
                "expert-cache budget resolves to \(slotCount) slots; this model needs at least \(geometry.minimumSlots)")
        }
        let (cacheBytes, overflow) = bytesPerSlot.multipliedReportingOverflow(by: UInt64(slotCount))
        guard !overflow else {
            throw ServerArgumentError.invalid("expert-cache allocation size overflows UInt64")
        }
        return ResolvedServerExpertCache(
            slotCount: slotCount,
            cacheBytes: cacheBytes,
            wasHeadroomClamped: headroomSlotLimit < min(requestedCount, geometry.expertsPerLayer))
    }

    private static func slotCount(forBytes bytes: UInt64, bytesPerSlot: UInt64) -> Int {
        let count = bytes / bytesPerSlot
        return count > UInt64(Int.max) ? Int.max : Int(count)
    }

    private static func gibibytesToBytes(_ value: Double) -> UInt64? {
        let bytes = value * 1_073_741_824
        guard bytes.isFinite, bytes >= 0, bytes < Double(UInt64.max) else { return nil }
        return UInt64(bytes.rounded(.down))
    }
}
