import Darwin
import Foundation
import MferenceServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let host = try arguments.bindAddress ?? arguments.bindMode.host()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let cacheGeometry = try ServerExpertCacheGeometry.load(modelDirectory: modelURL)
    let cache = try ServerExpertCacheConfiguration(
        requestedSlots: arguments.expertCacheSlots,
        targetGigabytes: arguments.expertCacheGigabytes,
        headroomGigabytes: arguments.expertCacheHeadroomGigabytes).resolve(
            geometry: cacheGeometry)
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        expertCacheSlots: cache.slotCount,
        globalExpertCacheBytes: arguments.expertCacheGigabytes == nil ? nil : cache.cacheBytes,
        promptCacheMode: arguments.promptCacheMode)
    let modelID = arguments.modelIDOverride ?? backend.defaultModelID
    let server = MferenceHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        chatDialect: backend.chatDialect)
    _ = try await server.start(host: host, port: arguments.port)
    let displayHost = host.contains(":") ? "[\(host)]" : host
    let cacheGiB = Double(cache.cacheBytes) / 1_073_741_824
    let cacheGiBText = String(format: "%.2f", cacheGiB)
    let cachePolicy = arguments.expertCacheGigabytes == nil ? "per-layer-lfu" : "global-byte-lfu"
    let displayedSlots = arguments.expertCacheGigabytes == nil
        ? cache.slotCount : Int(cache.cacheBytes / cacheGeometry.expertStride)
    print("MferenceServer ready at http://\(displayHost):\(arguments.port) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue) expert_cache_slots=\(displayedSlots) expert_cache_gib=\(cacheGiBText) policy=\(cachePolicy) headroom_clamped=\(cache.wasHeadroomClamped)")
    // Supervisors watch for the ready line through a pipe or log file, where
    // stdout is block-buffered and would otherwise hold it back indefinitely.
    fflush(stdout)

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
