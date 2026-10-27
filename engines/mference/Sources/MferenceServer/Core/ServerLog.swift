import Foundation
import Mference

/// Request lifecycle logging. Responses stay deliberately generic so runtime
/// details never reach a client; the operator needs the opposite, so the log
/// carries the underlying error verbatim. Local stderr only.
///
/// Start and completion are both logged: a long prefill emits no output for
/// minutes, and without a start line that is indistinguishable from a wedged
/// server.
enum ServerLog {
    static func requestStarted(id: String, streaming: Bool) {
        write("request \(id) started streaming=\(streaming)")
    }

    static func requestCompleted(id: String,
                                 duration: Duration,
                                 completion: ServerCompletion) {
        write(requestCompletedMessage(id: id, duration: duration, completion: completion))
    }

    static func requestCompletedMessage(id: String,
                                        duration: Duration,
                                        completion: ServerCompletion) -> String {
        let usage = completion.usage
        var message = """
        request \(id) completed in \(format(duration)) \
        prompt=\(usage.promptTokens) \
        cached=\(usage.promptTokensDetails.cachedTokens) \
        completion=\(usage.completionTokens) \
        finish=\(completion.finishReason)
        """
        if let hotPath = completion.hotPath {
            message += " hot_path_request{\(fields(hotPath.request))}"
            message += " hot_path_cumulative{\(fields(hotPath.cumulative))}"
        }
        return message
    }

    static func requestFailed(id: String, status: UInt, streaming: Bool, error: any Error) {
        let detail = switch error {
        case let error as ServerRequestError: describe(error)
        default: String(describing: error)
        }
        write("request \(id) failed status=\(status) streaming=\(streaming) error=\(detail)")
    }

    static func streamAborted(id: String, reason: String) {
        write("request \(id) stream aborted: \(reason)")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.1fs", seconds)
    }

    private static func describe(_ error: ServerRequestError) -> String {
        let detail = error.envelope.error
        return "\(detail.code): \(detail.message)"
    }

    private static func fields(_ metrics: HotPathMetrics) -> String {
        "command_buffers=\(metrics.commandBufferCommits),"
            + "waits=\(metrics.completionWaits),wait_ns=\(metrics.completionWaitNanos),"
            + "router_waits=\(metrics.routerWaits),router_wait_ns=\(metrics.routerWaitNanos),"
            + "indexer_waits=\(metrics.indexerWaits),indexer_wait_ns=\(metrics.indexerWaitNanos),"
            + "cache_plans=\(metrics.cachePlanCalls),cache_plan_ns=\(metrics.cachePlanNanos),"
            + "cache_hits=\(metrics.cachePlanHits),cache_misses=\(metrics.cachePlanMisses)"
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
