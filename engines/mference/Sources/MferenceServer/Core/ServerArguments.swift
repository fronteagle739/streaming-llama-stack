import Darwin
import Foundation

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let bindMode: ServerBindMode
    public let bindAddress: String?
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default (gemma-4-26b-a4b-it or qwen3.6-35b-a3b).
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let expertCacheSlots: Int?
    public let expertCacheGigabytes: Double?
    public let expertCacheHeadroomGigabytes: Double

    public static let usage = """
    usage: MferenceServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Listening port (default 8080).
      --bind <mode>          loopback or tailnet (default loopback). tailnet
                             binds only the machine's Tailscale IPv4 address
                             and fails when Tailscale is unavailable.
      --bind-address <IP>    Bind one exact IPv4 or IPv6 literal. Mutually
                             exclusive with --bind; wildcard addresses and
                             hostnames are rejected.
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it or
                             qwen3.6-35b-a3b).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>
                             Fixed routed-expert cache slots per layer (default
                             16). Any positive count is accepted, then clamped
                             to the model's experts per layer and RAM headroom.
      --expert-cache-gb <GiB>
                             Target total routed-expert cache allocation. This
                             is mutually exclusive with --expert-cache-slots.
      --expert-cache-headroom-gb <GiB>
                             RAM reserved beyond the resident model for OS,
                             KV, and scratch (default 32). Applies to either
                             sizing mode.
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var bindMode = ServerBindMode.loopback
        var bindModeExplicit = false
        var bindAddress: String?
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var expertCacheSlots: Int? = 16
        var expertCacheSlotsExplicit = false
        var expertCacheGigabytes: Double?
        var expertCacheHeadroomGigabytes = 32.0
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--bind":
                guard let parsed = ServerBindMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--bind must be loopback or tailnet")
                }
                guard bindAddress == nil else {
                    throw ServerArgumentError.invalid(
                        "--bind and --bind-address are mutually exclusive")
                }
                bindMode = parsed
                bindModeExplicit = true
            case "--bind-address":
                guard !bindModeExplicit else {
                    throw ServerArgumentError.invalid(
                        "--bind and --bind-address are mutually exclusive")
                }
                bindAddress = try ServerBindAddress.validate(value)
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--expert-cache-slots must be positive")
                }
                guard expertCacheGigabytes == nil else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots and --expert-cache-gb are mutually exclusive")
                }
                expertCacheSlots = parsed
                expertCacheSlotsExplicit = true
            case "--expert-cache-gb":
                guard let parsed = Double(value), parsed.isFinite, parsed > 0 else {
                    throw ServerArgumentError.invalid("--expert-cache-gb must be finite and positive")
                }
                guard !expertCacheSlotsExplicit else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots and --expert-cache-gb are mutually exclusive")
                }
                // The 16-slot value is an implicit default, not an explicit
                // conflicting selection.
                expertCacheSlots = nil
                expertCacheGigabytes = parsed
            case "--expert-cache-headroom-gb":
                guard let parsed = Double(value), parsed.isFinite, parsed >= 0 else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-headroom-gb must be finite and non-negative")
                }
                expertCacheHeadroomGigabytes = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               bindMode: bindMode,
                               bindAddress: bindAddress,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCacheGigabytes: expertCacheGigabytes,
                               expertCacheHeadroomGigabytes: expertCacheHeadroomGigabytes)
    }
}

/// Literal-only bind validation. No hostname resolution or interface
/// selection occurs, so the address handed to NIO is exactly the one supplied.
public enum ServerBindAddress {
    public static func validate(_ text: String) throws -> String {
        var ipv4 = in_addr()
        if text.withCString({ inet_pton(AF_INET, $0, &ipv4) }) == 1 {
            guard ipv4.s_addr != in_addr_t(0) else {
                throw ServerArgumentError.invalid("--bind-address must not be 0.0.0.0")
            }
            return text
        }

        var ipv6 = in6_addr()
        if text.withCString({ inet_pton(AF_INET6, $0, &ipv6) }) == 1 {
            let bytes = withUnsafeBytes(of: &ipv6) { Array($0) }
            let unspecified = bytes.allSatisfy { $0 == 0 }
            let mappedIPv4Wildcard = bytes.prefix(10).allSatisfy { $0 == 0 }
                && bytes[10] == 0xff && bytes[11] == 0xff
                && bytes.suffix(4).allSatisfy { $0 == 0 }
            guard !unspecified, !mappedIPv4Wildcard else {
                throw ServerArgumentError.invalid("--bind-address must not be ::")
            }
            return text
        }

        throw ServerArgumentError.invalid(
            "--bind-address must be an exact IPv4 or IPv6 literal")
    }
}

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `.tailnet` to a wildcard or LAN address.
public enum ServerBindMode: String, Equatable, Sendable {
    case loopback
    case tailnet

    /// Resolves the listening address. `.tailnet` asks the Tailscale CLI for
    /// this machine's IPv4 address; `tailnetAddresses` is the seam tests use to
    /// supply that output without a Tailscale install.
    public func host(
        tailnetAddresses: () throws -> String = ServerBindMode.tailscaleIPv4Output
    ) throws -> String {
        switch self {
        case .loopback: "127.0.0.1"
        case .tailnet: try Self.tailnetHost(from: tailnetAddresses())
        }
    }

    /// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
    /// malformed, and off-range output all fail; none of them fall back.
    static func tailnetHost(from output: String) throws -> String {
        let fields = output.split(whereSeparator: \.isWhitespace)
        guard let first = fields.first else {
            throw ServerArgumentError.invalid(
                "tailscale reported no IPv4 address; ensure Tailscale is running and connected")
        }
        guard fields.count == 1 else {
            throw ServerArgumentError.invalid(
                "tailscale reported \(fields.count) IPv4 addresses; refusing to guess which to bind")
        }
        guard let address = tailscaleIPv4(String(first)) else {
            throw ServerArgumentError.invalid(
                "tailscale reported \"\(first.prefix(64))\", which is not a Tailnet IPv4 address")
        }
        return address
    }

    /// Returns the address unchanged when it is a dotted-quad IPv4 inside
    /// 100.64.0.0/10, the range Tailscale allocates from. Restricting to that
    /// range keeps a wildcard, loopback, or LAN address from ever being bound.
    private static func tailscaleIPv4(_ text: String) -> String? {
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [UInt8] = []
        octets.reserveCapacity(4)
        for part in parts {
            guard (1...3).contains(part.count),
                  part.allSatisfy({ $0.isASCII && $0.isNumber }),
                  part.count == 1 || part.first != "0",
                  let octet = UInt8(part) else { return nil }
            octets.append(octet)
        }
        guard octets[0] == 100, (64...127).contains(octets[1]) else { return nil }
        return text
    }

    /// Raw stdout of `tailscale ip -4`. Spawned directly with no shell, so
    /// nothing is interpolated into a command line.
    public static func tailscaleIPv4Output() throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["tailscale", "ip", "-4"]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw ServerArgumentError.invalid(
                "could not run tailscale; install its CLI and keep it on PATH")
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ServerArgumentError.invalid(
                "tailscale ip -4 exited with status \(process.terminationStatus); ensure the tailscale CLI is on PATH and Tailscale is running")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ServerArgumentError.invalid("tailscale ip -4 returned non-UTF-8 output")
        }
        return text
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
