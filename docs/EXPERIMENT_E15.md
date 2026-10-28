# E15: RTX 5090 power and clocks

Verdict: reject changing the power limit. Current, default and maximum
supported limits all report575W. No supported increase is available.

An independent45-second NVML observation during the original-Q4 indexer
campaign contains27 samples at GPU utilization80% or higher. Active clocks
are2737/2760/2797MHz and power423.37/495.52/516.89W (minimum/median/maximum).
Software power capping is active in16 of those27 samples; hardware and
software thermal slowdown are inactive in every active sample.

These independent one-second readings can miss short events. They do not
prove continuous boost or zero throttling. A sampled average below575W does
not exclude transient software power capping. No power or clock setting was
changed, and no clock-intervention speed gain is claimed. Earlier32K readings
also showed the575W limit; the machine was not running with the reference
system's reported400W limit.
