# Native engine comparisons

The JSON examples compare 384 full-vocabulary Windows logit vectors and greedy choices against a zero-bias control, and a complete 32-token Mac state against an independent core/Metal build with identical settings. Mac source comparison normalizes CRLF line endings.

The carry-bank check inspects every page of a 4,536,143,872-byte anonymous locked bank and covers release and allocation failure. The producer comparison hashes the complete 10,559,816-byte state for 67 tokens in Accurate mode. These bounded comparisons do not establish equality across all arithmetic profiles, models or contexts.

No model weights or raw states are included. See [performance](../../docs/PERFORMANCE.md) for timing measurements and [YOCO](../../docs/ULTRAYOCO_QUALIFICATION.md) for its separate approximation error.
