"""Bound chat JSON independently of the small native session control protocol."""
def chat_body_limit(cfg):
    # One admitted 20 MiB image needs about 28 MiB as JSON/base64. Allow text
    # and multiple ordinary images too, with a fixed controller-memory ceiling.
    return min(64 << 20, max(32 << 20, int(cfg.get('context', 262144)) * 32))
