"""Studio's output filename helper; simulation backends are not exported."""


def _safe(text: str, fallback: str = "x") -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in (text or "")[:24]).strip("_") or fallback
