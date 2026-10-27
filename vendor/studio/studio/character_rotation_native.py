"""Studio's shared receipt context; application rotation tools are not exported."""
from contextvars import ContextVar

ROTATION_RECEIPT = ContextVar('rotation_native_receipt', default=False)
