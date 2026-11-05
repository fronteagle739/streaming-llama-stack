"""Bounded QFN peer recovery; guard/protection failures are never restarted."""
import re
from .paging_limits import shared_owned_paging_limit_bytes


def restart_allowed(record, used, limit, diagnostic=''):
    required={'exit_code','reason','release_allowed','cleanup_confirmed',
              'owned_paged_bytes','paging_telemetry_closed','child_pid'}
    if not required<=record.keys() or not 0<=used<limit:
        return False
    if (record['reason'] is not None or record['release_allowed'] is not True
            or record['cleanup_confirmed'] is not True
            or record['paging_telemetry_closed'] is not True
            or not 0 <= record['owned_paged_bytes'] <= shared_owned_paging_limit_bytes()):
        return False
    # Normal unexpected exit, externally terminated engine or console break.
    # Access violations, aborts, allocation failures and unknown exits stop.
    if record['exit_code'] not in (0,1,15,-9,-15,0xC000013A):
        return False
    return not re.search(r'\boom\b|out of memory|memory limit|allocation failed|'
                         r'host_pin_failed|unhandled exception|fatal:',diagnostic,re.I)
