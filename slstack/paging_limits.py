"""Optional stricter owned-paging limit using the shared guard's policy."""
from .native_broker import vendor


def shared_owned_paging_limit_bytes():
    """Return the canonical Studio allowance without a task-specific override."""
    vendor()
    from studio.job_paging import OWNED_PAGING_LIMIT
    return OWNED_PAGING_LIMIT


def owned_paging_limit_bytes(settings):
    """None retains the shared default; explicit values can only tighten it."""
    value = settings.get('owned_paging_limit_bytes')
    if value is None:
        return None
    vendor()
    from studio.job_paging import OWNED_PAGING_LIMIT
    if type(value) is not int or not 0 <= value <= OWNED_PAGING_LIMIT:
        raise ValueError(f'owned_paging_limit_bytes must be an integer in [0,{OWNED_PAGING_LIMIT}] or null')
    return value
