"""Use Studio's four delivered arithmetic profiles with mandatory RAM residency."""
from copy import deepcopy
from .native_broker import vendor

MODES = ('accurate', 'fast', 'ultra', 'hyper')


def profile(mode, techniques=None):
    vendor()
    from studio.dsv41_session_profiles import profile as delivered_profile
    result = deepcopy(delivered_profile(mode, techniques))
    # The older ULTRA preset allowed a pageable staging copy. The stack's
    # residency requirement supersedes that storage policy in every mode.
    result['arguments'].update({'--resident-pageable-copies': '0', '--tile-host-registration': '1'})
    result['storage_policy'] = 'Anonymous locked RAM carry; registered tile uploads; no pageable resident copy or disk state'
    result['stack_qualification'] = 'See examples/engine_parity; a delivered arithmetic profile does not qualify this build'
    return result
