"""Negotiated bounded-prefill provenance; native mode3 owner layout stays unchanged."""


def state_version(keep):
    if type(keep) is not int or keep not in (0, 140, 2541):
        raise ValueError('yoco_keep must be 0, 140 or 2541')
    return {0: 0, 140: 1, 2541: 2}[keep]


def require_state(keep, version):
    if type(version) is not int or version != state_version(keep):
        raise ValueError('Unsupported paired YOCO state provenance')
    return version
