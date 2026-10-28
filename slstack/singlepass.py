"""Carry registration policy for the existing fixed native owner."""


def registration(value):
    if value not in ('tile', 'bank'):
        raise ValueError('producer_carry_registration must be tile or bank')
    return value


def arguments(cfg, argv):
    mode = registration(cfg.get('producer_carry_registration', 'tile'))
    if mode == 'bank':
        if cfg.get('producer_keep_alive'):
            raise ValueError('Whole-bank carry registration is not covered by the switch arena')
        required = ('--pair-ffn-tiles', '--resident-hidden-bf16',
                    '--resident-compact-ids', '--resident-metadata', '--direct-carry-budget')
        if any(argv.get(key) != '1' for key in required):
            raise ValueError('Whole-bank carry registration requires the paired native resident plan')
        # The fixed owner already registers and unregisters its locked RAM bank
        # with an owning shared_ptr. Only its lifetime changes: compute shapes,
        # causal order, copy streams and capacity admission remain unchanged.
        argv['--tile-host-registration'] = '0'
    return argv
