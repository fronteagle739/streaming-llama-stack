from slstack.memory_counters import darwin_reclaimable_bytes


def test_active_file_cache_is_reclaimable_without_double_counting():
    text='Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 5.\nFile-backed pages: 100.\n'
    assert darwin_reclaimable_bytes(text)==110*16384
    assert darwin_reclaimable_bytes(text.replace('100.','15.'))==35*16384
