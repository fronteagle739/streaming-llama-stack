"""Verify the existing native mode1 payload in its admitted shared RAM owner."""
from __future__ import annotations
import hashlib
import struct
from .combo_contract import VERSION, owner_layout, validate

TRANSPORT_FIELDS=('context','position','mode','capture_end','capture_length','tokens_sha256','logits_sha256','owner_sha256')


def transport_checkpoint(row):
    from studio.dsv41_session_contract import checkpoint
    return checkpoint({key:row[key] for key in TRANSPORT_FIELDS},1)


def payload_bytes(position):
    if type(position) is not int or not 0<position<262144:raise ValueError('Mode1 checkpoint position is outside the qualified262K scope')
    return 52+position*4+129280*4+sum(owner_layout(position).values())


def contract_from_checkpoint(checkpoint,identity,session,epoch,generation,provenance):
    cp=transport_checkpoint(checkpoint)
    layout=owner_layout(cp['position'])
    owners=[dict(kind=key[0],index=key[1],bytes=n,sha256=digest)
            for (key,n),digest in zip(layout.items(),cp['owner_sha256'],strict=True)]
    row={**identity,'version':VERSION,'mode':1,'kv_dtype':'f32','context':cp['context'],
         'position':cp['position'],'request_id':session,'owner_epoch':epoch,'generation':generation,
         'tokens_sha256':cp['tokens_sha256'],'logits_sha256':cp['logits_sha256'],
         'owners':owners,'decode_provenance':provenance}
    validate(row)
    if 'owners' in checkpoint:
        actual={(x['kind'],x['index']):(x['bytes'],x['sha256']) for x in checkpoint['owners']}
        expected={(x['kind'],x['index']):(x['bytes'],x['sha256']) for x in owners}
        if actual!=expected or len(checkpoint['owners'])!=len(owners):raise ValueError('Native owner kinds/indices differ from the versioned transport')
    return row


def verify_payload(ram,row,expected_payload_sha256=None):
    validate(row)
    size=payload_bytes(row['position'])
    if size>ram.capacity:raise MemoryError('Validated native payload exceeds its admitted RAM owner')
    expected=(0x34565344,2,row['context'],1,128,128,row['context']+1,row['position'],40,512,128,129280,0x413431)
    header=ram.read(0,52)
    if struct.unpack('<13I',header)!=expected:raise ValueError('Native payload header differs from the contract')
    full=hashlib.sha256();full.update(header);cursor=52

    def verify_region(count,expected_sha,kind=None):
        nonlocal cursor
        digest=hashlib.sha256()
        for off in range(0,count,65536):
            block=ram.read(cursor+off,min(65536,count-off));digest.update(block);full.update(block)
            if kind:
                for (word,) in struct.iter_unpack('<I',block):
                    if kind=='tokens' and word>=129280:raise ValueError('Invalid native token ID')
                    if kind=='logits' and word&0x7f800000==0x7f800000:raise ValueError('Nonfinite native logit')
        if digest.hexdigest()!=expected_sha:raise ValueError('Actual native '+(kind or 'owner')+' SHA256 differs')
        cursor+=count

    with ram.lease():
        verify_region(row['position']*4,row['tokens_sha256'],'tokens')
        verify_region(129280*4,row['logits_sha256'],'logits')
        owners={(x['kind'],x['index']):x for x in row['owners']}
        for key,count in owner_layout(row['position']).items():verify_region(count,owners[key]['sha256'])
    actual=full.hexdigest()
    if cursor!=size or expected_payload_sha256 is not None and actual!=expected_payload_sha256:
        raise ValueError('Actual native payload extent/SHA256 differs')
    return dict(bytes=size,payload_sha256=actual,position=row['position'],all_owner_hashes_verified=True)
