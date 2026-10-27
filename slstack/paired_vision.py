"""Bounded authenticated native projection envelope (SLVP v1).

The receiver owns vision encoding. The RTX gets byte-identical SLVI embeddings,
their SHA256 and absolute positions; it never treats image IDs as text. Files
are immutable, task-owned and never shared by path supplied over the network.
"""
import hashlib,json,struct
from pathlib import Path

MEDIA='application/vnd.studio.dsv41.projected-v1'
MAX_BODY=64<<20

def unpack(data,*,base=0):
    def require(ok,why):
        if not ok:raise ValueError('Projected vision '+why)
    require(isinstance(data,bytes) and 12<=len(data)<=MAX_BODY,'body extent')
    require(data[:4]==b'SLVP','version')
    nt,nm=struct.unpack_from('<II',data,4)
    require(0<nt<=49152*4 and nt%4==0 and 0<nm<=65536 and 12+nt+nm<=len(data),'header extent')
    tokens=data[12:12+nt];ids=struct.unpack('<%dI'%(nt//4),tokens)
    require(all(t<129280 for t in ids),'token range')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate metadata');result[key]=value
        return result
    rows=json.loads(data[12+nt:12+nt+nm],object_pairs_hook=pairs)
    require(type(rows) is list and 1<=len(rows)<=16,'image count')
    cursor=12+nt+nm;previous=0;images=[];covered=set()
    for row in rows:
        require(type(row) is dict and set(row)=={'position','count','sha256'},'descriptor keys')
        start,count,digest=row['position'],row['count'],row['sha256']
        require(type(start) is int and type(count) is int and 0<count<=32768 and start>=previous and start+count<1000001,'span extent/overlap')
        n=48+count*5120*4
        require(cursor+n<=len(data),'embedding truncated')
        raw=data[cursor:cursor+n];cursor+=n
        require(struct.unpack_from('<4I',raw)==(0x49564c53,1,count,5120),'embedding geometry')
        require(hashlib.sha256(raw).hexdigest()==digest,'embedding SHA256')
        for at in range(max(base,start),min(base+len(ids),start+count)):
            require(ids[at-base]==129264,'image marker mismatch');covered.add(at-base)
        images.append(dict(row,data=raw));previous=start+count
    require(cursor==len(data),'trailing data')
    require(all(t!=129264 or i in covered for i,t in enumerate(ids)),'unbound image token')
    return tokens,images

def materialize(directory,images,*,previous=(),base=0):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    prior={row['position']:(row['count'],row['sha256']) for row in previous}
    current={row['position']:(row['count'],row['sha256']) for row in images}
    if any(current.get(pos)!=identity for pos,identity in prior.items() if pos<base):
        raise ValueError('Projected vision historical image identity changed')
    if any(pos<base and prior.get(pos)!=identity for pos,identity in current.items()):
        raise ValueError('Projected vision historical image inserted')
    rows=[]
    for image in images:
        raw=image['data'];digest=image['sha256']
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Projected vision staged SHA256')
        file=directory/(digest+'.vision')
        if file.exists():
            if file.read_bytes()!=raw:raise ValueError('Projected vision owned file changed')
        else:
            with file.open('xb') as f:f.write(raw)
        rows.append(dict(position=image['position'],count=image['count'],file=str(file),sha256=digest))
    encoded=json.dumps(rows,separators=(',',':')).encode()
    descriptor=directory/('spans-'+hashlib.sha256(encoded).hexdigest()+'.json')
    if not descriptor.exists():
        with descriptor.open('xb') as f:f.write(encoded)
    elif descriptor.read_bytes()!=encoded:raise ValueError('Projected vision descriptor changed')
    return descriptor
