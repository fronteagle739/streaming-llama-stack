"""Authenticate the loaded RTX engine before declaring a resident pair ready."""
import json
import socket
import time


def wait_ready(port,token,model,compatibility,stall_seconds,check=lambda:None):
    """No model work or state transfer; the engine emits ready after its warm-up."""
    deadline=time.monotonic()+stall_seconds
    while True:
        check()
        if time.monotonic()>=deadline:raise TimeoutError('RTX readiness made no progress')
        try:
            connection=socket.create_connection(('127.0.0.1',port),timeout=1)
            break
        except (ConnectionRefusedError,socket.timeout):time.sleep(.1)
    with connection:
        connection.settimeout(1)
        connection.sendall((json.dumps(dict(op='pair',token=token,model_sha256=model,
            compatibility_sha256=compatibility))+'\n').encode())
        pending=b''
        while True:
            check()
            if time.monotonic()>=deadline:raise TimeoutError('RTX readiness made no progress')
            try:block=connection.recv(65536)
            except socket.timeout:continue
            if not block:raise RuntimeError('RTX closed before authenticated readiness')
            deadline=time.monotonic()+stall_seconds;pending+=block
            if len(pending)>65536:raise ValueError('RTX readiness frame exceeds bound')
            while b'\n' in pending:
                line,pending=pending.split(b'\n',1);row=json.loads(line)
                if not row.get('ok'):raise RuntimeError('RTX readiness rejected: '+str(row.get('error')))
                if row.get('event')=='ready':continue
                if (row.get('event')!='paired' or row.get('model_sha256')!=model or
                    row.get('compatibility_sha256')!=compatibility or row.get('family')!='qwen4exp' or
                    row.get('kv_bits')!=16 or not row.get('context')):
                    raise ValueError('RTX readiness identity/format mismatch')
                return row
