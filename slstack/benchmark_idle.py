"""The existing 15-second CPU/GPU quiet admission used by prefill proofs."""
import statistics
import subprocess
import psutil


def admit(gpu_uuid):
    samples=[]
    for _ in range(15):
        busy=[p.info['name'] for p in psutil.process_iter(['name'])
              if (p.info['name'] or '').lower() in {'nvcc.exe','cicc.exe','ptxas.exe','cl.exe','link.exe','blender.exe'}]
        if busy:raise RuntimeError('WAITING-IDLE: compiler/render processes: '+str(busy))
        samples.append(psutil.cpu_percent(interval=1))
    raw=subprocess.check_output(['nvidia-smi','--id='+gpu_uuid,
        '--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
    gpu,memory,watts=map(float,raw.strip().split(','))
    result=dict(cpu_percent=samples,cpu_median=statistics.median(samples),gpu_percent=gpu,
                memory_mib=memory,watts=watts)
    if result['cpu_median']>10 or gpu>5 or memory>1600:
        raise RuntimeError('WAITING-IDLE: '+str(result))
    return result
