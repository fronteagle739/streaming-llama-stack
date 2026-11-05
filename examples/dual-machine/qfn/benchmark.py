"""Borrow an idle Studio Mac preset, run the QFN pair, then restore it exactly.

Run under qfn_dual_guard.py. The command after -- is pair.py with all of its
explicit flags; this controller supplies only its own parent Mac lease ID.
"""
import argparse,base64,datetime,json,subprocess,sys,time,urllib.parse,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from pair import mac_lease

def main():
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('--studio-url',required=True);p.add_argument('--studio-token-file',type=Path,required=True)
 p.add_argument('--hold-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--pair-output',type=Path,required=True);p.add_argument('--ssh-host',required=True);p.add_argument('--mac-python',required=True)
 p.add_argument('--mac-host-budget-gib',type=float,required=True);p.add_argument('--mac-floor-gib',type=float,required=True)
 p.add_argument('--idle-seconds',type=int,default=60);p.add_argument('--run-timeout',type=int,default=0)
 p.add_argument('--restore-timeout',type=int,default=300);p.add_argument('--lease-seconds',type=int,default=1140)
 p.add_argument('--progress-log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 command=a.command[1:] if a.command[:1]==['--'] else a.command
 u=urllib.parse.urlsplit(a.studio_url)
 if u.scheme!='http' or u.hostname not in ('127.0.0.1','::1') or u.path not in ('','/') or u.query or u.fragment or u.username:
  raise ValueError('Studio control URL must be a loopback HTTP origin')
 if min(a.idle_seconds,a.restore_timeout)<=0 or a.run_timeout<0 or not 0<a.lease_seconds<=1140 or (a.run_timeout and a.run_timeout+a.idle_seconds+a.restore_timeout+90>a.lease_seconds):
  raise ValueError('Reserve the full restoration window within the lease/outer guard deadline')
 def setting(name):
  if command.count(name)!=1:raise ValueError('Pair command requires exactly one '+name)
  return command[command.index(name)+1]
 if not command or '--parent-mac-lease' in command or Path(setting('--output')).resolve()!=a.pair_output.resolve():raise ValueError('Pair output/lease differs')
 if (setting('--ssh-host')!=a.ssh_host or setting('--mac-python')!=a.mac_python or
     float(setting('--mac-host-budget-gib'))!=a.mac_host_budget_gib or float(setting('--mac-floor-gib'))!=a.mac_floor_gib or
     Path(setting('--hold-file')).resolve()!=a.hold_file.resolve() or (a.run_timeout and int(setting('--seconds'))>a.run_timeout-20)):
  raise ValueError('Pair resource/endpoint flags differ from the benchmark controller')
 a.output.mkdir(parents=True,exist_ok=False);token=a.studio_token_file.read_text().strip()
 def api(route,post=False,timeout=60):
  req=urllib.request.Request(a.studio_url.rstrip('/')+route,data=b'{}' if post else None,headers={'X-Studio-Token':token,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
 def note(doing,latest,next_step):
  line=f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} doing={doing}; latest={latest}; next={next_step}'
  print(line,flush=True)
  with a.progress_log.open('a',encoding='utf8') as f:f.write(line+'\n')
 def inventory():
  code='''import json,psutil
rows=[]
for p in psutil.process_iter(['pid','create_time','cmdline']):
 c=p.info['cmdline'] or []
 if c and (c[0].endswith('/ds4-server') or c[0].endswith('/qfn-dual-decode')):rows.append(p.info)
print(json.dumps(dict(processes=rows,available=psutil.virtual_memory().available)))
'''
  payload=base64.b64encode(code.encode()).decode()
  import shlex
  args=[a.mac_python,'-B','-c','import base64;exec(base64.b64decode('+repr(payload)+'))']
  result=subprocess.run(['ssh','-o','BatchMode=yes',a.ssh_host,shlex.join(args)],capture_output=True,text=True,check=True)
  return json.loads(result.stdout)
 before=api('/presets/active');presets=api('/presets');mac_before=inventory()
 if not before.get('server_live') or not before.get('mac_live') or not before.get('preset') or before.get('switch_in_flight'):
  raise ValueError('A healthy selected Mac serving preset is required before borrowing it')
 (a.output/'before.json').write_text(json.dumps(dict(active=before,presets=presets,mac=mac_before),indent=2))
 settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};settings['command']=command
 (a.output/'settings.json').write_text(json.dumps(settings,indent=2))
 identity=uuid.uuid4().hex;ejected=False;child=None;code=1
 with mac_lease(a.hold_file,identity,a.mac_host_budget_gib,a.mac_floor_gib,a.lease_seconds) as lease:
  try:
   deadline=time.monotonic()+a.idle_seconds
   while time.monotonic()<deadline:
    q=api('/llm/queue?peers=true');peers=q.get('peers',[])
    if q.get('count') or q.get('engines') or any(p.get('engines') for p in peers):raise RuntimeError('LLM queue became busy before ejection')
    mac_host=before['preset'].get('mac',{}).get('host') or urllib.parse.urlsplit(before['endpoint']).hostname
    peer=next((p for p in peers if p.get('host')==mac_host),None)
    if not peer or peer.get('error'):raise RuntimeError('The serving Mac queue is unreachable; idle cannot be verified')
    time.sleep(min(5,max(.01,deadline-time.monotonic())))
   if api('/presets/active')['preset']!=before['preset']:raise RuntimeError('Selected definition changed during idle admission')
   note('benchmark eject',f'queues idle{a.idle_seconds}s; full preset/launch contract saved','normal Studio eject')
   ejected=True;(a.output/'eject.json').write_text(json.dumps(api('/llm/eject',True),indent=2))
   deadline=time.monotonic()+90
   while True:
    clean=inventory()
    if not clean['processes']:break
    if time.monotonic()>deadline:raise RuntimeError('Normal eject did not free the native instance; no direct kill/bypass')
    time.sleep(3)
   (a.output/'clean.json').write_text(json.dumps(clean,indent=2));note('Mac clean',str(clean),'guarded QFN pair')
   child=subprocess.Popen([*command,'--parent-mac-lease',identity],cwd=ROOT);code=child.wait(timeout=a.run_timeout or None)
  finally:
   cleanup_errors=[]
   try:
    if child and child.poll()is None:
     try:child.wait(timeout=70)
     except subprocess.TimeoutExpired:
      child.terminate()
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:cleanup_errors.append('Owned controller did not terminate')
    if ejected:
     cert=a.pair_output/'mac.guard.json'
     if cert.exists() and not json.loads(cert.read_text()).get('release_allowed'):
      cleanup_errors.append('Owned QFN guard cleanup is unconfirmed')
     if any(p['cmdline'][0].endswith('/qfn-dual-decode') for p in inventory()['processes']):
      cleanup_errors.append('Owned QFN decoder remains')
   except BaseException as error:cleanup_errors.append(repr(error))
   finally:
    # Restoration is outermost: a wait, certificate, inventory or SSH error
    # above can never skip the normal switch back to the original preset.
    if ejected:
     note('restoring prior preset','owned QFN job gone','normal same-preset switch and full readback')
     job=api('/presets/'+before['preset']['id']+'/switch',True,120);(a.output/'restore-job.json').write_text(json.dumps(job,indent=2))
     deadline=time.monotonic()+a.restore_timeout;restored=False
     while time.monotonic()<deadline:
      active=api('/presets/active')
      if active.get('server_live') and active.get('mac_live') and not active.get('switch_in_flight'):
       after=inventory()
       if active['preset']!=before['preset'] or active.get('endpoint')!=before.get('endpoint') or api('/presets')!=presets:
        raise RuntimeError('Preset definition/store/endpoint differs after restoration')
       if len(after['processes'])!=1 or len(mac_before['processes'])!=1 or after['processes'][0]['cmdline']!=mac_before['processes'][0]['cmdline']:
        raise RuntimeError('Native launch contract differs after restoration')
       result=dict(preset_id=active['preset']['id'],endpoint=active['endpoint'],native_pid=after['processes'][0]['pid'],
        preset_store_equal=True,launch_contract_equal=True,health=api('/health'),mac=after,
        original_process_continuity=after['processes'][0]['pid']==mac_before['processes'][0]['pid'] and after['processes'][0]['create_time']==mac_before['processes'][0]['create_time'])
       (a.output/'restored.json').write_text(json.dumps(result,indent=2));restored=True
       note('prior preset restored',f'full definition/endpoint/native argv match; PID{result["native_pid"]}','release benchmark lease');break
      time.sleep(3)
     if not restored:raise RuntimeError('Prior serving preset did not become healthy before restoration deadline')
   if cleanup_errors:
    (a.output/'cleanup-errors.json').write_text(json.dumps(cleanup_errors,indent=2))
    raise RuntimeError('Production restore attempted; cleanup errors: '+repr(cleanup_errors))
   lease['cleanup_confirmed']=True
 return code
if __name__=='__main__':raise SystemExit(main())
