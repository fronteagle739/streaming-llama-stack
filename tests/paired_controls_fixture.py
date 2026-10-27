"""Real WebUI/toolbar CPU fixture. No engine, model mapping or decode allowed.

Run under the stack's guarded command, passing a scratch output directory and
a Node browser driver. This is not the required two-machine inference gate.
"""
import argparse,json,subprocess,sys,threading
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slstack.config import load
from slstack.engines import Engines
from slstack.router import Stack,handler
from slstack.pairing import Pair
from slstack.paired_generation import decode_request

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--driver',required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=load(overrides=dict(runtime_root=str(out/'runtime'),host_budget_gib=2,min_available_gib=8,
                            mac='http://example.test:8180',gpu_uuid='GPU-cpu-fixture',webui=True))
    class CPUOnly(Engines):
        def ready(self):raise AssertionError('CPU WebUI fixture must never start an engine')
    stack=Stack(cfg,CPUOnly(cfg));base=handler(stack)
    requests=[]
    class Fixture(base):
        def do_POST(self):
            if self.path!='/test/relay':return super().do_POST()
            n=int(self.headers['Content-Length']);assert 0<n<1<<20
            body=json.loads(self.rfile.read(n));pair=Pair(cfg)
            actual=decode_request(pair.decode_request(body))
            requests.append(actual)
            (out/('mac-request-'+str(len(requests))+'.json')).write_text(json.dumps(actual,indent=2))
            (out/'mac-request.json').write_text(json.dumps(actual,indent=2))
            self.send(200,actual)
    server=ThreadingHTTPServer(('127.0.0.1',0),Fixture);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        subprocess.run(['node',args.driver,'http://127.0.0.1:'+str(server.server_port),str(out)],check=True)
    finally:
        server.shutdown();server.server_close();thread.join();stack.engines.close()
if __name__=='__main__':main()
