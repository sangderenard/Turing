import pytest
from examples.meowdoku.client_ray_process_bridge import ClientRayProcessBridge
from examples.meowdoku.client_ray_relay import RelayError

def test_process_credentials_are_separate_and_no_cross_owner():
    b=ClientRayProcessBridge();secret=b.register('job','account:1')
    browser=b.join('job','account:1')['credential']
    assert browser!=secret
    with pytest.raises(RelayError):b.join('job','account:2')
    with pytest.raises(RelayError):b.engine('job',browser,'take',{})
    assert b.engine('job',secret,'take',{})['result'] is None
    b.close('job')
    with pytest.raises(RelayError):b.join('job','account:1')

def test_actual_server_http_mailbox(tmp_path):
    import hashlib,http.client,json,sys,threading
    from pathlib import Path
    from http.server import ThreadingHTTPServer
    root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'examples')]
    from meowdoku import server
    web=tmp_path/'web';web.mkdir();(web/'boards.json').write_text('{"boards":[],"levels":{}}')
    sessions=server.Sessions(web,None);sessions.lifecycle=None;sessions.access.lookup=lambda url:None
    token,_,_=sessions.get(None);owner='browser:'+hashlib.sha256(token.encode()).hexdigest()
    bridge=sessions.client_ray_bridge;secret=bridge.register('job',owner)
    browser=bridge.join('job',owner)['credential']
    httpd=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(sessions));thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    def call(path,body,engine=False,credential=None):
        headers={'Content-Type':'application/json','Authorization':'Bearer '+(credential or (secret if engine else browser))}
        if engine:headers['X-Meowdoku-Engine-Job']='job'
        else:headers.update({'X-Meowdoku-Rays':'1','Cookie':server.COOKIE+'='+token})
        c=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5);c.request('POST','/api/client_rays/'+path,json.dumps(body),headers)
        r=c.getresponse();data=json.loads(r.read());status=r.status;c.close();return status,data
    scene={'revision':'r','geometry_hash':'g','triangles':[{'owner':3,'v0':[0,0,1],'e1':[1,0,0],'e2':[0,1,0]}]}
    work={'token':'lease','revision':'r','geometry_hash':'g','sample_prefix':'s','kernel':'engine','rays':[[0,0,0,0,0,0,1,100]]}
    try:
        assert call('engine/take',{},True,browser)[0]==403
        assert call('engine/scene',{'scene':scene,'query_kernel':'query'},True)[0]==200
        assert call('engine/ready',{},True)[1]['ready'] is False
        assert call('poll',{'job':'job'})[1]['state']=='waiting'
        assert call('engine/ready',{},True)[1]['ready'] is True
        assert call('engine/publish',{'work':work,'query_kernel':'query'},True)[0]==200
        __import__('time').sleep(.1)
        status,poll=call('poll',{'job':'job'})
        assert status==200 and poll['scene_ack_required']
        identity={k:poll['work'][k] for k in ('revision','geometry_hash','query_kernel')}
        assert call('scene_ack',{'job':'job','scene':identity})[0]==200
        result={**work,'query_kernel':'query','hits':[[1,3,1,0]]};result.pop('rays')
        assert call('result',{'job':'job','result':result})[1]['committed'] is False
        assert call('engine/take',{},True)[1]['result']==result
        assert call('engine/finish',{'outcome':'committed'},True)[1]['committed'] is True
    finally:httpd.shutdown();httpd.server_close();thread.join();sessions.marble_photos.close()
