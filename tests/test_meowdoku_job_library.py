import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku import job_library as lib

def job(tmp_path,digest,state='complete',live=False):
    p=tmp_path/digest;p.mkdir(parents=True)
    (p/'config.json').write_text(json.dumps({'live':{'session_id':'private'} if live else None,'private':'never return'}))
    (p/'status.json').write_text(json.dumps({'digest':digest,'state':state,'image':'photo.png','secret':'not for API'}))

def test_owners_and_account_binding(tmp_path):
    digest='a'*64
    lib.claim(tmp_path,digest,['browser:one'])
    assert lib.allowed(tmp_path,digest,['browser:one'])
    assert not lib.allowed(tmp_path,digest,['browser:two'])
    with pytest.raises(PermissionError):lib.claim(tmp_path,digest,['browser:two'])
    with pytest.raises(PermissionError):lib.claim(tmp_path,digest,['account:7'])
    assert not lib.allowed(tmp_path,digest,['account:7'])
    other='b'*64
    lib.claim(tmp_path,other,['account:7'])
    assert not lib.allowed(tmp_path,other,['browser:one'])
    assert not lib.allowed(tmp_path,other,['account:8'])

def test_list_real_states_and_legacy(tmp_path):
    a,b,c,d=[v*64 for v in 'abcd']
    for digest,state,live in [(a,'complete',False),(b,'exposing',False),(c,'live',True),(d,'complete',False)]:job(tmp_path,digest,state,live)
    lib.claim(tmp_path,a,['browser:one']);lib.claim(tmp_path,b,['browser:one']);lib.claim(tmp_path,c,['browser:two'])
    jobs=SimpleNamespace(directory=tmp_path,lock=threading.Lock(),active_digest=None,process=None)
    result=lib.entries(jobs,['browser:one'],[d,c])
    entries={v['digest']:v for v in result['entries']}
    assert set(entries)=={a,b,d}
    assert entries[b]['state']=='interrupted'
    assert not entries[b]['can_cancel']
    assert not result['queue_available'] and result['scheduling']=='single_gpu'
    assert 'secret' not in str(result) and 'private' not in str(result)
    assert lib.entries(jobs,['browser:stranger'])['entries']==[]
    assert lib.allowed(tmp_path,d,['browser:stranger']) # legacy access unchanged, NOT enumerated

def test_http_account_switch_same_browser(tmp_path):
    import http.client
    from http.server import ThreadingHTTPServer
    from meowdoku import server
    web=tmp_path/'web';web.mkdir();(web/'boards.json').write_text('{"boards":[],"levels":{}}')
    sessions=server.Sessions(web,None);sessions.lifecycle=None;sessions.access.lookup=lambda url:None
    a=sessions.accounts.signup('job_owner_a','test-password-one')
    b=sessions.accounts.signup('job_owner_b','test-password-two')
    ta=sessions.accounts.create_session(a['id']);tb=sessions.accounts.create_session(b['id'])
    digest='e'*64;job(sessions.marble_photos.directory,digest)
    lib.claim(sessions.marble_photos.directory,digest,['account:'+str(a['id'])])
    # Deliberately stale legacy association must not defeat indexed ownership.
    sessions.accounts.claim(b['id'],'marble_photo',digest)
    httpd=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(sessions))
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    browser=''
    def get(path,token):
        nonlocal browser
        c=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
        c.request('GET',path,headers={'Cookie':'; '.join(v for v in [browser,server.ACCOUNT_COOKIE+'='+token] if v)})
        r=c.getresponse();data=json.loads(r.read());status=r.status
        browser='; '.join(v.split(';')[0] for k,v in r.getheaders() if k.lower()=='set-cookie') or browser
        c.close();return status,data
    try:
        assert get('/api/marble_photo/jobs',ta)[1]['entries'][0]['digest']==digest
        assert get('/api/marble_photo/jobs',tb)[1]['entries']==[]
        assert get('/api/marble_photo?digest='+digest,tb)[0]==403
        assert get('/api/account/assets',tb)[1]['marble_photos']==[]
        assert get('/api/marble_photo/jobs','')[1]['entries']==[]
    finally:
        httpd.shutdown();httpd.server_close();thread.join();sessions.marble_photos.close()
