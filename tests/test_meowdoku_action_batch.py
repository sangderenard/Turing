from __future__ import annotations
import http.client
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku import server as srv, action_batch, idle_trainer

@pytest.fixture
def service(tmp_path, monkeypatch):
    # Real puzzle implementation, temporary tables; never start a trainer/network game.
    monkeypatch.setattr(idle_trainer,'touch',lambda:None)
    board=json.loads((ROOT/'build/meowdoku/web/boards.json').read_text())['boards'][0]
    web=tmp_path/'web'; web.mkdir()
    (web/'boards.json').write_text(json.dumps({'boards':[board],'levels':{}}))
    sessions=srv.Sessions(web,None); sessions.lifecycle=None
    httpd=ThreadingHTTPServer(('127.0.0.1',0),srv.make_handler(sessions))
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    def call(body,path='/api/action/batch',cookie=None,header=True):
        headers={'Content-Type':'application/json'}
        if header: headers['X-Meowdoku-Batch']='1'
        if cookie: headers['Cookie']=cookie
        c=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=10)
        c.request('POST',path,json.dumps(body),headers)
        r=c.getresponse(); result=json.loads(r.read()); cookies=r.getheaders();status=r.status;c.close()
        cookie='; '.join(v.split(';')[0] for k,v in cookies if k.lower()=='set-cookie') or cookie
        return status,result,cookie
    yield call,sessions,board
    httpd.shutdown();httpd.server_close();thread.join()

def new(call,board):
    status,state,cookie=call({'game':'meowdoku','id':board['id']},'/api/new')
    assert status==200
    _,meta,_=call({'op':'describe'},cookie=cookie)
    return cookie,meta

def packet(meta,actions,seq=0):
    return {'table_epoch':meta['table_epoch'],'sequence':seq,'actions':actions}

CELL={'type':'cell','r':0,'c':0,'button':'mouse'}

def test_real_two_session_and_receipt(service):
    call,sessions,board=service
    a,ma=new(call,board);b,mb=new(call,board)
    p=packet(ma,[CELL])
    status,result,_=call(p,cookie=a)
    assert status==200 and result['completed']==1 and result['status']=='completed'
    assert call(p,cookie=a)[1]==result
    assert call(p,cookie=b)[0]==400
    assert call(packet(mb,[CELL]),cookie=b)[1]['completed']==1
    assert call(packet(ma,[dict(CELL,c=1)]),cookie=a)[0]==400
    assert call(packet(ma,[CELL],seq=2),cookie=a)[0]==400
    assert call(packet(ma,[CELL],seq=1),cookie=a)[1]['completed']==1
    assert call(p,cookie=a)[0]==400 # old receipt evicted, no replay

def test_stop_partial_and_epoch(service):
    call,_,board=service;a,m=new(call,board)
    p=packet(m,[CELL,{'type':'not_a_real_action'},CELL])
    r=call(p,cookie=a)[1]
    assert r['completed']==1 and r['failed_index']==1
    assert r['failed_action_may_have_mutated'] and r['status']=='stopped'
    assert call(p,cookie=a)[1]==r
    call({'game':'meowdoku','id':board['id']},'/api/new',cookie=a)
    assert call(packet(m,[CELL],1),cookie=a)[0]==400

def test_auth_preflight_and_bounds(service):
    call,sessions,board=service;a,m=new(call,board)
    sessions.access.config()['policy']['lan']='play'
    p=packet(m,[CELL,{'type':'demo_learning','on':True}])
    assert call(p,cookie=a)[0]==403
    assert call({'op':'describe'},cookie=a)[1]['next_sequence']==0
    assert call(packet(m,[CELL]),cookie=a,header=False)[0]==403
    assert call(packet(m,[CELL]*33),cookie=a)[0]==400
    assert call({**packet(m,[CELL]),'table':'other'},cookie=a)[0]==400
    assert call(packet(m,[dict(CELL,extra='x'*70000)]),cookie=a)[0]==413

@pytest.mark.parametrize('bad',[None,[],{'op':'describe','owner':'other'}, {'table_epoch':'a'*32,'sequence':True,'actions':[CELL]}, {'table_epoch':'a'*32,'sequence':0,'actions':[{'type':'cell','x':float('nan')}]}])
def test_invalid(bad):
    with pytest.raises(ValueError):action_batch.validate(bad)
