import copy
import json
import pytest
from examples.meowdoku.client_ray_relay import ClientRayRelay, RelayError


def work():
    return dict(token='lease',revision='scene',sample_prefix='scene:0:0:0',geometry_hash='geometry',kernel='engine',
                rays=[[0,0,0,0,0,0,1,100]],triangles=[dict(owner=3,v0=[0,0,1],e1=[1,0,0],e2=[0,1,0])])

@pytest.fixture
def setup():
    clock=[0.]
    relay=ClientRayRelay(clock=lambda:clock[0])
    key=relay.register('job','origin')
    relay.publish('job',work(),'query',seconds=10)
    return relay,key,clock


def result(relay,key):
    w=relay.poll('job','origin',key)['work'];w.pop('rays');w.pop('triangles');w['hits']=[[1,3,1,0]]
    return w


def test_auth_and_no_cross_session(setup):
    r,key,_=setup
    for owner,token in [('other',key),('origin','wrong'),(None,key)]:
        with pytest.raises(RelayError,match='unauthorized'):r.poll('job',owner,token)
    with pytest.raises(RelayError):r.register('second','origin')


def test_received_is_not_committed_and_consumed_once(setup):
    r,key,t=setup;value=result(r,key)
    assert r.submit('job','origin',key,value)=={'job':'job','state':'submitted','committed':False}
    with pytest.raises(RelayError,match='duplicate'):r.submit('job','origin',key,value)
    assert r.take_result('job')==value
    assert r.take_result('job') is None
    with pytest.raises(RelayError):r.cancel('job','origin',key)
    r.finish('job','committed');t[0]=1
    assert r.poll('job','origin',key)['state']=='committed'


def test_expired_retains_no_commit_and_engine_reissues(setup):
    r,key,t=setup;value=result(r,key);t[0]=11
    with pytest.raises(RelayError):r.submit('job','origin',key,value)
    assert r.take_result('job') is None
    assert r.jobs['job']['work']['sample_prefix']==value['sample_prefix']
    next_work=work();next_work['token']='new-lease';r.publish('job',next_work,'query')
    with pytest.raises(RelayError,match='identity'):r.submit('job','origin',key,value)


def test_inflight_result_expiry_cannot_commit(setup):
    r,key,t=setup;r.submit('job','origin',key,result(r,key));t[0]=11
    assert r.take_result('job') is None
    assert r.jobs['job']['state']=='expired'

@pytest.mark.parametrize('field',['revision','sample_prefix','geometry_hash','kernel','query_kernel','token'])
def test_all_identity_fields_bound(setup,field):
    r,key,_=setup;v=result(r,key);v[field]='wrong'
    with pytest.raises(RelayError,match='identity'):r.submit('job','origin',key,v)

@pytest.mark.parametrize('hits',[[[float('nan'),3,1,0]],[[101,3,1,0]],[[1,7,1,0]],[[1,3,.5,0]],[]])
def test_invalid_results_reject(setup,hits):
    r,key,_=setup;v=result(r,key);v['hits']=hits
    with pytest.raises(RelayError):r.submit('job','origin',key,v)
    assert r.jobs['job']['state']=='pending'


def test_caps_and_rate_limit(setup):
    r,key,_=setup;r.poll('job','origin',key)
    with pytest.raises(RelayError,match='rate'):r.poll('job','origin',key)
    for index in range(7):r.register(str(index),str(index))
    with pytest.raises(RelayError,match='full'):r.register('overflow','overflow')
    r.close('job');r.register('new','origin')
    w=work();w['rays']*=4097
    with pytest.raises(RelayError):r.publish('new',w,'query')


def test_cancel_and_uncertain_never_report_commit(setup):
    r,key,t=setup
    assert r.cancel('job','origin',key)['state']=='canceled'
    with pytest.raises(RelayError):r.publish('job',work(),'query')
    assert r.take_result('job') is None


def test_actual_http_relay_origin_session_and_result(setup):
    import threading
    import http.client
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from http.cookies import SimpleCookie
    from examples.meowdoku.client_ray_relay import handle_relay_post
    r,key,t=setup
    renewed=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def _json(self,value,status=200):
            raw=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_POST(self):
            jar=SimpleCookie(self.headers.get('Cookie',''))
            owner=jar['session'].value if 'session' in jar else None
            if not handle_relay_post(self,r,owner,on_poll=lambda job,owner:renewed.append((job,owner))):self._json({'error':'no route'},404)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(action,payload=None,**headers):
        conn=http.client.HTTPConnection(*server.server_address)
        base={'Authorization':'Bearer '+key,'X-Meowdoku-Rays':'1','Content-Type':'application/json','Cookie':'session=origin'};base.update(headers)
        conn.request('POST','/api/client_rays/'+action,json.dumps(payload or {'job':'job'}),base)
        response=conn.getresponse();value=json.loads(response.read());conn.close();return response.status,value
    try:
        assert post('poll',Cookie='session=other')[0]==403
        assert post('poll',Origin='https://attacker.example')[0]==403
        assert post('register')[0]==404
        assert post('poll',Authorization='Bearer wrong')[0]==403
        assert renewed==[]
        status,v=post('poll');assert status==200 and v['state']=='pending'
        assert renewed==[('job','origin')]
        w=v['work'];w.pop('rays');w.pop('triangles');w['hits']=[[1,3,1,0]]
        assert post('result',{'job':'job','result':w})==(202,{'job':'job','state':'submitted','committed':False})
        assert r.take_result('job')==w;r.finish('job','committed');t[0]=1
        assert post('poll')[1]['state']=='committed'
    finally:server.shutdown();server.server_close();thread.join()


def test_huge_json_integer_rejects_without_overflow(setup):
    r,key,_=setup;value=result(r,key);value['hits'][0][0]=10**1000
    with pytest.raises(RelayError,match='payload'):r.submit('job','origin',key,value)
    assert r.jobs['job']['state']=='pending'

def test_scene_ack_elides_only_exact_resident_identity():
    clock=[0.];r=ClientRayRelay(clock=lambda:clock[0]);key=r.register('job','owner',continuous=True)
    w=work();w.update(query_variant='region-bvh',bvh_nodes=[[-1,-1,-1,-1,1,1,1,-1,0,1,0,0]],bvh_ids=[3],bvh_epsilon=1e-7)
    r.publish('job',w,'composite')
    first=r.poll('job','owner',key)
    assert first['scene_ack_required'] and first['work']['query_variant']=='region-bvh'
    with pytest.raises(RelayError):r.scene_ack('job','owner',key,{'revision':'scene','geometry_hash':'wrong','query_kernel':'composite'})
    r.scene_ack('job','owner',key,{'revision':'scene','geometry_hash':'geometry','query_kernel':'composite'})
    result={k:first['work'][k] for k in ('token','revision','sample_prefix','geometry_hash','kernel','query_kernel')};result['hits']=[[1,3,1,0]]
    r.submit('job','owner',key,result);r.take_result('job');r.finish('job','committed')
    next_work={k:v for k,v in work().items() if k!='triangles'};next_work['token']='second'
    r.publish('job',next_work,'composite');clock[0]=1
    cached=r.poll('job','owner',key)
    assert 'triangles' not in cached['work'] and 'bvh_nodes' not in cached['work']
    r.scene_reset('job','owner',key);clock[0]=2
    assert r.poll('job','owner',key)['work']['triangles']==w['triangles']

def test_query_geometry_survives_transport_revision_without_stale_result_authority():
    clock=[0.];r=ClientRayRelay(clock=lambda:clock[0]);key=r.register('job','owner',continuous=True)
    w=work();w['query_geometry_hash']='query-geometry'
    r.publish('job',w,'query')
    first=r.poll('job','owner',key)['work']
    r.scene_ack('job','owner',key,{'query_geometry_hash':'query-geometry','query_kernel':'query'})
    returned={k:first[k] for k in ('token','revision','sample_prefix','geometry_hash','kernel','query_kernel','query_geometry_hash')};returned['hits']=[[1,3,1,0]]
    r.submit('job','owner',key,returned);r.take_result('job');r.finish('job','committed')
    next_work={k:v for k,v in w.items() if k!='triangles'}
    next_work.update(token='new-token',revision='material-revision-2',geometry_hash='changed-transport',sample_prefix='new-sample')
    r.publish('job',next_work,'query');clock[0]=1
    packet=r.poll('job','owner',key)
    assert 'triangles' not in packet['work']
    assert packet['work']['revision']=='material-revision-2'
    assert packet['work']['geometry_hash']=='changed-transport'
    with pytest.raises(RelayError,match='identity'):r.submit('job','owner',key,returned)
    valid={k:packet['work'][k] for k in returned if k!='hits'};valid['hits']=returned['hits']
    bad={**valid,'query_geometry_hash':'other'}
    with pytest.raises(RelayError,match='identity'):r.submit('job','owner',key,bad)
    r.submit('job','owner',key,valid);r.take_result('job');r.finish('job','committed')
    wrong=copy.deepcopy(w);wrong['triangles'][0]['v0'][0]=3
    with pytest.raises(RelayError,match='different geometry'):r.set_scene('job',wrong,'query')


def test_retained_parent_is_bounded_separately_from_compute_packet():
    r=ClientRayRelay();r.register('job','owner')
    w=work();w['rays']*=4097
    with pytest.raises(RelayError):r.publish('job',w,'query')
    w['parent_dispatch']=True
    r.publish('job',w,'query')
    r.close('job');r.register('job','owner')
    w['rays']=[w['rays'][0]]*65537
    with pytest.raises(RelayError):r.publish('job',w,'query')
