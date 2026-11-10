"""Bounded origin-session mailbox. No GL work executes on HTTP threads.

Only the owning engine can register/publish/take/finish. HTTP acceptance is not
an estimator commit. Tokens are ephemeral, never persisted or put in URLs.
"""
import copy
import math
import secrets
import threading
import time

MAX_RAYS = 4096
MAX_PARENT_RAYS = 65536
MAX_TRIANGLES = 4096
MAX_BVH_TRIANGLES = 200000
MAX_SCENE_BYTES = 64 * 1024 * 1024
SCENE_FIELDS = ("triangles", "bvh_nodes", "bvh_ids", "bvh_epsilon", "query_variant", "query_scope", "unsupportedGeometry")
MAX_BODY = 16 * 1024 * 1024
QUERY_IDENTITY = ('query_geometry_hash','query_kernel')
IDENTITY = ('token', 'revision', 'sample_prefix', 'geometry_hash', 'kernel', 'query_kernel')

class RelayError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _text(value, maximum=256):
    return isinstance(value, str) and 0 < len(value) <= maximum


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _numbers(row, size):
    return isinstance(row, (list, tuple)) and len(row) == size and all(_finite(v) for v in row)


class ClientRayRelay:
    def __init__(self, clock=time.monotonic, max_jobs=8):
        self.clock, self.max_jobs = clock, max_jobs
        self.lock = threading.RLock()
        self.jobs = {}

    def register(self, job, owner_session, *, continuous=False):
        """Engine-only: bind a new isolated job to its originating browser."""
        if not _text(job) or not _text(owner_session):
            raise RelayError('Invalid job or owner')
        with self.lock:
            if job in self.jobs or len(self.jobs) >= self.max_jobs:
                raise RelayError('Job already registered or relay full', 409)
            if any(x['owner'] == owner_session for x in self.jobs.values()):
                raise RelayError('Session already owns a relay job', 409)
            credential = secrets.token_urlsafe(32)
            self.jobs[job] = dict(owner=owner_session, credential=credential, state='waiting',
                                  work=None, result=None, deadline=0, last_poll=-math.inf, scene=None, scene_ack=None, continuous=bool(continuous),ready_credit=False)
            return credential

    def publish(self, job, issued, query_kernel, seconds=10):
        """Engine-only. Lease lifetime must not exceed FirstHitLoan's lifetime."""
        if not _finite(seconds) or not 0 < seconds <= 60:
            raise RelayError('Lease duration must be 0..60 seconds')
        if not isinstance(issued,dict):raise RelayError('Expected issued work')
        work = copy.deepcopy(issued)
        if hasattr(work.get('rays'), 'tolist'):
            work['rays'] = work['rays'].tolist()
        work['query_kernel'] = query_kernel
        if 'query_geometry_hash' in work and not _text(work['query_geometry_hash']):raise RelayError('Invalid query geometry identity')
        if any(not _text(work.get(k)) for k in IDENTITY):
            raise RelayError('Missing immutable work identity')
        rays = work.get('rays')
        limit=MAX_PARENT_RAYS if work.get('parent_dispatch') is True else MAX_RAYS
        if not isinstance(rays, list) or not 1 <= len(rays) <= limit or not all(_numbers(r,8) for r in rays):
            raise RelayError('Invalid bounded ray batch')
        if any(r[3] < 0 or r[7] <= r[3] or sum(x*x for x in r[4:7]) <= 0 for r in rays):
            raise RelayError('Invalid ray interval or direction')
        if 'triangles' in work:
            self.set_scene(job, work, query_kernel)
        for field in SCENE_FIELDS:work.pop(field,None)
        with self.lock:
            entry = self.jobs[job]
            if entry['state'] not in ('waiting','expired','rejected','committed'):
                raise RelayError('Unconsumed batch already exists',409)
            if not entry['scene'] or any(entry['scene'].get(k)!=work.get(k) for k in (QUERY_IDENTITY if 'query_geometry_hash' in work else ('revision','geometry_hash','query_kernel'))):
                raise RelayError('Authoritative scene descriptor is not registered',409)
            entry.update(work=work,result=None,state='pending',deadline=self.clock()+seconds,published_at=self.clock(),ready_credit=False)

    def set_scene(self, job, descriptor, query_kernel):
        """Engine-only bounded authoritative packed scene; never an arbitrary page mesh."""
        import json
        if not isinstance(descriptor,dict):raise RelayError('Expected scene descriptor')
        scene={k:copy.deepcopy(descriptor[k]) for k in SCENE_FIELDS if k in descriptor}
        scene.update(revision=descriptor.get('revision'),geometry_hash=descriptor.get('geometry_hash'),query_kernel=query_kernel)
        if 'query_geometry_hash' in descriptor:
            if not _text(descriptor['query_geometry_hash']):raise RelayError('Invalid query geometry identity')
            scene['query_geometry_hash']=descriptor['query_geometry_hash']
        if any(not _text(scene.get(k)) for k in ('revision','geometry_hash','query_kernel')):
            raise RelayError('Invalid scene identity')
        if len(json.dumps(scene,allow_nan=False).encode())>MAX_SCENE_BYTES:
            raise RelayError('Scene descriptor exceeds 64 MiB',413)
        triangles=scene.get('triangles');bvh=scene.get('query_variant') in ('plain-bvh','region-bvh')
        limit=MAX_BVH_TRIANGLES if bvh else MAX_TRIANGLES
        if not isinstance(triangles,list) or not 1<=len(triangles)<=limit:
            raise RelayError('Unsupported empty or oversized triangle scene')
        owners=set()
        for t in triangles:
            owner=t.get('owner') if isinstance(t,dict) else None
            if type(owner) is not int or not 0<=owner<=16777215 or owner in owners:
                raise RelayError('Invalid triangle owner')
            owners.add(owner)
            if any(not _numbers(t.get(k),3) for k in ('v0','e1','e2')):
                raise RelayError('Invalid triangle geometry')
        if bvh:
            nodes,ids=scene.get('bvh_nodes'),scene.get('bvh_ids')
            if (not isinstance(nodes,list) or not 1<=len(nodes)<=400000 or not all(_numbers(n,12) for n in nodes)
                    or not isinstance(ids,list) or len(ids)!=len(triangles) or set(ids)!=owners
                    or scene.get('bvh_epsilon')!=1e-7):
                raise RelayError('Invalid bounded BVH descriptor')
        if scene.get('unsupportedGeometry'):raise RelayError('Unsupported geometry cannot be omitted')
        with self.lock:
            e=self.jobs[job]
            if e['state'] in ('pending','submitted','validating'):
                raise RelayError('Cannot replace scene while work is in flight',409)
            previous=e['scene']
            same_query=(previous is not None and 'query_geometry_hash' in scene and
                        all(previous.get(k)==scene.get(k) for k in QUERY_IDENTITY))
            if same_query:
                if any(previous.get(k)!=scene.get(k) for k in SCENE_FIELDS):
                    raise RelayError('Same query identity supplied different geometry',409)
                e.update(scene=scene,owners=owners) # Retain browser geometry acknowledgement across transport revisions.
            elif previous!=scene:e.update(scene=scene,scene_ack=None,owners=owners)

    def scene_ack(self, job, owner, credential, scene):
        with self.lock:
            e=self._auth(job,owner,credential)
            fields=QUERY_IDENTITY if e['scene'] and 'query_geometry_hash' in e['scene'] else ('revision','geometry_hash','query_kernel')
            if not isinstance(scene,dict) or not e['scene'] or any(scene.get(k)!=e['scene'][k] for k in fields):
                raise RelayError('Scene acknowledgement mismatch',409)
            e['scene_ack']={k:scene[k] for k in fields}
            return {'job':job,'scene_acknowledged':True}

    def scene_reset(self, job, owner, credential):
        with self.lock:
            e=self._auth(job,owner,credential)
            e['scene_ack']=None
            return {'job':job,'scene_acknowledged':False}

    def _auth(self, job, owner, credential):
        entry = self.jobs.get(job)
        if not entry or not isinstance(credential,str) or not isinstance(owner,str) or not secrets.compare_digest(entry['credential'],credential) or not secrets.compare_digest(entry['owner'],owner):
            raise RelayError('Unknown or unauthorized ray job',403)
        return entry

    def _expire(self, entry):
        if entry['state'] in ('pending','submitted') and self.clock() >= entry['deadline']:
            entry.update(state='expired',result=None)

    def poll(self, job, owner, credential):
        with self.lock:
            e=self._auth(job,owner,credential)
            if self.clock()-e['last_poll'] < .09:
                raise RelayError('Poll rate exceeds ten per second',429)
            e['last_poll']=self.clock();self._expire(e)
            if e['state'] in ('waiting','expired','rejected','committed'):e['ready_credit']=True
            result={'job':job,'state':e['state'],'continuous':e['continuous']}
            if e['state']=='pending':
                work=copy.deepcopy(e['work'])
                if e['scene_ack'] is None:
                    work.update({k:copy.deepcopy(e['scene'][k]) for k in (*SCENE_FIELDS,'query_geometry_hash') if k in e['scene']})
                    result['scene_ack_required']=True
                result.update(work=work,remaining_ms=max(0,int((e['deadline']-self.clock())*1000)))
            return result

    def submit(self, job, owner, credential, result):
        with self.lock:
            e=self._auth(job,owner,credential);self._expire(e)
            if e['state']!='pending':raise RelayError('Stale, duplicate or expired result',409)
            w=e['work']
            if not isinstance(result,dict) or any(result.get(k)!=w[k] for k in (*IDENTITY,*(['query_geometry_hash'] if 'query_geometry_hash' in w else []))):
                raise RelayError('Result identity differs from pending work',409)
            hits=result.get('hits')
            if not isinstance(hits,list) or len(hits)!=len(w['rays']) or not all(_numbers(h,4) for h in hits):
                raise RelayError('Invalid hit payload')
            owners=e['owners']
            for ray,hit in zip(w['rays'],hits):
                if hit[2] not in (0,1) or (hit[2]==1 and (hit[1] not in owners or not ray[3]<hit[0]<ray[7])):
                    raise RelayError('Hit outside permitted geometry or interval')
            profile=result.get('profile')
            if profile is not None:
                if (not isinstance(profile,dict) or set(profile)-{'client_compute_ms','rays','backend'} or
                        not _finite(profile.get('client_compute_ms')) or not 0<=profile['client_compute_ms']<=3600000 or
                        type(profile.get('rays')) is not int or profile['rays']!=len(w['rays']) or
                        not _text(profile.get('backend'),64)):
                    raise RelayError('Invalid client-reported timing profile')
            e.update(result=copy.deepcopy(result),state='submitted')
            return {'job':job,'state':'submitted','committed':False}

    def take_result(self, job):
        """Engine thread takes once, then validates and commits on its GL context."""
        with self.lock:
            e=self.jobs[job];self._expire(e)
            if e['state']!='submitted':return None
            result=e['result'];e.update(result=None,state='validating');return result

    def finish(self, job, outcome):
        if outcome not in ('committed','rejected','uncertain'):
            raise RelayError('Invalid engine outcome')
        with self.lock:
            e=self.jobs[job]
            if e['state']!='validating':raise RelayError('No result under validation',409)
            e.update(state=outcome,result=None)

    def cancel(self, job, owner, credential):
        with self.lock:
            e=self._auth(job,owner,credential)
            if e['state']=='validating':raise RelayError('Commit in progress; outcome must be resolved',409)
            if e['state'] not in ('committed','uncertain'):
                e.update(state='canceled',result=None)
            return {'job':job,'state':e['state']}

    def close(self, job):
        """Engine-only, after releasing/canceling its loan on the GL thread."""
        with self.lock:
            self.jobs.pop(job,None)


def handle_relay_post(handler, relay, owner_session, *, on_poll=None):
    """Existing authenticated server supplies owner from its HttpOnly session.

    No registration endpoint. This adapter is also used by the isolated engine
    harness; production callers must apply their existing edit/viewer checks.
    """
    import json
    import ssl
    from urllib.parse import urlsplit
    path=urlsplit(handler.path).path
    routes={'/api/client_rays/poll':'poll','/api/client_rays/result':'submit','/api/client_rays/cancel':'cancel','/api/client_rays/scene_ack':'scene_ack','/api/client_rays/scene_reset':'scene_reset'}
    if path not in routes:return False
    try:
        secure=isinstance(handler.connection,ssl.SSLSocket)
        if not secure and handler.client_address[0] not in ('127.0.0.1','::1'):
            raise RelayError('Ray participation requires HTTPS',403)
        origin=handler.headers.get('Origin')
        if origin and origin != ('https://' if secure else 'http://')+handler.headers.get('Host',''):
            raise RelayError('Cross-origin participation is not allowed',403)
        if handler.headers.get('X-Meowdoku-Rays')!='1':raise RelayError('Missing ray request header',403)
        if handler.headers.get('Content-Type','').split(';')[0]!='application/json':raise RelayError('Expected JSON')
        length=int(handler.headers.get('Content-Length','0'))
        if not 0<length<=MAX_BODY:raise RelayError('Invalid bounded request size',413)
        authorization=handler.headers.get('Authorization','')
        if not authorization.startswith('Bearer '):raise RelayError('Missing job credential',403)
        credential=authorization[7:]
        value=json.loads(handler.rfile.read(length))
        if not isinstance(value,dict):raise RelayError('Expected request object')
        job=value.get('job')
        if not _text(job):raise RelayError('Invalid job')
        args=(job,owner_session,credential)
        result=(relay.submit(*args,value.get('result')) if routes[path]=='submit' else relay.scene_ack(*args,value.get('scene')) if routes[path]=='scene_ack' else getattr(relay,routes[path])(*args))
        if routes[path]=='poll' and on_poll is not None:
            on_poll(job,owner_session)  # Only after relay credential and ownership validation.
        handler._json(result,202 if routes[path]=='submit' else 200)
    except (ValueError,TypeError) as error:
        handler.close_connection=True
        handler._json({'error':str(error)},getattr(error,'status',400))
    return True
