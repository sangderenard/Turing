"""Authenticated process mailbox for Nodus-owned first-hit loans.

This does not schedule or execute ray work. Only the registered engine worker
may publish/take/finish; only the originating identity may join the browser side.
HTTP acceptance is not a Nodus receipt or an estimator commit.
"""
import secrets
import threading
from .client_ray_relay import ClientRayRelay, RelayError

class ClientRayProcessBridge:
    def __init__(self):
        self.relay=ClientRayRelay()
        self.lock=threading.RLock()
        self.workers={}

    def register(self,job,owner):
        with self.lock:
            if job in self.workers:raise RelayError('Worker already enrolled',409)
            credential=self.relay.register(job,owner,continuous=True)
            secret=secrets.token_urlsafe(32)
            self.workers[job]={'owner':owner,'secret':secret,'credential':credential}
            return secret

    def join(self,job,owner):
        with self.lock:
            entry=self.workers.get(job)
            if not entry or not secrets.compare_digest(entry['owner'],owner):
                raise RelayError('No authorized client-ray worker for this job',403)
            return {'job':job,'credential':entry['credential'],'scope':'primary intersections; server spheres and transport',
                    'scene_residency':'authoritative packed BVH after acknowledgement','continuous':True}

    def authenticate(self,job,secret):
        with self.lock:
            entry=self.workers.get(job)
            if not entry or not isinstance(secret,str) or not secrets.compare_digest(entry['secret'],secret):
                raise RelayError('Unknown engine job',403)

    def engine(self,job,secret,action,body):
        with self.lock:
            entry=self.workers.get(job)
            if not entry or not isinstance(secret,str) or not secrets.compare_digest(entry['secret'],secret):
                raise RelayError('Unknown engine job',403)
        if action=='ready':
            with self.relay.lock:
                e=self.relay.jobs[job]
                self.relay._expire(e)
                return {'ready':bool(e['ready_credit'] and self.relay.clock()-e['last_poll']<1.0), 'state':e['state']}
        if action=='scene':
            self.relay.set_scene(job,body.get('scene'),body.get('query_kernel'))
            return {'scene_registered':True}
        if action=='publish':
            with self.relay.lock:
                e=self.relay.jobs[job]
                if not e['ready_credit'] or self.relay.clock()-e['last_poll']>=1.0 or e['state'] not in ('waiting','expired','rejected','committed'):
                    raise RelayError('Browser has not requested another batch',409)
            self.relay.publish(job,body.get('work'),body.get('query_kernel'),body.get('seconds',60))
            return {'published':True,'committed':False}
        if action=='take':
            result=self.relay.take_result(job)
            with self.relay.lock:state=self.relay.jobs[job]['state']
            return {'state':state,'result':result,'committed':False}
        if action=='finish':
            self.relay.finish(job,body.get('outcome'))
            return {'state':body.get('outcome'),'committed':body.get('outcome')=='committed'}
        raise RelayError('Unknown engine mailbox operation')

    def close(self,job):
        # Host calls only after worker exited/canceled, never as timeout fallback.
        with self.lock:
            self.workers.pop(job,None);self.relay.close(job)
