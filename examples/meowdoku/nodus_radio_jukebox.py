"""Persistent first-channel request rotation and song ratings in accounts SQL.

One turn per nonempty user list, oldest last turn first. New/rejoining listeners
join the tail of the current rotation. Selecting a song is atomic, not a GET
side effect. Native playback owns claim/finish; HTTP never supplies paths.
"""
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid
from meowdoku import schema_version
from meowdoku.nodus_radio_control import ControlError

DDL = '''
CREATE TABLE IF NOT EXISTS radio_channel (id INTEGER PRIMARY KEY CHECK(id=1), paused INTEGER NOT NULL DEFAULT 1, turn INTEGER NOT NULL DEFAULT 0, current_id INTEGER);
INSERT OR IGNORE INTO radio_channel(id) VALUES(1);
CREATE TABLE IF NOT EXISTS radio_turns (user_id INTEGER PRIMARY KEY, turn INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS radio_requests (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, song_id TEXT NOT NULL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS radio_request_owner ON radio_requests(user_id,id);
CREATE TABLE IF NOT EXISTS radio_plays (id INTEGER PRIMARY KEY, request_id INTEGER NOT NULL, user_id INTEGER NOT NULL, song_id TEXT NOT NULL, token TEXT NOT NULL UNIQUE, status TEXT NOT NULL, started REAL NOT NULL, ended REAL);
CREATE TABLE IF NOT EXISTS radio_votes (song_id TEXT NOT NULL, user_id INTEGER NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 0 AND 5), updated REAL NOT NULL, PRIMARY KEY(song_id,user_id));
CREATE TABLE IF NOT EXISTS radio_actions (id INTEGER PRIMARY KEY, actor INTEGER NOT NULL, action TEXT NOT NULL, target INTEGER, created REAL NOT NULL);
CREATE VIEW IF NOT EXISTS radio_billboard AS SELECT song_id, COUNT(*) AS votes, SUM(rating) AS points, AVG(rating) AS average FROM radio_votes GROUP BY song_id;
'''

class Jukebox:
    def __init__(self, db_path, config_path, *, state=None):
        self.db_path, self.config_path = str(db_path), Path(config_path)
        self.state = state

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            # Version refusal precedes owned table changes.
            schema_version.declare(db, 'nodus_radio_jukebox', 1)
            db.executescript(DDL)
            try:
                db.execute('BEGIN IMMEDIATE')
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def catalogue(self):
        try:
            config = json.loads(self.config_path.read_text(encoding='utf-8')) if self.config_path.is_file() else {}
            songs = config.get('jukebox', {}).get('songs', [])
            if not isinstance(songs, list) or len(songs) > 10000:
                raise ValueError()
            result = {}
            for song in songs:
                key, title = song['id'], song['title']
                if not isinstance(key, str) or not 1 <= len(key) <= 128 or not isinstance(title, str) or not 1 <= len(title) <= 200 or key in result:
                    raise ValueError()
                result[key] = dict(song)
            if self.state is not None:
                from meowdoku.suno_radio_assets import station_catalogue
                result.update(station_catalogue(self.state, self.db_path))
            return result
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise ControlError('Radio song catalogue is not configured', 503)

    @staticmethod
    def _audit(db, actor, action, target=None):
        db.execute('INSERT INTO radio_actions(actor,action,target,created) VALUES(?,?,?,?)', (actor,action,target,time.time()))

    def request(self, user, song_id):
        if song_id not in self.catalogue():
            raise ControlError('Choose a song from the station catalogue', 404)
        with self.connect() as db:
            if db.execute('SELECT COUNT(*) FROM radio_requests WHERE user_id=?',(user,)).fetchone()[0] >= 100:
                raise ControlError('Your request list is full', 409)
            pending = db.execute('SELECT 1 FROM radio_requests WHERE user_id=?',(user,)).fetchone()
            if not pending:
                turn = db.execute('SELECT turn FROM radio_channel WHERE id=1').fetchone()[0]
                db.execute('INSERT INTO radio_turns(user_id,turn) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET turn=excluded.turn',(user,turn))
            row = db.execute('INSERT INTO radio_requests(user_id,song_id,created) VALUES(?,?,?)',(user,song_id,time.time()))
            self._audit(db,user,'request',row.lastrowid)
            return row.lastrowid

    def remove(self, user, request_id, *, admin=False):
        with self.connect() as db:
            row=db.execute('SELECT user_id FROM radio_requests WHERE id=?',(request_id,)).fetchone()
            if row is None: raise ControlError('Request is no longer queued',404)
            if row['user_id'] != user and not admin: raise ControlError('This is not your request',403)
            db.execute('DELETE FROM radio_requests WHERE id=?',(request_id,))
            self._audit(db,user,'remove',request_id)

    def audit(self, actor, action, target=None):
        with self.connect() as db:
            self._audit(db, actor, action, target)

    def pause(self, actor, paused):
        with self.connect() as db:
            db.execute('UPDATE radio_channel SET paused=? WHERE id=1',(int(paused),))
            self._audit(db,actor,'pause' if paused else 'resume')

    def claim(self):
        catalogue=self.catalogue()
        with self.connect() as db:
            channel=db.execute('SELECT * FROM radio_channel WHERE id=1').fetchone()
            if channel['paused'] or channel['current_id'] is not None: return None
            rows=db.execute('SELECT r.* FROM radio_requests r JOIN radio_turns t ON t.user_id=r.user_id ORDER BY t.turn,r.id').fetchall()
            row=next((r for r in rows if r['song_id'] in catalogue),None)
            if row is None: return None
            token=uuid.uuid4().hex
            play=db.execute("INSERT INTO radio_plays(request_id,user_id,song_id,token,status,started) VALUES(?,?,?,?, 'playing',?)",(row['id'],row['user_id'],row['song_id'],token,time.time())).lastrowid
            db.execute('DELETE FROM radio_requests WHERE id=?',(row['id'],))
            db.execute('UPDATE radio_turns SET turn=? WHERE user_id=?',(channel['turn']+1,row['user_id']))
            db.execute('UPDATE radio_channel SET turn=turn+1,current_id=? WHERE id=1',(play,))
            return {'id':play,'token':token,'user_id':row['user_id'],'song_id':row['song_id']}

    def finish(self, token, status='completed'):
        if status not in ('completed','failed','skipped','interrupted'): raise ValueError('invalid playback outcome')
        with self.connect() as db:
            row=db.execute('SELECT p.id FROM radio_plays p JOIN radio_channel c ON c.current_id=p.id WHERE p.token=?',(token,)).fetchone()
            if row is None: return False
            db.execute('UPDATE radio_plays SET status=?,ended=? WHERE id=?',(status,time.time(),row['id']))
            db.execute('UPDATE radio_channel SET current_id=NULL WHERE id=1')
            return True

    def rate(self, user, play_id, rating):
        if type(rating) is not int or not 0 <= rating <= 5: raise ControlError('Rating must be an integer from 0 to 5')
        with self.connect() as db:
            play=db.execute('SELECT p.song_id FROM radio_plays p JOIN radio_channel c ON c.current_id=p.id WHERE p.id=?',(play_id,)).fetchone()
            if play is None: raise ControlError('That song is no longer playing; refresh before voting',409)
            db.execute('INSERT INTO radio_votes(song_id,user_id,rating,updated) VALUES(?,?,?,?) ON CONFLICT(song_id,user_id) DO UPDATE SET rating=excluded.rating,updated=excluded.updated',(play['song_id'],user,rating,time.time()))

    def status(self, user, *, admin=False):
        catalogue=self.catalogue()
        with self.connect() as db:
            channel=dict(db.execute('SELECT * FROM radio_channel WHERE id=1').fetchone())
            current=db.execute('SELECT id,song_id,status FROM radio_plays WHERE id=?',(channel['current_id'],)).fetchone()
            requests=[dict(r) for r in db.execute('SELECT r.id,r.user_id,r.song_id FROM radio_requests r JOIN radio_turns t ON t.user_id=r.user_id ORDER BY t.turn,r.id')]
            for row in requests:
                row['mine']=row['user_id']==user
                if not admin: row.pop('user_id')
            charts=[dict(r) for r in db.execute('SELECT * FROM radio_billboard ORDER BY average DESC,votes DESC,song_id LIMIT 100')]
            vote=None if current is None else db.execute('SELECT rating FROM radio_votes WHERE song_id=? AND user_id=?',(current['song_id'],user)).fetchone()
            return {'paused':bool(channel['paused']),'current':dict(current) if current else None,'requests':requests,'charts':charts,'my_vote':vote[0] if vote else None,'catalogue':[{'id':s['id'],'title':s['title']} for s in catalogue.values()],'can_admin':admin}


def http_route(handler, method, store, runner=None):
    handler._classify()
    user=handler._principal().user_id
    if user is None: return handler._json({'error':'Sign in to use the jukebox'},401)
    admin='account_manage' in handler._caps()
    try:
        def status():
            data=store.status(user,admin=admin)
            if runner is not None: data['runner']=runner.status()
            return data
        if method=='GET': return handler._json(status())
        origin=handler.headers.get('Origin')
        expected=('https://' if handler._is_secure() else 'http://')+handler.headers.get('Host','')
        if (origin and origin!=expected) or handler.headers.get('Sec-Fetch-Site')=='cross-site' or handler.headers.get('X-Meowdoku-Radio')!='1':
            raise ControlError('Same-origin Radio control header required',403)
        if handler.headers.get('Content-Type','').split(';',1)[0].strip().lower()!='application/json': raise ControlError('JSON only',415)
        if handler.headers.get('Transfer-Encoding') or len(handler.headers.get_all('Content-Length',[]))!=1: raise ControlError('One Content-Length is required')
        body=handler._body_json(2048)
        if body is None:return
        if not isinstance(body,dict):raise ControlError('Expected an action object')
        action=body.get('action')
        fields={'request':{'action','song_id'},'remove':{'action','request_id'},'rate':{'action','play_id','rating'},'pause':{'action'},'resume':{'action'},'skip':{'action'}}
        if not isinstance(action,str) or action not in fields or set(body)!=fields[action]:raise ControlError('Invalid jukebox action or fields')
        if action in ('pause','resume','skip') and not admin:raise ControlError('Jukebox control requires account_manage',403)
        for key in ('request_id','play_id'):
            if key in body and (type(body[key]) is not int or body[key]<1):raise ControlError('Invalid record id')
        if action=='request':
            if not isinstance(body['song_id'],str):raise ControlError('Invalid song id')
            store.request(user,body['song_id'])
        elif action=='remove':store.remove(user,body['request_id'],admin=admin)
        elif action=='rate':store.rate(user,body['play_id'],body['rating'])
        else:
            if runner is None:raise ControlError('Native jukebox runner is not configured',503)
            getattr(runner,action)(user)
        return handler._json(status())
    except ControlError as error:return handler._json({'error':str(error)},error.status)
    except (sqlite3.Error, schema_version.SchemaVersionError):
        return handler._json({'error':'Jukebox storage unavailable; no retry was assumed'},503)
