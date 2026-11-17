"""Connection governor: an even, randomly ordered chance to be served, and staged pruning when connections compete.

STATUS: SCAFFOLD (owner request, 2026-11-16). Not wired into server.py. Every number below is a first guess, named in
``Policy`` so it can be refined; every quantity the stages look at is counted in ``Metrics``.

What the owner asked for, in their words, and where each part lives:

* "total connections cycled in their speed ... round robin of opportunity for commands, but randomly selected order"
  -> ``_dispatch``: when a service slot frees, the next request comes from a connection chosen at random among the
  connections that are waiting, one chance per connection, not per request.  New arrivals never jump a queue.
* "individual ips can have 16 connections" -> ``Policy.lanes_per_ip``.
* "pruned when there's competition, a queue of requests or requests are being denied" -> ``_measure`` sets the stage
  from the queue and the recent refusals.
* "a staged set of priorities":
    stage 1  ``_prune_idle``: connections are dropped by their idleness, the longest idle first.
    stage 2  shared connections feel it: an IP holding many connections gets a smaller cap (``_ip_cap``) and each of
             its connections a smaller chance in the draw (``_weight``).
    stage 3  ``_penalize``: slowdown penalties per IP for saturating the slots, growing with repeated strikes.

What a "connection" is here.  Behind Apache one backend socket carries many visitors' requests in turn (see
server.py ``handle_one_request``), so a TCP socket is not a visitor.  A connection is a *lane*: one client IP (from
proxy_trust.client_origin, the forwarded address when the peer is a trusted proxy) plus one identity (the signed-in
account, else the browser's session cookie, else nothing).  An IP's lanes are its connections.

Not decided here, for refinement:
* the owner's earlier order for guests (memory: guests are throttled first, 2026-11-13) is not applied; ``Lane.kind``
  is carried so a stage can prefer to cut guests;
* where the governor sits: the intended seam is the start of each request in server.py, after ``_classify`` has set
  ``_client_ip`` and before routing, as ``with governor.slot(ip, identity): ...``; a refusal answers 429 with
  ``Refused.retry_after`` and ``Refused.prose``;
* Apache's backend cap (``max=16`` on the BalancerMember in the live conf) limits the backend sockets of all visitors
  together, not per IP; with this governor in place it bounds how many requests reach it at once.

The core is non-blocking (``request`` / ``release``) so it is tested with a fake clock and a seeded draw; ``admit``
and ``slot`` add the waiting a request thread does.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

PROSE = ("It is rude to interrupt so quickly so often. Please leave a friendly gap between requests, then try again.")


@dataclass
class Policy:
    """Every knob. Times are seconds."""
    slots: int = 16                        # requests this instance serves at once
    lanes_per_ip: int = 16                 # connections one IP may hold when nothing competes
    wait_max: float = 30.0                 # longest a request waits in the queue before it is refused
    # -- how the stage is measured (``_measure``)
    denial_window: float = 60.0            # refusals this recent count as "requests being denied"
    stage2_queue: int = 16                 # this many waiting requests ... (stage 1 is any queue at all)
    stage2_denials: int = 1                # ... or this many recent refusals -> stage 2
    stage3_queue: int = 48
    stage3_denials: int = 8
    calm_after: float = 30.0               # a stage steps down one level after this long without its trigger
    # -- stage 1: idleness
    idle_drop_after: float = 120.0         # a connection idle this long is dropped at stage 1 or above
    idle_drop_after_stage3: float = 15.0   # and this long at stage 3
    # -- stage 2: shared connections
    shared_lanes_floor: int = 2            # an IP's cap never shrinks below this
    shared_cap_ratio: float = 0.5          # at stage 2 an IP's cap is this share of lanes_per_ip (stage 3: squared)
    shared_weight_exponent: float = 1.0    # chance of each of an IP's n connections = 1 / n ** exponent
    # -- stage 3: saturation penalties
    penalty_base: float = 2.0              # seconds of slowdown per unit of saturation above the fair share
    penalty_max: float = 60.0
    strike_forget_after: float = 300.0     # an IP's strikes reset after this long without a new one


@dataclass
class Metrics:
    """Every quantity the stages look at, plus what they did. ``Governor.snapshot`` reports these."""
    stage: int = 0
    admitted: int = 0                      # granted at once
    queued: int = 0                        # had to wait
    served_from_queue: int = 0
    refused: dict = field(default_factory=dict)       # reason -> count
    pruned: dict = field(default_factory=dict)        # reason -> connections dropped
    penalties: int = 0
    wait_total: float = 0.0                # seconds waited by requests served from the queue
    wait_longest: float = 0.0
    stage_entered: dict = field(default_factory=dict)  # stage -> times entered


@dataclass(eq=False)
class Lane:
    ip: str
    identity: str
    kind: str = "unknown"                  # "account", "guest", ... (not used by any stage yet)
    in_flight: int = 0
    waiting: deque = field(default_factory=deque)     # Tickets, oldest first
    last_seen: float = 0.0

    def idle(self) -> bool:
        return self.in_flight == 0 and not self.waiting


@dataclass
class IpState:
    lanes: dict = field(default_factory=dict)         # identity -> Lane
    in_flight: int = 0
    strikes: int = 0
    last_strike: float = 0.0
    penalty_until: float = 0.0


class Refused(Exception):
    def __init__(self, reason: str, retry_after: float):
        super().__init__(reason)
        self.reason, self.retry_after, self.prose = reason, max(1.0, retry_after), PROSE


@dataclass(eq=False)
class Ticket:
    lane: Lane
    asked: float
    granted: bool = False
    refused: Refused | None = None
    released: bool = False
    event: threading.Event = field(default_factory=threading.Event)


class Governor:
    def __init__(self, policy: Policy | None = None, *, clock=time.monotonic, rng: random.Random | None = None):
        self.policy = policy or Policy()
        self.clock, self.rng = clock, rng or random.Random()
        self.metrics = Metrics()
        self.ips: dict[str, IpState] = {}
        self.busy = 0
        self.denials: deque = deque()                  # refusal times
        self._stage_trigger_seen = {1: 0.0, 2: 0.0, 3: 0.0}
        self.lock = threading.Lock()

    # ---- the request side

    def request(self, ip: str, identity: str, kind: str = "unknown") -> Ticket:
        """A request asks for a slot: granted now, queued (``granted`` False), or refused (``refused`` set)."""
        with self.lock:
            now = self.clock()
            self._measure(now)
            state = self.ips.setdefault(ip, IpState())
            lane = state.lanes.get(identity)
            if lane is None:
                if len(state.lanes) >= self._ip_cap(state):
                    self._prune_idle(now, only=state, force=True)
                if len(state.lanes) >= self._ip_cap(state):
                    return self._refuse(Ticket(Lane(ip, identity, kind), now), "connections_per_ip", 5.0, now)
                lane = state.lanes[identity] = Lane(ip, identity, kind)
            lane.last_seen = now
            ticket = Ticket(lane, now)
            if state.penalty_until - now > self.policy.wait_max:
                return self._refuse(ticket, "ip_penalty", state.penalty_until - now, now)
            if self.busy < self.policy.slots and not self._anyone_waiting() and state.penalty_until <= now:
                self._grant(ticket)
                self.metrics.admitted += 1
                self._measure(now)         # a granted slot changes each IP's share (stage 3)
            else:
                lane.waiting.append(ticket)
                self.metrics.queued += 1
                self._dispatch(now)        # a free slot is never left empty, even for a penalized IP waiting alone
                self._measure(now)
            return ticket

    def release(self, ticket: Ticket) -> None:
        """The request is answered: its slot goes to a connection drawn at random among those waiting."""
        with self.lock:
            if not ticket.granted or ticket.released:
                return
            ticket.released = True
            now = self.clock()
            lane, state = ticket.lane, self.ips.get(ticket.lane.ip)
            lane.in_flight -= 1
            lane.last_seen = now
            self.busy -= 1
            if state is not None:
                state.in_flight -= 1
            self._measure(now)
            self._dispatch(now)

    def expire(self, ticket: Ticket) -> None:
        """A queued request waited ``wait_max``: it is refused and counted as a denial."""
        with self.lock:
            if ticket.granted or ticket.refused:
                return
            now = self.clock()
            try:
                ticket.lane.waiting.remove(ticket)
            except ValueError:
                pass
            self._refuse(ticket, "queue_wait", self.policy.wait_max, now)
            self._measure(now)

    def admit(self, ip: str, identity: str, kind: str = "unknown") -> Ticket:
        """For a request thread: wait for the slot, or raise ``Refused``."""
        ticket = self.request(ip, identity, kind)
        if not ticket.granted and not ticket.refused:
            ticket.event.wait(self.policy.wait_max)
            if not ticket.granted:
                self.expire(ticket)
        if ticket.refused:
            raise ticket.refused
        return ticket

    @contextmanager
    def slot(self, ip: str, identity: str, kind: str = "unknown"):
        ticket = self.admit(ip, identity, kind)
        try:
            yield ticket
        finally:
            self.release(ticket)

    def tick(self) -> None:
        """Housekeeping with no request in hand (a timer may call it): stage, idle pruning, waking penalized lanes."""
        with self.lock:
            now = self.clock()
            self._measure(now)
            self._dispatch(now)

    def snapshot(self) -> dict:
        with self.lock:
            m = self.metrics
            return {"stage": m.stage, "busy": self.busy, "slots": self.policy.slots,
                    "waiting": sum(len(l.waiting) for s in self.ips.values() for l in s.lanes.values()),
                    "ips": len(self.ips), "lanes": sum(len(s.lanes) for s in self.ips.values()),
                    "recent_denials": len(self.denials), "admitted": m.admitted, "queued": m.queued,
                    "served_from_queue": m.served_from_queue, "refused": dict(m.refused), "pruned": dict(m.pruned),
                    "penalties": m.penalties, "wait_longest": m.wait_longest,
                    "wait_mean": m.wait_total / m.served_from_queue if m.served_from_queue else 0.0,
                    "stage_entered": dict(m.stage_entered)}

    # ---- the draw

    def _anyone_waiting(self) -> bool:
        return any(lane.waiting for state in self.ips.values() for lane in state.lanes.values())

    def _weight(self, lane: Lane) -> float:
        """One chance per connection; at stage 2 or above an IP's many connections share theirs."""
        if self.metrics.stage < 2:
            return 1.0
        return 1.0 / len(self.ips[lane.ip].lanes) ** self.policy.shared_weight_exponent

    def _dispatch(self, now: float) -> None:
        while self.busy < self.policy.slots:
            waiting = [lane for state in self.ips.values() for lane in state.lanes.values() if lane.waiting]
            if not waiting:
                return
            ready = [lane for lane in waiting if self.ips[lane.ip].penalty_until <= now]
            pool = ready or waiting        # never leave a slot empty: a penalized lane is served if no one else waits
            lane = self.rng.choices(pool, weights=[self._weight(l) for l in pool])[0]
            ticket = lane.waiting.popleft()
            waited = now - ticket.asked
            self.metrics.served_from_queue += 1
            self.metrics.wait_total += waited
            self.metrics.wait_longest = max(self.metrics.wait_longest, waited)
            self._grant(ticket)

    def _grant(self, ticket: Ticket) -> None:
        ticket.granted = True
        ticket.lane.in_flight += 1
        self.ips[ticket.lane.ip].in_flight += 1
        self.busy += 1
        ticket.event.set()

    def _refuse(self, ticket: Ticket, reason: str, retry_after: float, now: float) -> Ticket:
        ticket.refused = Refused(reason, retry_after)
        self.metrics.refused[reason] = self.metrics.refused.get(reason, 0) + 1
        self.denials.append(now)
        ticket.event.set()
        return ticket

    # ---- the stages

    def _measure(self, now: float) -> None:
        p = self.policy
        while self.denials and now - self.denials[0] > p.denial_window:
            self.denials.popleft()
        queue = sum(len(l.waiting) for s in self.ips.values() for l in s.lanes.values())
        denials = len(self.denials)
        triggered = {1: queue > 0 or denials > 0,
                     2: queue >= p.stage2_queue or denials >= p.stage2_denials,
                     3: queue >= p.stage3_queue or denials >= p.stage3_denials}
        for level, on in triggered.items():
            if on:
                self._stage_trigger_seen[level] = now
        target = max([0] + [level for level, on in triggered.items() if on])
        stage = self.metrics.stage
        if target > stage:
            stage = target
        elif stage > target and now - self._stage_trigger_seen.get(stage, 0.0) >= p.calm_after:
            stage -= 1                     # steps down one level at a time, after a calm spell
        if stage != self.metrics.stage:
            self.metrics.stage = stage
            self.metrics.stage_entered[stage] = self.metrics.stage_entered.get(stage, 0) + 1
        if stage >= 1:
            self._prune_idle(now)
        if stage >= 2:
            for state in self.ips.values():
                self._trim_shared(state, now)
        if stage >= 3:
            for ip in list(self.ips):
                self._penalize(ip, now)
        for state in self.ips.values():
            if state.strikes and now - state.last_strike > p.strike_forget_after:
                state.strikes = 0
        for ip in [ip for ip, s in self.ips.items() if not s.lanes and s.penalty_until <= now and not s.strikes]:
            del self.ips[ip]

    def _prune_idle(self, now: float, only: IpState | None = None, force: bool = False) -> None:
        """Stage 1: drop connections by their idleness, the longest idle first. ``force`` (an IP at its cap asking
        for one more) drops that IP's longest-idle connection whatever the stage."""
        p = self.policy
        after = p.idle_drop_after_stage3 if self.metrics.stage >= 3 else p.idle_drop_after
        for state in ([only] if only is not None else list(self.ips.values())):
            idle = sorted((l for l in state.lanes.values() if l.idle()), key=lambda l: l.last_seen)
            for lane in idle:
                if now - lane.last_seen >= after:
                    self._drop(state, lane, "idle")
            if force and len(state.lanes) >= self._ip_cap(state):
                idle = [l for l in idle if l.identity in state.lanes]
                if idle:
                    self._drop(state, idle[0], "idle_at_cap")

    def _ip_cap(self, state: IpState) -> int:
        """Stage 2: an IP's cap shrinks when connections compete (squared at stage 3)."""
        p = self.policy
        if self.metrics.stage < 2:
            return p.lanes_per_ip
        ratio = p.shared_cap_ratio ** (self.metrics.stage - 1)
        return max(p.shared_lanes_floor, int(p.lanes_per_ip * ratio))

    def _trim_shared(self, state: IpState, now: float) -> None:
        """Stage 2: an IP above its shrunken cap loses its idle connections, longest idle first."""
        cap = self._ip_cap(state)
        for lane in sorted((l for l in state.lanes.values() if l.idle()), key=lambda l: l.last_seen):
            if len(state.lanes) <= cap:
                return
            self._drop(state, lane, "shared")

    def _penalize(self, ip: str, now: float) -> None:
        """Stage 3: an IP holding more than its fair share of the slots is slowed, longer with each strike. A penalty
        running now is not stacked: the next strike comes only after it ends."""
        p, state = self.policy, self.ips[ip]
        if state.penalty_until > now:
            return
        active = max(1, sum(1 for s in self.ips.values() if s.in_flight or any(l.waiting for l in s.lanes.values())))
        fair = p.slots / active
        saturation = state.in_flight / fair
        if saturation <= 1.0:
            return
        delay = min(p.penalty_max, p.penalty_base * (saturation - 1.0) * 2 ** state.strikes)
        state.penalty_until = max(state.penalty_until, now + delay)
        state.strikes += 1
        state.last_strike = now
        self.metrics.penalties += 1

    def _drop(self, state: IpState, lane: Lane, reason: str) -> None:
        if state.lanes.get(lane.identity) is lane and lane.idle():
            del state.lanes[lane.identity]
            self.metrics.pruned[reason] = self.metrics.pruned.get(reason, 0) + 1
