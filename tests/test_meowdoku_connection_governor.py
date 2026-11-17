"""connection_governor.py (scaffold): an even, random chance per connection, and the staged pruning under competition."""
import random
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import connection_governor as cg  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def governor(seed=0, **policy):
    clock = Clock()
    return cg.Governor(cg.Policy(**policy), clock=clock, rng=random.Random(seed)), clock


def test_calm_requests_are_served_at_once_and_nothing_is_pruned():
    gov, _ = governor()
    tickets = [gov.request(f"10.0.0.{i}", "a") for i in range(16)]
    assert all(t.granted for t in tickets) and gov.metrics.stage == 0
    for t in tickets:
        gov.release(t)
    assert gov.snapshot()["admitted"] == 16 and gov.snapshot()["pruned"] == {}


def test_each_waiting_connection_has_one_chance_however_many_requests_it_queued():
    wins = Counter()
    for seed in range(2000):
        gov, _ = governor(seed, slots=1)
        holder = gov.request("10.0.0.9", "holder")
        busy = [gov.request("10.0.0.1", "busy") for _ in range(10)]
        quiet = gov.request("10.0.0.2", "quiet")
        gov.release(holder)
        wins["busy" if any(t.granted for t in busy) else "quiet" if quiet.granted else "none"] += 1
    assert wins["none"] == 0
    assert 0.45 < wins["quiet"] / 2000 < 0.55, wins


def test_the_order_is_drawn_not_rotated():
    orders = set()
    for seed in range(20):
        gov, _ = governor(seed, slots=1)
        holder = gov.request("10.0.0.9", "holder")
        tickets = {name: gov.request("10.0.0.1", name) for name in "abcd"}
        order, last = [], holder
        for _ in range(4):
            gov.release(last)
            last = next(t for n, t in tickets.items() if t.granted and n not in order)
            order.append(next(n for n, t in tickets.items() if t is last))
        orders.add("".join(order))
    assert len(orders) > 5, orders


def test_an_ip_holds_sixteen_connections_and_an_idle_one_makes_room():
    gov, _ = governor(slots=64)
    held = [gov.request("10.0.0.1", f"id{i}") for i in range(16)]
    assert all(t.granted for t in held)
    gov.release(held[3])                           # id3 is now idle: the next new connection takes its place
    assert gov.request("10.0.0.1", "id16").granted
    assert gov.metrics.pruned == {"idle_at_cap": 1} and "id3" not in gov.ips["10.0.0.1"].lanes
    extra = gov.request("10.0.0.1", "id17")       # sixteen in use: the seventeenth is refused
    assert extra.refused and extra.refused.reason == "connections_per_ip" and extra.refused.prose == cg.PROSE


def test_stage_one_drops_connections_by_idleness_longest_first():
    gov, clock = governor(slots=1, idle_drop_after=120)
    for age, name in ((300, "oldest"), (200, "older"), (60, "recent")):
        clock.now = 1000 + 300 - age
        gov.release(gov.request("10.0.0.5", name))
    clock.now = 1300
    holder = gov.request("10.0.0.9", "holder")
    gov.request("10.0.0.8", "waiter")              # a queue forms: stage 1
    assert gov.metrics.stage == 1
    assert set(gov.ips["10.0.0.5"].lanes) == {"recent"} and gov.metrics.pruned == {"idle": 2}
    gov.release(holder)


def test_stage_two_shrinks_shared_ips_and_shares_their_chance():
    gov, clock = governor(slots=1)
    gov.request("10.0.0.9", "holder")
    gov.metrics.refused["test"] = 1
    gov.denials.append(clock.now)                  # a recent refusal: stage 2
    gov.tick()
    assert gov.metrics.stage == 2 and gov._ip_cap(cg.IpState()) == 8
    shared = [gov.request("10.0.0.1", f"s{i}") for i in range(4)]
    single = gov.request("10.0.0.2", "only")
    assert gov._weight(shared[0].lane) == pytest.approx(0.25) and gov._weight(single.lane) == 1.0
    wins = Counter()
    for seed in range(2000):
        gov.rng = random.Random(seed)
        lanes = [l for s in gov.ips.values() for l in s.lanes.values() if l.waiting]
        wins[gov.rng.choices(lanes, weights=[gov._weight(l) for l in lanes])[0].ip] += 1
    assert 0.45 < wins["10.0.0.2"] / 2000 < 0.55, wins


def test_stage_two_trims_an_ip_above_its_shrunken_cap_idle_first():
    gov, clock = governor(slots=64)
    for i in range(12):
        clock.now += 1
        gov.release(gov.request("10.0.0.1", f"id{i}"))
    busy = gov.request("10.0.0.1", "id0")          # id0 is in use, so it is kept
    gov.denials.append(clock.now)
    gov.tick()
    lanes = gov.ips["10.0.0.1"].lanes
    assert gov.metrics.stage == 2 and len(lanes) == 8 and "id0" in lanes
    assert gov.metrics.pruned == {"shared": 4} and not {"id1", "id2", "id3", "id4"} & set(lanes)
    gov.release(busy)


def test_stage_three_slows_an_ip_that_saturates_the_slots():
    gov, clock = governor(slots=4, stage3_denials=1, stage2_denials=1)
    gov.denials.append(clock.now)
    gov.tick()
    assert gov.metrics.stage == 3
    hog = [gov.request("10.0.0.1", f"h{i}") for i in range(3)]
    other = gov.request("10.0.0.2", "o")
    state = gov.ips["10.0.0.1"]
    assert all(t.granted for t in hog) and state.strikes >= 1 and state.penalty_until > clock.now
    hog_wait = gov.request("10.0.0.1", "h9")
    other_wait = gov.request("10.0.0.2", "o2")
    gov.release(other)
    assert other_wait.granted and not hog_wait.granted, "a penalized IP waits while others are served"
    state.penalty_until = clock.now + 600
    refused = gov.request("10.0.0.1", "h0")        # an existing connection: stage 3 caps the IP at 4 new ones
    assert refused.refused.reason == "ip_penalty" and refused.refused.retry_after == pytest.approx(600)


def test_a_penalized_ip_waiting_alone_still_gets_a_free_slot():
    gov, clock = governor(slots=2)
    gov.ips.setdefault("10.0.0.1", cg.IpState()).penalty_until = clock.now + 5
    assert gov.request("10.0.0.1", "a").granted


def test_the_stage_steps_down_one_level_after_a_calm_spell():
    gov, clock = governor(slots=4, denial_window=10, calm_after=30, stage3_denials=1)
    gov.denials.append(clock.now)
    gov.tick()
    assert gov.metrics.stage == 3
    clock.now += 11
    gov.tick()
    assert gov.metrics.stage == 3
    for expected in (2, 1, 0):
        clock.now += 30
        gov.tick()
        assert gov.metrics.stage == expected


def test_a_request_thread_waits_its_turn_or_is_refused_after_wait_max():
    gov = cg.Governor(cg.Policy(slots=1, wait_max=0.2))
    first = gov.admit("10.0.0.1", "a")
    with pytest.raises(cg.Refused) as refused:
        gov.admit("10.0.0.2", "b")
    assert refused.value.reason == "queue_wait"
    got = []
    worker = threading.Thread(target=lambda: got.append(gov.admit("10.0.0.3", "c")))
    worker.start()
    gov.release(first)
    worker.join(2)
    assert got and got[0].granted
    gov.release(got[0])
    with gov.slot("10.0.0.4", "d") as ticket:
        assert ticket.granted and gov.snapshot()["busy"] == 1
    assert gov.snapshot()["busy"] == 0
