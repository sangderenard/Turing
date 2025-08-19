import numpy as np
import threading
import time
import random
from .base import verbose_log, physics_keys
from ..quad_buffer.tribuffer import Tribuffer
from .lock import LockManagerThread, LockGraph, LockCommand

class DoubleBuffer:
    """
    Thin agent-index/cursor tracker for hyperlocal concurrent buffer access.
    All data operations are callback-driven; this class holds *no* locks.
    """
    def __init__(self, roll_length=2, num_agents=2, reference=None):
        verbose_log(f"DoubleBuffer.__init__(roll_length={roll_length}, num_agents={num_agents})")
        self.roll_length = roll_length
        self.num_agents = num_agents
        self.reference = reference or physics_keys
        self.read_idx = [0] * num_agents
        self.write_idx = [1] * num_agents
        self.frames = [None] * roll_length
        phase_distance = roll_length // num_agents
        for i in range(num_agents):
            self.read_idx[i] = self.read_idx[i-1]+1 if i > 0 else 0
            self.write_idx[i] = (self.read_idx[i] + phase_distance) % roll_length

    def get_read_page(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.get_read_page(agent_idx={agent_idx})")
        return self.read_idx[agent_idx]

    def get_write_page(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.get_write_page(agent_idx={agent_idx})")
        return self.write_idx[agent_idx]

    def advance(self, agent_idx=0):
        verbose_log(f"DoubleBuffer.advance(agent_idx={agent_idx})")
        self.read_idx[agent_idx] = (self.read_idx[agent_idx] + 1) % self.roll_length
        self.write_idx[agent_idx] = (self.write_idx[agent_idx] + 1) % self.roll_length

    def for_read(self, agent_idx=0, keys=None, callback=None):
        verbose_log(f"DoubleBuffer.for_read(agent_idx={agent_idx}, keys={keys})")
        """Invoke `callback(page_idx, keys, agent_idx)` for agent's current read page."""
        page_idx = self.get_read_page(agent_idx)
        if callback is not None:
            return callback(page_idx, keys, agent_idx)

    def for_write(self, agent_idx=0, keys=None, callback=None):
        verbose_log(f"DoubleBuffer.for_write(agent_idx={agent_idx}, keys={keys})")
        """Invoke `callback(page_idx, keys, agent_idx)` for agent's current write page."""
        page_idx = self.get_write_page(agent_idx)
        if callback is not None:
            return callback(page_idx, keys, agent_idx)

    # Optional: a universal accessor if you want to specify r/w or have more metadata
    def access(self, agent_idx=0, mode='read', keys=None, callback=None):
        verbose_log(f"DoubleBuffer.access(agent_idx={agent_idx}, mode={mode}, keys={keys})")
        idx = self.get_read_page(agent_idx) if mode == 'read' else self.get_write_page(agent_idx)
        if callback:
            return callback(idx, keys, agent_idx)

    # Lightweight frame exchange helpers ---------------------------------
    def write_frame(self, frame, agent_idx=0):
        verbose_log(f"DoubleBuffer.write_frame(agent_idx={agent_idx})")
        idx = self.get_write_page(agent_idx)
        self.frames[idx] = frame
        self.advance(agent_idx)

    def read_frame(self, agent_idx=1):
        verbose_log(f"DoubleBuffer.read_frame(agent_idx={agent_idx})")
        idx = self.get_read_page(agent_idx)
        frame = self.frames[idx]
        if frame is not None:
            self.frames[idx] = None
            self.advance(agent_idx)
            return frame
        return None

class NumpyActionHistory:
    def __init__(self, num_agents, num_pages, num_keys, window_size=256):
        verbose_log(f"NumpyActionHistory.__init__(num_agents={num_agents}, num_pages={num_pages}, num_keys={num_keys}, window_size={window_size})")
        self.window = window_size
        self.na = num_agents
        self.np = num_pages
        self.nk = num_keys
        self.ptr = 0
        self.actions = np.zeros((window_size, num_agents, num_pages, num_keys, 2), dtype=np.uint8)
        self.timestamps = np.zeros(window_size, dtype=np.float64)  # optional

    def record(self, agent, page, key, action):
        verbose_log(f"NumpyActionHistory.record(agent={agent}, page={page}, key={key}, action={action})")
        # action: 0=read, 1=write
        self.actions[self.ptr, agent, page, key, action] = 1
        self.timestamps[self.ptr] = time.time()
        self.ptr = (self.ptr + 1) % self.window

    def get_recent(self, agent=None, page=None, key=None, action=None, kernel=None):
        verbose_log(f"NumpyActionHistory.get_recent(agent={agent}, page={page}, key={key}, action={action}, kernel={kernel})")
        """
        Returns a view or sum over the history window for the given indices.
        Use `kernel=(start, end)` for time slices.
        """
        idx = slice(None) if kernel is None else slice(*kernel)
        sl = [
            idx,
            agent if agent is not None else slice(None),
            page if page is not None else slice(None),
            key if key is not None else slice(None),
            action if action is not None else slice(None),
        ]
        return self.actions[tuple(sl)]

    def last_write_idx(self, agent, page, key):
        verbose_log(f"NumpyActionHistory.last_write_idx(agent={agent}, page={page}, key={key})")
        """Returns the latest window idx (or -1) where a write occurred."""
        # Get all indices where a write occurred for given (a,p,k)
        writes = np.nonzero(self.actions[:, agent, page, key, 1])[0]
        return writes[-1] if writes.size else -1

    def reads_since_last_write(self, agent, page, key):
        verbose_log(f"NumpyActionHistory.reads_since_last_write(agent={agent}, page={page}, key={key})")
        last_write = self.last_write_idx(agent, page, key)
        # All reads since last write (could broadcast across agents if needed)
        reads = self.actions[last_write+1:, agent, page, key, 0]
        return np.sum(reads)

    def unique_agents_since_last_write(self, page, key):
        verbose_log(f"NumpyActionHistory.unique_agents_since_last_write(page={page}, key={key})")
        """Which agents have read (page,key) since the last write by anyone?"""
        # Find last write index for any agent
        writes = np.nonzero(self.actions[:, :, page, key, 1])
        if writes[0].size == 0:
            start = 0
        else:
            start = writes[0].max() + 1
        reads = self.actions[start:, :, page, key, 0]
        return np.where(reads.sum(axis=0) > 0)[0]  # agent indices

    # Add fast reductions, kernel/stride ops, etc. as needed


import time
import threading
import random
import numpy as np
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# ----------- Gold Standard Stress Test ------------

class EnginePerfTracker:
    """Tracks running averages and break-even points for each engine."""
    def __init__(self):
        verbose_log("EnginePerfTracker.__init__()")
        self.stats = {'numpy': [], 'torch': []}
    def record(self, engine, batch_size, duration):
        verbose_log(f"EnginePerfTracker.record(engine={engine}, batch_size={batch_size}, duration={duration})")
        self.stats[engine].append((batch_size, duration))
    def avg_time(self, engine):
        verbose_log(f"EnginePerfTracker.avg_time(engine={engine})")
        d = [x[1] for x in self.stats[engine]] if self.stats[engine] else [0.0]
        return sum(d) / len(d)
    def avg_batch(self, engine):
        verbose_log(f"EnginePerfTracker.avg_batch(engine={engine})")
        b = [x[0] for x in self.stats[engine]] if self.stats[engine] else [0]
        return sum(b) / max(1, len(b))
    def report(self):
        verbose_log("EnginePerfTracker.report()")
        print("Engine Perf:")
        for eng in self.stats:
            print(f"  {eng} - avg batch: {self.avg_batch(eng):.1f}, avg time: {self.avg_time(eng):.4f}s")

class MultiAgentEngineSplitter:
    """
    Routes each job to numpy or torch engine based on delta mask, age, perf table.
    """
    def __init__(self, perf_tracker, torch_threshold=32, old_age=10):
        verbose_log(f"MultiAgentEngineSplitter.__init__(torch_threshold={torch_threshold}, old_age={old_age})")
        self.perf = perf_tracker
        self.torch_threshold = torch_threshold
        self.old_age = old_age  # in frames

    def choose_engine(self, batch_size, age):
        verbose_log(f"MultiAgentEngineSplitter.choose_engine(batch_size={batch_size}, age={age})")
        # If large/young, torch. If small/old, numpy.
        if batch_size >= self.torch_threshold and age < self.old_age:
            return 'torch'
        else:
            return 'numpy'

    def profile(self, engine, batch_size, fn):
        verbose_log(f"MultiAgentEngineSplitter.profile(engine={engine}, batch_size={batch_size})")
        t0 = time.time()
        result = fn()
        t1 = time.time()
        self.perf.record(engine, batch_size, t1-t0)
        return result

    def split_and_run(self, mask, ages, do_numpy, do_torch):
        verbose_log("MultiAgentEngineSplitter.split_and_run()")
        """
        mask: bool array, which indices to process
        ages: int array, age of each particle/region
        """
        idxs = np.where(mask)[0]
        jobs_torch = [i for i in idxs if ages[i] < self.old_age]
        jobs_numpy = [i for i in idxs if ages[i] >= self.old_age]
        res_numpy = []
        res_torch = []
        if jobs_numpy:
            res_numpy = self.profile('numpy', len(jobs_numpy), lambda: do_numpy(jobs_numpy))
        if jobs_torch:
            res_torch = self.profile('torch', len(jobs_torch), lambda: do_torch(jobs_torch))
        return res_numpy, res_torch

class ThreadSafeBuffer:
    def __init__(self, shape, dtype, agent_specs, manager, buffer_size=2, key_library=physics_keys, late_join=True):
        verbose_log(f"ThreadSafeBuffer.__init__(shape={shape}, dtype={dtype}, buffer_size={buffer_size}, late_join={late_join})")
        self.shape = tuple([buffer_size] + list(shape))
        self.dtype = dtype
        self.manager = manager
        if manager is None:
            self.manager = LockManagerThread(LockGraph())
            if not self.manager.running:
                self.manager.start()
        self.responses = self.manager.get_response_queue()
        self.blacklist = None  # Optional: set of agent IDs that cannot join
        self.whitelist = None
        self.agents_on_auto_advance = set()  # Agents that auto-advance
        self.auto_advance = False  # If True, auto-advance agents on read/write
        self.late_join = late_join
        self.readstates = DoubleBuffer(roll_length=buffer_size, num_agents=len(agent_specs), reference=physics_keys)
        self.host = Tribuffer(physics_keys, self.shape, dtype, '32', init='zeroes', manager=self.manager)
        self.mailboxes = {spec.agent_id: [] for spec in agent_specs}
        self.agent_specs = {spec.agent_id: spec for spec in agent_specs}
        self.mailbox_locks = {spec.agent_id: threading.Lock() for spec in agent_specs}
        self.mailbox_tokens = {spec.agent_id: None for spec in agent_specs}
        
    def late_join_agent(self, agent_specs):
        verbose_log(f"ThreadSafeBuffer.late_join_agent(agent_specs={agent_specs})")
        """Allow an agent to join late, initializing its mailbox and state."""
        rejected_agents = []
        accepted_agents = []
        for i, agent_spec in enumerate(agent_specs):
            agent_id = agent_spec.agent_id
            if agent_id in self.agent_specs:
                print(f"Agent {agent_id} already registered.")
                rejected_agents.append(i)
            elif self.blacklist and agent_id not in self.blacklist:
                print(f"Agent {agent_id} is blacklisted and cannot join.")
                rejected_agents.append(i)
            elif self.whitelist and agent_id in self.whitelist:
                print(f"Agent {agent_id} is not whitelisted and cannot join whitelisted buffer.")
                rejected_agents.append(i)
            elif self.late_join and agent_id not in self.agent_specs:
                accepted_agents.append(i)
            elif not self.late_join:
                print(f"Late joining not allowed for agent {agent_id}.")
                rejected_agents.append(i)
        if rejected_agents:
            print(f"Agents {', '.join(str(agent_specs[i].agent_id) for i in rejected_agents)} were rejected due to existing registration or policy.")
        for i in accepted_agents:
            agent_spec = agent_specs[i]
            agent_id = agent_spec.agent_id
            self.agent_specs[agent_id] = agent_spec
            self.mailboxes[agent_id] = []
            self.mailbox_locks[agent_id] = threading.Lock()
            self.mailbox_tokens[agent_id] = None
            self.readstates = self.readstates.insert_agents(agent_specs)

    def __getitem__(self, idx, agent=None, backend=None, device=None, blocking=True, readonly=False, callback=lambda x: x, reply_event=None, timeout=None, *args, **kwargs):
        verbose_log(f"ThreadSafeBuffer.__getitem__(idx={idx}, agent={agent}, backend={backend}, device={device}, blocking={blocking})")
        self.manager.submit(LockCommand('read', f"buf:{idx}", blocking=blocking, callback=callback, reply_event=reply_event, timeout=timeout, *args, **kwargs))
        cmd = self.manager.get_response_queue().get()
        if self.auto_advance and agent is not None:
            # Auto-advance the agent if it has auto-advance enabled
            if agent in self.agents_on_auto_advance:
                self.advance_agent(agent)
        return cmd.result
        
    def __setitem__(self, idx, value, agent=None, backend=None, device=None, blocking=True):
        verbose_log(f"ThreadSafeBuffer.__setitem__(idx={idx}, value_type={type(value)}, agent={agent}, backend={backend}, device={device}, blocking={blocking})")
        self.manager.submit(LockCommand('write', f"buf:{idx}", value=value, blocking=blocking))
        cmd = self.manager.get_response_queue().get()
        if self.auto_advance and agent is not None:
            # Auto-advance the agent if it has auto-advance enabled
            if agent in self.agents_on_auto_advance:
                self.advance_agent(agent)
        return cmd.result

    

    def sync(self):
        verbose_log("ThreadSafeBuffer.sync()")
        # In real version: must union/copy states so both numpy and torch arrays have the freshest view
        pass

    def register_agent(self, agent_id, allow_clipping=False):
        verbose_log(f"ThreadSafeBuffer.register_agent(agent_id={agent_id}, allow_clipping={allow_clipping})")
        # Register agent as mailbox owner; maybe allow special privilege
        pass

    def advance_agent(self, agent_id):
        verbose_log(f"ThreadSafeBuffer.advance_agent(agent_id={agent_id})")
        # Move this agent’s “moment” forward
        pass

    def query(self, region, agent=None, query={}):
        verbose_log(f"ThreadSafeBuffer.query(region={region}, agent={agent}, query={query})")
        # Query region state, token, who’s got it, queue length, etc.
        pass

    # ...additional helper methods as needed



# ----------- Smoke Test Scenario ------------

