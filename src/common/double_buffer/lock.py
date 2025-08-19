import threading
import queue
import time
import random
from collections import defaultdict, deque
from .base import verbose_log, nx

class LockCommand:
    def __init__(self, op, name, blocking=True, callback=None, reply_event=None, timeout=None, *args, **kwargs):
        verbose_log(f"LockCommand.__init__(op={op}, name={name}, blocking={blocking})")
        self.op = op              # "acquire", "release", etc
        self.name = name
        self.blocking = blocking
        self.callback = callback  # Called under exclusive lock, if supplied
        self.reply_event = reply_event or threading.Event()
        self.timeout = timeout
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self.thread = threading.current_thread().name
class RegionToken:
    """
    A unique handle given to agents for persistent region ownership.
    Agent must present this token to continue or release the region.
    """
    _counter = 0
    _lock = threading.Lock()
    def __init__(self, region_name):
        verbose_log(f"RegionToken.__init__(region_name={region_name})")
        with RegionToken._lock:
            RegionToken._counter += 1
            self.id = RegionToken._counter
        self.region_name = region_name
        self.created = time.time()
    def __repr__(self):
        return f"<RegionToken #{self.id} for {self.region_name}>"
class LockManagerThread(threading.Thread):
    """
    The core of the abstraction. Owns all lock state, runs callbacks for
    small ops, issues tokens for persistent/complex access, manages queue.
    """
    def __init__(self, lock_graph):
        verbose_log("LockManagerThread.__init__()")
        super().__init__(daemon=True)
        self.lock_graph = lock_graph
        self.cmd_queue = queue.Queue()
        self.running = True
        self.region_tokens = {} # region -> token (if locked persistently)
        self.thread_safe_buffer_response_queue = queue.Queue()  # for async buffer responses
    def get_response_queue(self):
        verbose_log("LockManagerThread.get_response_queue()")
        return self.thread_safe_buffer_response_queue
    def submit(self, cmd: LockCommand):
        verbose_log(f"LockManagerThread.submit(cmd={cmd.op}, name={cmd.name})")
        self.cmd_queue.put(cmd)
        return cmd.reply_event
    def register_buffer_sync(self, buffer_sync):
        verbose_log("LockManagerThread.register_buffer_sync()")
        """
        Register a buffer sync manager to handle async sync operations.
        """
        if buffer_sync.manager is not None and buffer_sync.manager != self:
            need_to_start = False
            if not self.running and self.buffer_sync.manager.running:
                need_to_start = True
            self.buffer_sync.manager.shutdown()
            self.buffer_sync.manager = self
            if need_to_start:
                self.start()
        self.buffer_sync = buffer_sync

    def start(self):
        verbose_log("LockManagerThread.start()")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        else:
            raise RuntimeError("LockManagerThread is already running")

    def run(self):
        verbose_log("LockManagerThread.run() started")
        while self.running:
            cmd = self.cmd_queue.get()
            if cmd is None:
                verbose_log("LockManagerThread.run() received shutdown signal")
                break
            def deploy_runner():
                try:
                    verbose_log(f"LockManagerThread.deploy_runner() op={cmd.op} name={cmd.name} [START]")
                    # Print lock graph state before handling command
                    extra_verbose_node_print(self.lock_graph)
                    # ----------- SMALL OP: INLINE READ -----------
                    if cmd.op == "read":
                        verbose_log(f"LockManagerThread.deploy_runner() [READ] Trying to acquire lock for {cmd.name}")
                        if self.lock_graph.try_acquire(cmd.name):
                            try:
                                verbose_log(f"LockManagerThread.deploy_runner() [READ] Lock acquired for {cmd.name}")
                                if cmd.callback is not None:
                                    verbose_log(f"LockManagerThread.deploy_runner() [READ] Running callback for {cmd.name}")
                                    cmd.result = self.lock_graph.run_callback(cmd.callback, *cmd.args, **cmd.kwargs)
                            finally:
                                verbose_log(f"LockManagerThread.deploy_runner() [READ] Releasing lock for {cmd.name}")
                                self.lock_graph.release(cmd.name)
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [READ] Done for {cmd.name}")
                        else:
                            verbose_log(f"LockManagerThread.deploy_runner() [READ] Region busy for {cmd.name}")
                            cmd.error = RuntimeError("Region busy")
                            cmd.reply_event.set()

                    # ----------- PERSISTENT/COMPLEX OP: TOKEN GRANT -----------
                    elif cmd.op == "acquire":
                        verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Trying to acquire lock for {cmd.name}")
                        if self.lock_graph.try_acquire(cmd.name):
                            token = RegionToken(cmd.name)
                            self.region_tokens[cmd.name] = token
                            cmd.result = token
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Token granted for {cmd.name}")
                        else:
                            verbose_log(f"LockManagerThread.deploy_runner() [ACQUIRE] Region busy for {cmd.name}")
                            cmd.error = RuntimeError("Region busy")
                            cmd.reply_event.set()

                    # ----------- RELEASE (MUST PRESENT TOKEN) -----------
                    elif cmd.op == "release":
                        verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] Attempting release for {cmd.name}")
                        token = cmd.token
                        if not token:
                            verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] No token provided for {cmd.name}")
                            cmd.error = RuntimeError("Token required for release")
                            cmd.reply_event.set()
                        else:
                            self.lock_graph.release(cmd.name, token)
                            self.region_tokens.pop(cmd.name, None)
                            cmd.result = True
                            cmd.reply_event.set()
                            verbose_log(f"LockManagerThread.deploy_runner() [RELEASE] Released {cmd.name}")

                    # ----------- EXTENSION: BATCH, QUERY, ETC. -----------
                    # (Handle more ops as needed)

                except Exception as e:
                    verbose_log(f"LockManagerThread.deploy_runner() [ERROR] Exception: {e}")
                    cmd.error = e
                    cmd.reply_event.set()

                # Print lock graph state after handling command
                extra_verbose_node_print(self.lock_graph)
                verbose_log(f"LockManagerThread.deploy_runner() op={cmd.op} name={cmd.name} [END]")
                return cmd.result
            # Run the command in a separate thread to avoid blocking
            runner_thread = threading.Thread(target=deploy_runner, daemon=True)
            runner_thread.start()
            # if the op was read, wait and deliver the result directly
            return_message = None
            if cmd.op == "read":
                runner_thread.join()
                return_message = cmd.result

            if return_message is not None:
                self.thread_safe_buffer_response_queue.put(return_message)
    def shutdown(self):
        verbose_log("LockManagerThread.shutdown()")
        self.running = False
        self.cmd_queue.put(None)

class LockNode:
    def __init__(self, name):
        verbose_log(f"LockNode.__init__(name={name})")
        self.name = name
        self.lock = threading.Lock()
        self.holder = None  # Which thread owns the lock
        self.authority_edges = set()   # children: has authority over
        self.submission_edges = set()  # parents: submits to
        self.waiting_queue = deque()

class LockGraph:
    def __init__(self, *,
                 num_pages=None,
                 key_shapes=None,
                 kernels=None,
                 default_kernel_size=4,
                 default_stride=2,
                 boundary='clip'):
        """
        LockGraph: Dense region lock graph for ND buffer concurrency.

        Node/edge structure (all-to-all, not sparse):

            [Page Layer]      [Key Layer]      [Kernel Layer]      [Vertex Layer]
            ┌───────────┐     ┌──────────┐     ┌────────────┐      ┌────────────┐
            │ page:0    │────▶│ key:pos  │────▶│ kernel:... │─────▶│ vtx:...    │
            │ page:1    │────▶│ key:vel  │────▶│ ...        │─────▶│ ...        │
            │ ...       │────▶│ ...      │────▶│            │─────▶│            │
            └───────────┘     └──────────┘     └────────────┘      └────────────┘

        - Every page is connected to every key, every key to every kernel region, every kernel region to every vertex it covers.
        - All nodes are LockNode instances, all possible edges are present for immediate access and analysis.
        - All nodes and edges are also present in self.nx_graph (networkx.DiGraph) for full graph ops.

        Example node names:
            page:0
            buf:positions
            buf:positions:p0_d0_0_16_d1_0_16   # kernel region for page 0, positions, 2D window
            buf:positions:p0_d0_0_16_d1_0_16:vtx_5_7  # vertex region (optional)

        """
        verbose_log("LockGraph.__init__()")
        self.master_lock = threading.Lock()
        self.nodes = {}   # name -> LockNode
        self.nx_graph = nx.DiGraph()
        self.interval_index = {}  # prefix -> IntervalTree
        # --- for fast edge correlation lookups ---
        self.edge_table = None        # pandas DataFrame of edges + metadata
        self.crash_buffer = None      # circular buffer of verbose results
        self.verbose_mode = False
        # --- Build all nodes and all possible edges ---
        # 1. Pages
        page_nodes = []
        if num_pages is not None:
            for p in range(num_pages):
                pname = f"page:{p}"
                self.add_node(pname)
                page_nodes.append(pname)
        # 2. Keys
        key_nodes = []
        if key_shapes is not None:
            for k in key_shapes:
                kname = f"buf:{k}"
                self.add_node(kname)
                key_nodes.append(kname)
        # 3. Kernel regions
        kernel_nodes = []
        if num_pages is not None and key_shapes is not None:
            for k, shape in key_shapes.items():
                if isinstance(shape, int):
                    dims = (shape,)
                elif isinstance(shape, (tuple, list)):
                    dims = tuple(shape)
                else:
                    raise ValueError(f"Unsupported shape type for key '{k}': {shape}")
                if kernels and k in kernels:
                    kernel = kernels[k]
                    if isinstance(kernel, int):
                        kernel_shape = tuple([kernel] * len(dims))
                        stride = tuple([kernel] * len(dims))
                    elif isinstance(kernel, (tuple, list)):
                        kernel_shape = tuple(kernel)
                        stride = tuple(kernel)
                    elif isinstance(kernel, dict):
                        kernel_shape = tuple(kernel.get('shape', [default_kernel_size]*len(dims)))
                        stride = tuple(kernel.get('stride', kernel_shape))
                    else:
                        raise ValueError(f"Unsupported kernel type for key '{k}': {kernel}")
                else:
                    kernel_shape = tuple([default_kernel_size] * len(dims))
                    stride = tuple([default_stride] * len(dims))
                for page in range(num_pages):
                    # ND sliding window
                    from itertools import product
                    ranges = [range(0, dims[d] - kernel_shape[d] + 1, stride[d]) for d in range(len(dims))]
                    for idxs in product(*ranges):
                        idx_str = "_".join(f"d{d}{idxs[d]}_{idxs[d]+kernel_shape[d]}" for d in range(len(dims)))
                        region_name = f"buf:{k}:p{page}_{idx_str}"
                        self.add_node(region_name)
                        kernel_nodes.append(region_name)

                        # — now add only axis-aligned extreme vertices —
                        dcount = len(dims)
                        # build list of vertex offsets
                        if dcount <= 4:
                            # all corners: 2^d
                            corner_bits = product(*([[0,1]] * dcount))
                            offsets = [
                                tuple(
                                    idxs[d] + bit * (kernel_shape[d] - 1)
                                    for d, bit in enumerate(bits)
                                )
                                for bits in corner_bits
                            ]
                        else:
                            # only low/high on each axis: 2*d
                            center = [
                                idxs[d] + kernel_shape[d] // 2
                                for d in range(dcount)
                            ]
                            offsets = []
                            for d in range(dcount):
                                low  = center.copy(); high = center.copy()
                                low[d]  = idxs[d]
                                high[d] = idxs[d] + kernel_shape[d] - 1
                                offsets.extend([tuple(low), tuple(high)])

                        # add vertex nodes & edges
                        for off in offsets:
                            coord_str = "_".join(str(c) for c in off)
                            vname = f"{region_name}:vtx_{coord_str}"
                            self.add_node(vname)
                            # hierarchical edge region → vertex
                            self.add_authority(region_name, vname)
        # 4. Vertices (optional, for full density)
        # For each kernel region, add all possible vertex nodes it covers
        # (This can be omitted if not needed for your use case.)

        # --- Add all possible edges ---
        # page -> key (all-to-all)
        for p in page_nodes:
            for k in key_nodes:
                self.add_authority(p, k)
        # key -> kernel (all-to-all)
        for k in key_nodes:
            for kr in kernel_nodes:
                # Only connect keys to their own kernel regions
                if kr.startswith(k):
                    self.add_authority(k, kr)


        self.monitor_event = threading.Event()
        # once nx_graph is fully built, build the edge correlation table
        self._build_edge_table()

        # start monitor thread (unchanged)
        self.monitor_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_queues, daemon=True
        )
        self.monitor_thread.start()
    # ——————————— Correlation table machinery ———————————
    def _build_edge_table(self):
        """Construct a pandas DataFrame with one row per edge, plus parsed metadata."""
        import pandas as pd, re
        rows = []
        for src, dst in self.nx_graph.edges():
            m_src = self._parse_node_meta(src)
            m_dst = self._parse_node_meta(dst)
            row = {'src': src, 'dst': dst}
            # prefix each src-meta with “src_” and dst-meta with “dst_”
            for k, v in m_src.items():
                row[f"src_{k}"] = v
            for k, v in m_dst.items():
                row[f"dst_{k}"] = v
            rows.append(row)
        self.edge_table = pd.DataFrame(rows)

    def _parse_node_meta(self, name):
        """Extract page/key/region/vertex info from a node name."""
        # defaults
        info = {'page': None, 'key': None, 'region': None, 'vertex': None}
        # page:<n>
        if name.startswith("page:"):
            info['page'] = int(name.split(":",1)[1])
            return info
        # buf:<key> or buf:<key>:p<page>_<region>...
        if name.startswith("buf:"):
            parts = name.split(":p",1)
            if len(parts) < 2:
                print(parts)
                raise ValueError(f"Invalid node name format: {name}")
            
            prospective_key = parts[0].split("buf:",1)
            if len(prospective_key) < 2:
                print(prospective_key)
                raise ValueError(f"Invalid key format in node name: {name}")    
            info['key'] = prospective_key[1]
            if len(parts)==2:
                pg, rest = parts[1].split("_",1)
                info['page'] = int(pg)
                # region e.g. d0_0_16_d1_0_16…
                region = tuple(tuple(map(int,x.split("_")[-2:]))
                               for x in rest.split("_d") if x)
                info['region'] = region
        # vertex appended
        if ":vtx_" in name:
            base, coords = name.split(":vtx_",1)
            info_v = tuple(map(int, coords.split("_")))
            info['vertex'] = info_v
            # also inherit page/key/region from the base part
            base_meta = self._parse_node_meta(base)
            info.update({k: base_meta[k] for k in ['page','key','region']})
        return info

    def query_edges(self, **filters):
        """
        Slice the edge_table by arbitrary filters:
          e.g. page=2, src_key='positions',
                dst_region=lambda r: any(start<10 for start,end in r)
        Filters map column names (src_page, dst_key, src_region, dst_vertex, etc.)
        to either:
          - a literal to ==-compare
          - a callable f(col_series)→bool mask
        Returns a DataFrame of matching rows.
        """
        df = self.edge_table
        for col, cond in filters.items():
            if callable(cond):
                mask = cond(df[col])
                df = df[mask]
            else:
                df = df[df[col] == cond]
        return df.copy()

    def enable_verbose(self, interval=60, buffer_size=1000):
        """Turn on verbose crash‐buffering, dump every `interval` seconds."""
        import threading
        from collections import deque
        self.verbose_mode = True
        self.crash_buffer = deque(maxlen=buffer_size)
        self._crash_interval = interval

        def _dumper():
            from time import strftime, localtime
            if self.crash_buffer:
                with open("lockgraph_crash.log","a") as f:
                    for ts, res in list(self.crash_buffer):
                        tstr = strftime("%Y-%m-%d %H:%M:%S", localtime(ts))
                        f.write(f"[{tstr}] {res}\n")
                self.crash_buffer.clear()
            threading.Timer(self._crash_interval, _dumper).start()

        # schedule first dump
        threading.Timer(self._crash_interval, _dumper).start()
    def add_node(self, name):
        verbose_log(f"LockGraph.add_node(name={name})")
        with self.master_lock:
            if name not in self.nodes:
                self.nodes[name] = LockNode(name)
                self.nx_graph.add_node(name)

    def add_authority(self, parent, child):
        verbose_log(f"LockGraph.add_authority(parent={parent}, child={child})")
        with self.master_lock:
            self.nodes[parent].authority_edges.add(child)
            self.nodes[child].submission_edges.add(parent)
            self.nx_graph.add_edge(parent, child)

    def try_acquire(self, name, blocking=True):
        verbose_log(f"LockGraph.try_acquire(name={name}, blocking={blocking})")
        current_thread = threading.current_thread()
        with self.master_lock:
            if self._can_acquire(name, current_thread):
                node = self.nodes[name]
                acquired = node.lock.acquire(blocking)
                if acquired:
                    node.holder = current_thread
                return acquired
            else:
                if blocking:
                    node = self.nodes[name]
                    node.waiting_queue.append(current_thread)
                    self.monitor_event.set()
                return False

    def release(self, name):
        verbose_log(f"LockGraph.release(name={name})")
        with self.master_lock:
            node = self.nodes[name]
            if node.holder != threading.current_thread():
                raise RuntimeError("Cannot release a lock not held by this thread")
            node.holder = None
            node.lock.release()
            self.monitor_event.set()  # Wake the monitor thread

    def _can_acquire(self, name, thread):
        verbose_log(f"LockGraph._can_acquire(name={name}, thread={thread})")
        """Return True if no ancestor or descendant lock is held by any other thread."""
        # Breadth-first search up and down
        visited = set()
        queue = deque([name])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            node = self.nodes[n]
            # If the node is locked by someone else, can't acquire
            if node.holder and node.holder != thread:
                return False
            # Recurse to parents (submission_edges) and children (authority_edges)
            queue.extend(node.submission_edges)
            queue.extend(node.authority_edges)
        return True

    def _monitor_queues(self):
        verbose_log("LockGraph._monitor_queues() started")
        while True:
            self.monitor_event.wait()
            with self.master_lock:
                for node in self.nodes.values():
                    if node.waiting_queue and self._can_acquire(node.name, node.waiting_queue[0]):
                        t = node.waiting_queue.popleft()
                        # You would need a more robust notification system here.
                        # For now, just acquire for them (dangerous: demo only)
                        node.lock.acquire()
                        node.holder = t
                        # (In practice, notify thread via event/condition, not this)
            self.monitor_event.clear()

    def read_only_traverse(self, name, fn, direction='both'):
        verbose_log(f"LockGraph.read_only_traverse(name={name}, direction={direction})")
        """
        Traverse the graph from `name`, calling `fn(node)` for every
        node that is NOT currently locked. Skips locked nodes entirely.
        direction: 'authority', 'submission', or 'both'
        """
        with self.master_lock:
            visited = set()
            queue = deque([name])
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                node = self.nodes[n]
                # If locked, skip this node (do not visit or descend)
                if node.lock.locked():
                    continue
                fn(node)
                if direction in ('authority', 'both'):
                    queue.extend(node.authority_edges)
                if direction in ('submission', 'both'):
                    queue.extend(node.submission_edges)

def extra_verbose_node_print(lock_graph):
    """
    Print an ASCII visualization of the lock graph.
    o = open, - = waiting, x = locked.
    Nodes are grouped by their level (distance from root).
    """
    # Find roots (nodes with no submission_edges)
    with lock_graph.master_lock:
        nodes = lock_graph.nodes
        roots = [n for n in nodes if not nodes[n].submission_edges]
        # BFS to assign levels
        level_map = {}
        visited = set()
        queue = [(r, 0) for r in roots]
        while queue:
            n, lvl = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            level_map.setdefault(lvl, []).append(n)
            for c in nodes[n].authority_edges:
                queue.append((c, lvl + 1))
        # Print by level
        print("\n[LockGraph State]")
        for lvl in sorted(level_map):
            line = []
            for n in sorted(level_map[lvl]):
                node = nodes[n]
                if node.lock.locked():
                    ch = "x"
                elif node.waiting_queue:
                    ch = "-"
                else:
                    ch = "o"
                line.append(f"{n}:{ch}")
            print(f"Level {lvl}: " + "  ".join(line))
        print("[End LockGraph State]\n")

