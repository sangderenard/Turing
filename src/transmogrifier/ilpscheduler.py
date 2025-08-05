import sympy
import networkx as nx
class ILPScheduler:
    def __init__(self, process_graph):
        self.G = process_graph.G
        self.operator_signatures = process_graph.role_schemas if hasattr(process_graph, 'role_schemas') else {}
        self.levels_asap = {}
        self.levels_alap = {}
        self.level_symbols = {}
        self.phase_symbols = {}
        self.ilp_constraints = []
        self.phase_constraints = []

    def compute_levels(self, method, order):
        return_value = None
        match method:
            case 'asap':
                return_value = self.compute_asap_levels()
            case 'alap':
                return_value = self.compute_alap_levels()
            case 'max_local_slack':
                return_value = self.compute_max_local_slack_levels()
            case _:
                return_value = self.compute_alap_levels()  # Default to ALAP if unknown method

        return return_value
        match order:
            case 'dependency':
                return return_value
            case 'processing':
                # Reverse the levels explicitly
                inverted_levels = {nid: (max(return_value.values()) - lvl) for nid, lvl in return_value.items()}
                return inverted_levels
            case _:
                return return_value


    # ----------------------
    # Integer level scheduling
    # ----------------------
    def compute_asap_levels(self):
        levels = {}
        def dfs(n):
            if n in levels:
                return levels[n]
            preds = [p for p, _ in self.G.nodes[n]['parents']]
            lvl = 0 if not preds else 1 + max(dfs(p) for p in preds)
            levels[n] = lvl
            return lvl
        for nid in list(self.G.nodes):
            dfs(nid)
        self.levels_asap = levels
        return levels

    def compute_alap_levels(self):
        asap = self.compute_asap_levels()
        max_level = max(asap.values())
        alap = {}

        def dfs(node):
            if node in alap:
                return alap[node]
            children = [c for c, _ in self.G.nodes[node]['children']]
            if not children:
                alap[node] = asap[node]  # keep output at its ASAP level to preserve makespan
            else:
                alap[node] = min(dfs(c) - 1 for c in children)
            return alap[node]

        for nid in self.G.nodes:
            dfs(nid)
        self.levels_alap = alap
        return alap

    def compute_max_local_slack_levels(self):
        """
        Compute schedule where each node is delayed as long as it can be
        under its own immediate successors, maximizing local slack,
        even if it stretches overall makespan.
        """
        asap = self.compute_asap_levels()
        slack_schedule = asap.copy()

        changed = True
        while changed:
            changed = False
            for nid in self.G.nodes:
                children = [c for c, _ in self.G.nodes[nid]['children']]
                if not children:
                    continue
                min_child_level = min(slack_schedule[c] for c in children)
                max_local_level = min_child_level - 1
                if max_local_level > slack_schedule[nid]:
                    slack_schedule[nid] = max_local_level
                    changed = True

        self.levels_max_local_slack = slack_schedule
        return slack_schedule
    def compute_asap_maxslack_interference(self, mode="asap-maxslack"):
        """
        Run ASAP and max local slack scheduling,
        then compute lifespans and build conservative interference graph.
        This gives the 'race-free' union interference graph.
        """
        # Compute the bounds
        asap = {}
        max_slack = {}
        if mode == "asap":
            asap = self.compute_asap_levels()
            max_slack = asap.copy()
        elif mode == "alap":
            asap = self.compute_alap_levels()
            max_slack = asap.copy()
        elif mode == "asap-alap":
            asap = self.compute_asap_levels()
            max_slack = self.compute_alap_levels()
        elif mode == "alap-maxslack":
            asap = self.compute_alap_levels()
            max_slack = self.compute_max_local_slack_levels()
        elif mode == "asap-maxslack":
            asap = self.compute_asap_levels()
            max_slack = self.compute_max_local_slack_levels()
        elif mode == "maxslack":
            max_slack = self.compute_max_local_slack_levels()
            asap = max_slack.copy()

        # Build lifespan ranges
        lifespans = {nid: (asap[nid] - 1, max_slack[nid]) for nid in asap} # writes happen previous round, reads happen current round
        
        # Build minimal interference graph (pure node lifespans overlap)
        interference_graph = nx.Graph()
        interference_graph.add_nodes_from(asap.keys())

        nodes = list(asap.keys())
        for i, n1 in enumerate(nodes):
            s1, e1 = lifespans[n1]
            for n2 in nodes[i+1:]:
                s2, e2 = lifespans[n2]
                if not (e1 < s2 or e2 < s1):
                    # lifespans overlap -> interference
                    interference_graph.add_edge(n1, n2)

        # Attach to the object for later exploration
        self.lifespans_union = lifespans
        self.interference_graph = interference_graph



        return interference_graph, lifespans

    # ----------------------
    # Integer symbolic constraints
    # ----------------------
    def build_level_symbols(self):
        self.level_symbols = {nid: sympy.Symbol(f"L_{nid}", integer=True) for nid in self.G.nodes}

    def build_ilp_constraints(self):
        """
        Build classic scheduling constraints: L_parent + 1 <= L_child for each edge.
        """
        constraints = []
        for nid in self.G.nodes:
            L_nid = self.level_symbols[nid]
            for child, _ in self.G.nodes[nid]['children']:
                L_child = self.level_symbols[child]
                constraints.append(L_nid + 1 <= L_child)
        self.ilp_constraints = constraints
        return constraints

    def solve_ilp_symbolically(self):
        solution = sympy.solve(self.ilp_constraints, list(self.level_symbols.values()), dict=True)
        return solution

    # ----------------------
    # Phase (rotor) harmonic system
    # ----------------------
    def build_phase_symbols(self):
        self.phase_symbols = {nid: sympy.Symbol(f"θ_{nid}", real=True) for nid in self.G.nodes}

    def build_harmonic_constraints(self, harmonics=3):
        """
        Build harmonic phase lock constraints:
        sum_m sin(m*(θ_c - θ_p - latency)) / m
        captures multi-harmonic coupling between parent and child.
        """
        constraints = []
        for nid in self.G.nodes:
            θ_n = self.phase_symbols[nid]
            for child, _ in self.G.nodes[nid]['children']:
                θ_c = self.phase_symbols[child]
                harmonic_sum = sum(sympy.sin(m * (θ_c - θ_n - 1)) / m for m in range(1, harmonics+1))
                constraints.append(harmonic_sum)
        self.phase_constraints = constraints
        return constraints

    def solve_symbolic_phase_system(self):
        solution = sympy.solve(self.phase_constraints, list(self.phase_symbols.values()), dict=True)
        return solution

    # ----------------------
    # Diagnostics
    # ----------------------
    def print_asap_levels(self):
        print("\n=== ASAP Levels ===")
        for nid, lvl in sorted(self.levels_asap.items(), key=lambda x: x[1]):
            print(f"Node {nid}: Level {lvl}")

    def print_alap_levels(self):
        print("\n=== ALAP Levels ===")
        for nid, lvl in sorted(self.levels_alap.items(), key=lambda x: x[1]):
            print(f"Node {nid}: Level {lvl}")

    def print_ilp_constraints(self):
        print("\n=== ILP Constraints ===")
        for c in self.ilp_constraints:
            print(c)
    def print_max_local_slack_levels(self):
        print("\n=== Maximum Local Slack Levels ===")
        for nid, lvl in sorted(self.levels_max_local_slack.items(), key=lambda x: x[1]):
            print(f"Node {nid}: Level {lvl}")

    def print_phase_constraints(self):
        print("\n=== Harmonic Phase Constraints ===")
        for c in self.phase_constraints:
            print(c)
