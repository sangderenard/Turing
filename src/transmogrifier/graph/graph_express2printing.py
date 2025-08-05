import colorsys
import networkx as nx
import matplotlib.pyplot as plt
from ..solver_types import Operation
from matplotlib.animation import FuncAnimation
try:
    from colorama import Style
except ImportError:
    class Style:
        RESET_ALL = '\033[0m'

MAX_HUES = 16

def get_color(level, num_levels, max_hues=MAX_HUES):
    """Generate ANSI TrueColor escape for a level normalized to num_levels (capped at max_hues)."""
    # Determine effective hue count
    hue_count = min(num_levels, max_hues)
    # Normalize level into [0,1)
    h = (level % hue_count) / hue_count
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return f"\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

# fallback static list (unused)
colors = []

class GraphExpresss2Printer:
    def print_parallel_bands(self):
        """
        Colorized print of parallel execution bands.
        """
        bands = self.serialize_bands()
        num_levels = len(bands)
        print("\n=== Parallel execution bands ===")
        for lvl in sorted(bands):
            color = get_color(lvl, num_levels)
            print(f"{color}Level {lvl}:{Style.RESET_ALL}")
            for tp, labels in bands[lvl].items():
                print(f"{color}  {tp}:{Style.RESET_ALL}")
                for lbl in labels:
                    print(f"{color}    - {lbl}{Style.RESET_ALL}")

    def print_colorized_operations(self):
        """
        Colorized print of operations sorted by dependencies.
        """
        proc = self.extract_full_process_graph()
        _, _, _, ops = self.parse_requirements(proc)
        print("\n=== Operations ===")
        # normalize hues to sequence levels count
        seqs = [op.sequence_order for op in ops.values()]
        num_levels = max(seqs) + 1 if seqs else 1
        for op in Operation.default_sort(ops.values()):
            lvl = op.sequence_order
            color = get_color(lvl, num_levels)
            print(f"{color}{op}{Style.RESET_ALL}")

    def print_colorized_expressions(self):
        """
        Inline serial colorization: show each node's full symbolic expression with component levels colored.
        """
        proc = self.extract_full_process_graph()
        nodes_meta = proc['nodes']
        num_levels = len(proc['levels'])  # for hue normalization
        print("\n=== Dependency fabric with color-coded parent inclusions ===")
        for nid, data in nodes_meta.items():
            text = data['label']
            for pid, _ in data['parents']:
                parent_label = nodes_meta[pid]['label']
                parent_level = nodes_meta[pid]['level']
                color = get_color(parent_level, num_levels)
                if parent_label in text:
                    import re
                    pattern = re.compile(re.escape(parent_label))
                    text = pattern.sub(f"{color}[{parent_label}]{Style.RESET_ALL}", text, count=1)


            print(f"Expr: {text}")

    def print_bands_and_ops(self):
        # Colorized parallel bands, expressions, and operations
        self.print_parallel_bands()
        self.print_colorized_expressions()
        self.print_colorized_operations()


    def plot_simple_graph(self, graph, layout='spring'):
        """
        Plots a simple flowchart-like view of the graph without edge labels.
        """
        plt.figure(figsize=(12, 8))
        if layout == 'spring':
            pos = nx.spring_layout(graph, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=800,
                node_color='lightblue', edge_color='gray', font_size=10)
        plt.show()

    def plot_graph_with_roles(self, layout='spring'):
        """
        Plots the graph with edge labels showing producer->consumer roles from Edge.extra.
        """
        plt.figure(figsize=(12, 8))
        if layout == 'spring':
            pos = nx.spring_layout(self.G, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        nx.draw(self.G, pos, with_labels=True, node_size=800, node_color='lightblue',
                edge_color='gray', font_size=10)
        
        # Extract edge roles
        edge_labels = {}
        for u, v, data in self.G.edges(data=True):
            # Build label from your Edge extras
            extras = data.get('extra', [])
            label_parts = []
            for e in extras:
                if hasattr(e, 'id') and len(e.id) >= 4:
                    label_parts.append(f"{e.id[2]}→{e.id[3]}")
            edge_labels[(u, v)] = ", ".join(label_parts)
        
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)
        plt.show()


    def get_rgb_color(level, num_levels):
        """Convert a level to an RGB tuple."""
        hue = (level % num_levels) / num_levels
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        return (r, g, b)

    def animate_data_flow(self, dataG, duration=5, fps=30):
        """
        Animate the data flow graph by cycling through datasets present in edge 'extras'.
        Colors nodes and edges belonging to the current dataset.
        """
        # 1) layout
        pos = nx.shell_layout(dataG)

        # 2) get nested grouping: { level → { type → { role → [ (src, tgt), … ] } } }
        grouped = self.group_edges_by_dataset(dataG)

        # 3) flatten that grouping into an ordered sequence (level→type→role)
        ordered_keys = self.sort_roles(grouped)
        total = len(ordered_keys)

        # 4) draw static background
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Data Flow Animation")
        ax.axis('off')
        nx.draw_networkx_edges(dataG, pos, ax=ax, edge_color='lightgray')
        node_collection = nx.draw_networkx_nodes(dataG, pos, ax=ax, node_color='lightgray')
        nx.draw_networkx_labels(dataG, pos, ax=ax, font_size=8)

        # 5) keep handles to each edge line + its extras
        edge_lines = {}
        for (u, v, attrs) in dataG.edges(data=True):
            line = ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color='lightgray',
                linewidth=2
            )[0]
            edge_lines[(u, v)] = (line, attrs.get('extras', []))

        # 6) animation update uses only ordered_keys
        def update(frame):
            current = ordered_keys[frame % total]
            ax.set_title(f"Dataset: {current}")

            # highlight edges in this dataset
            for (u, v), (line, extras) in edge_lines.items():
                if current in extras:
                    line.set_color('red')
                    line.set_linewidth(3)
                else:
                    line.set_color('lightgray')
                    line.set_linewidth(1)

            # highlight connected nodes
            highlights = {
                u for (u, v), (_, extras) in edge_lines.items() if current in extras
            } | {
                v for (u, v), (_, extras) in edge_lines.items() if current in extras
            }
            node_colors = [
                'blue' if n in highlights else 'lightgray'
                for n in dataG.nodes()
            ]
            node_collection.set_color(node_colors)
            return list(edge_lines.keys()) + [node_collection]

        frames = total * fps
        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False, repeat=True)
        plt.show()