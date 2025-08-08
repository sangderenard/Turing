Mechanical split of `memory_graph.py`:

memory_graph_pkg/
  __init__.py
  memory_graph.py          # ONLY BitTensorMemoryGraph (+ from .helpers import *)
  helpers/
    __init__.py            # imports each helper in original order
    <one file per class>   # each file has the original header + class body
Preserved nearby globals:
  - META_GRAPH_TRANSFER_BUFFER_SIZE kept with MetaGraphEdge
  - meta_nodes, root_meta_nodes, master_graph kept with GraphSearch
No logic changed; only organization and intra-package imports added.
