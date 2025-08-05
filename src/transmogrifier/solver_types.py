SIMD_DEFAULT_CONCURRENCY = 4  # Default concurrency for SIMD operations
SCATTER_LIMIT = 8             # Maximum number of outputs for scatter operations

class Operation:
    def __init__(self, id, inputs, max_inputs, outputs, max_outputs, string, type, sequence_order, time_penalty,
                 concurrency=SIMD_DEFAULT_CONCURRENCY, allows_inplace=True):
        self.id = id
        self.inputs = inputs
        self.max_inputs = max_inputs
        self.outputs = outputs
        self.max_outputs = max_outputs
        self.string = string
        self.type = type
        self.sequence_order = sequence_order
        self.time_penalty = time_penalty
        self.concurrency = concurrency
        self.allows_inplace = allows_inplace

    @staticmethod
    def get_matmul_fuse(inputs, outputs):
        return Operation(inputs, 3, outputs, SCATTER_LIMIT, "MatMulFuseAdd", ((0,1,2),(0)), 0.0)

    @staticmethod
    def to_string(inputs):
        if isinstance(inputs, list):
            return ','.join(str(x) for x in inputs)
        return str(inputs)

    @staticmethod
    def default_sort(operations):
        return sorted(operations, key=lambda op: (op.sequence_order, op.type, op.string))

    def __repr__(self):
        return (f"<\"{self.string}\":\"{self.type}\", WAVEPOINT: {self.sequence_order}, "
                f"IN({Operation.to_string(self.inputs)})[{self.max_inputs}] -> "
                f"O({Operation.to_string(self.outputs)})[{self.max_outputs}], "
                f"SIMD: {self.concurrency}, inplace: {self.allows_inplace}>")

READ = 1
WRITE = 2
READWRITE = 3


class NodeSet:
    """
    NodeSet is your 3D storage space (n,m,k) that also automatically supports arbitrary ND -> 3D (or flat)
    index mapping. It is the canonical space for storing and addressing your nodes.
    """
    def __init__(self, n, m, k):
        self.shape = (n, m, k)
        self.size = n * m * k

    def flat_from_nd(self, nd_idx):
        """
        Compute flat index in the 3D storage array from ND index of any dimension count.
        The ND index is first converted into the equivalent flat ordinal in its own shape,
        then remapped into this NodeSet's flat space.
        """
        if len(nd_idx) == 3 and self.shape == (len(set(nd_idx)) or 0):
            # direct 3D shortcut (unlikely to be exact)
            return nd_idx[0] * self.shape[1] * self.shape[2] + nd_idx[1] * self.shape[2] + nd_idx[2]
        else:
            # general ND to flat
            raise ValueError("Use translate_nd_index for full ND mapping.")

    def nd_from_flat(self, flat_idx):
        """
        Given a flat index in this 3D NodeSet, return its (n,m,k) index.
        """
        n, m, k = self.shape
        i = flat_idx // (m * k)
        j = (flat_idx % (m * k)) // k
        l = flat_idx % k
        return (i, j, l)

    def translate_nd_index(self, source_shape, source_index):
        """
        Given a coordinate in another ND shape, compute what (n,m,k) index
        it should map to in this NodeSet's 3D shape by preserving flat ordinal.
        """
        flat_in_source = self._nd_to_flat(source_index, source_shape)
        return self._flat_to_nd(flat_in_source, self.shape)

    def _nd_to_flat(self, nd_idx, shape):
        """
        Generic ND to flat index calculator.
        """
        strides = []
        prod = 1
        for s in reversed(shape[1:]):
            prod *= s
            strides.insert(0, prod)
        strides.append(1)
        return sum(i * s for i, s in zip(nd_idx, strides))

    def _flat_to_nd(self, flat_idx, shape):
        """
        Generic flat to ND index calculator.
        """
        strides = []
        prod = 1
        for s in reversed(shape[1:]):
            prod *= s
            strides.insert(0, prod)
        strides.append(1)

        result = []
        for stride in strides:
            result.append(flat_idx // stride)
            flat_idx %= stride
        return tuple(result)

    def __getitem__(self, nd_idx):
        """
        Allow indexing: node_set[ (i,j,k) ] returns flat index (optional customization)
        """
        return self._nd_to_flat(nd_idx, self.shape)

    def __repr__(self):
        return f"<NodeSet shape={self.shape}, size={self.size}>"

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
import numpy as np

@dataclass#(frozen=True)
class DomainNode:
    """
    Represents a multi-dimensional allocation of unit nodes in the computational memory graph.
    """
    shape: tuple
    unit_size: int = 1  # in fundamental allocation units (could be floats, bytes, or abstract slots)
    id: int = field(default_factory=lambda: DomainNode._generate_id())

    total_elements: int = field(init=False)
    total_allocation: int = field(init=False)
    _id_counter: int = 0

    @staticmethod
    def _generate_id():
        DomainNode._id_counter += 1
        return DomainNode._id_counter

    def __post_init__(self):
        # compute total elements and allocation
        self.memory = {}  # Initialize memory as a dictionary for dynamic storage
        if not self.shape:
            self.shape = (1,)  # Default to a single unit if no shape is provided
        object.__setattr__(self, 'shape', tuple(self.shape))  # Ensure shape is immutable
        object.__setattr__(self, 'total_elements', self._compute_total_elements())
        object.__setattr__(self, 'total_allocation', self.total_elements * self.unit_size)

    def _compute_total_elements(self):
        if not self.shape:
            return 1
        if isinstance(self.shape, int):
            return self.shape * self.unit_size
        return reduce(mul, self.shape, 1)

    def locations(self):
        """
        Generator for iterating over all multi-dimensional indices in this domain.
        Example: for shape (2,3), yields (0,0),(0,1),(0,2),(1,0),...
        """
        return np.ndindex(*self.shape) if self.shape else iter([()])

    def put(self, node_id, value):
        """
        Store a value in memory for the given node ID.
        :param node_id: ID of the node to store the value for.
        :param value: Value to store.
        """
        self.memory[node_id] = value

    def get(self, node_id):
        """
        Retrieve a value from memory for the given node ID.
        :param node_id: ID of the node to retrieve the value for.
        :return: Value stored for the node ID, or None if not found.
        """
        return self.memory.get(node_id)
    
    def get_all(self):
        """
        Retrieve all values stored in memory.
        :return: Dictionary of all node IDs and their corresponding values.
        """
        return self.memory.copy()

    def __repr__(self):
        return (f"DomainNode(shape={self.shape}, unit_size={self.unit_size}, "
                f"total_elements={self.total_elements}, total_allocation={self.total_allocation}, "
                f"memory={self.memory})")

    def flatten_index(self, idx):
        """
        Given a multi-dimensional index tuple, flatten to a single index (row-major).
        Example: shape (2,3), idx (1,2) => 5
        """
        if not self.shape:
            return 0
        strides = [reduce(mul, self.shape[i+1:], 1) for i in range(len(self.shape))]
        return sum(i * s for i, s in zip(idx, strides))

    def expand_index(self, flat_idx):
        """
        Given a flat index, convert to multi-dimensional index tuple.
        Example: shape (2,3), flat_idx 5 => (1,2)
        """
        if not self.shape:
            return ()
        result = []
        for dim in self.shape:
            result.append(flat_idx % dim)
            flat_idx //= dim
        return tuple(reversed(result))


class Node:
    def __init__(self, id, location_in_set, location_in_memory, readwrite=None, exclusionary=True):
        self.id = id
        self.location_in_set = location_in_set
        self.location_in_memory = location_in_memory
        self.readwrite = readwrite
        self.exclusionary = exclusionary

    def __repr__(self):
        return (f"<Node id={self.id} in_set={self.location_in_set} "
                f"mem={self.location_in_memory} rw={self.readwrite} excl={self.exclusionary}>")

class Edge:
    def __init__(self, id, operation, source, target, weight=None, store_id=None):
        self.id = id
        self.operation = operation
        self.source = source
        self.target = target
        self.store_id = store_id
        self.weight = weight

    def __repr__(self):
        return (f"<Edge op={self.operation} from={self.source} to={self.target} weight={self.weight}>")
