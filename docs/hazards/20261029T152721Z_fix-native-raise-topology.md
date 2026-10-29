# 2026-10-29 Native codec reaches protected conversion failure edges

The reducer now records that a slice of a source value whose exact class is
`src.common.tensors.abstraction.AbstractTensor` remains that class. The
class row derives from the base's `SOURCE_VALUE_CLASS` cell and both value
identities; the source implementation of `AbstractTensor.__getitem__`
wraps a tensor slice in `type(self)`. The planned scalar type fixed point
reads that class row and the ProgramABI's rank-1 element dtype. It carries
the dtype across the slice's exact base edge and the tensor `tolist()`
operand edge into `enumerate()`'s second loop target. No dtype default is
assigned to a name or position.

Before this change the bounded public compile of the real
`tensor_octets_to_bytes` source refused at `region value 4 (input 'value')
has no settled dtype`. After the change the same compile reached SSA and
refused at the two exact protected calls: `protected call has no lowered
native failure edge: call=65, try=56` and `call=80, try=56`. This is the
next actual compiler error and remains red. This change proves element
type propagation, not native `tolist()` storage materialization.

The two existing native C raise corpus cases passed (2 passed, 651
deselected). Both edited modules passed `py_compile`. The broad `view`
concordance audit still stops at its prior `selected return has no exact
source call reachability authority` refusal. The next raise work is the
protected call status edge, matched to its exact source transfer and
handler-entry cell; the compiled codec artifact remains unfinished.
