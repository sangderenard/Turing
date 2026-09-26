# **Tape Intermediate Representation (IR) and Architecture**

This document outlines the architecture for compiling a ProvenanceGraph into a binary format suitable for execution on the simulated TapeMachine. The core idea is to treat the ProvenanceGraph as a high-level program that is compiled into low-level "machine code." This machine code is then written to a virtual tape and executed by a processor that reads its instructions directly from that tape.

## **1\. Overall Workflow**

The new end-to-end process follows a standard compiler and virtual machine pipeline:

1. **Trace**: A high-level calculation (e.g., 5 \* 3\) is performed using BitOpsTranslator, which generates a ProvenanceGraph. This graph represents the abstract data-flow of the computation.  
2. **Compile**: A new **TapeCompiler** takes the ProvenanceGraph as input. It performs two main tasks:  
   * **Memory Allocation**: It maps every variable and intermediate value from the graph to a specific address in the tape's data section.  
   * **Code Generation**: It translates each node in the graph into one or more 16-bit binary instructions.  
3. **Assemble**: The compiler uses TapeMap to assemble a complete tape image. This image is a sequence of bit-frames containing:  
   * A **BIOS Header**: Defines tape parameters and the starting address of the instruction code.  
   * An **Instruction Section**: The sequence of 16-bit machine code instructions generated in the previous step.  
   * A **Data Section**: The space allocated for all variables, initialized to zero.  
4. **Execute**: A new **TapeMachine** is initialized with a CassetteTapeBackend that has been "primed" with the compiled tape image. The machine then enters a fetch-decode-execute loop:  
   * **Fetch**: It reads the next 16-bit instruction from the tape.  
   * **Decode**: It parses the 16 bits to determine the operation and its operands.  
   * **Execute**: It invokes the corresponding *physical analog operator* (e.g., nand\_wave), using read\_wave and write\_wave to interact with the data on the tape.

## **2\. 16-Bit Instruction Format**

The IR is a sequence of 16-bit instructions. Each instruction follows a fixed-width format. This is a simple Register-Immediate-like format where "registers" are direct memory addresses on the tape.

| Bits | Size | Purpose |
| :---- | :---- | :---- |
| 15:12 | 4 | **Opcode** (see analog\_spec.Opcode) |
| 11:10 | 2 | **Source A register** (reg\_a) |
| 9:8 | 2 | **Source B register** (reg\_b) |
| 7:6 | 2 | **Destination register** (dest) |
| 5:0 | 6 | **Parameter / third-register selector** (param) |

### **Field Descriptions:**

* **Opcode**: A 4-bit value corresponding to an operation in the Opcode enum (e.g., NAND, SIGL, READ). This allows for 16 unique operations.  
* **dest**: The 2-bit register index for the operation's output. The machine maps this to a full tape address.
* **reg\_a** and **reg\_b**: Two 2-bit source-register indices.
* **param**: A 6-bit immediate. Ternary operations may interpret it as a third-register selector. `LOAD` and `STORE` interpret it as one of 64 fixed-width spill slots following the physical register region.

The terminal storage opcodes are `LOAD` (`0xB`) and `STORE` (`0xC`). A spill
slot begins at `data_start + (REGISTERS + param) * bit_width`. `LOAD` copies a
slot into `dest`; `STORE` copies `reg_a` into a slot. They are tape mechanics,
not additions to the universal Turing operator vocabulary.

This structure provides a simple, clean target for the compiler and a straightforward format for the TapeMachine to decode.

## **3. Recursive structural super-reduction**

The executable bridge does not require arithmetic-specific tape opcodes. Its
general path is:

`machine instruction -> vector Turing graph -> scalar Turing graph -> NAND/data DAG -> tape instructions -> physical events`

The provenance recorder distinguishes carrier arguments from scalar structural
literals. Slice bounds, motion distances, and zero counts are stored as literal
token data; they are never inferred through Python object identity. This avoids
false edges caused by interned small integers and makes the Turing graph
self-describing enough to lower independently of the tracing process.

`scalarize_turing_operator_graph` turns vector structure into graph topology:

* `CONCAT`, `SLICE`, `SIGL`, `SIGR`, `LENGTH`, and `ZEROS` become compile-time
  carrier layout.
* `MU` becomes its general four-NAND Boolean selector per scalar lane.
* Word inputs become ordered scalar leaves; output leaves retain their word-bit
  order.
* Equal constants and equal commutative NAND expressions are hash-consed, while
  every scalar node retains the set of vector Turing parents that produced it.

The spill allocator schedules the live DAG and reuses a slot immediately after
its last operand load. The 6-bit parameter therefore limits *simultaneously
live physical slots* to 64, rather than limiting a program to 64 total values.
Outputs beyond the three-register file remain in spill slots and are observed
there; a 32-bit result is not truncated to three scalar outputs.

`ScalarMachineTapeAssembly` preserves ownership across four graph layers and
can reconstruct a word from its scalar output witnesses. Its execution-cost
preflight follows both the physical cassette head and `TapeTransport`'s logical
cursor. This includes discard reads and the one-frame rewinds caused by reading
all 16 instruction lanes at one tape frame. On compact physical programs, the
predicted seek, read, write, event, and storage counters are checked against the
actual cassette witness.

The current `add eax, 1; ret` specimen lowers without an adder-specific rewrite:

* 2,654 scalar graph nodes, with 1,776 live NAND instructions after slicing;
* 5,425 tape instructions (`1,872 LOAD`, `1,776 NAND`, `1,776 STORE`, `1 HALT`);
* 34 reused spill slots;
* three register outputs plus 29 spill outputs;
* a static execution estimate of 29,596,257 seek frames, 95,778 seeks,
  123,930 reads, 5,424 writes, and 89,176,833 exposed noise sources.

Concurrency is reported before serialization. `analyze_graph_concurrency`
assigns each live operator its dependency depth and reports level widths,
critical-path work, maximum frontier width, and average available parallelism.
For the same ADD image, 1,776 NAND events have a 258-event critical path, a
maximum independent frontier of 32 NANDs, and average available parallelism of
about 6.884. The present cassette has one physical execution lane, so this
parallel topology is preserved as evidence while tape scheduling remains
serial. It can later guide multi-head, multi-track, or replicated-tape layouts
without reconstructing concurrency from flattened instructions.

Static cost estimation also emits one cost vector per encoded instruction.
Ownership descendants can therefore attribute seek distance, reads, writes,
latency, mechanical work, signal energy, noise exposure, and a reliability
bound back to one originating machine instruction. Compact executions provide
the same query over observed physical events. Aggregate totals are the serial
composition of these event vectors, so attribution and whole-program accounting
share one algebra.

That estimate is intentionally visible. Under the present analog timing model
it represents about 98,956 seconds of modeled tape time, so compact programs
are executed for physical witnesses while large images are preflighted before a
user elects to incur the full simulation.

The structural route is also joined to the source-facing bridge. ProcessGraph
ingestion retains the full Turing provenance metadata, including structural
literals, and records the chosen bit width on the graph. When an object method
contains arithmetic, its journey gains two strictly decreasing Turing ranks:
vector structure at depth 1 and scalar NAND topology at depth 0. A two-bit
`WordOps.add(x, y)` specimen executes all seven visible stages:

`OBJECT -> PROCESS -> BITOPS -> TURING(vector) -> TURING(scalar) -> TAPE -> PHYSICAL`

For `1 + 1`, it emits 165 tape instructions and physically observes the word
value `2`. The object-method ancestor reaches every owned physical event through
the added vector-to-scalar morphism; scalarization is not an opaque codegen
step.

`ExecutedReductionArtifact` exposes that composition as a query. Given any
stage index and node identity, it deduplicates all descendant physical events
and serially combines their cost vectors. Distance, latency, storage, energy,
noise, and reliability can therefore be requested for an object method, a
ProcessGraph element, a BitOps operation, either Turing rank, or one tape
instruction through the same API. The artifact also selects its active terminal
graph and reports dependency concurrency alongside the observed physical-lane
count.
