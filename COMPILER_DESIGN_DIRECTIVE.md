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
| 11:8 | 4 | **Destination Address** (dest) |
| 7:4 | 4 | **Source A Address** (reg\_a) |
| 3:0 | 4 | **Source B / Parameter** (reg\_b) |

### **Field Descriptions:**

* **Opcode**: A 4-bit value corresponding to an operation in the Opcode enum (e.g., NAND, SIGL, READ). This allows for 16 unique operations.  
* **dest**: The 4-bit "register" index for the operation's output. The compiler maps this to a full tape address.  
* **reg\_a**: The 4-bit "register" index for the first input operand.  
* **reg\_b**: The 4-bit "register" index for the second input operand. For immediate operations (like SIGL which takes a shift amount k), this field holds the parameter value directly.

This structure provides a simple, clean target for the compiler and a straightforward format for the TapeMachine to decode.