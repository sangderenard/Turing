#ifndef CGPT_H
#define CGPT_H
#include <stddef.h>

// ########## STUB: CGPT data types ##########
// PURPOSE: Placeholder structs representing the model weights.
// EXPECTED BEHAVIOR: When implemented, these will store matrices for
// attention and feed-forward layers.
// INPUTS: N/A
// OUTPUTS: N/A
// KEY ASSUMPTIONS/DEPENDENCIES: Relies on BLIS/BLASFEO for GEMM.
// TODO:
//   - Define weight structures and initialization logic.
// NOTES: This is only a skeleton header.
// ###########################################

typedef struct {
    size_t n_layers;
} CGPTModel;

void cgpt_forward(const CGPTModel* model,
                  const double* input,
                  size_t n_tokens,
                  double* output);

#endif // CGPT_H
