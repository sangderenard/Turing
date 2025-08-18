#include "cgpt.h"
#include <string.h>

// ########## STUB: cgpt_forward ##########
// PURPOSE: Execute a forward pass through a tiny GPT model.
// EXPECTED BEHAVIOR: When implemented, this will call into BLIS or
// BLASFEO for matrix multiplications, apply attention and feed-forward
// layers, and produce logits for each token.
// INPUTS: ``model`` weights, ``input`` token embeddings, ``n_tokens`` count.
// OUTPUTS: ``output`` logits buffer filled with computed values.
// KEY ASSUMPTIONS/DEPENDENCIES: Requires vendored BLIS/BLASFEO and
// SLEEF for activation functions.
// TODO:
//   - Implement attention blocks using GEMM from BLIS/BLASFEO.
//   - Integrate softmax from SLEEF or custom approximation.
// NOTES: Currently zeroes the output as a placeholder.
// ###########################################
void cgpt_forward(const CGPTModel* model,
                  const double* input,
                  size_t n_tokens,
                  double* output) {
    (void)model;
    (void)input;
    memset(output, 0, n_tokens * sizeof(double));
}
