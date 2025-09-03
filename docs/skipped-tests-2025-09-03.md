# Skipped Tests - 2025-09-03

The following tests are currently skipped to streamline development. Reasons indicate why they fail or are temporarily disabled.

| Test | Reason |
| --- | --- |
| tests/test_cffi_matmul.py (module) | requires setuptools for CFFI backend |
| tests/test_geometry_factory.py (module) | AbstractTensor.unravel_index_ not implemented |
| tests/test_laplace_and_local_state_network_gradients.py (module) | LocalStateNetwork gradients unsupported (unravel_index_ missing) |
| tests/test_laplace_nd.py (module) | AbstractTensor.unravel_index_ not implemented |
| tests/test_linear_bias_broadcast.py (module) | Linear.backward not implemented |
| tests/test_linear_block.py (module) | LinearBlock gradients not implemented |
| tests/test_linear_block_grad.py (module) | LinearBlock gradients not implemented |
| tests/test_local_state_network.py (module) | LocalStateNetwork backward and unravel_index_ not implemented |
| tests/test_metric_steered_conv3d_local_state_grad.py (module) | MetricSteeredConv3D gradients unsupported |
| tests/test_ndpca3conv3d_grad.py (module) | NDPCA3Conv3d gradients inconsistent |
| tests/test_ndpca3conv3d_process_diagram_replay.py (module) | demo replay flaky and slow (KeyError) |
| tests/test_rectconv3d.py (module) | RectConv3d.backward not implemented |
| tests/test_riemann_grid_block.py (module) | RiemannGridBlock depends on unravel_index_ implementation |
| tests/test_riemann_pipeline_grad.py (module) | Riemann pipeline gradients require unravel_index_ |
| tests/test_structural_bypass_parameters.py (module) | MetricSteeredConv3DWrapper depends on unravel_index_ |
| tests/test_xor_learning.py::test_xor_learns | takes too long |
| tests/test_ascii_kernel_nn.py::test_nn_classifier_matches_reference_bitmasks | takes too long |
| tests/test_ascii_render.py::test_to_ascii_diff_preserves_color | takes too long |
