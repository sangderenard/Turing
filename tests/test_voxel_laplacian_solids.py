import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.cells.bath.voxel_fluid import VoxelMACFluid, VoxelFluidParams

def test_constant_field_invariant_under_laplacian_with_solids():
    params = VoxelFluidParams(nx=4, ny=4, nz=4)
    vf = VoxelMACFluid(params)
    solid = np.zeros((params.nx, params.ny, params.nz), dtype=bool)
    solid[1,1,1] = True
    solid[2,2,2] = True
    vf.set_solid_mask(solid)

    X = np.ones((params.nx, params.ny, params.nz))
    Lcc = vf._laplace_cc(X, vf.solid)
    assert np.allclose(Lcc[~vf.solid], 0.0)

    Fu = np.ones((params.nx+1, params.ny, params.nz))
    Fu[vf.solid_u] = 0.0
    H = vf._helmholtz_face_apply(Fu, a=0.25, axis=0, solid=vf.solid_u)
    assert np.allclose(H[~vf.solid_u], Fu[~vf.solid_u])
