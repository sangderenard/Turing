"""Native batch receipts for the four arithmetic sites in Spectral T4."""

import numpy as np
import pytest
import shutil
import subprocess

from src.compiler.spectral_wave_laws import (
    compile_wave_compute_shader, compile_wave_kernel)


@pytest.mark.parametrize("kind", (
    "angular_propagating", "angular_evanescent",
    "aperture_material", "jones_interface"))
def test_wave_law_native_batch_matches_t4_arithmetic(tmp_path, kind):
    batch = 4
    piece = compile_wave_kernel(kind, batch, directory=tmp_path / kind)
    assert piece.batch == batch
    assert piece.source and piece.module and piece.compiler

    if kind == "angular_propagating":
        values = {
            "ar": np.array([1., .5, -.25, .125]),
            "ai": np.array([0., -.3, .4, -.2]),
            "kx": np.array([0., 1e6, 2e6, 3e6]),
            "ky": np.array([0., .5e6, .2e6, .1e6]),
            "wavelength": np.array([550e-9, 500e-9, 650e-9, 450e-9]),
            "distance": np.array([1e-6, 2e-6, 3e-6, 4e-6]),
            "direction": np.array([1., -1., 1., -1.]),
        }
        kz = np.sqrt((2 * np.pi / values["wavelength"]) ** 2
                     - values["kx"] ** 2 - values["ky"] ** 2)
        phase = values["distance"] * kz
        tr, ti = np.cos(phase), values["direction"] * np.sin(phase)
        expected = (values["ar"] * tr - values["ai"] * ti,
                    values["ar"] * ti + values["ai"] * tr)
    elif kind == "angular_evanescent":
        values = {
            "ar": np.array([1., .5, -.25, .125]),
            "ai": np.array([0., -.3, .4, -.2]),
            "gap": np.array([1e12, 4e12, 9e12, 16e12]),
            "distance": np.array([1e-7, 2e-7, 3e-7, 4e-7]),
        }
        factor = np.exp(-values["distance"] * np.sqrt(values["gap"]))
        expected = values["ar"] * factor, values["ai"] * factor
    elif kind == "aperture_material":
        values = {
            "ar": np.array([1., .5, -.25, .125]),
            "ai": np.array([0., -.3, .4, -.2]),
            "wavelength": np.array([550e-9, 500e-9, 650e-9, 450e-9]),
            "distance": np.array([1e-6, 2e-6, 3e-6, 4e-6]),
            "direction": np.array([1., -1., 1., -1.]),
            "real_index": np.array([1.5, 1.4, 1.3, 1.2]),
            "background_index": np.ones(batch),
            "imaginary_index": np.array([0., .01, .03, .05]),
        }
        k0 = 2 * np.pi / values["wavelength"]
        phase = values["direction"] * k0 * (
            values["real_index"] - values["background_index"]
        ) * values["distance"]
        attenuation = np.exp(-k0 * values["imaginary_index"]
                             * values["distance"])
        tr, ti = attenuation * np.cos(phase), attenuation * np.sin(phase)
        expected = (values["ar"] * tr - values["ai"] * ti,
                    values["ar"] * ti + values["ai"] * tr)
    else:
        values = {
            name: np.array([base + i * .1 for i in range(batch)])
            for name, base in (
                ("sr", 1.), ("si", -.1), ("pr", .5), ("pi", .2),
                ("a_re", .8), ("a_im", .1), ("b_re", -.2), ("b_im", .3),
                ("c_re", .4), ("c_im", -.15), ("d_re", .7), ("d_im", .05))
        }
        expected = (
            values["a_re"] * values["sr"] - values["a_im"] * values["si"]
            + values["b_re"] * values["pr"] - values["b_im"] * values["pi"],
            values["a_re"] * values["si"] + values["a_im"] * values["sr"]
            + values["b_re"] * values["pi"] + values["b_im"] * values["pr"],
            values["c_re"] * values["sr"] - values["c_im"] * values["si"]
            + values["d_re"] * values["pr"] - values["d_im"] * values["pi"],
            values["c_re"] * values["si"] + values["c_im"] * values["sr"]
            + values["d_re"] * values["pi"] + values["d_im"] * values["pr"],
        )
    actual = piece(*(values[name] for name in piece.argument_names))
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", (
    "angular_propagating", "angular_evanescent",
    "aperture_material", "jones_interface"))
def test_wave_law_emits_valid_deployable_gpu_kernel(tmp_path, kind):
    artifact = compile_wave_compute_shader(kind)
    assert artifact.complete, artifact.shortfalls
    assert artifact.bindings and artifact.component_abi and artifact.api
    assert artifact.local_size == (64, 1, 1)
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("SPIR-V assembler/validator unavailable")
    source = artifact.write(tmp_path)
    binary = tmp_path / (artifact.name + ".spv")
    assembled = subprocess.run(
        ["spirv-as", "--target-env", artifact.target_env, str(source),
         "-o", str(binary)], capture_output=True, text=True)
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        ["spirv-val", "--target-env", artifact.target_env, str(binary)],
        capture_output=True, text=True)
    assert validated.returncode == 0, validated.stderr
