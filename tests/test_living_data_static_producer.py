"""Static canonical projection must not compile or start gameplay."""
import pytest
from meowdoku.storybook_environment import build_storybook_world
from src.compiler import abstract_ui_div_map as module


def test_static_projection_is_safe_standalone(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Gameplay producer invoked by static projection')
    for name in ('spawn_world_player','symbolic_world_physics_wasm_plugin','symbolic_vehicle_physics_wasm_plugin','symbolic_wheel_contact_wasm_plugin','living_map_loop_deployment','compile_audio_fft_wasm','engine_pcm_kernel_bank_model'):
        monkeypatch.setattr(module,name,forbidden)
    ui=module.project_world_to_div_map(build_storybook_world(),static_only=True)
    assert not set(ui.model)&{'entity_mezzanine','vehicle_slot','physics_program','loop_deployment','engine_audio','music_room'}
    script=ui.packet('javascript').source
    assert script.count('globalThis.livingDataStaticHost = Object.freeze')==1
    assert 'navigationRuntime.ready = Promise.resolve()' in script
    assert 'throw Error("Physics is unavailable in the static map host")' in script
    assert 'throw Error("Game updates are unavailable in the static map host")' in script
    assert 'Static AbstractUI projection' in ui.html
    assert len(ui.model['world']['objects'])==33
    assert script.index("document.body.classList.remove('qualification-pending')") < script.index('const livingDataStaticReady = new Promise')


def test_static_projection_rejects_game_mezzanine():
    with pytest.raises(ValueError,match='static map'):
        module.project_world_to_div_map(build_storybook_world(),static_only=True,entity_mezzanine=object())
