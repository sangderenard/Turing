"""Small real-Rich terminal tests; no long compiler probes."""
from __future__ import annotations

import builtins
import io
import sys

import pytest

from src.compiler.shell_telemetry import (
    CompileProgress, CompileStage, active_compile_progress,
    active_rule_batch_progress, compile_rule_batch,
)

pytest.importorskip('rich')

STAGES = tuple(CompileStage(f's{i}', f'stage {i}') for i in range(10))


class Terminal(io.StringIO):
    def isatty(self):
        return True


def layer(**options):
    stream = Terminal()
    clock = [100.0]
    progress = CompileProgress(enabled=True, renderer='rich', stream=stream,
                               stages=STAGES, heartbeat=None, clock=lambda: clock[0],
                               rss=lambda: 1024 ** 3, min_interval=0, **options)
    return progress, stream, clock


def frame_text(progress, width=100, height=30):
    from rich.console import Console
    output = io.StringIO()
    progress.bars.console.width = width
    progress.bars.console.height = height
    console = Console(file=output, width=width, height=height, color_system=None)
    console.print(progress.bars._frame())
    return output.getvalue()


def test_one_live_owns_logs_timing_and_completed_work_then_restores_streams():
    progress, output, clock = layer()
    original = sys.stdout, sys.stderr
    callback = lambda message: print('callback ' + message)
    with progress:
        assert progress.renderer == 'rich'
        assert progress.progress_callback(callback) is callback
        assert progress.bars.live.is_started
        assert not progress.bars.progress.live.is_started
        assert progress.bars.live._refresh_thread is None
        progress.log('literal [red]compiler[/red] message')
        callback('output')
        print('ordinary diagnostic', file=sys.stderr)
        for index in progress.iterate([1, 2, 3], 'nodes'):
            assert progress._counts['nodes']['n'] == index - 1
        clock[0] += 3
        progress.stage('s1')
        progress.count('fold', 7, label='fold rounds')
        progress.count('functions', 2, total=4, label='functions')
        current = frame_text(progress)
        assert 'Compiler timings' in current and 'Wall(s)' in current
        assert 'stage 0' in current and 'done' in current and '3.00' in current
        assert 'stage 1' in current and 'running' in current
        assert '7 completed' in current and '2/4 (50%)' in current
        assert '100%' not in current and 'remaining' not in current
    assert (sys.stdout, sys.stderr) == original
    assert active_compile_progress() is None
    assert progress.bars.live is None
    assert progress.summary()['work']['s0']['nodes']['done'] == 3
    text = output.getvalue()
    assert '[red]compiler[/red]' in text
    assert 'callback output' in text and 'ordinary diagnostic' in text
    assert 'compile done (2 timed stages)' in text


def test_rule_batch_clock_is_the_top_line_and_bars_remain_at_the_bottom():
    progress, _, clock = layer()
    with compile_rule_batch(4, clock=lambda: clock[0]) as batch:
        assert active_rule_batch_progress() is batch
        with batch.rule('first'):
            clock[0] += 10
        with batch.rule('second'):
            with progress:
                clock[0] += 5
                for index in range(6):
                    progress.count(f'work{index}', index + 1, total=10,
                                   label=f'work {index}')
                current = frame_text(progress, width=100, height=18)
                lines = current.splitlines()
                assert lines[0].startswith(
                    'Rules 1/4 | wall 15s | average 10s | projected total 40s')
                assert len(lines) <= 18
                assert current.index('Compiler timings') < current.index('6/10 (60%)')
                assert '+2 other active counters' in current
    assert active_rule_batch_progress() is None


def test_one_live_renders_concurrent_compilers_as_columns_and_keeps_failure():
    progress, _, _ = layer()
    failed_stages = [
        {"label": f"stage {index}", "state": "done" if index < 4 else
         "failed" if index == 4 else "pending"}
        for index in range(10)
    ]
    failed_stages[4]["label"] = "deployment select"
    failed_stages[8]["label"] = "SSA lowering"
    working_stages = [
        {"label": failed_stages[index]["label"],
         "state": "done" if index < 8 else
         "running" if index == 8 else "pending"}
        for index in range(10)
    ]
    working_stages[8]["label"] = "SSA lowering"
    with progress:
        progress.worker(
            'failed', label='failed_unit', status='failed',
            stage_number=5, stage_count=10, stage_label='deployment select',
            elapsed_seconds=12.5, rss_bytes=2 * 1024 ** 3,
            stages=failed_stages,
        )
        progress.worker(
            'working', label='working_unit', status='running',
            stage_number=9, stage_count=10, stage_label='SSA lowering',
            substage='lower-functions', elapsed_seconds=20,
            rss_bytes=3 * 1024 ** 3,
            stages=working_stages,
        )
        current = frame_text(progress, width=160)
        assert 'Compiler workers' in current
        assert 'Compiler stage' in current
        assert 'failed_unit' in current and 'deployment select' in current
        assert 'working_unit' in current and 'SSA lowering' in current
        assert 'failed' in current and 'running' in current
        assert 'compile 1/10' not in current


@pytest.mark.parametrize('width,height', [(100, 30), (80, 14), (40, 10), (20, 6)])
def test_resize_keeps_active_stage_visible_without_terminal_overflow(width, height):
    progress, _, _ = layer()
    with progress:
        for index in range(1, 9):
            progress.stage(f's{index}')
        for index in range(8):
            progress.count(f'work{index}', index, total=10)
        current = frame_text(progress, width, height)
        assert len(current.splitlines()) <= height
        assert 'stage 8' in current
        if width >= 55:
            assert 'Wall(s)' in current
        assert len(progress.summary()['stages']) == 10
        progress.bars.heartbeat(force=True)


def test_keyboard_interrupt_restores_stdout_stderr_and_keeps_failed_profile():
    progress, output, _ = layer()
    original = sys.stdout, sys.stderr
    with pytest.raises(KeyboardInterrupt):
        with progress:
            progress.count('nodes', 1, total=3)
            raise KeyboardInterrupt()
    assert (sys.stdout, sys.stderr) == original
    assert progress.summary()['stages'][0]['state'] == 'failed'
    assert progress.summary()['work']['s0']['nodes']['done'] == 1
    assert 'compile FAILED' in output.getvalue()


def test_render_failure_degrades_to_plain_logs_not_failed_compile(monkeypatch):
    progress, output, clock = layer()
    original = sys.stdout, sys.stderr
    with progress:
        monkeypatch.setattr(progress.bars.live, 'update',
                            lambda *a, **kw: (_ for _ in ()).throw(OSError('draw failure')))
        clock[0] += 1
        progress.count('nodes', 1, total=2)
        progress.log('still compiling')
        progress.stage('s1')
    assert (sys.stdout, sys.stderr) == original
    assert progress.summary()['stages'][0]['state'] == 'done'
    assert 'live display disabled' in output.getvalue()
    assert 'still compiling' in output.getvalue()


def test_heartbeat_drawing_fault_is_reported_once_and_releases_terminal(monkeypatch):
    progress, output, clock = layer()
    original = sys.stdout, sys.stderr
    with progress:
        monkeypatch.setattr(progress.bars.live, 'update',
                            lambda *a, **kw: (_ for _ in ()).throw(OSError('heartbeat draw')))
        clock[0] += 1
        progress.bars.heartbeat()
        progress.bars.heartbeat()
        assert (sys.stdout, sys.stderr) == original
        progress.log('plain after heartbeat')
    assert output.getvalue().count('live display disabled') == 1
    assert 'plain after heartbeat' in output.getvalue()


def test_auto_falls_back_when_rich_is_absent_and_headless_imports_no_renderer(monkeypatch):
    original_import = builtins.__import__
    imported = []

    def without_rich(name, *args, **kwargs):
        imported.append(name)
        if name.startswith('rich'):
            raise ImportError('optional Rich absent')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_rich)
    quiet = CompileProgress(enabled=False, collect_profile=True, renderer='rich')
    assert quiet.bars is None
    assert not any(n.startswith(('rich', 'tqdm')) for n in imported)
    progress = CompileProgress(enabled=True, renderer='auto', heartbeat=None, stream=io.StringIO())
    assert progress.renderer == 'tqdm'
    with progress:
        progress.log('fallback')


def test_environment_renderer_selection_and_invalid_choice(monkeypatch):
    monkeypatch.setenv('TURING_COMPILE_RENDERER', 'tqdm')
    progress = CompileProgress(enabled=True, heartbeat=None, stream=io.StringIO())
    assert progress.renderer == 'tqdm'
    with pytest.raises(ValueError, match='auto, rich or tqdm'):
        CompileProgress(enabled=False, renderer='garbage')


def test_nested_public_progress_scope_reuses_the_live_display():
    from src.compiler.shell_telemetry import compile_progress
    progress, _, _ = layer()
    with progress:
        live = progress.bars.live
        nested = compile_progress(True, renderer='tqdm')
        assert nested is progress
        with nested:
            assert progress.bars.live is live
        assert live.is_started
    assert not live.is_started


def test_vt_terminal_uses_one_cursor_region_and_restores_cursor(monkeypatch):
    # The command harness sets TERM=dumb. Exercise the VT-capable terminal
    # explicitly rather than mistaking Rich's plain-text fallback for redraws.
    monkeypatch.setenv('TERM', 'xterm-256color')
    progress, output, clock = layer()
    progress.bars.console.legacy_windows = False
    with progress:
        progress.log('scrolling diagnostic')
        progress.count('work', 1, total=3)
        clock[0] += 1
        progress.bars.console.width = 30
        progress.bars.console.height = 10
        progress.bars.heartbeat()
        progress.bars.console.width = 100
        progress.bars.console.height = 30
        clock[0] += 1
        progress.bars.heartbeat()
    text = output.getvalue()
    assert '\x1b[?25l' in text and '\x1b[?25h' in text
    assert '\x1b[2K' in text  # one compositor clearing and redrawing its frame
    assert 'scrolling diagnostic' in text


def test_public_source_entry_accepts_renderer_and_retains_same_work_contract():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    module, _, _ = lower_ast_source_to_ssa('def root():\n    return 2 + 3\n', 'root',
        extraction_contract='extraction_contracts/program_extraction.yaml',
        progress_bars=True, progress_renderer='rich', progress=lambda _: None)
    summary = module.metadata['compile_profile']
    assert all(row['state'] == 'done' for row in summary['stages'])
    work = summary['work']['ssa-lowering']['functions']
    assert work['done'] == work['total'] > 0
