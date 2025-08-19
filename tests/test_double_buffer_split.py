import numpy as np
import pytest
from src.common.double_buffer import (
    AgentSpec, random_tensor,
    Tribuffer, LockGraph, DoubleBuffer,
    ThreadSafeBuffer, NumpyActionHistory,
    EnginePerfTracker, MultiAgentEngineSplitter,
)


def test_random_tensor_cpu():
    arr = random_tensor(np.float32, (2, 2), 'cpu')
    assert isinstance(arr, np.ndarray)


def test_agent_spec_init():
    spec = AgentSpec(0, 'numpy', 'cpu')
    assert spec.backend == 'numpy'


def test_lock_graph_init():
    lg = LockGraph()
    assert isinstance(lg.nodes, dict)
    lg.monitor_event.set(); lg.monitor_thread.join(timeout=0.1)

def test_tribuffer_basic():
    try:
        tb = Tribuffer(keys=['a'], shapes=[(2,)], type='float', depth='32')
    except (NotImplementedError, AttributeError):
        pytest.skip("torch or pin_memory not supported")
    assert 'a' in tb.data
    tb.sync_manager.shutdown()


def test_double_buffer_basic():
    db = DoubleBuffer(roll_length=4, num_agents=2)
    db.write_frame('frame0', agent_idx=0)
    assert db.read_frame(agent_idx=1) is None


def test_double_buffer_custom_frames():
    frames = np.zeros(4)
    db = DoubleBuffer(num_agents=1, frames=frames)
    db.write_frame(1.0, agent_idx=0)
    assert db.read_frame(agent_idx=0) == pytest.approx(1.0)
    # Second read should yield nothing even though frames container retains value
    assert db.read_frame(agent_idx=0) is None


def test_thread_safe_buffer_init():
    spec = AgentSpec(0, 'numpy', 'cpu')
    try:
        buf = ThreadSafeBuffer(shape=(2, 2), dtype='float', agent_specs=[spec], manager=None)
    except (NotImplementedError, AttributeError):
        pytest.skip("torch or pin_memory not supported")
    assert buf.shape[0] == 2
    buf.manager.shutdown()


def test_numpy_action_history():
    hist = NumpyActionHistory(num_agents=1, num_pages=1, num_keys=1)
    assert hist.actions.shape[1] == 1


def test_engine_perf_tracker():
    perf = EnginePerfTracker()
    perf.record('numpy', 10, 0.1)
    assert perf.avg_time('numpy') == 0.1


def test_multi_agent_engine_splitter():
    perf = EnginePerfTracker()
    splitter = MultiAgentEngineSplitter(perf_tracker=perf)
    engine = splitter.choose_engine(batch_size=10, age=5)
    assert engine in ('numpy', 'torch')
