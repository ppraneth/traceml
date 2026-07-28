import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml
from traceml_ai.runtime.arming import _set_tracing_armed, is_tracing_armed

try:
    import accelerate  # noqa: F401

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

INPUT_DIM = 16
HIDDEN_DIM = 32
NUM_CLASSES = 4
BATCH_SIZE = 8

FORWARD = "_traceml_internal:forward_time"
BACKWARD = "_traceml_internal:backward_time"
OPTIMIZER = "_traceml_internal:optimizer_step"
STEP = "_traceml_internal:step_time"
DATALOADER = "_traceml_internal:dataloader_next"


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class _FakeAcceleratedOptimizer(torch.optim.Optimizer):
    # Mirrors accelerate's AcceleratedOptimizer: it subclasses Optimizer but
    # never calls Optimizer.__init__, so torch never wraps this class's
    # step() with the global step hooks. Only the inner optimizer fires
    # them, which is why one accelerate step yields one optimizer event.
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        self.optimizer.step(closure)

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class _FakeAccelerator:
    # Reproduces only the surface the documented TraceML recipe touches.
    def prepare(self, *objects):
        prepared = []
        for obj in objects:
            if isinstance(obj, torch.optim.Optimizer):
                prepared.append(_FakeAcceleratedOptimizer(obj))
            else:
                prepared.append(obj)
        return tuple(prepared)

    def backward(self, loss):
        # Real Accelerate scales the loss and then reaches
        # torch.Tensor.backward, which is the call TraceML times.
        loss.backward()


def _drain_step_time_queue() -> list:
    """Drain all StepTimeBatch entries from the shared queue."""
    from traceml_ai.utils.timing import get_step_time_queue

    queue = get_step_time_queue()
    batches = []
    while not queue.empty():
        batches.append(queue.get_nowait())
    return batches


def _reset_traceml_state() -> None:
    """Reset TraceML's step counter, recording state, and step-time queue."""
    from traceml_ai.runtime.state import (
        configure_trace_recording,
        reset_trace_session_state,
    )
    from traceml_ai.utils.timing import _STEP_BUFFER

    reset_trace_session_state()
    configure_trace_recording(max_steps=None)
    _drain_step_time_queue()
    _STEP_BUFFER.clear()


def _install_auto_instrumentation() -> None:
    from traceml_ai.instrumentation.hooks.optimizer_hooks import (
        ensure_optimizer_timing_installed,
    )
    from traceml_ai.instrumentation.patches.backward_auto_timer_patch import (
        patch_backward,
    )
    from traceml_ai.instrumentation.patches.dataloader_patch import (
        patch_dataloader,
    )
    from traceml_ai.instrumentation.patches.forward_auto_timer_patch import (
        patch_forward,
    )
    from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
        patch_h2d,
    )

    # The full set that init(mode="auto") installs.
    patch_forward()
    patch_backward()
    patch_h2d()
    patch_dataloader()
    ensure_optimizer_timing_installed()


@pytest.fixture(autouse=True)
def _armed_tracing():
    # The patches are gated on the process-wide flag that init() raises once
    # its patches install cleanly. These tests install the patches directly,
    # so arm the flag here and restore it, keeping the file runnable on its
    # own rather than depending on some earlier test having called init().
    previous = is_tracing_armed()
    _set_tracing_armed(True)
    yield
    _set_tracing_armed(previous)


def _count(batch, event_name: str) -> int:
    return sum(1 for evt in batch.events if evt.name == event_name)


def _batches_missing(batches: list, event_name: str) -> list:
    return [i for i, b in enumerate(batches) if _count(b, event_name) == 0]


def test_accelerate_recipe_brackets_step():
    from traceml_ai.runtime.state import get_trace_session_state

    _reset_traceml_state()
    _install_auto_instrumentation()

    accelerator = _FakeAccelerator()
    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    criterion = nn.CrossEntropyLoss()
    num_steps = 3

    step_before = get_trace_session_state().step

    for _ in range(num_steps):
        # This is exactly the recipe shown in the docs and example.
        with traceml.trace_step(model):
            x = torch.randn(BATCH_SIZE, INPUT_DIM)
            y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            accelerator.backward(loss)
            optimizer.step()

    step_after = get_trace_session_state().step
    assert step_after - step_before == num_steps, (
        "trace_step must advance the TraceML step counter once per "
        f"Accelerate step; advanced by {step_after - step_before}."
    )

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    assert not _batches_missing(
        batches, FORWARD
    ), "forward timing should be captured on the prepared model."
    assert not _batches_missing(batches, BACKWARD), (
        "backward timing should be captured because accelerator.backward "
        "reaches torch.Tensor.backward()."
    )
    assert not _batches_missing(batches, OPTIMIZER), (
        "optimizer timing should be captured because the accelerate "
        "optimizer wrapper delegates to the inner torch optimizer."
    )
    assert not _batches_missing(
        batches, STEP
    ), "step timing should be captured once per trace_step block."

    doubled = [i for i, b in enumerate(batches) if _count(b, OPTIMIZER) != 1]
    assert not doubled, (
        "Exactly one optimizer event per step is expected. The accelerate "
        "optimizer wrapper must not be double counted with the optimizer "
        f"it wraps; StepTimeBatch(es) {doubled} disagree."
    )


def test_accelerate_recipe_emits_dataloader_next_over_real_loader():
    # dataloader_next rides a class-level patch of DataLoader.__iter__, so it
    # only lands when a real torch DataLoader is iterated. The recipe fetches
    # OUTSIDE trace_step; the event is buffered and flushed into that step.
    _reset_traceml_state()
    _install_auto_instrumentation()

    num_steps = 3
    dataset = TensorDataset(
        torch.randn(num_steps * BATCH_SIZE, INPUT_DIM),
        torch.randint(0, NUM_CLASSES, (num_steps * BATCH_SIZE,)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    accelerator = _FakeAccelerator()
    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y in loader:
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            accelerator.backward(loss)
            optimizer.step()

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    dark = _batches_missing(batches, DATALOADER)
    assert not dark, (
        "dataloader_next stream is dark for the Accelerate recipe over a "
        f"real DataLoader: StepTimeBatch(es) {dark} of {num_steps} carry no "
        "event."
    )


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_real_accelerate_recipe_emits_every_stream():
    # The fakes above cannot catch an upstream Accelerate refactor. This runs
    # the documented recipe through a real Accelerator on CPU.
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    _reset_traceml_state()
    _install_auto_instrumentation()

    num_steps = 3
    dataset = TensorDataset(
        torch.randn(num_steps * BATCH_SIZE, INPUT_DIM),
        torch.randint(0, NUM_CLASSES, (num_steps * BATCH_SIZE,)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    try:
        accelerator = Accelerator(cpu=True)
        model = _TinyMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer, loader = accelerator.prepare(
            model, optimizer, loader
        )

        criterion = nn.CrossEntropyLoss()

        for batch_x, batch_y in loader:
            with traceml.trace_step(model):
                optimizer.zero_grad(set_to_none=True)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                accelerator.backward(loss)
                optimizer.step()
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    for name in (FORWARD, BACKWARD, OPTIMIZER, STEP, DATALOADER):
        dark = _batches_missing(batches, name)
        assert not dark, (
            f"{name} is dark through real Accelerate wrappers: "
            f"StepTimeBatch(es) {dark} of {num_steps} carry no event."
        )

    doubled = [i for i, b in enumerate(batches) if _count(b, OPTIMIZER) != 1]
    assert not doubled, (
        "AcceleratedOptimizer must not be double counted with the optimizer "
        f"it wraps; StepTimeBatch(es) {doubled} disagree."
    )


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_accelerate_public_api_available():
    """When Accelerate is installed, the symbols the example needs exist."""
    from accelerate import Accelerator

    assert hasattr(Accelerator, "prepare")
    assert hasattr(Accelerator, "backward")
    assert hasattr(Accelerator, "accumulate")
    assert hasattr(Accelerator, "unwrap_model")
    assert hasattr(Accelerator, "end_training")
