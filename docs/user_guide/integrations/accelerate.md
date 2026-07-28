# Hugging Face Accelerate

Use TraceML with a Hugging Face Accelerate training loop to find training
bottlenecks without changing how your job runs.

Accelerate leaves the training loop to you: you call `model(...)`,
`accelerator.backward(loss)`, and `optimizer.step()` yourself. So the
integration is the same recipe used for plain PyTorch, DDP, and DeepSpeed:
call `traceml.init(mode="auto")` once, then wrap each step with
`traceml.trace_step(...)`. There is no Accelerate-specific callback or wrapper
to install.

If you use `transformers.Trainer` instead of your own loop, use
[Hugging Face Trainer](huggingface.md), which has a callback.

## 1. Install

TraceML does not depend on Accelerate. Install Accelerate yourself:

```bash
pip install "traceml-ai[accelerate]"
```

or follow the [Accelerate installation guide](https://huggingface.co/docs/accelerate/basic_tutorials/install).
`traceml-ai[hf]` already includes Accelerate, so you do not need both.

Accelerate runs on CPU or GPU, so the example below works on either.

## 2. Wrap The Step

Two lines are added to the loop from Accelerate's own
[migration guide](https://huggingface.co/docs/accelerate/basic_tutorials/migration):

```diff
  from accelerate import Accelerator
+ import traceml_ai as traceml

  accelerator = Accelerator()
  model, optimizer, training_dataloader, scheduler = accelerator.prepare(
      model, optimizer, training_dataloader, scheduler
  )
+ traceml.init(mode="auto")

  for batch in training_dataloader:
+     with traceml.trace_step(model):
          optimizer.zero_grad()
          inputs, targets = batch
          outputs = model(inputs)
          loss = loss_function(outputs, targets)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
```

Two details matter:

- **Wrap the prepared model.** `accelerator.prepare(...)` returns a model
  wrapped for your distributed setup, which is a `DistributedDataParallel`
  wrapper on multi-GPU. Pass that returned object to `trace_step(...)`, not
  the module you built. TraceML times the outermost forward either way, but
  the prepared model is also what lets TraceML detect that the run is DDP.
  Keep using `accelerator.unwrap_model(model)` for saving, as Accelerate
  documents.
- **Iterate the dataloader outside `trace_step`.** The fetch is recorded as
  input time and folded into the step that consumes the batch. Bracketing
  only the compute keeps input wait separate from step time.

`traceml.init(mode="auto")` installs TraceML's process-wide auto-timers, so it
records DataLoader fetch timing, host-to-device (H2D) copies, forward,
backward, optimizer, and step timing, plus GPU and process memory, and writes
an end-of-run summary. Backward is captured because `accelerator.backward(loss)`
reaches `torch.Tensor.backward()`; optimizer time is captured from the torch
optimizer that Accelerate's `AcceleratedOptimizer` wraps. You do not need to
add anything else per step.

Call `accelerator.end_training()` at the end of the script. It tears down the
process group that `Accelerator()` created, the same way the DDP example calls
`destroy_process_group()`.

## 3. Launch The Run

`traceml run` launches your script through torchrun, and Accelerate supports
being launched that way. Its
[launch tutorial](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
says plainly that if you are used to launching scripts with `torchrun` you can
still do that, and that `accelerate launch` is not required. `Accelerator()`
reads the same `RANK` / `LOCAL_RANK` / `WORLD_SIZE` environment variables
torchrun sets.

Single GPU or CPU:

```bash
traceml run train.py --mode=summary
```

Single-node multi-GPU (e.g. 2 GPUs):

```bash
traceml run train.py --mode=summary --nproc-per-node=2
```

For multi-node launch commands, see
[Distributed Training](../distributed-training.md).

### `accelerate launch` is not the TraceML path yet

`traceml run` does not wrap `accelerate launch`. Use `traceml run` for now.
The practical consequence is that your `accelerate config` file and the
`accelerate launch` CLI flags are not applied, so set those options on the
`Accelerator` in your script instead:

```python
accelerator = Accelerator(mixed_precision="fp16")
```

Accelerate's launch tutorial shows a bare `MIXED_PRECISION=fp16` environment
variable next to its `torchrun` example. Current Accelerate reads
`ACCELERATE_MIXED_PRECISION`, so prefer setting mixed precision in the script.

Note that `traceml run` goes through torchrun even at `--nproc-per-node=1`.
Accelerate therefore sees a distributed launch, initializes a process group of
size one, and wraps your model in DDP. That is expected and needs no
configuration.

## Gradient Accumulation

The minimal example does not use gradient accumulation. If you add it, keep
`trace_step` around the `accumulate` block:

```python
accelerator = Accelerator(gradient_accumulation_steps=2)

for batch in training_dataloader:
    with traceml.trace_step(model):
        with accelerator.accumulate(model):
            ...
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
```

One TraceML step is then one micro-batch: forward and backward are timed every
step, but `optimizer.step()` only reaches the real optimizer on the steps where
`accelerator.sync_gradients` is `True`, so optimizer time appears on those
steps only. Accelerate also scales the loss inside `accelerator.backward(...)`
for you, and expects only one forward/backward inside `accumulate(...)`.

If you want one TraceML step to mean one optimizer step, pull the micro-batches
yourself inside a single `trace_step`, as
`examples/advanced/bert_gradient_accum.py` does.

## Limitations

- **H2D time is folded into input time.** A prepared dataloader puts each batch
  on the device before yielding it, which happens during the fetch and so
  outside your `trace_step`. TraceML reports that cost as input fetch time and
  shows no separate H2D time. Accelerate's automatic placement is also blocking
  by default. If you are chasing transfer cost specifically, either enable
  non-blocking transfers:

    ```python
    from accelerate import Accelerator, DataLoaderConfiguration

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )
    ```

    with `pin_memory=True` on your `DataLoader`, or opt out of device placement
    for the dataloader and move the batch yourself inside `trace_step`.

- **No explicit collective timing.** DDP's gradient all-reduce overlaps the
  backward pass, so its cost is folded into backward and step time rather than
  reported as a separate collective number.

- **Optimizer timing is "where available".** TraceML times the torch optimizer
  reached inside `AcceleratedOptimizer.step()`. Accelerate's own work around it
  (loss scaling, gradient clipping) is outside the traced phases and shows up
  as residual.

- **`accelerate launch` is not supported.** See the launch section above.

## Troubleshooting

### Multi-GPU run only shows one rank

Make sure you launched through TraceML with `--nproc-per-node`, not plain
`python` or `accelerate launch`:

```bash
traceml run train.py --nproc-per-node=2
```

### I want a baseline without TraceML

Run the same script with TraceML disabled:

```bash
traceml run train.py --disable-traceml
```

This launches your script natively through `torchrun` without TraceML
telemetry.

## Full Example

A runnable example lives in the repo:

- `examples/integrations/accelerate_minimal.py`

Run it with:

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary
```

or on two GPUs:

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary --nproc-per-node=2
```

The example exits cleanly when Accelerate is not installed.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [Hugging Face Trainer](huggingface.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
