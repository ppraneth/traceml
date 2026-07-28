import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 8192
BATCH_SIZE = 64
EPOCHS = 2


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def prepare_data():
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    # No DistributedSampler here. accelerator.prepare(...) rebuilds the
    # loader with a per-process batch sampler, unlike the DDP example.
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def main() -> None:
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError:
        print(
            "This example requires Hugging Face Accelerate. Install it "
            "with:\n"
            '  pip install "traceml-ai[accelerate]"\n'
            "or see https://huggingface.co/docs/accelerate."
        )
        sys.exit(0)

    # Accelerator() reads the torchrun environment that `traceml run` sets
    # up, so the same script works on CPU, one GPU, and several GPUs.
    accelerator = Accelerator()
    set_seed(SEED)

    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = prepare_data()

    # prepare() moves the model to the right device, wraps it for the
    # distributed setup, and returns a per-process dataloader whose
    # batches already sit on accelerator.device.
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Enable TraceML auto instrumentation (dataloader, forward, backward,
    # optimizer, and H2D timing) for this process.
    traceml.init(mode="auto")

    criterion = nn.CrossEntropyLoss()

    model.train()
    global_step = 0

    for epoch in range(EPOCHS):
        running_loss = torch.zeros((), device=accelerator.device)

        for batch_x, batch_y in train_loader:
            # Wrap the prepared model, not the TinyMLP you built. On
            # multi-GPU prepare() returns a DDP wrapper, and TraceML reads
            # the training strategy off it.
            with traceml.trace_step(model):
                optimizer.zero_grad(set_to_none=True)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                accelerator.backward(loss)
                optimizer.step()

            # Keep loss logging OUTSIDE trace_step so it never inflates the
            # measured step, and accumulate on-device so there is no
            # per-step device-to-host sync (only when we actually print).
            running_loss += loss.detach()
            global_step += 1

            if accelerator.is_main_process and global_step % 50 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"loss: {float(running_loss) / 50:.4f}"
                )
                running_loss.zero_()

    if accelerator.is_main_process:
        print("Done.")

    # Accelerator() created the process group, so let it tear the group
    # down. This matches destroy_process_group() in the DDP example.
    accelerator.end_training()


if __name__ == "__main__":
    main()
