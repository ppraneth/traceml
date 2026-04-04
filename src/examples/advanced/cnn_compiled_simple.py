import warnings

import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

# -------------------------
# Medium CNN for MNIST
# -------------------------


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# TraceML wrappers
# -------------------------


def forward_step(model, x):
    return model(x)


def backward_step(loss):
    loss.backward()


def optimizer_step(opt):
    opt.step()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)

    print(
        "Compiling model via torch.compile (this may take a bit on the first step)..."
    )
    compiled_model = torch.compile(model)

    # ⚠️ Notice the absence of `trace_model_instance()` here.
    # We are not deeply profiling individual layers, allowing `torch.compile`
    # to absorb the ENTIRE model into a single fast graph.

    opt = optim.Adam(compiled_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training with compiled model (NO deep instrumentation)...")
    for step, (xb, yb) in enumerate(loader):

        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad(set_to_none=True)

        out = forward_step(compiled_model, xb)
        loss = loss_fn(out, yb)

        backward_step(loss)
        optimizer_step(opt)

        if step % 50 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

        if step == 500:
            break

    print("\n--- torch.compile verification ---")
    stats = torch._dynamo.utils.compile_times()
    print(f"Dynamo Compile Times (Stringent checks): {stats}")
    print("\nNotice how there should only be 1 or 2 `unique_graphs` compiled ")
    print(
        "because Dynamo did not encounter any uncompilable python code on the layers."
    )


if __name__ == "__main__":
    main()
