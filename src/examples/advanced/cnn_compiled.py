import warnings

import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_step

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

    # ⚠️ Deep instrumentation
    # Applying TraceML to the compiled Module
    trace_model_instance(
        compiled_model,
        trace_layer_forward_memory=True,
        trace_layer_backward_memory=True,
        trace_layer_forward_time=True,
        trace_layer_backward_time=True,
    )

    opt = optim.Adam(compiled_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training with compiled model...")
    for step, (xb, yb) in enumerate(loader):

        with trace_step(compiled_model):
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
    # Dynamo keeps track of how many graphs it successfully compiled
    # If this is completely empty or 0, Compilation failed completely.
    # High cache misses or many small compilations indicate graph breaks!
    stats = torch._dynamo.utils.compile_times()
    print(f"Dynamo Compile Times (Stringent checks): {stats}")
    print(
        "\nTip: To see exact reasons for graph breaks, run this script with:"
    )
    print('TORCH_LOGS="graph_breaks" python cnn_compiled.py')


if __name__ == "__main__":
    main()
