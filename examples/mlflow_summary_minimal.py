# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Log the TraceML summary to MLflow. MLflow is optional (pip install mlflow).

traceml run examples/mlflow_summary_minimal.py --mode=summary
"""

from __future__ import annotations

import torch
from torch import nn

import traceml_ai as traceml


def log_to_mlflow(summary: dict, full: dict | None) -> None:
    """Log summary metrics/tags to MLflow and attach the full report."""
    try:
        import mlflow
    except ImportError:
        print("\nMLflow not installed; skipping (pip install mlflow).")
        return

    numeric = {
        k: v
        for k, v in summary.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    tags = {
        k.replace("/", "."): v
        for k, v in summary.items()
        if isinstance(v, str)
    }

    with mlflow.start_run():
        mlflow.log_metrics(numeric)
        mlflow.set_tags(tags)
        if full is not None:
            mlflow.log_dict(full, "traceml/final_summary.json")

    print(f"\nLogged {len(numeric)} metrics and {len(tags)} tags to MLflow.")


def main() -> None:
    traceml.init()

    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))

    for _ in range(128):
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    summary = traceml.summary(print_text=True)
    if summary is None:
        return

    full = traceml.final_summary()
    log_to_mlflow(summary, full)


if __name__ == "__main__":
    main()
