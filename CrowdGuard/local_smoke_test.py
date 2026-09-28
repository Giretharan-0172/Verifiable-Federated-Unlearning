#!/usr/bin/env python
"""CPU smoke test for the CrowdGuard project implementation.

This intentionally avoids OpenFL so it can validate the model, trigger,
constrain-and-scale attack, historical update recording, and CrowdGuard HLBIM
implementation on a normal local Python environment.  The full OpenFL
workflow remains in cifar10_crowdguard.py.
"""

import copy
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_backdoor import adaptive_train_and_scale, sample_trigger
from CrowdGuardClientValidation import CrowdGuardClientValidation
from experiment_config import smoke_config
from lightweight_resnet18 import LightweightResNet18, count_parameters


def make_synthetic_client_data(num_clients, samples_per_client, seed=10):
    generator = torch.Generator().manual_seed(seed)
    total = num_clients * samples_per_client
    x = torch.randn(total, 3, 32, 32, generator=generator)
    y = torch.arange(total) % 10
    loaders = {}
    for client in range(num_clients):
        start = client * samples_per_client
        end = start + samples_per_client
        loaders[client] = DataLoader(
            TensorDataset(x[start:end], y[start:end]),
            batch_size=32,
            shuffle=False,
        )
    return loaders


def main():
    config = smoke_config()
    torch.manual_seed(config.seed)
    device = torch.device("cpu")

    model = LightweightResNet18()
    print(f"[1/6] Model parameters: {count_parameters(model):,}")
    assert 200_000 <= count_parameters(model) <= 350_000

    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    states = model.predict_internal_states(x)
    assert output.shape == (4, 10)
    assert len(states) > 0
    print(f"[2/6] Forward + internal states: OK ({len(states)} states)")

    trigger = sample_trigger(6, seed=config.seed)
    assert 0 <= trigger.top <= 26
    assert 0 <= trigger.left <= 26
    print(f"[3/6] Random trigger: OK at ({trigger.top}, {trigger.left}), "
          f"target={trigger.target_label}")

    loaders = make_synthetic_client_data(
        config.num_clients, config.samples_per_client, config.seed
    )
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Train two benign models and two malicious models from the same global state.
    models = []
    for client in range(config.num_clients):
        local = copy.deepcopy(model)
        optimizer = torch.optim.SGD(local.parameters(), lr=0.01, momentum=0.9)
        if client == config.num_clients - 1:
            stats = adaptive_train_and_scale(
                local,
                global_state,
                loaders[client],
                optimizer,
                device,
                trigger,
                poison_rate=config.poison_rate,
                alpha=config.alpha,
                local_epochs=1,
                scale_factor=config.num_clients / config.num_malicious_clients,
            )
            assert stats["scale_factor"] > 1.0
        else:
            criterion = torch.nn.CrossEntropyLoss()
            local.train()
            for data, labels in loaders[client]:
                optimizer.zero_grad()
                loss = criterion(local(data), labels)
                loss.backward()
                optimizer.step()
        models.append(local.cpu())
    print("[4/6] Benign + adaptive malicious local updates: OK")

    # Run the actual CrowdGuard validator on the tiny synthetic clients.
    # Four clients and 64 samples/client are enough to exercise the code path.
    validation_loader = loaders[0]
    detected = CrowdGuardClientValidation.validate_models(
        model.cpu(), models, 0, validation_loader, device
    )
    print(f"[5/6] CrowdGuard HLBIM + pruning path: OK; detected={sorted(detected)}")

    # Verify historical updates can be constructed without filtering.
    updates = {}
    for client, local in enumerate(models):
        updates[client] = {
            name: value.detach().clone() - global_state[name]
            for name, value in local.state_dict().items()
            if torch.is_floating_point(value)
        }
    assert len(updates) == config.num_clients
    assert all(len(update) > 0 for update in updates.values())
    print("[6/6] Historical update capture + detection-only handoff: OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
