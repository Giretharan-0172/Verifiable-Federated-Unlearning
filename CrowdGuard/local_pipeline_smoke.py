#!/usr/bin/env python
"""End-to-end local smoke test without requiring OpenFL.

It exercises the same project components as the OpenFL workflow on a tiny
synthetic CIFAR-shaped dataset: multi-client training, adaptive attack,
CrowdGuard HLBIM/stacked clustering, detection-only FedAvg, historical update
capture, and final-M handoff.
"""

import copy
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from adaptive_backdoor import adaptive_train_and_scale, sample_trigger
from CrowdGuardClientValidation import CrowdGuardClientValidation
from experiment_config import smoke_config
from lightweight_resnet18 import LightweightResNet18


def model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def model_delta(model, global_state):
    return {
        k: (v.detach().cpu() - global_state[k]).clone()
        for k, v in model.state_dict().items()
        if torch.is_floating_point(v)
    }


def fedavg(models):
    result = copy.deepcopy(models[0])
    state = result.state_dict()
    states = [m.state_dict() for m in models]
    for name in state:
        if torch.is_floating_point(state[name]):
            state[name] = torch.stack([s[name].cpu() for s in states]).mean(0)
        else:
            state[name] = states[0][name].cpu().clone()
    result.load_state_dict(state)
    return result


def stacked_vote(votes):
    n = len(votes)
    if n < 3:
        return votes[0]
    ac = AgglomerativeClustering(
        n_clusters=2, metric="euclidean", linkage="single",
        compute_full_tree=True, compute_distances=True,
    ).fit(votes)
    biggest = max(set(ac.labels_), key=lambda label: sum(ac.labels_ == label))
    indices = [i for i, label in enumerate(ac.labels_) if label == biggest]
    db_input = [votes[i] for i in indices]
    db = DBSCAN(eps=0.5, min_samples=1).fit(db_input)
    largest = max(set(db.labels_), key=lambda label: sum(db.labels_ == label))
    first = next(i for i, label in enumerate(db.labels_) if label == largest)
    return db_input[first]


def make_data(num_clients, samples_per_client, seed):
    g = torch.Generator().manual_seed(seed)
    total = num_clients * samples_per_client
    x = torch.randn(total, 3, 32, 32, generator=g)
    y = torch.arange(total) % 10
    return {
        i: DataLoader(
            TensorDataset(
                x[i * samples_per_client:(i + 1) * samples_per_client],
                y[i * samples_per_client:(i + 1) * samples_per_client],
            ),
            batch_size=32,
            shuffle=False,
        )
        for i in range(num_clients)
    }


def main():
    cfg = smoke_config()
    # Make the smoke test explicitly exercise 4 logical clients for two rounds.
    cfg.rounds = 2
    cfg.num_clients = 4
    cfg.samples_per_client = 64
    cfg.pmr = 0.25
    cfg.attack_start_round = 1

    torch.manual_seed(cfg.seed)
    loaders = make_data(cfg.num_clients, cfg.samples_per_client, cfg.seed)
    global_model = LightweightResNet18()
    trigger = sample_trigger(cfg.trigger_size, seed=cfg.seed)
    malicious = {cfg.num_clients - 1}
    history = []

    for round_num in range(cfg.rounds):
        global_state = model_state(global_model)
        local_models = []
        active_attack = {}

        for client in range(cfg.num_clients):
            local = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
            is_attack = client in malicious and round_num >= cfg.attack_start_round
            active_attack[client] = is_attack

            if is_attack:
                adaptive_train_and_scale(
                    local, global_state, loaders[client], optimizer, "cpu", trigger,
                    poison_rate=cfg.poison_rate,
                    alpha=cfg.alpha,
                    local_epochs=cfg.local_epochs,
                    scale_factor=cfg.num_clients / cfg.num_malicious_clients,
                )
            else:
                criterion = torch.nn.CrossEntropyLoss()
                local.train()
                for _ in range(cfg.local_epochs):
                    for data, labels in loaders[client]:
                        optimizer.zero_grad()
                        loss = criterion(local(data), labels)
                        loss.backward()
                        optimizer.step()
            local_models.append(local.cpu())

        votes = []
        for validator in range(cfg.num_clients):
            detected = CrowdGuardClientValidation.validate_models(
                global_model, local_models, validator, loaders[validator], "cpu"
            )
            row = [1] * cfg.num_clients
            for index in detected:
                row[index] = 0
            row[validator] = 1
            votes.append(row)

        final_vote = stacked_vote(votes)
        detected_clients = [i for i, vote in enumerate(final_vote) if vote == 0]

        # Detection-only: aggregation deliberately uses ALL local models.
        aggregated = fedavg(local_models)
        history.append({
            "round": round_num,
            "updates": {str(i): model_delta(local_models[i], global_state)
                         for i in range(cfg.num_clients)},
            "votes": {str(i): votes[i] for i in range(cfg.num_clients)},
            "detected_clients": detected_clients,
            "attack_active": active_attack,
            "global_model_before": global_state,
            "global_model_after": model_state(aggregated),
        })
        global_model = aggregated
        print(f"round {round_num}: detected={detected_clients}, attack_active={active_attack}")

    final_M = sorted({i for record in history for i in record["detected_clients"]})
    handoff = {
        "theta_0": history[0]["global_model_before"],
        "theta_k": history[-1]["global_model_after"],
        "round_history": history,
        "final_M": final_M,
        "ground_truth_malicious_clients": sorted(malicious),
        "config": cfg.to_dict(),
    }
    output_dir = "smoke_outputs"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "crowdguard_handoff.pt")
    torch.save(handoff, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert len(loaded["round_history"]) == cfg.rounds
    assert all(len(r["updates"]) == cfg.num_clients for r in loaded["round_history"])
    assert all(len(r["votes"]) == cfg.num_clients for r in loaded["round_history"])
    assert "theta_0" in loaded and "theta_k" in loaded
    print(f"handoff written: {path}")
    print(f"final M (temporary union rule): {final_M}")
    print("END-TO-END LOCAL PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
