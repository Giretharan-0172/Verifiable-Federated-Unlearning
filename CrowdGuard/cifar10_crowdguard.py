#!/usr/bin/env python
# coding: utf-8

"""CrowdGuard experiment runner for the verifiable federated-unlearning project.

The repository's original CrowdGuard HLBIM + stacked-clustering detector is
retained.  The project modification is detection-only operation: detected
clients are recorded but are NOT filtered from equal-weight FedAvg, because the
next Wu stage requires their historical submitted updates.
"""

import argparse
import copy
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from CrowdGuardClientValidation import CrowdGuardClientValidation
from adaptive_backdoor import adaptive_train_and_scale, sample_trigger, apply_trigger
from experiment_config import ExperimentConfig
from lightweight_resnet18 import LightweightResNet18, count_parameters

from openfl.experimental.workflow.interface import Aggregator, Collaborator, FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

warnings.filterwarnings("ignore")

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD_DEV = torch.tensor([0.2023, 0.1994, 0.2010])
VOTE_FOR_BENIGN = 1
VOTE_FOR_POISONED = 0
LOG_INTERVAL = 10


# ---------------------------------------------------------------------------
# Reproducibility / model utilities
# ---------------------------------------------------------------------------

def seed_random_generators(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def default_optimizer(model, learning_rate, momentum=0.9, optimizer_type="SGD"):
    if optimizer_type.upper() == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if optimizer_type.upper() == "ADAM":
        return optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def model_state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_delta(local_model, global_state):
    delta = {}
    local_state = local_model.state_dict()
    for name, value in local_state.items():
        reference = global_state[name]
        if torch.is_floating_point(value):
            delta[name] = (value.detach().cpu() - reference.detach().cpu()).clone()
        else:
            # Integer buffers such as num_batches_tracked are not part of the
            # mathematical FedAvg update.  Retain them separately as metadata.
            delta[name] = torch.zeros_like(reference.detach().cpu())
    return delta


def apply_delta_to_state(global_state, delta):
    result = {}
    for name, value in global_state.items():
        if torch.is_floating_point(value):
            result[name] = value + delta[name]
        else:
            result[name] = value.clone()
    return result


def fed_avg(models):
    if not models:
        raise ValueError("FedAvg received no models")
    result = copy.deepcopy(models[0])
    state = result.state_dict()
    state_dicts = [model.state_dict() for model in models]
    for name in state:
        if torch.is_floating_point(state[name]):
            stacked = torch.stack([sd[name].detach().cpu() for sd in state_dicts])
            state[name] = stacked.mean(dim=0)
        else:
            state[name] = state_dicts[0][name].detach().cpu().clone()
    result.load_state_dict(state)
    return result


def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * target.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    model.cpu()
    return loss_sum / max(1, total), correct / max(1, total)


# ---------------------------------------------------------------------------
# CrowdGuard stacked clustering helper functions
# ---------------------------------------------------------------------------

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    assert len(clustering_labels) == expected_number_of_labels
    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        clusters.setdefault(cluster, []).append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}


def determine_biggest_cluster(clustering):
    if not clustering:
        raise ValueError("Cannot choose a cluster from an empty clustering")
    return max(clustering, key=lambda cluster_id: len(clustering[cluster_id]))


def select_final_m(round_history):
    """Project-default temporary rule: union of per-round detections.

    The final-M rule is deliberately isolated so the team can replace it later
    without changing the detector or historical-update recording.
    """
    malicious = set()
    for round_record in round_history:
        malicious.update(round_record["detected_clients"])
    return sorted(malicious)


# ---------------------------------------------------------------------------
# OpenFL workflow
# ---------------------------------------------------------------------------

class FederatedFlow(FLSpec):
    def __init__(self, model, optimizer_templates, config, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.theta_0 = model_state_cpu(model)
        self.optimizer_templates = optimizer_templates
        self.config = config
        self.device = device
        self.round_num = 0
        self.round_history = []
        self.all_models = {}
        self.all_votes_by_name = {}
        self.start_time = None
        self.trigger = sample_trigger(
            config.trigger_size,
            target_label=config.target_label,
            seed=config.seed,
        )
        self.malicious_client_names = {
            f"malicious_{i:02d}" for i in range(config.num_malicious_clients)
        }

    @aggregator
    def start(self):
        self.start_time = time.time()
        self.collaborators = self.runtime.collaborators
        self.private = 10
        print("#" * 60)
        print("CrowdGuard project run")
        print(f"Clients: {self.config.num_clients}")
        print(f"Malicious clients: {sorted(self.malicious_client_names)}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Trigger: {self.trigger.size}x{self.trigger.size}, "
              f"position=({self.trigger.top},{self.trigger.left}), "
              f"target={self.trigger.target_label}")
        print("CrowdGuard mode: DETECTION ONLY (all models aggregated)")
        print("#" * 60)
        self.next(self.train, foreach="collaborators", exclude=["private"])

    @collaborator
    def train(self):
        self.collaborator_name = self.input
        global_state = model_state_cpu(self.global_model)
        self.model.load_state_dict(global_state)
        self.model.to(self.device)

        optimizer = default_optimizer(
            self.model,
            self.config.learning_rate,
            self.config.momentum,
            self.optimizer_templates[self.input],
        )

        is_malicious = self.collaborator_name in self.malicious_client_names
        attack_active = is_malicious and self.round_num >= self.config.attack_start_round

        if attack_active:
            scale_factor = self.config.num_clients / max(1, self.config.num_malicious_clients)
            attack_stats = adaptive_train_and_scale(
                self.model,
                global_state,
                self.train_loader,
                optimizer,
                self.device,
                self.trigger,
                poison_rate=self.config.poison_rate,
                alpha=self.config.alpha,
                local_epochs=self.config.local_epochs,
                scale_factor=scale_factor,
            )
            self.attack_stats = attack_stats
        else:
            criterion = nn.CrossEntropyLoss()
            losses = []
            self.model.train()
            for _ in range(self.config.local_epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % LOG_INTERVAL == 0:
                        losses.append(float(loss.detach().cpu()))
            self.attack_stats = {"mean_loss": float(np.mean(losses)) if losses else 0.0,
                                 "scale_factor": 1.0}

        self.model.cpu()
        self.training_completed = True
        self.attack_active = attack_active
        self.next(self.collect_models, exclude=["training_completed"])

    @aggregator
    def collect_models(self, inputs):
        # This is the project handoff boundary: retain every submitted model
        # update before any CrowdGuard detection result is used.
        self.all_models = {
            item.collaborator_name: item.model.cpu() for item in inputs
        }
        global_state = model_state_cpu(self.global_model)

        updates = {
            name: model_delta(model, global_state)
            for name, model in self.all_models.items()
        }
        attack_status = {
            item.collaborator_name: bool(item.attack_active) for item in inputs
        }

        self.current_round_record = {
            "round": self.round_num,
            "global_model_before": global_state,
            "updates": updates,
            "attack_active": attack_status,
        }
        self.next(self.local_validation, foreach="collaborators")

    @collaborator
    def local_validation(self):
        self.collaborator_name = self.input
        all_names = sorted(self.all_models.keys())
        all_models = [self.all_models[name] for name in all_names]
        own_client_index = all_names.index(self.collaborator_name)

        detected = CrowdGuardClientValidation.validate_models(
            self.global_model,
            all_models,
            own_client_index,
            self.train_loader,
            self.device,
        )
        detected = sorted(detected)
        print(f"Round {self.round_num}: {self.collaborator_name} detected indices {detected}")

        votes = []
        for index in range(len(all_models)):
            if index == own_client_index:
                votes.append(VOTE_FOR_BENIGN)
            elif index in detected:
                votes.append(VOTE_FOR_POISONED)
            else:
                votes.append(VOTE_FOR_BENIGN)

        self.votes_of_this_client = dict(zip(all_names, votes))
        self.next(self.defend)

    @aggregator
    def defend(self, inputs):
        all_names = sorted(self.all_models.keys())
        all_votes_by_name = {
            item.collaborator_name: item.votes_of_this_client for item in inputs
        }
        binary_votes = [
            [all_votes_by_name[validator][candidate] for candidate in all_names]
            for validator in all_names
        ]

        # Original CrowdGuard stacked-clustering aggregation.
        if len(all_names) >= 3:
            ac = AgglomerativeClustering(
                n_clusters=2,
                distance_threshold=None,
                compute_full_tree=True,
                metric="euclidean",
                linkage="single",
                compute_distances=True,
            ).fit(binary_votes)
            ac_labels = ac.labels_.tolist()
            agglomerative_result = create_cluster_map_from_labels(len(all_names), ac_labels)
            biggest = agglomerative_result[determine_biggest_cluster(agglomerative_result)]
            db_input = [binary_votes[index] for index in biggest]
            db = DBSCAN(eps=0.5, min_samples=1).fit(db_input)
            db_clusters = create_cluster_map_from_labels(len(biggest), db.labels_.tolist())
            largest_db = db_clusters[determine_biggest_cluster(db_clusters)]
            final_voting = db_input[int(largest_db[0])]
        else:
            # Small smoke-test fallback; the real experiment uses 20 clients.
            final_voting = [
                1 if sum(row[index] for row in binary_votes) >= len(binary_votes) / 2
                else 0
                for index in range(len(all_names))
            ]

        detected_names = [
            name for name, vote in zip(all_names, final_voting)
            if vote == VOTE_FOR_POISONED
        ]
        detected_names = sorted(set(detected_names))

        # Project modification: DO NOT filter detected models.  Equal-weight
        # FedAvg receives every participating client update.
        aggregated_model = fed_avg([self.all_models[name] for name in all_names])
        self.model = aggregated_model
        self.global_model = copy.deepcopy(aggregated_model)

        self.current_round_record["votes"] = all_votes_by_name
        self.current_round_record["detected_clients"] = detected_names
        self.current_round_record["global_model_after"] = model_state_cpu(aggregated_model)
        self.round_history.append(self.current_round_record)

        self.round_num += 1
        if self.round_num < self.config.rounds:
            self.next(self.train, foreach="collaborators")
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        final_m = select_final_m(self.round_history)
        elapsed = time.time() - self.start_time
        handoff = {
            "theta_0": self.theta_0,
            "theta_k": model_state_cpu(self.model),
            "round_history": self.round_history,
            "final_M": final_m,
            "ground_truth_malicious_clients": sorted(self.malicious_client_names),
            "config": self.config.to_dict(),
            "trigger": self.trigger.to_dict(),
            "elapsed_seconds": elapsed,
            "model_parameter_count": count_parameters(self.model),
        }

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, "crowdguard_handoff.pt")
        torch.save(handoff, output_path)

        metadata = {
            "final_M": final_m,
            "ground_truth_malicious_clients": sorted(self.malicious_client_names),
            "config": self.config.to_dict(),
            "trigger": {
                "size": self.trigger.size,
                "top": self.trigger.top,
                "left": self.trigger.left,
                "target_label": self.trigger.target_label,
                "pattern": self.trigger.pattern.tolist(),
            },
            "elapsed_seconds": elapsed,
            "model_parameter_count": count_parameters(self.model),
        }
        with open(os.path.join(self.config.output_dir, "crowdguard_metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2)

        print("#" * 60)
        print("CrowdGuard run completed")
        print(f"Final M: {final_m}")
        print(f"Handoff: {output_path}")
        print(f"Elapsed: {elapsed:.2f}s")
        print("#" * 60)


def build_datasets(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD_DEV.tolist()),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                    transform=transform)

    if config.num_clients * config.samples_per_client > len(train_dataset):
        raise ValueError(
            f"Need {config.num_clients * config.samples_per_client} training samples "
            f"but CIFAR-10 has only {len(train_dataset)}."
        )

    # Deterministic IID partition of the 50,000 training samples.
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_dataset))
    indices = indices[:config.num_clients * config.samples_per_client]

    train_x = torch.stack([train_dataset[i][0] for i in indices])
    train_y = torch.tensor([train_dataset[i][1] for i in indices], dtype=torch.long)
    test_x = torch.stack([item[0] for item in test_dataset])
    test_y = torch.tensor([item[1] for item in test_dataset], dtype=torch.long)

    client_loaders = {}
    for client_id in range(config.num_clients):
        start = client_id * config.samples_per_client
        end = start + config.samples_per_client
        x = train_x[start:end]
        y = train_y[start:end]
        client_loaders[client_id] = DataLoader(
            TensorDataset(x, y), batch_size=config.batch_size, shuffle=True
        )

    # Per-collaborator diagnostics use a bounded subset; the complete 10,000-image
    # test set is intentionally not duplicated into every collaborator state.
    eval_count = min(1000, len(test_x))
    clean_test_loader = DataLoader(
        TensorDataset(test_x[:eval_count], test_y[:eval_count]),
        batch_size=1000, shuffle=False
    )
    return client_loaders, clean_test_loader


def make_backdoor_test_loader(test_dataset, trigger, batch_size=1000, max_samples=1000):
    # Keep the per-collaborator diagnostic loader small; the complete 10,000-image
    # CIFAR-10 test set remains reserved for downstream evaluation.
    count = min(len(test_dataset), max_samples)
    data = torch.stack([test_dataset[i][0] for i in range(count)])
    labels = torch.full((count,), trigger.target_label, dtype=torch.long)
    poisoned = torch.stack([apply_trigger(image, trigger) for image in data])
    return DataLoader(TensorDataset(poisoned, labels), batch_size=batch_size, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comm_round", "--rounds", dest="rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=2500)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--pmr", type=float, default=0.05)
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--attack_start_round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--optimizer_type", type=str, default="SGD")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rounds=args.rounds,
        pmr=args.pmr,
        poison_rate=args.poison_rate,
        alpha=args.alpha,
        attack_start_round=args.attack_start_round,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.validate()
    seed_random_generators(config.seed)

    aggregator_object = Aggregator()
    aggregator_object.private_attributes = {}
    collaborator_names = [
        f"benign_{i:02d}" for i in range(config.num_benign_clients)
    ] + [
        f"malicious_{i:02d}" for i in range(config.num_malicious_clients)
    ]
    collaborators = [Collaborator(name=name) for name in collaborator_names]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, clean_test_loader = build_datasets(config)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD_DEV.tolist()),
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                    transform=transform)
    trigger = sample_trigger(config.trigger_size, config.target_label, seed=config.seed)

    for idx, collab in enumerate(collaborators):
        train_loader = client_loaders[idx]
        backdoor_loader = make_backdoor_test_loader(test_dataset, trigger)
        collab.private_attributes = {
            "train_loader": train_loader,
            "test_loader": clean_test_loader,
            "backdoor_test_loader": backdoor_loader,
        }

    local_runtime = LocalRuntime(
        aggregator=aggregator_object,
        collaborators=collaborators,
    )

    model = LightweightResNet18()
    optimizer_templates = {
        collaborator.name: args.optimizer_type.upper()
        for collaborator in collaborators
    }

    flflow = FederatedFlow(
        model=model,
        optimizer_templates=optimizer_templates,
        config=config,
        device=device,
    )
    flflow.runtime = local_runtime
    flflow.run()


if __name__ == "__main__":
    main()
