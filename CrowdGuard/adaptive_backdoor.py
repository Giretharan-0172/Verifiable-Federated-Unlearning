"""Adaptive pixel-backdoor attack used by the CrowdGuard project.

The attack follows the generic constrain-and-scale formulation from
Bagdasaryan et al.: classification loss is combined with an anomaly/distance
penalty and the resulting model update is scaled before submission.

The particular model-distance anomaly term below is a project implementation
choice.  It is deliberately not described as an HLBIM-specific loss from the
paper.
"""

from dataclasses import dataclass
from copy import deepcopy
import random

import torch
import torch.nn as nn


@dataclass
class TriggerConfig:
    size: int
    top: int
    left: int
    pattern: torch.Tensor  # shape [3, size, size], normalized input space
    target_label: int

    def to_dict(self):
        return {
            "size": self.size,
            "top": self.top,
            "left": self.left,
            "target_label": self.target_label,
            "pattern": self.pattern.cpu(),
        }


def sample_trigger(size, target_label=None, num_classes=10, seed=None):
    rng = random.Random(seed)
    top = rng.randint(0, 32 - size)
    left = rng.randint(0, 32 - size)
    target = rng.randrange(num_classes) if target_label is None else int(target_label)

    # Continuous RGB values in the normalized input domain.  Sampling the
    # underlying [0,1] RGB values avoids hard-coded red/corner behavior.
    raw = torch.rand(3, size, size, generator=torch.Generator().manual_seed(
        rng.randrange(2**31)
    ))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    pattern = (raw - mean) / std
    return TriggerConfig(size, top, left, pattern, target)


def apply_trigger(image, trigger):
    image = image.clone()
    t = trigger.pattern.to(device=image.device, dtype=image.dtype)
    image[:, trigger.top:trigger.top + trigger.size,
          trigger.left:trigger.left + trigger.size] = t
    return image


def poison_batch(data, labels, trigger, poison_rate, generator=None):
    if poison_rate <= 0:
        return data, labels, torch.empty(0, dtype=torch.long, device=labels.device)

    batch_size = data.shape[0]
    count = max(1, int(round(batch_size * poison_rate)))
    count = min(count, batch_size)
    if generator is None:
        indices = torch.randperm(batch_size, device=data.device)[:count]
    else:
        indices = torch.randperm(batch_size, generator=generator, device=data.device)[:count]

    poisoned = data.clone()
    poisoned[indices] = torch.stack([
        apply_trigger(data[idx], trigger) for idx in indices
    ])
    poisoned_labels = labels.clone()
    poisoned_labels[indices] = trigger.target_label
    return poisoned, poisoned_labels, indices


def state_dict_distance(model, reference_state):
    total = None
    for name, parameter in model.named_parameters():
        diff = parameter - reference_state[name].to(parameter.device)
        value = torch.sum(diff * diff)
        total = value if total is None else total + value
    return total if total is not None else torch.tensor(0.0, device=next(model.parameters()).device)


def adaptive_train_and_scale(
    model,
    global_state,
    loader,
    optimizer,
    device,
    trigger,
    poison_rate=0.10,
    alpha=0.70,
    local_epochs=1,
    scale_factor=1.0,
):
    """Train a malicious local model and scale its submitted update.

    L = alpha * L_class + (1-alpha) * L_ano
    where L_ano is a normalized squared parameter-distance penalty.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    # The raw squared distance can be much larger than CE.  Normalizing by the
    # number of parameters makes alpha's behavior less architecture-dependent.
    parameter_count = sum(p.numel() for p in model.parameters())

    for _ in range(local_epochs):
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            poisoned_data, poisoned_labels, _ = poison_batch(
                data, labels, trigger, poison_rate
            )
            optimizer.zero_grad()
            logits = model(poisoned_data)
            classification_loss = criterion(logits, poisoned_labels)
            anomaly_loss = state_dict_distance(model, global_state) / parameter_count
            loss = alpha * classification_loss + (1.0 - alpha) * anomaly_loss
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu()))

    # Scale the complete model delta relative to the round's incoming global
    # model.  N / malicious_count is the project default for equal-weight FedAvg.
    with torch.no_grad():
        current = model.state_dict()
        parameter_names = {name for name, _ in model.named_parameters()}
        scaled = {}
        for name, value in current.items():
            reference = global_state[name].to(value.device)
            if name in parameter_names:
                scaled[name] = reference + scale_factor * (value - reference)
            else:
                # BatchNorm running statistics/counters are buffers rather than
                # trainable parameters.  Never multiply these buffers by the
                # model-replacement scale: in particular, scaling running_var can
                # make it invalid and produce NaNs on the next forward pass.
                scaled[name] = value.clone()
        model.load_state_dict(scaled)

    model.cpu()
    return {
        "mean_loss": sum(loss_history) / max(1, len(loss_history)),
        "scale_factor": float(scale_factor),
    }


def scale_benign_model_update(model, global_state):
    """Return a CPU state dict representing a benign model's submitted update."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
