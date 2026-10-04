"""Evaluate a CrowdGuard handoff.

    python evaluate_handoff.py outputs/crowdguard_handoff.pt [--device cuda]

Writes <handoff dir>/evaluation.json and prints two tables:

1. Detection: per-round and final-M precision / recall / F1 against the ground
   truth, for the union rule and for the frequency rule at every tau.
2. Models: clean accuracy and attack success rate (ASR) of theta_0 and of the
   global model after every round (the last one is theta_k), measured ONLY on
   the held-out part of the test set.  The 2,500 KD-reference images are never
   used here.  ASR counts triggered images whose true class is not the target
   and that the model classifies as the target class.
"""

import argparse
import json
import os

import torch
from torchvision import datasets, transforms

from adaptive_backdoor import TriggerConfig, apply_trigger
from handoff_utils import detection_report
from lightweight_resnet18 import LightweightResNet18

# Must match cifar10_crowdguard.py
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def build_eval_tensors(handoff, data_root):
    """Held-out clean images + labels, and triggered non-target images."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    split = handoff["test_split"]
    eval_idx = split["eval_indices"]
    assert not set(eval_idx) & set(split["kd_indices"]), "KD and eval splits overlap"

    trig = handoff["trigger"]
    trigger = TriggerConfig(size=int(trig["size"]), top=int(trig["top"]),
                            left=int(trig["left"]), pattern=trig["pattern"].float(),
                            target_label=int(trig["target_label"]))

    images = torch.stack([test[i][0] for i in eval_idx])
    labels = torch.tensor([int(test.targets[i]) for i in eval_idx], dtype=torch.long)

    keep = labels != trigger.target_label            # drop images already of the target class
    triggered = torch.stack([apply_trigger(img, trigger) for img in images[keep]])
    return images, labels, triggered, trigger.target_label


@torch.no_grad()
def accuracy(model, images, labels, device, batch=500):
    correct = 0
    for start in range(0, len(images), batch):
        out = model(images[start:start + batch].to(device))
        correct += (out.argmax(1).cpu() == labels[start:start + batch]).sum().item()
    return correct / len(images)


@torch.no_grad()
def attack_success_rate(model, triggered, target, device, batch=500):
    hits = 0
    for start in range(0, len(triggered), batch):
        out = model(triggered[start:start + batch].to(device))
        hits += (out.argmax(1).cpu() == target).sum().item()
    return hits / len(triggered)


def evaluate_models(handoff, images, labels, triggered, target, device):
    states = [("theta_0", handoff["theta_0"])]
    for rec in handoff["round_history"]:
        name = f"after_round_{rec['round']}"
        if rec["round"] == len(handoff["round_history"]) - 1:
            name += " (=theta_k)"
        states.append((name, rec["global_model_after"]))

    model = LightweightResNet18().to(device)
    results = []
    for name, state in states:
        model.load_state_dict(state)
        model.eval()
        results.append({
            "model": name,
            "clean_accuracy": accuracy(model, images, labels, device),
            "asr": attack_success_rate(model, triggered, target, device),
            "n_clean": len(images),
            "n_asr": len(triggered),
        })
    return results


def print_detection(report):
    print("\nDETECTION (ground truth:", ", ".join(report["ground_truth"]) or "none", ")")
    print(f"{'round':>5}  {'TP':>2} {'FP':>2} {'FN':>2}  {'prec':>5} {'rec':>5}  detected")
    for r in report["per_round"]:
        print(f"{r['round']:>5}  {r['tp']:>2} {r['fp']:>2} {r['fn']:>2}  "
              f"{_fmt(r['precision']):>5} {_fmt(r['recall']):>5}  {r['detected']}")
    print(f"\n{'final-M rule':<22} {'TP':>2} {'FP':>2} {'FN':>2}  {'prec':>5} {'rec':>5} {'F1':>5}")
    for name, r in report["final_m_rules"].items():
        print(f"{name:<22} {r['tp']:>2} {r['fp']:>2} {r['fn']:>2}  "
              f"{_fmt(r['precision']):>5} {_fmt(r['recall']):>5} {_fmt(r['f1']):>5}")


def print_models(results):
    print("\nGLOBAL MODEL (held-out test images only)")
    print(f"{'model':<28} {'clean acc':>9} {'ASR':>7}")
    for r in results:
        print(f"{r['model']:<28} {r['clean_accuracy']:>9.4f} {r['asr']:>7.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("handoff")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    handoff = torch.load(args.handoff, map_location="cpu", weights_only=True)
    if handoff.get("schema_version") != 2:
        raise SystemExit("this script expects a schema_version 2 handoff")
    if handoff["config"].get("tamper_test"):
        print("WARNING: tamper_test run, results must not be used")

    report = detection_report(handoff)
    images, labels, triggered, target = build_eval_tensors(handoff, args.data_root)
    models = evaluate_models(handoff, images, labels, triggered, target, args.device)

    print_detection(report)
    print_models(models)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.handoff)),
                                        "evaluation.json")
    with open(out_path, "w") as handle:
        json.dump({
            "config": handoff["config"],
            "final_M": handoff["final_M"],
            "final_M_rule": handoff["final_M_rule"],
            "detection": report,
            "models": models,
            "target_label": target,
            "eval_images": len(images),
            "asr_images": len(triggered),
        }, handle, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
