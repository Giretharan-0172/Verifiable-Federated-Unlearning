"""Helpers for the CrowdGuard handoff.

Everything here is importable WITHOUT torch (torch/sklearn are imported lazily
inside the few functions that need them), so the logic can be unit-tested and
the handoff can be audited on any machine.
"""

import hashlib
import os
import platform
import subprocess
import sys
from collections import Counter

import numpy as np

FINAL_M_RULES = ("union", "frequency")

SOURCE_FILES = (
    "cifar10_crowdguard.py",
    "CrowdGuardClientValidation.py",
    "adaptive_backdoor.py",
    "lightweight_resnet18.py",
    "experiment_config.py",
    "handoff_utils.py",
    "commit_reveal.py",
)


# ---------------------------------------------------------------------------
# Final malicious-set rule
# ---------------------------------------------------------------------------

def detection_counts(round_history):
    """Number of rounds in which each client appeared in M_t."""
    counts = Counter()
    for record in round_history:
        counts.update(set(record["detected_clients"]))
    return dict(counts)


def select_final_m(round_history, rule="frequency", tau=0.5):
    """Final malicious-client set M from the per-round detections M_t.

    union:      M = union of all M_t (provisional rule used in the first run).
    frequency:  client is in M if it was detected in at least tau * K of the
                K rounds.  Repeated detection counts, one-off flags do not.
    """
    if rule not in FINAL_M_RULES:
        raise ValueError(f"unknown final-M rule {rule!r}; expected {FINAL_M_RULES}")
    rounds = len(round_history)
    if rounds == 0:
        return []
    counts = detection_counts(round_history)
    if rule == "union":
        return sorted(counts)
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    needed = tau * rounds
    # small epsilon so e.g. tau=0.3, K=10 (3.0000000000000004) means "3 rounds"
    return sorted(c for c, n in counts.items() if n >= needed - 1e-9)


# ---------------------------------------------------------------------------
# Test-set split (KD reference / held-out evaluation)
# ---------------------------------------------------------------------------

def split_test_indices(labels, kd_size, seed):
    """Stratified, deterministic split of the test set.

    Returns (kd_indices, eval_indices): sorted lists of python ints.  The KD
    part holds ``kd_size`` images spread evenly over the classes; the rest is
    held out for evaluation.  Same (labels, kd_size, seed) -> same split.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if not 0 < kd_size < n:
        raise ValueError("kd_size must be between 1 and len(labels)-1")
    classes = np.unique(labels)
    base, extra = divmod(kd_size, len(classes))
    rng = np.random.default_rng(seed)
    kd = []
    for i, cls in enumerate(classes):
        members = np.flatnonzero(labels == cls)
        take = base + (1 if i < extra else 0)
        if take > len(members):
            raise ValueError(f"class {cls} has only {len(members)} samples")
        kd.extend(int(j) for j in rng.choice(members, size=take, replace=False))
    kd = sorted(kd)
    kd_set = set(kd)
    held_out = [i for i in range(n) if i not in kd_set]
    return kd, held_out


# ---------------------------------------------------------------------------
# Stacked-clustering vote aggregation (single source of truth)
# ---------------------------------------------------------------------------

def _cluster_map(labels):
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)
    return clusters


def _biggest(clusters):
    return max(clusters, key=lambda cid: len(clusters[cid]))


def stacked_clustering_vote(binary_votes, num_candidates):
    """CrowdGuard Alg. 3: agglomerative -> biggest cluster -> DBSCAN -> biggest.

    binary_votes: list of equal-length vote vectors (1 = benign, 0 = poisoned),
    one per validator whose reveal was valid.  Returns the final vote vector.
    The same function is used by the experiment and by the audit script, so
    the tally can be recomputed exactly from the revealed votes.
    """
    from sklearn.cluster import AgglomerativeClustering, DBSCAN

    rows = [list(map(int, row)) for row in binary_votes]
    if not rows:
        return [1] * num_candidates          # nothing valid: default benign
    if len(rows) >= 3:
        ac = AgglomerativeClustering(
            n_clusters=2, distance_threshold=None, compute_full_tree=True,
            metric="euclidean", linkage="single", compute_distances=True,
        ).fit(rows)
        clusters = _cluster_map(ac.labels_.tolist())
        biggest = clusters[_biggest(clusters)]
        db_input = [rows[i] for i in biggest]
        db = DBSCAN(eps=0.5, min_samples=1).fit(db_input)
        db_clusters = _cluster_map(db.labels_.tolist())
        largest = db_clusters[_biggest(db_clusters)]
        return list(db_input[int(largest[0])])
    # fewer than 3 valid validators (smoke tests / heavy tampering): majority
    return [
        1 if sum(row[j] for row in rows) >= len(rows) / 2 else 0
        for j in range(num_candidates)
    ]


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _prf(pred, truth, universe):
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    tn = len(universe - pred - truth)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def detection_report(handoff):
    """Per-round and final-M detection metrics against the ground truth."""
    history = handoff["round_history"]
    truth = set(handoff["ground_truth_malicious_clients"])
    universe = set(history[0]["updates"].keys())
    rounds = len(history)

    per_round = []
    for record in history:
        pred = set(record["detected_clients"])
        entry = {"round": record["round"],
                 "detected": sorted(pred),
                 "attack_active": sorted(c for c, v in record["attack_active"].items() if v)}
        entry.update(_prf(pred, truth, universe))
        per_round.append(entry)

    rules = {"union": _prf(set(select_final_m(history, "union")), truth, universe)}
    for k in range(1, rounds + 1):
        tau = k / rounds
        m = set(select_final_m(history, "frequency", tau))
        rules[f"frequency_tau={tau:.2f}"] = dict(_prf(m, truth, universe), M=sorted(m))
    rules["union"]["M"] = select_final_m(history, "union")

    return {
        "ground_truth": sorted(truth),
        "detection_counts": detection_counts(history),
        "per_round": per_round,
        "final_m_rules": rules,
    }


# ---------------------------------------------------------------------------
# Run metadata / atomic saving
# ---------------------------------------------------------------------------

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args, cwd):
    try:
        out = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_environment_metadata(repo_dir=None):
    """Code and library versions, so the handoff can be loaded/reproduced later.

    Source-file hashes are recorded because the notebook environment may not
    be a git checkout.
    """
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    info = {"python": sys.version.split()[0], "platform": platform.platform()}

    from importlib import metadata
    packages = {}
    for pkg in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "openfl"):
        try:
            packages[pkg] = metadata.version(pkg)
        except Exception:
            packages[pkg] = None
    info["packages"] = packages

    commit = _git(["rev-parse", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    info["git_commit"] = commit
    info["git_dirty"] = bool(status) if status is not None else None

    hashes = {}
    for name in SOURCE_FILES:
        path = os.path.join(repo_dir, name)
        hashes[name] = _sha256_file(path) if os.path.exists(path) else None
    info["source_sha256"] = hashes
    return info


def atomic_torch_save(obj, path):
    """torch.save to a temp file, then rename: a crash never leaves half a file."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
