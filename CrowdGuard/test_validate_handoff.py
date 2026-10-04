"""Tests for validate_handoff on a synthetic schema-2 handoff (needs numpy+sklearn,
not torch).  Run:  python test_validate_handoff.py"""

import copy
import unittest

import numpy as np

import commit_reveal as cr
from handoff_utils import detection_counts, select_final_m, split_test_indices
from validate_handoff import validate

NAMES = [f"benign_{i:02d}" for i in range(4)] + ["malicious_00"]
SEED, PMR, ROUNDS, SAMPLES = 10, 0.2, 3, 10


def build_handoff():
    rng = np.random.default_rng(0)
    ctx = cr.make_context(SEED, len(NAMES), PMR)
    theta_0 = {
        "w": rng.normal(size=(4, 3)).astype(np.float32),
        "bn.running_var": (np.abs(rng.normal(size=4)) + 1).astype(np.float32),
        "bn.num_batches_tracked": np.array(0, dtype=np.int64),
    }
    float_keys = ["w", "bn.running_var"]
    current = {k: v.copy() for k, v in theta_0.items()}
    history = []
    for r in range(ROUNDS):
        updates = {n: {k: (rng.normal(size=current[k].shape) * 0.1).astype(np.float32)
                       for k in float_keys} for n in NAMES}
        before = {k: v.copy() for k, v in current.items()}
        after = dict(before)
        for k in float_keys:
            mean = sum(updates[n][k].astype(np.float64) for n in NAMES) / len(NAMES)
            after[k] = (before[k].astype(np.float64) + mean).astype(np.float32)
        current = {k: v.copy() for k, v in after.items()}

        # honest votes: everybody flags malicious_00, which votes all-benign
        votes = {v: [1, 1, 1, 1, 0] for v in NAMES[:4]}
        votes["malicious_00"] = [1, 1, 1, 1, 1]
        commits, reveals = {}, {}
        for v in NAMES:
            nonce = cr.new_nonce()
            commits[v] = cr.make_commitment(votes[v], r, v, NAMES, nonce, ctx)
            reveals[v] = {"votes": votes[v], "nonce": nonce}
        status, valid = cr.verify_round(commits, reveals, NAMES, r, ctx)
        history.append({
            "round": r,
            "updates": updates,
            "attack_active": {n: (n == "malicious_00" and r > 0) for n in NAMES},
            "global_model_before": before,
            "global_model_after": after,
            "votes": {v: dict(zip(NAMES, reveals[v]["votes"])) for v in valid},
            "commit_reveal": {"context": ctx, "commitments": commits,
                              "commitment_digest": cr.commitment_digest(commits, r, ctx),
                              "own_commitment_confirmed": {v: True for v in NAMES},
                              "reveals": reveals,
                              "status": status, "valid_validators": valid},
            "detected_clients": ["malicious_00"],
        })

    kd, ev = split_test_indices(np.repeat(np.arange(10), 1000), 2500, 0)
    perm = np.random.default_rng(1).permutation(50000)
    return {
        "schema_version": 2,
        "theta_0": theta_0,
        "theta_k": current,
        "round_history": history,
        "final_M": select_final_m(history, "frequency", 0.5),
        "final_M_rule": {"rule": "frequency", "tau": 0.5, "rounds": ROUNDS},
        "detection_counts": detection_counts(history),
        "ground_truth_malicious_clients": ["malicious_00"],
        "config": {"seed": SEED, "num_clients": len(NAMES), "pmr": PMR, "rounds": ROUNDS,
                   "samples_per_client": SAMPLES, "kd_reference_size": 2500,
                   "tamper_test": False},
        "trigger": {},
        "client_train_indices": {n: perm[i * SAMPLES:(i + 1) * SAMPLES].tolist()
                                 for i, n in enumerate(NAMES)},
        "test_split": {"seed": 0, "kd_reference_size": 2500,
                       "kd_indices": kd, "eval_indices": ev},
        "environment": {},
    }


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.good = build_handoff()

    def run_validate(self, mutate):
        h = copy.deepcopy(self.good)
        mutate(h)
        return validate(h)["errors"]

    def test_clean_handoff_passes(self):
        result = validate(self.good)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])

    def test_weight_identity_break_is_caught(self):
        def m(h): h["round_history"][1]["updates"]["benign_00"]["w"] += 1.0
        self.assertTrue(any("mean(updates)" in e for e in self.run_validate(m)))

    def test_theta_k_mismatch_is_caught(self):
        def m(h): h["theta_k"]["w"] = h["theta_k"]["w"] + 0.5
        self.assertTrue(self.run_validate(m))

    def test_changed_revealed_vote_is_caught(self):
        def m(h): h["round_history"][0]["commit_reveal"]["reveals"]["benign_00"]["votes"][0] = 0
        errors = self.run_validate(m)
        self.assertTrue(any("status disagrees" in e for e in errors))

    def test_forged_detection_is_caught(self):
        def m(h): h["round_history"][2]["detected_clients"] = []
        self.assertTrue(any("recomputed from revealed votes" in e for e in self.run_validate(m)))

    def test_wrong_final_m_is_caught(self):
        def m(h): h["final_M"] = []
        self.assertTrue(any("final_M" in e for e in self.run_validate(m)))

    def test_wrong_counts_are_caught(self):
        def m(h): h["detection_counts"]["malicious_00"] = 1
        self.assertTrue(any("detection_counts" in e for e in self.run_validate(m)))

    def test_missing_client_is_caught(self):
        def m(h): del h["round_history"][1]["updates"]["benign_03"]
        self.assertTrue(any("client set differs" in e for e in self.run_validate(m)))

    def test_overlapping_partitions_are_caught(self):
        def m(h):
            h["client_train_indices"]["benign_01"][0] = h["client_train_indices"]["benign_00"][0]
        self.assertTrue(any("overlap" in e for e in self.run_validate(m)))

    def test_overlapping_test_split_is_caught(self):
        def m(h): h["test_split"]["kd_indices"][0] = h["test_split"]["eval_indices"][0]
        self.assertTrue(any("overlap" in e or "cover" in e for e in self.run_validate(m)))

    def test_rejected_reveal_is_reported_when_consistent(self):
        # A genuinely tampered round, recorded honestly, is a warning not an error.
        h = copy.deepcopy(self.good)
        rec = h["round_history"][0]
        reveals = cr.tamper_reveal(rec["commit_reveal"]["reveals"], "benign_00")
        ctx = rec["commit_reveal"]["context"]
        status, valid = cr.verify_round(rec["commit_reveal"]["commitments"],
                                        reveals, NAMES, 0, ctx)
        rec["commit_reveal"].update(reveals=reveals, status=status, valid_validators=valid)
        rec["votes"] = {v: dict(zip(NAMES, reveals[v]["votes"])) for v in valid}
        result = validate(h)
        self.assertEqual(result["errors"], [])
        self.assertTrue(any("rejected" in w for w in result["warnings"]))

    def test_digest_mismatch_is_caught(self):
        def m(h): h["round_history"][1]["commit_reveal"]["commitment_digest"] = "0" * 64
        self.assertTrue(any("digest" in e for e in self.run_validate(m)))

    def test_swapped_commitment_list_is_caught(self):
        # commitment list altered after the digest was published
        def m(h):
            c = h["round_history"][0]["commit_reveal"]["commitments"]
            c["benign_00"], c["benign_01"] = c["benign_01"], c["benign_00"]
        errors = self.run_validate(m)
        self.assertTrue(any("digest" in e for e in errors))

    def test_unconfirmed_commitment_is_reported(self):
        h = copy.deepcopy(self.good)
        h["round_history"][0]["commit_reveal"]["own_commitment_confirmed"]["benign_02"] = False
        self.assertTrue(any("refused to reveal" in w for w in validate(h)["warnings"]))

    def test_old_schema_is_refused(self):
        h = copy.deepcopy(self.good)
        h["schema_version"] = 1
        self.assertTrue(validate(h)["errors"])

    def test_tamper_flag_warns(self):
        h = copy.deepcopy(self.good)
        h["config"]["tamper_test"] = True
        self.assertTrue(any("tamper_test" in w for w in validate(h)["warnings"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
