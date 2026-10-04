"""Unit tests for the torch-free helpers.  Run:  python test_handoff_utils.py"""

import unittest

import numpy as np

import commit_reveal as cr
from handoff_utils import (
    detection_counts, detection_report, select_final_m,
    split_test_indices, stacked_clustering_vote,
)

CTX = cr.make_context(seed=10, num_clients=5, pmr=0.05)
NAMES = [f"c{i}" for i in range(5)]


def honest_round(round_num=1, votes=None):
    votes = votes or {v: [1] * 5 for v in NAMES}
    commits, reveals = {}, {}
    for v in NAMES:
        nonce = cr.new_nonce()
        commits[v] = cr.make_commitment(votes[v], round_num, v, NAMES, nonce, CTX)
        reveals[v] = {"votes": list(votes[v]), "nonce": nonce}
    return commits, reveals


class CommitRevealTests(unittest.TestCase):
    def test_honest_round_all_valid(self):
        commits, reveals = honest_round()
        status, valid = cr.verify_round(commits, reveals, NAMES, 1, CTX)
        self.assertEqual(valid, NAMES)
        self.assertTrue(all(s == cr.STATUS_VALID for s in status.values()))

    def test_changed_vote_is_rejected(self):
        commits, reveals = honest_round()
        reveals = cr.tamper_reveal(reveals, "c0")
        status, valid = cr.verify_round(commits, reveals, NAMES, 1, CTX)
        self.assertEqual(status["c0"], cr.STATUS_MISMATCH)
        self.assertNotIn("c0", valid)
        self.assertEqual(len(valid), 4)

    def test_replay_in_other_round_is_rejected(self):
        commits, reveals = honest_round(round_num=1)
        status, valid = cr.verify_round(commits, reveals, NAMES, 2, CTX)
        self.assertEqual(valid, [])
        self.assertTrue(all(s == cr.STATUS_MISMATCH for s in status.values()))

    def test_commitment_bound_to_validator(self):
        commits, reveals = honest_round()
        commits = dict(commits)
        commits["c0"], commits["c1"] = commits["c1"], commits["c0"]
        _, valid = cr.verify_round(commits, reveals, NAMES, 1, CTX)
        self.assertNotIn("c0", valid)
        self.assertNotIn("c1", valid)

    def test_wrong_context_is_rejected(self):
        commits, reveals = honest_round()
        other = cr.make_context(seed=11, num_clients=5, pmr=0.05)
        _, valid = cr.verify_round(commits, reveals, NAMES, 1, other)
        self.assertEqual(valid, [])

    def test_missing_and_malformed(self):
        commits, reveals = honest_round()
        del reveals["c0"]
        del commits["c1"]
        reveals["c2"] = {"votes": [1, 1], "nonce": "x"}          # wrong length
        reveals["c3"] = {"votes": [1, 1, 1, 1, 2], "nonce": "x"}  # not 0/1
        status, valid = cr.verify_round(commits, reveals, NAMES, 1, CTX)
        self.assertEqual(status["c0"], cr.STATUS_MISSING_REVEAL)
        self.assertEqual(status["c1"], cr.STATUS_MISSING_COMMITMENT)
        self.assertEqual(status["c2"], cr.STATUS_MALFORMED)
        self.assertEqual(status["c3"], cr.STATUS_MALFORMED)
        self.assertEqual(valid, ["c4"])


class DigestTests(unittest.TestCase):
    def test_digest_properties(self):
        commits, _ = honest_round(round_num=1)
        d = cr.commitment_digest(commits, 1, CTX)
        self.assertEqual(d, cr.commitment_digest(dict(reversed(list(commits.items()))), 1, CTX))
        altered = dict(commits)
        altered["c0"] = "0" * 64
        self.assertNotEqual(d, cr.commitment_digest(altered, 1, CTX))
        self.assertNotEqual(d, cr.commitment_digest(commits, 2, CTX))
        missing = {k: v for k, v in commits.items() if k != "c3"}
        self.assertNotEqual(d, cr.commitment_digest(missing, 1, CTX))


class FinalMTests(unittest.TestCase):
    def history(self):
        # 10 rounds: 'bad' flagged 9 times, four benign clients flagged once each
        det = [["b0"], ["b1", "bad"], ["bad"], ["bad"], ["bad"],
               ["b2", "bad"], ["bad"], ["bad"], ["b3", "bad"], ["bad"]]
        return [{"round": i, "detected_clients": d} for i, d in enumerate(det)]

    def test_union_keeps_one_off_flags(self):
        self.assertEqual(select_final_m(self.history(), "union"),
                         ["b0", "b1", "b2", "b3", "bad"])

    def test_frequency_drops_one_off_flags(self):
        for tau in (0.2, 0.5, 0.9):
            self.assertEqual(select_final_m(self.history(), "frequency", tau), ["bad"])

    def test_frequency_boundaries(self):
        h = self.history()
        self.assertEqual(len(select_final_m(h, "frequency", 0.1)), 5)   # >=1 round
        self.assertEqual(select_final_m(h, "frequency", 1.0), [])       # all 10 rounds
        self.assertEqual(select_final_m(h, "frequency", 0.9), ["bad"])  # exactly 9

    def test_float_edge(self):
        # 0.3 * 10 == 3.0000000000000004: a client flagged in exactly 3 rounds is in
        h = [{"round": i, "detected_clients": ["x"] if i < 3 else []} for i in range(10)]
        self.assertEqual(select_final_m(h, "frequency", 0.3), ["x"])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            select_final_m(self.history(), "median")
        with self.assertRaises(ValueError):
            select_final_m(self.history(), "frequency", 0)
        self.assertEqual(select_final_m([], "frequency", 0.5), [])

    def test_counts(self):
        self.assertEqual(detection_counts(self.history())["bad"], 9)


class SplitTests(unittest.TestCase):
    def labels(self):
        return np.repeat(np.arange(10), 1000)       # CIFAR-10 test layout

    def test_sizes_disjoint_stratified(self):
        y = self.labels()
        kd, ev = split_test_indices(y, 2500, seed=0)
        self.assertEqual((len(kd), len(ev)), (2500, 7500))
        self.assertEqual(set(kd) & set(ev), set())
        self.assertEqual(set(kd) | set(ev), set(range(10000)))
        self.assertTrue(all(c == 250 for c in np.bincount(y[kd])))

    def test_deterministic_and_seed_sensitive(self):
        y = self.labels()
        self.assertEqual(split_test_indices(y, 2500, 0), split_test_indices(y, 2500, 0))
        self.assertNotEqual(split_test_indices(y, 2500, 0)[0], split_test_indices(y, 2500, 1)[0])

    def test_uneven_size(self):
        kd, ev = split_test_indices(self.labels(), 2505, seed=0)
        self.assertEqual(len(kd), 2505)
        self.assertEqual(len(kd) + len(ev), 10000)

    def test_bad_size(self):
        with self.assertRaises(ValueError):
            split_test_indices(self.labels(), 0, 0)
        with self.assertRaises(ValueError):
            split_test_indices(self.labels(), 10000, 0)


class VoteAggregationTests(unittest.TestCase):
    def test_honest_majority_flags_attacker(self):
        # 20 validators; 19 benign validators flag candidate 19 and the attacker
        # (validator 19) votes everything benign.
        n = 20
        benign_row = [1] * 19 + [0]
        rows = [benign_row[:] for _ in range(19)] + [[1] * 20]
        final = stacked_clustering_vote(rows, n)
        self.assertEqual([i for i, v in enumerate(final) if v == 0], [19])

    def test_all_benign(self):
        rows = [[1] * 20 for _ in range(20)]
        self.assertEqual(stacked_clustering_vote(rows, 20), [1] * 20)

    def test_no_valid_rows_defaults_benign(self):
        self.assertEqual(stacked_clustering_vote([], 7), [1] * 7)

    def test_small_matrix_majority(self):
        rows = [[1, 0], [1, 0]]
        self.assertEqual(stacked_clustering_vote(rows, 2), [1, 0])

    def test_one_tampered_row_dropped_does_not_change_result(self):
        n = 20
        benign_row = [1] * 19 + [0]
        rows = [benign_row[:] for _ in range(19)]       # attacker's row excluded
        final = stacked_clustering_vote(rows, n)
        self.assertEqual([i for i, v in enumerate(final) if v == 0], [19])


class DetectionReportTests(unittest.TestCase):
    def test_report(self):
        names = ["b0", "b1", "bad"]
        det = [["b0"], ["bad"], ["bad"], ["bad"]]
        history = [{"round": i, "detected_clients": d,
                    "attack_active": {"b0": False, "b1": False, "bad": i > 0},
                    "updates": {n: None for n in names}} for i, d in enumerate(det)]
        report = detection_report({"round_history": history,
                                   "ground_truth_malicious_clients": ["bad"]})
        self.assertEqual(report["final_m_rules"]["union"]["fp"], 1)
        self.assertEqual(report["final_m_rules"]["frequency_tau=0.50"]["fp"], 0)
        self.assertEqual(report["final_m_rules"]["frequency_tau=0.50"]["recall"], 1.0)
        self.assertEqual(report["per_round"][0]["fp"], 1)
        self.assertEqual(report["per_round"][0]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
