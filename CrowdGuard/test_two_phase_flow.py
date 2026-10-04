"""Checks the two-phase commit-reveal STRUCTURE on the real OpenFL workflow engine.

A small torch-free flow with the same steps and the same private-vault pattern as
cifar10_crowdguard.py:

    local_validation (collab)  -> collect_commitments (agg) -> reveal_votes (collab) -> defend (agg)

It asserts what the protocol relies on:
  * during the commit phase the aggregator sees commitments but no votes, no nonces
    and no vault (private attributes are stripped before transfer)
  * a validator's secret survives the aggregator step (private attribute write-back)
  * collaborators never receive other collaborators' votes
  * reveals are verified against the commitments fixed in phase 1
  * a validator whose commitment was altered between phases refuses to reveal
  * (OpenFL reuses collaborator state objects across rounds, so stale values from the
    previous round must be cleared explicitly; the toy flow and the real flow both do)
  * the vaults are empty at the end

Run:  python test_two_phase_flow.py        (needs openfl; skipped if unavailable)
"""

import os
import tempfile
import unittest

import commit_reveal as cr
from handoff_utils import stacked_clustering_vote

try:
    from openfl.experimental.workflow.interface import Aggregator, Collaborator, FLSpec
    from openfl.experimental.workflow.placement import aggregator, collaborator
    from openfl.experimental.workflow.runtime import LocalRuntime
    HAVE_OPENFL = True
except Exception as exc:                                   # pragma: no cover
    HAVE_OPENFL = False
    IMPORT_ERROR = repr(exc)

SEED, PMR = 10, 0.2
NAMES = ["benign_00", "benign_01", "benign_02", "benign_03", "malicious_00"]
BAD = "malicious_00"


if HAVE_OPENFL:
    class ToyFlow(FLSpec):
        def __init__(self, rounds=2, substitute=None, **kwargs):
            super().__init__(**kwargs)
            self.rounds = rounds
            self.round_num = 0
            self.history = []
            self.observations = []
            self.substitute = substitute        # name whose commitment the aggregator swaps

        @aggregator
        def start(self):
            self.collaborators = self.runtime.collaborators
            self.names = sorted(NAMES)
            self.next(self.local_validation, foreach="collaborators")

        @collaborator
        def local_validation(self):
            self.collaborator_name = self.input
            names = list(self.names)
            self.reveal_record = None               # same explicit reset as the real flow
            self.own_commitment_confirmed = None
            votes = [1] * len(names)
            if self.collaborator_name != BAD:
                votes[names.index(BAD)] = 0
            ctx = cr.make_context(SEED, len(names), PMR)
            nonce = cr.new_nonce()
            commitment = cr.make_commitment(
                votes, self.round_num, self.collaborator_name, names, nonce, ctx)
            self.vote_vault[self.round_num] = {
                "votes": votes, "nonce": nonce, "commitment": commitment}
            self.commit_record = {"validator": self.collaborator_name,
                                  "commitment": commitment}
            self.next(self.collect_commitments)

        @aggregator
        def collect_commitments(self, inputs):
            leaked = {}
            for item in inputs:
                leaked[item.collaborator_name] = {
                    "vault": hasattr(item, "vote_vault"),
                    "reveal": getattr(item, "reveal_record", None) is not None,
                    "own_confirmed": getattr(item, "own_commitment_confirmed", None) is not None,
                    "commit": hasattr(item, "commit_record"),
                }
            self.observations.append(("commit_phase", self.round_num, leaked))

            ctx = cr.make_context(SEED, len(self.names), PMR)
            self.round_commitments = {i.collaborator_name: i.commit_record["commitment"]
                                      for i in inputs}
            if self.substitute:
                self.round_commitments[self.substitute] = "f" * 64   # aggregator cheats
            self.round_digest = cr.commitment_digest(self.round_commitments, self.round_num, ctx)
            self.next(self.reveal_votes, foreach="collaborators")

        @collaborator
        def reveal_votes(self):
            self.collaborator_name = self.input
            entry = self.vote_vault.pop(self.round_num)
            published = self.round_commitments.get(self.collaborator_name)
            self.own_commitment_confirmed = (published == entry["commitment"])
            self.reveal_record = ({"votes": entry["votes"], "nonce": entry["nonce"]}
                                  if self.own_commitment_confirmed else None)
            # what can a collaborator see about others?  Only the public commitments.
            self.saw_other_votes = any(
                hasattr(self, attr) for attr in ("votes_of_this_client", "all_votes", "reveals"))
            self.next(self.defend)

        @aggregator
        def defend(self, inputs):
            ctx = cr.make_context(SEED, len(self.names), PMR)
            reveals = {i.collaborator_name: (dict(i.reveal_record) if i.reveal_record else None)
                       for i in inputs}
            confirmed = {i.collaborator_name: i.own_commitment_confirmed for i in inputs}
            peeked = any(i.saw_other_votes for i in inputs)
            status, valid = cr.verify_round(
                self.round_commitments, reveals, self.names, self.round_num, ctx)
            rows = [reveals[v]["votes"] for v in valid]
            final = stacked_clustering_vote(rows, len(self.names))
            detected = sorted(n for n, x in zip(self.names, final) if x == 0)
            self.history.append({"round": self.round_num, "status": status,
                                 "valid": valid, "detected": detected,
                                 "confirmed": confirmed, "peeked": peeked})
            self.round_num += 1
            if self.round_num < self.rounds:
                self.next(self.local_validation, foreach="collaborators")
            else:
                self.next(self.end)

        @aggregator
        def end(self):
            pass


def run_flow(**kwargs):
    agg = Aggregator()
    agg.private_attributes = {}
    collabs = [Collaborator(name=n) for n in NAMES]
    for c in collabs:
        c.private_attributes = {"vote_vault": {}}            # one distinct dict each
    flow = ToyFlow(**kwargs)
    flow.runtime = LocalRuntime(aggregator=agg, collaborators=collabs)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            flow.run()
        finally:
            os.chdir(cwd)
    return flow, collabs


@unittest.skipUnless(HAVE_OPENFL, "openfl workflow engine not importable")
class TwoPhaseFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow, cls.collabs = run_flow(rounds=2)

    def test_runs_all_rounds_and_detects_attacker(self):
        self.assertEqual(len(self.flow.history), 2)
        for rec in self.flow.history:
            self.assertEqual(rec["valid"], sorted(NAMES))
            self.assertEqual(rec["detected"], [BAD])

    def test_aggregator_sees_only_commitments_in_commit_phase(self):
        commit_obs = [o for o in self.flow.observations if o[0] == "commit_phase"]
        self.assertEqual(len(commit_obs), 2)
        for _, _, leaked in commit_obs:
            self.assertEqual(sorted(leaked), sorted(NAMES))
            for name, seen in leaked.items():
                self.assertTrue(seen["commit"], f"{name}: commitment missing")
                self.assertFalse(seen["vault"], f"{name}: vault leaked to aggregator")
                self.assertFalse(seen["reveal"], f"{name}: reveal existed before commit closed")
                self.assertFalse(seen["own_confirmed"])

    def test_secret_survives_aggregator_step(self):
        # reveal_votes pops the vault entry created in local_validation; if the vault
        # had not persisted across the aggregator step this would have raised KeyError
        # and the flow would not have completed all rounds with valid reveals.
        for rec in self.flow.history:
            self.assertTrue(all(rec["confirmed"].values()))

    def test_collaborators_never_see_other_votes(self):
        for rec in self.flow.history:
            self.assertFalse(rec["peeked"])

    def test_vaults_empty_at_end(self):
        for c in self.collabs:
            self.assertEqual(c.private_attributes["vote_vault"], {})


@unittest.skipUnless(HAVE_OPENFL, "openfl workflow engine not importable")
class SubstitutedCommitmentTests(unittest.TestCase):
    def test_validator_refuses_to_reveal_if_commitment_was_altered(self):
        flow, _ = run_flow(rounds=1, substitute="benign_01")
        rec = flow.history[0]
        self.assertFalse(rec["confirmed"]["benign_01"])
        self.assertEqual(rec["status"]["benign_01"], cr.STATUS_MISSING_REVEAL)
        self.assertNotIn("benign_01", rec["valid"])
        # everybody else is unaffected and the attacker is still found
        self.assertEqual(len(rec["valid"]), len(NAMES) - 1)
        self.assertEqual(rec["detected"], [BAD])


if __name__ == "__main__":
    if not HAVE_OPENFL:
        print("openfl not importable, skipping:", IMPORT_ERROR)
    unittest.main(verbosity=2)
