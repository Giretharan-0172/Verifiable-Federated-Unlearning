"""Commit-reveal for CrowdGuard validator votes (replaces the TEE vote channel).

Each validator publishes   C = SHA-256(scheme | context | round | validator |
                                       candidate order | votes | nonce)
and later reveals (votes, nonce).  Anyone can recompute C.  Only valid reveals
enter the vote matrix that the stacked clustering sees.

What this gives: votes are binding (cannot be changed after seeing others'),
bound to a round and validator (no replay), and the tally is recomputable.
What it does NOT give: honest voting, model confidentiality, or code
attestation (those were TEE properties and are out of scope here).

Protocol order (enforced by the OpenFL flow structure, see cifar10_crowdguard.py):
  1. local_validation   validators compute votes, keep (votes, nonce) in a
                        private vault, publish only the commitment
  2. collect_commitments aggregator fixes the commitment list + digest
  3. reveal_votes       each validator checks its commitment was published
                        unchanged, then reveals (votes, nonce)
  4. defend             aggregator verifies reveals against the commitments
                        fixed in step 2
Still a single-process simulation: there is no network and no adversarial
timing; the guarantee is that the flow itself never lets a reveal exist before
the commitment list is closed.
"""

import copy
import hashlib
import hmac
import secrets

SCHEME_VERSION = "crowdguard-commit-v1"

STATUS_VALID = "valid"
STATUS_MISSING_COMMITMENT = "missing_commitment"
STATUS_MISSING_REVEAL = "missing_reveal"
STATUS_MALFORMED = "malformed_reveal"
STATUS_MISMATCH = "hash_mismatch"


def make_context(seed, num_clients, pmr):
    """Experiment-level string bound into every commitment."""
    return f"{SCHEME_VERSION}|seed={seed}|clients={num_clients}|pmr={pmr}"


def new_nonce():
    return secrets.token_hex(16)


def _payload(votes, round_num, validator, candidates, nonce, context):
    vote_str = "".join(str(int(v)) for v in votes)
    return "|".join([
        SCHEME_VERSION,
        context,
        f"round={int(round_num)}",
        f"validator={validator}",
        "candidates=" + ",".join(candidates),
        f"votes={vote_str}",
        f"nonce={nonce}",
    ])


def make_commitment(votes, round_num, validator, candidates, nonce, context):
    payload = _payload(votes, round_num, validator, candidates, nonce, context)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _well_formed(reveal, num_candidates):
    if not isinstance(reveal, dict):
        return False
    votes, nonce = reveal.get("votes"), reveal.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        return False
    if not isinstance(votes, (list, tuple)) or len(votes) != num_candidates:
        return False
    return all(isinstance(v, int) and not isinstance(v, bool) and v in (0, 1)
               for v in votes)


def commitment_digest(commitments, round_num, context):
    """Digest of the whole published commitment list for one round.

    The aggregator publishes it when the commit phase closes; anyone holding
    the commitment list can recompute it, so the list cannot be silently
    changed after reveals start.  Independent of dict order.
    """
    lines = [f"{name}:{commitments[name]}" for name in sorted(commitments)]
    payload = "|".join([SCHEME_VERSION, "digest", context, f"round={int(round_num)}"] + lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_reveal(commitment, reveal, round_num, validator, candidates, context):
    """Return (ok, status) for one validator in one round."""
    if commitment is None:
        return False, STATUS_MISSING_COMMITMENT
    if reveal is None:
        return False, STATUS_MISSING_REVEAL
    if not _well_formed(reveal, len(candidates)):
        return False, STATUS_MALFORMED
    expected = make_commitment(reveal["votes"], round_num, validator,
                               candidates, reveal["nonce"], context)
    if not hmac.compare_digest(expected, str(commitment)):
        return False, STATUS_MISMATCH
    return True, STATUS_VALID


def verify_round(commitments, reveals, validators, round_num, context):
    """Check every validator.  Returns (status_by_validator, valid_validators).

    valid_validators keeps the order of ``validators`` so the vote matrix rows
    are in a fixed, reproducible order.
    """
    status, valid = {}, []
    for validator in validators:
        ok, reason = verify_reveal(
            commitments.get(validator), reveals.get(validator),
            round_num, validator, list(validators), context,
        )
        status[validator] = reason
        if ok:
            valid.append(validator)
    return status, valid


def tamper_reveal(reveals, validator):
    """Test helper: flip one revealed vote WITHOUT updating the commitment."""
    out = copy.deepcopy(reveals)
    votes = list(out[validator]["votes"])
    votes[0] = 1 - votes[0]
    out[validator]["votes"] = votes
    return out
