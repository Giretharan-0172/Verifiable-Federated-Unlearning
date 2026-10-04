# CrowdGuard — Version 1 Experimental Results

This document records the results of the completed **CrowdGuard v1 reference experiment**.

These results are preserved as a baseline for comparison with the updated v2 implementation. **The v2 experiment has not yet been executed**, so the results below must not be interpreted as v2 results.

---

## 1. Experiment Overview

The v1 experiment evaluated the CrowdGuard detection pipeline in a simulated federated learning environment using CIFAR-10.

### Configuration

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Number of clients | 20 |
| FL rounds | 10 |
| Local epochs | 10 |
| Batch size | 64 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Partial Model Replacement (PMR) | 0.05 |
| Poison rate | 0.10 |
| Attack alpha | 0.70 |
| Trigger size | 6 × 6 |
| Random seed | 10 |

The experiment used one malicious client, `malicious_00`, together with 19 benign clients.

---

## 2. Runtime

The complete v1 experiment required:

**15,246.08 seconds**

which is approximately:

**4 hours 14 minutes**

The run completed successfully.

---

## 3. CrowdGuard Detection Results

The following shows which clients were included in the detected set during each FL round.

```text
Round 0: benign_00

Round 1: benign_04, malicious_00

Round 2: malicious_00

Round 3: malicious_00

Round 4: malicious_00

Round 5: benign_06, malicious_00

Round 6: malicious_00

Round 7: malicious_00

Round 8: benign_11, malicious_00

Round 9: malicious_00
```

### Detection Frequency

| Client | Detection frequency |
|---|---:|
| `malicious_00` | 9/10 rounds |
| `benign_00` | 1/10 rounds |
| `benign_04` | 1/10 rounds |
| `benign_06` | 1/10 rounds |
| `benign_11` | 1/10 rounds |

The malicious client was therefore detected in **9 out of 10 rounds**, while each false-positive benign client was detected only once.

---

## 4. Final Detected Set in v1

The final v1 run produced:

```text
M = [
    'benign_00',
    'benign_04',
    'benign_06',
    'benign_11',
    'malicious_00'
]
```

The v1 implementation formed the final set using the **union of clients detected across rounds**.

Therefore, a client only needed to be detected in a single round to appear in the final set.

---

## 5. Interpretation

The v1 experiment shows that CrowdGuard consistently identified the malicious client:

- `malicious_00` was detected in **9/10 rounds**.
- The malicious client was detected in every round except Round 0.
- Four benign clients were detected once each:
  - `benign_00`
  - `benign_04`
  - `benign_06`
  - `benign_11`

This indicates substantially stronger persistence of the malicious client's detection compared with the benign false positives.

However, because v1 used a union rule for the final set, the one-off benign detections were also included in the final set `M`.

---

## 6. Important Round-9 Detail

In the Round 9 console output, the following appeared:

```text
Round 9: benign_12 detected indices [1,16,19]
```

This does **not** mean that `benign_12` itself was detected.

Here, `benign_12` was the validator, and it reported candidate model/client indices:

```text
[1, 16, 19]
```

The candidate at index `19` corresponded to `malicious_00`.

Therefore, this output should be interpreted as a validator vote rather than a detection of `benign_12`.

---

## 7. Changes Introduced in Version 2

The v2 implementation changes how the final malicious set is selected.

Instead of using the v1 union rule, v2 supports a **frequency-based final-M rule**.

The default configuration is:

```text
final_M_rule = "frequency"
final_M_tau = 0.5
```

Under this rule, a client is included in the final set only if it is detected in at least:

```text
tau × number_of_rounds
```

For a 10-round experiment with:

```text
tau = 0.5
```

a client must be detected in at least **5 rounds**.

This is intended to distinguish persistent detections from isolated false positives.

The union rule remains available as an alternative for comparison.

---

## 8. Version 1 vs Version 2

| Component | v1 | v2 |
|---|---|---|
| Final-M selection | Union of detected clients | Frequency-based by default |
| Default threshold | None | `tau = 0.5` |
| Commit-reveal | Not present | Added |
| Vote integrity | Direct vote collection | Two-phase commit-reveal |
| Test-data split | Previous setup | 2,500 KD + 7,500 held-out evaluation |
| Handoff validation | Basic | Dedicated validation tooling |
| Crash recovery | Limited | Per-round checkpoints |
| Reproducibility metadata | Limited | Expanded metadata and source hashes |

---

## 9. Status of These Results

These results correspond to the **completed v1 experiment**.

They should be treated as a **reference/baseline experiment**, not as validation of the v2 implementation.

At the time of writing:

- v1 full experiment: **completed**
- v1 results: **recorded above**
- v2 implementation: **prepared**
- v2 automated tests: **completed**
- v2 full 20-client experiment: **not yet run**
- v2 experimental results: **not yet available**

The v2 results should be added separately after the updated full experiment is executed.

---

## 10. Archived Run

The original v1 completed run is retained locally for reproducibility/reference.

The large completed-run archive is intentionally **not committed to GitHub**, because GitHub imposes a 100 MB per-file limit.

The Markdown results in this file provide the important experimental record without requiring the large archive to be stored in the repository.
