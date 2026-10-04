# CrowdGuard — Federated Backdoor Detection and Handoff

## 1. Project Overview

This repository implements the **CrowdGuard detection stage** of a larger federated-learning backdoor-defense pipeline.

The overall project is:

```text
Federated Training
        │
        ▼
CrowdGuard Detection
        │
        ▼
Detected Malicious Clients M
        │
        ▼
Wu Historical Unlearning
        │
        ▼
Wu Knowledge-Distillation Repair
        │
        ▼
BAERASER-Inspired Trigger Recovery
        │
        ▼
Post-Unlearning Verification
        │
        ▼
Commit–Reveal Audit
```

The current repository focuses on:

- CIFAR-10 federated training
- CrowdGuard HLBIM-style detection
- stacked-clustering vote aggregation
- detection-only federated aggregation
- historical client-update capture
- temporal malicious-client selection
- commit–reveal vote verification
- structured handoff generation
- handoff validation and evaluation

The downstream Wu and BAERASER-inspired stages use the CrowdGuard handoff as their input.

This is a reproduction/extension project. It does **not** claim to reproduce every component of the original CrowdGuard, Wu, or BAERASER implementations exactly.

---

# 2. Project Structure and Methodological Scope

There are two main implementation responsibilities in the overall project.

### CrowdGuard

Responsible for:

- federated training
- poisoning experiment
- HLBIM-style detection
- stacked clustering
- detection history
- malicious-client selection
- historical-update capture
- handoff generation

### Downstream Wu / BAERASER stages

Responsible for:

- historical malicious-update subtraction
- knowledge-distillation repair
- trigger recovery
- post-unlearning verification

Wu is an **integration stage**, not a separate project pair.

BAERASER is being used as the basis for the trigger-recovery component, with project-specific modifications.

---

# 3. What This Repository Implements

The CrowdGuard detection pipeline is:

```text
Client Models
      │
      ▼
Hidden-Layer Activations
      │
      ▼
Distance Computation
      │
      ▼
Relative Distances
      │
      ▼
PCA
      │
      ▼
Significance / Iterative Pruning
      │
      ▼
Binary Validator Votes
      │
      ▼
Stacked Clustering
      │
      ▼
Detected Clients M_t
```

The implementation retains the HLBIM-style detection mechanism and CrowdGuard's stacked-clustering approach.

The stacked-clustering logic is centralized in:

```text
stacked_clustering_vote()
```

so the same procedure can be reused when validating the recorded results.

---

# 4. Detection-Only Modification

The original CrowdGuard system uses its detection result to filter suspicious models before aggregation.

This project intentionally uses **detection-only mode**.

All participating client models continue into FedAvg:

```text
20 Client Models
      │
      ├──────────────► CrowdGuard Detection
      │                       │
      │                       ▼
      │                    M_t
      │
      └──────────────────► Equal-Weight FedAvg
```

Therefore:

> CrowdGuard identifies suspicious clients, but detected clients are not immediately removed from aggregation.

This is a project-level modification and should not be described as the original CrowdGuard aggregation procedure.

The reason is that the later Wu stage needs access to the historical contributions of the clients identified as malicious.

---

# 5. Experimental Configuration

The main experiments use CIFAR-10.

| Setting | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Training samples | 50,000 |
| Test samples | 10,000 |
| Clients | 20 |
| Samples/client | 2,500 |
| Data partition | IID |
| Client participation | All clients every round |
| Aggregation | Equal-weight FedAvg |
| FL rounds | 10 |
| Local epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Model | Lightweight ResNet-18-style |
| Trigger | 6 × 6 RGB |
| Poison rate | 0.10 |
| Attack alpha | 0.70 |
| Primary PMR | 0.05 |
| Additional PMR | 0.10 |
| Stress PMR | 0.45 |
| Seed | 10 |

The 50,000 training samples are partitioned as:

```text
20 clients × 2,500 samples = 50,000 samples
```

The experiment configuration is defined in:

```text
experiment_config.py
```

---

# 6. Client Participation

All 20 clients participate in every FL round.

There is no client dropout or random client-selection mechanism in the current experiment.

```text
Client 0  ─┐
Client 1  ─┤
Client 2  ─┤
...        ├── every round
Client 19 ─┘
```

This is an explicit project-level threat-model boundary.

It also allows the historical-update stage to retain one contribution from every client for every round.

---

# 7. Model

The project uses a lightweight ResNet-18-style model suitable for CIFAR-10.

Current parameter count:

```text
275,730 parameters
```

The model exposes internal states required by CrowdGuard's HLBIM computation.

Therefore it supports:

1. normal CIFAR-10 classification
2. hidden-layer/internal-state extraction for CrowdGuard

The architecture is project-specific and should be described as a **lightweight ResNet-18-style model**, not as an exact reproduction of a paper architecture.

---

# 8. Backdoor Attack

The malicious clients use an adaptive constrain-and-scale backdoor attack.

The implementation includes:

- clean classification objective
- anomaly/constrain objective
- adaptive optimization
- trigger injection
- malicious-update scaling

The attack configuration uses:

```text
alpha = 0.70
```

The general attack objective follows the constrain-and-scale formulation:

```text
L = alpha * L_class + (1 - alpha) * L_ano
```

The anomaly component and implementation details are project-level choices and should not be described as an exact reproduction of a specific anomaly-loss implementation from the reference attack paper.

---

# 9. Trigger

The project uses a:

```text
6 × 6 RGB pixel trigger
```

One trigger is shared by malicious clients during an attack run.

The trigger size is fixed, while the following can vary between attack runs:

```text
Trigger position
RGB pattern
Target label
```

The verifier therefore knows:

```text
Trigger size = 6 × 6
Dataset = CIFAR-10
```

but does not receive the:

```text
Position
RGB pattern
Target label
```

The trigger configuration is therefore hidden from the recovery procedure.

---

# 10. Malicious Client Configuration

For 20 clients:

```text
PMR = 0.05
→ approximately 1 malicious client

PMR = 0.10
→ approximately 2 malicious clients

PMR = 0.45
→ 9 malicious clients
```

The primary experiments use PMR 0.05 and 0.10.

PMR 0.45 is used as a higher-adversary stress condition.

The malicious update is scaled approximately according to:

```text
number of clients / number of malicious clients
```

Therefore attack strength changes with PMR.

---

# 11. CrowdGuard Detection

CrowdGuard detection is based on the HLBIM-style hidden-layer analysis.

The general procedure is:

```text
Local Models
    │
    ▼
Hidden-Layer Outputs
    │
    ▼
Distance Measurements
    │
    ▼
Relative Distances
    │
    ▼
PCA
    │
    ▼
Significance Testing
    │
    ▼
Iterative Pruning
    │
    ▼
Validator Vote
```

Each validator produces binary decisions over candidate models.

The validator votes are then processed by CrowdGuard's stacked-clustering procedure.

The output for round `t` is:

```text
M_t
```

where `M_t` contains the clients detected during that round.

---

# 12. Stacked Clustering

The within-round voting procedure follows the CrowdGuard stacked-clustering approach.

Conceptually:

```text
Validator Votes
      │
      ▼
Agglomerative Clustering
      │
      ▼
Largest / Representative Cluster
      │
      ▼
DBSCAN
      │
      ▼
Final Vote Interpretation
```

The implementation is centralized in:

```text
stacked_clustering_vote()
```

A fallback is available for cases where there are too few valid validators to perform the normal procedure.

The same implementation is used when auditing the recorded votes.

---

# 13. Detection-Only Aggregation

For each round:

```text
All client models
       │
       ├── CrowdGuard
       │       │
       │       └── detected clients M_t
       │
       └── Equal-weight FedAvg
               │
               ▼
         Global model
```

Detected clients are therefore **recorded but not filtered**.

This ensures the final global model remains representative of the poisoned FL state for the downstream unlearning experiment.

---

# 14. Historical Update Capture

For every FL round, the system stores the client updates associated with:

```text
Round
Client ID
Global model before the round
Client update
Attack-active status
```

Conceptually:

```text
Δ_i^t
```

represents client `i`'s update in round `t`.

The historical updates are required by the downstream Wu stage.

The later unlearning process can therefore identify the historical contributions associated with the clients in the final malicious set.

---

# 15. Detection History and Final M

The system stores detection results for every round:

```text
M_0
M_1
M_2
...
M_9
```

The final malicious set is:

```text
M
```

The current v2 implementation uses a **frequency/persistence rule** by default.

```text
final_M_rule = "frequency"
final_M_tau  = 0.5
```

A client is included in `M` if it is detected in at least:

```text
tau × number_of_rounds
```

For 10 rounds and:

```text
tau = 0.5
```

a client must be detected in at least 5 rounds.

The original union rule is still available:

```text
final_M_rule = "union"
```

where:

```text
M = union of all M_t
```

The intended design is therefore:

```text
Within each round
    ↓
Stacked clustering
    ↓
M_t

Across rounds
    ↓
Detection frequency / persistence
    ↓
Final M
```

A second stacked-clustering stage across rounds is intentionally not used.

---

# 16. Ground Truth

The actual malicious-client identities are stored separately from CrowdGuard's detections.

This distinction is important:

```text
Ground-truth malicious clients
            ≠
CrowdGuard detected clients
```

Ground truth is used only for offline evaluation.

It is not provided to CrowdGuard during detection.

---

# 17. Test-Set Split

The CIFAR-10 test set contains 10,000 images.

The v2 implementation creates a deterministic stratified split:

```text
2,500 images → KD reference set
7,500 images → held-out evaluation set
```

The split is stratified so that the class distribution is preserved.

The indices are stored in the handoff.

The KD reference set is reserved for the downstream Wu knowledge-distillation stage.

The held-out set is used for evaluation.

---

# 18. ASR Evaluation

Backdoor Attack Success Rate (ASR) is evaluated using held-out evaluation images.

Images whose true class is already equal to the target class are excluded.

This avoids artificially inflating ASR.

```text
Held-out evaluation images
          │
          ▼
Remove true-target images
          │
          ▼
Apply trigger
          │
          ▼
Evaluate target-class predictions
```

Clean accuracy is evaluated separately on clean images.

---

# 19. Commit–Reveal Protocol

The v2 implementation includes a two-phase commit–reveal protocol for validator votes.

The flow is:

```text
local_validation
        ↓
collect_commitments
        ↓
reveal_votes
        ↓
defend
```

During `local_validation`, each validator:

1. computes its CrowdGuard vote vector
2. generates a random nonce
3. creates a SHA-256 commitment
4. stores the vote and nonce privately
5. sends only the commitment to the aggregator

The commitment binds:

```text
scheme
context
round
validator
candidate order
votes
nonce
```

---

# 20. Commit Phase

Conceptually:

```text
Votes + nonce
     │
     ▼
SHA-256
     │
     ▼
Commitment
```

The aggregator collects all commitments and records the commitment list and digest before the reveal phase.

The commitment digest binds the set of commitments associated with that round.

---

# 21. Reveal Phase

During the reveal phase, each validator:

1. retrieves its private vote and nonce
2. verifies that its published commitment has not changed
3. reveals only if the commitment still matches
4. submits its vote vector and nonce

The aggregator verifies the reveal against the original commitment.

Possible verification statuses are:

```text
valid
missing_commitment
missing_reveal
malformed_reveal
hash_mismatch
```

Only valid reveals are used for the final stacked-clustering vote.

Per-round commit–reveal information is saved in the handoff.

---

# 22. Single-Process Simulation

The current commit–reveal implementation is a **simulation of the distributed protocol**.

In a real deployment, validators and the aggregator would be independent processes or machines:

```text
Validator 1 ──┐
Validator 2 ──┤
Validator 3 ──┼── Network ──► Aggregator
...            │
Validator 20 ─┘
```

In the current implementation, these roles are simulated inside one Python/OpenFL execution:

```text
One Python/OpenFL process
 ├── Validator state
 ├── Private vote vaults
 ├── Aggregator state
 └── Commit/reveal flow
```

The protocol ordering is still enforced:

```text
Validate
   ↓
Commit
   ↓
Commitments fixed
   ↓
Reveal
   ↓
Verify
   ↓
Tally
```

However, this does **not** test:

- real network communication
- independent machine/process isolation
- network-level adversarial timing
- malicious network participants

Therefore:

> The implementation demonstrates the commit–reveal protocol logic and verification sequence, but it is not a complete distributed security deployment.

---

# 23. Commit–Reveal vs TEE

The project does not implement the original CrowdGuard TEE assumption.

Commit–reveal provides integrity/auditability for the vote commitment and reveal process.

It does **not** provide all guarantees of a TEE.

In particular, it does not guarantee:

- model confidentiality
- code attestation
- validator honesty
- aggregator honesty
- protection against a compromised execution environment

Therefore commit–reveal should not be described as a complete replacement for all TEE functionality.

---

# 24. OpenFL Checkpointing

The workflow should keep:

```text
checkpoint=False
```

Checkpointing is disabled because collaborator state, including private vote-vault information, should not be serialized as part of the OpenFL checkpoint.

Collaborator state can persist across rounds, so reveal-related state is explicitly reset where required.

Raw detection console logs may reveal detection/voting information. These logs are useful during development but should be removed or restricted for strict privacy-preserving experiments.

---

# 25. Reproducibility Metadata

The v2 handoff stores reproducibility information including:

- random seed
- per-client training indices
- KD/evaluation test indices
- git commit when available
- Python/library versions
- SHA-256 hashes of source files
- experiment configuration

The same seed produces the same deterministic client partition and trigger configuration.

Use different seeds when independent experimental realizations are required.

---

# 26. Crash Insurance

Each completed FL round is saved independently:

```text
outputs/
└── rounds/
    ├── round_00.pt
    ├── round_01.pt
    ├── ...
    └── round_09.pt
```

The final handoff is written atomically.

These round files provide recovery/salvage artifacts.

They do **not** currently provide automatic experiment resumption.

---

# 27. Handoff

The v2 handoff uses:

```text
schema_version = 2
```

Important fields include:

```text
schema_version
final_M_rule
detection_counts
client_train_indices
test_split
environment
commit_reveal
```

The handoff also stores the main model and experiment state:

```text
theta_0
theta_K
round_history
final_M
ground_truth_malicious_clients
configuration
trigger information
parameter count
elapsed time
historical client updates
```

The handoff is the main interface between CrowdGuard and the downstream Wu stage.

---

# 28. Wu Historical Unlearning

The downstream Wu stage uses the CrowdGuard handoff to identify and remove historical malicious contributions.

Conceptually:

```text
CrowdGuard Handoff
        │
        ▼
Final malicious set M
        │
        ▼
Historical malicious updates
        │
        ▼
Historical subtraction
        │
        ▼
Unlearned model
```

CrowdGuard itself does not perform the historical subtraction.

---

# 29. Wu Knowledge Distillation

The downstream repair stage uses:

```text
Teacher:
pre-unlearning global model

Student:
historical-subtracted model

Reference data:
2,500-image KD reference set
```

The purpose is to recover useful clean-model behavior after historical subtraction.

The KD reference indices are stored in the CrowdGuard handoff so that the downstream stage can reconstruct the same split.

---

# 30. BAERASER-Inspired Trigger Recovery

The project uses a trigger-recovery procedure inspired by BAERASER.

This is **not claimed to be a complete reproduction of BAERASER**.

The current trigger geometry is fixed to:

```text
6 × 6
```

For a 32 × 32 CIFAR-10 image:

```text
(32 - 6 + 1)^2 = 729 possible positions
```

With 10 possible target labels:

```text
729 × 10 = 7,290 position/target combinations
```

The trigger pattern itself is optimized as a:

```text
6 × 6 × 3 RGB tensor
```

The position, RGB pattern, and target label are hidden from the verifier.

---

# 31. Evaluation Metrics

## Detection

- true-positive rate
- true-negative rate
- false-positive rate
- false-negative rate
- precision
- recall
- F1
- detection persistence
- final-M precision/recall

## Classification

- clean accuracy
- clean loss
- macro-F1

## Backdoor

- pre-unlearning ASR
- post-unlearning ASR
- recovered-trigger ASR

## Trigger Recovery

- recovery success
- position error
- pattern similarity
- target-label recovery
- recovery runtime

## Commit–Reveal

- verifier agreement
- commitment consistency
- valid reveal count
- invalid reveal count
- tally consistency

## Efficiency

- training time
- detection runtime
- recovery runtime
- total runtime
- memory usage where relevant

---

# 32. Experimental Comparisons

The intended evaluation includes:

```text
1. Clean-model recovery baseline
2. No defense
3. CrowdGuard only
4. Wu only
5. CrowdGuard + Wu
6. Full pipeline
7. Known trigger
8. Recovered trigger
```

Additional analysis can vary:

```text
PMR
final-M tau
Dirichlet alpha
```

where applicable.

---

# 33. PMR Experiments

The main PMR configurations are:

```text
PMR = 0.05
PMR = 0.10
PMR = 0.45
```

The number of malicious clients and the malicious-update scaling both change with PMR.

Therefore PMR experiments should be interpreted as different attack-strength/adversary settings rather than simply different percentages.

---

# 34. Environment

The successful full experiment used:

```text
OS:              Ubuntu 22.04
Python:          3.10.21
uv:              0.12.9

OpenFL:          1.9
Metaflow:        2.7.15
setuptools:      80.10.2

PyTorch:         2.14.1+cu130
torchvision:     0.29.1
NumPy:           2.2.6
SciPy:           1.15.3
scikit-learn:    1.6.1
pandas:          2.3.3
matplotlib:      3.10.9

ray:             2.59.0
dill:            0.4.1
tabulate:         0.10.0
nbformat:        5.11.1
nbdev:           3.3.24
```

The final successful environment used:

```text
CUDA-enabled PyTorch 2.14.1+cu130
```

The exact GPU depends on the execution platform.

---

# 35. Installation

## 35.1 Recommended Kaggle Setup

The Kaggle environment should use Python 3.10.21 rather than the default Python 3.12 environment.

Install `uv`:

```bash
python -m pip install "uv==0.12.9"
```

Install Python 3.10.21:

```bash
uv python install 3.10.21
```

Create the environment:

```bash
uv venv /kaggle/working/crowdguard-env --python 3.10.21
```

Install the exact tested versions:

```bash
uv pip install \
    --python /kaggle/working/crowdguard-env/bin/python \
    "openfl==1.9" \
    "metaflow==2.7.15" \
    "setuptools==80.10.2" \
    "torch==2.14.1" \
    "torchvision==0.29.1" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "scikit-learn==1.6.1" \
    "pandas==2.3.3" \
    "matplotlib==3.10.9" \
    "ray==2.59.0" \
    "dill==0.4.1" \
    "tabulate==0.10.0" \
    "nbformat==5.11.1" \
    "nbdev==3.3.24"
```

Verify Python:

```bash
/kaggle/working/crowdguard-env/bin/python --version
```

Expected:

```text
Python 3.10.21
```

---

# 36. Verify Installation

Check the major versions:

```bash
/kaggle/working/crowdguard-env/bin/python -c "
import sys, torch, openfl, metaflow, numpy, scipy, sklearn, pandas
print('Python:', sys.version.split()[0])
print('OpenFL:', openfl.__version__)
print('Metaflow:', metaflow.__version__)
print('PyTorch:', torch.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('sklearn:', sklearn.__version__)
print('pandas:', pandas.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

Expected core versions:

```text
Python: 3.10.21
OpenFL: 1.9
Metaflow: 2.7.15
PyTorch: 2.14.1+cu130
NumPy: 2.2.6
SciPy: 1.15.3
sklearn: 1.6.1
pandas: 2.3.3
```

Verify OpenFL Workflow:

```bash
/kaggle/working/crowdguard-env/bin/python -c "
from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
print('OpenFL Workflow API OK')
"
```

Expected:

```text
OpenFL Workflow API OK
```

Verify CUDA:

```bash
nvidia-smi
```

---

# 37. Kaggle Environment Variables

Metaflow requires the user identity to be defined in the Kaggle environment:

```python
import os

os.environ["USERNAME"] = "kaggle"
os.environ["USER"] = "kaggle"

print(os.environ["USERNAME"])
print(os.environ["USER"])
```

---

# 38. Local Linux Setup

For a local Ubuntu environment:

```bash
cd /path/to/CrowdGuard

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
```

For the closest match to the tested environment, install the same pinned package versions listed above.

Then verify:

```bash
python --version
python -c "import openfl; print(openfl.__version__)"
```

Expected:

```text
Python 3.10.x
1.9
```

For CUDA-enabled local execution, verify:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 39. Smoke Tests

Run the utility tests:

```bash
python test_handoff_utils.py
```

Run handoff validation tests:

```bash
python test_validate_handoff.py
```

Run the two-phase-flow tests:

```bash
python test_two_phase_flow.py
```

The two-phase-flow test requires OpenFL.

Run the original component smoke tests:

```bash
python local_smoke_test.py
```

and:

```bash
python local_pipeline_smoke.py
```

The local smoke tests validate the core model, trigger, attack, detection, historical-update capture, and detection-only handoff without requiring a full 20-client experiment.

---

# 40. Tamper Test

The commit–reveal implementation provides an intentional tampering test.

Run:

```bash
python cifar10_crowdguard.py \
    --num_clients 5 \
    --samples_per_client 200 \
    --local_epochs 1 \
    --rounds 3 \
    --pmr 0.2 \
    --tamper_test \
    --output_dir outputs/smoke_tamper
```

Validate the resulting handoff:

```bash
python validate_handoff.py \
    outputs/smoke_tamper/crowdguard_handoff.pt
```

The tamper test is intentionally expected to produce verification warnings.

**Do not use tamper-test output as an experimental result.**

---

# 41. Main Experiment

The main entry point is:

```bash
python cifar10_crowdguard.py
```

The workflow contains the relevant stages:

```text
start
  ↓
train
  ↓
collect_models
  ↓
local_validation
  ↓
collect_commitments
  ↓
reveal_votes
  ↓
defend
  ↓
end
```

The default experiment uses:

```text
20 clients
10 rounds
10 local epochs
```

A full run can take several hours depending on the hardware.

---

# 42. PMR Runs

Run PMR 0.05:

```bash
python cifar10_crowdguard.py \
    --pmr 0.05 \
    --output_dir outputs/pmr005
```

Run PMR 0.10:

```bash
python cifar10_crowdguard.py \
    --pmr 0.10 \
    --output_dir outputs/pmr010
```

Run PMR 0.45:

```bash
python cifar10_crowdguard.py \
    --pmr 0.45 \
    --output_dir outputs/pmr045
```

---

# 43. Validate a Handoff

After a run:

```bash
python validate_handoff.py \
    outputs/pmr010/crowdguard_handoff.pt
```

The validator checks the schema, required fields, detection history, commit–reveal records, metadata, and consistency constraints.

---

# 44. Evaluate a Handoff

Run:

```bash
python evaluate_handoff.py \
    outputs/pmr010/crowdguard_handoff.pt
```

This provides offline analysis of the recorded detection/handoff information without retraining the federated model.

This is useful for testing different final-M thresholds and detection-history interpretations without paying the cost of another FL run.

---

# 45. Recommended Execution Order

```text
1. Create environment
        ↓
2. Install pinned dependencies
        ↓
3. Verify OpenFL / PyTorch / CUDA
        ↓
4. Run unit tests
        ↓
5. Run local smoke tests
        ↓
6. Run commit–reveal tamper smoke test
        ↓
7. Validate smoke handoff
        ↓
8. Run fresh 20-client CrowdGuard experiment
        ↓
9. Validate real handoff
        ↓
10. Evaluate real handoff
        ↓
11. Run PMR = 0.10
        ↓
12. Run PMR = 0.45
        ↓
13. Perform offline final-M / tau sensitivity
        ↓
14. Pass handoff to Wu
        ↓
15. Wu historical unlearning
        ↓
16. Wu knowledge-distillation repair
        ↓
17. BAERASER-inspired trigger recovery
        ↓
18. Final verification
```

---

# 46. Reproducibility

For every experimental run, record:

```text
Dataset
Number of clients
Samples/client
Client partition
Random seed
PMR
Poison rate
Malicious client IDs
Trigger size
Trigger position
Trigger RGB pattern
Target label
Local epochs
Batch size
Learning rate
Momentum
Attack alpha
Number of FL rounds
Final-M rule
Final-M tau
Detected clients per round
Final M
```

The v2 handoff automatically records much of this information.

---

# 47. Known Limitations

## Single-Process Commit–Reveal

The protocol is simulated within one Python/OpenFL execution.

It does not establish security against real network-level adversaries.

## No TEE

The project does not reproduce the original TEE execution environment.

Commit–reveal provides vote commitment/reveal integrity but not the confidentiality or attestation guarantees of a TEE.

## Detection-Only CrowdGuard

Detected clients remain in FedAvg.

This is intentional and required for the historical-unlearning experiment, but it differs from the original CrowdGuard filtering behavior.

## Final-M Sensitivity

The final malicious set depends on the selected temporal aggregation rule and threshold.

The frequency rule is currently the default, but tau should be evaluated experimentally.

## PMR-Dependent Attack Scaling

The malicious update scale changes with the number of malicious clients.

Therefore PMR changes both adversary prevalence and effective attack scaling.

## OpenFL State

Collaborator state persists across rounds.

Checkpointing remains disabled to avoid serializing private protocol state.

## Logging

Raw detection logs may reveal validator detections and should be cleaned for strict privacy-preserving experiments.

## Same Seed

The same seed reproduces the same deterministic partition and attack configuration.

Independent experiments should use different seeds.

## No Automatic Resume

Round files provide crash insurance but do not implement automatic experiment resumption.

## Trigger Recovery

Failure to recover a trigger does not formally prove that a model is backdoor-free.

The correct interpretation is:

> No successful trigger recovery was found under the specified recovery procedure and search space.

---

# 48. Methodological Boundaries

The main components have distinct roles:

```text
CrowdGuard
→ hidden-layer anomaly detection

Stacked Clustering
→ within-round validator-vote aggregation

Temporal Frequency
→ across-round malicious-client selection

Wu
→ historical unlearning + knowledge-distillation repair

BAERASER-inspired recovery
→ trigger recovery / verification

Commit–Reveal
→ vote commitment and reveal integrity
```

The project does not claim:

- exact reproduction of the original CrowdGuard model architecture
- complete TEE functionality
- real multi-machine secure commit–reveal execution
- complete BAERASER reproduction
- formal proof that a failed trigger search means no backdoor exists

---

# 49. Current Implementation Status

## Implemented

- [x] CIFAR-10 federated training
- [x] 20-client configuration
- [x] deterministic IID client partition
- [x] lightweight ResNet-18-style model
- [x] internal-state extraction
- [x] adaptive constrain-and-scale attack
- [x] 6 × 6 RGB trigger
- [x] HLBIM-style CrowdGuard detection
- [x] iterative pruning
- [x] stacked clustering
- [x] detection-only aggregation
- [x] per-round detection history
- [x] historical client-update capture
- [x] frequency-based final-M rule
- [x] configurable final-M threshold
- [x] deterministic KD/evaluation test split
- [x] corrected held-out ASR evaluation
- [x] two-phase commit–reveal
- [x] commitment verification
- [x] reveal verification
- [x] per-round commit–reveal records
- [x] per-round output files
- [x] atomic final handoff
- [x] reproducibility metadata
- [x] handoff validation
- [x] handoff evaluation
- [x] unit tests
- [x] smoke tests
- [x] tamper testing

## Remaining downstream work

- [ ] Full Wu historical-unlearning integration
- [ ] Wu knowledge-distillation repair
- [ ] BAERASER-inspired trigger recovery integration
- [ ] Post-unlearning verification
- [ ] Full cross-stage pipeline evaluation
- [ ] Independent PMR/seed repetitions

---

# 50. Repository Structure

```text
CrowdGuard/
│
├── Core implementation
│   ├── cifar10_crowdguard.py
│   ├── CrowdGuardClientValidation.py
│   ├── adaptive_backdoor.py
│   ├── lightweight_resnet18.py
│   ├── experiment_config.py
│   ├── commit_reveal.py
│   └── handoff_utils.py
│
├── Validation / evaluation
│   ├── validate_handoff.py
│   ├── evaluate_handoff.py
│   ├── test_handoff_utils.py
│   ├── test_two_phase_flow.py
│   ├── test_validate_handoff.py
│   ├── local_smoke_test.py
│   └── local_pipeline_smoke.py
│
├── Handoff / previous experimental artifacts
│   └── handoff_files/
│
├── Original OpenFL deployment examples
│   ├── Amsterdam/
│   ├── Bangalore/
│   ├── Chandler/
│   ├── Detroit/
│   └── director/
│
├── Reference / v1 notebooks
│   ├── FederatedCrowdGuard.ipynb
│   ├── PoisoningAttackDemo.ipynb
│   └── PoisoningAttackDemoReduced.ipynb
│
├── README.md
└── .gitignore
```

The `Amsterdam`, `Bangalore`, `Chandler`, `Detroit`, and `director` directories are retained from the original OpenFL setup and are not required to conceptually define the 20-client experiment.

`handoff_files/` contains previous/output handoff artifacts and is not required for a fresh run.

The existing notebooks are retained as v1/reference material. The updated v2 Kaggle notebook is the reproducibility/run notebook for the current implementation.

---

# 51. One-Line Description

> A detection-first federated-learning backdoor-defense pipeline combining CrowdGuard hidden-layer anomaly detection, stacked-clustering voting, temporal malicious-client identification, historical-update capture, Wu-style unlearning, BAERASER-inspired trigger recovery, and commit–reveal auditing.
