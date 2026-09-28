# CrowdGuard — Federated Backdoor Detection and Handoff

## 1. Project Overview

This repository implements the **CrowdGuard stage** of the larger federated-learning pipeline.

The overall project pipeline is:

```text
Federated Training
       │
       ▼
CrowdGuard Detection
       │
       ▼
Estimated Malicious Client Set M
       │
       ▼
Wu Historical-Update Subtraction + Knowledge Distillation
       │
       ▼
BAERASER-Inspired Trigger Recovery
       │
       ▼
Post-Unlearning Verification
       │
       ▼
Commit–Reveal Audit / Voting
```

This repository is responsible for the **CrowdGuard detection and historical-update handoff** portion.

### Team structure

There are **two pairs** in the project:

* **Pair 1 — CrowdGuard:** detection, poisoning experiment, historical-update capture, and handoff.
* **Pair 2 — BAERASER:** trigger recovery and verification, constructed independently on centralized ML.

**Wu is not a separate pair.** Wu's historical-update subtraction and knowledge-distillation stage is an integration stage in the overall pipeline.

---

# 2. What This Repository Implements

The implementation retains the core CrowdGuard detection mechanism:

```text
Local model updates
       │
       ▼
HLBIM computation
       │
       ▼
Significance testing / votes
       │
       ▼
Stacked clustering
       │
       ▼
Detected clients M_t
```

The project modifies the original CrowdGuard pipeline in one important way.

### Original CrowdGuard

CrowdGuard detects suspicious clients and removes their models before aggregation.

### This project

CrowdGuard is used in **detection-only mode**.

All 20 client updates continue into equal-weight FedAvg:

```text
20 client updates
      │
      ├── CrowdGuard detection
      │       └── produces M_t
      │
      └── all 20 updates
              │
              ▼
         Equal-weight FedAvg
```

The detected clients are **not immediately filtered from aggregation**.

Their identities are instead passed to the later Wu stage, which uses the historical updates to remove their past contributions.

This is a **project-level modification** and should not be described as the original CrowdGuard aggregation procedure.

---

# 3. Experimental Configuration

The current project configuration is based on CIFAR-10.

| Setting                   |                             Value |
| ------------------------- | --------------------------------: |
| Dataset                   |                          CIFAR-10 |
| Training samples          |                            50,000 |
| Test samples              |                            10,000 |
| Number of logical clients |                                20 |
| Samples per client        |                             2,500 |
| Data partition            |                               IID |
| Aggregation               |               Equal-weight FedAvg |
| Local epochs              |                                10 |
| Batch size                |                                64 |
| Learning rate             |                              0.01 |
| Model                     | Lightweight ResNet-18-style model |
| Trigger                   |                         6 × 6 RGB |
| Primary PMR               |                        0.05, 0.10 |
| Stress PMR                |                              0.45 |

The 50,000 CIFAR-10 training samples are partitioned as:

```text
20 clients × 2,500 samples = 50,000 samples
```

The 10,000-image CIFAR-10 test set is reserved for downstream evaluation and verification.

---

# 4. Client Participation

All 20 clients participate in **every FL round**.

This is a deliberate project-level threat-model boundary.

There is no client selection/dropout mechanism in the experimental setup.

Therefore, the system maintains:

```text
Client 0  ─┐
Client 1  ─┤
Client 2  ─┤
...        ├── every round
Client 19 ─┘
```

This is also important for the historical-update stage because every client's contribution is retained for every round.

---

# 5. Model

The implementation uses a lightweight **ResNet-18-style model suitable for CIFAR-10**.

The current implementation has approximately:

```text
275,730 parameters
```

The model also exposes internal states required by CrowdGuard's HLBIM computation.

The model therefore supports both:

1. normal classification, and
2. internal-state extraction for CrowdGuard validation.

The exact architecture should be described as the project's lightweight ResNet-18-style implementation rather than claiming that it is an exact reconstruction of the lightweight ResNet-18 architecture used in the Bagdasaryan experiments.

---

# 6. Backdoor Attack

The poisoning component implements an adaptive constrain-and-scale attack.

The attack contains:

* clean classification objective,
* anomaly/constrain objective,
* adaptive optimization,
* trigger injection,
* scaling of the submitted model update.

The project uses:

```text
alpha = 0.7
```

for the adaptive attack configuration.

The attack is based on the general constrain-and-scale formulation:

$$
L = \alpha L_{class} + (1-\alpha)L_{ano}
$$

followed by scaling of the trained malicious update relative to the global model.

The anomaly component used in this implementation is a **project-level implementation choice**. It should not be described as an exact CrowdGuard-specific anomaly loss from the Bagdasaryan paper.

---

# 7. Trigger Configuration

The trigger is:

```text
6 × 6 RGB patch
```

The trigger configuration is randomized between attack runs.

The verifier is not given the trigger details.

The randomized components are:

* trigger position,
* RGB pattern,
* target label.

Only the trigger size is fixed.

Conceptually:

```text
Known to verifier:
    trigger size = 6 × 6
    label space = CIFAR-10

Hidden:
    position
    RGB values
    target label
    poisoned training data
```

The 6 × 6 trigger size is a project choice based on the reference CrowdGuard repository's trigger example. The original repository used a fixed red patch near the upper-left corner; this project randomizes the configuration instead.

---

# 8. Malicious Client Configuration

The primary experiments use:

```text
PMR = 0.05
PMR = 0.10
```

For 20 clients, these correspond to approximately:

```text
PMR 0.05 → 1 malicious client
PMR 0.10 → 2 malicious clients
```

A stress experiment uses:

```text
PMR = 0.45
```

which corresponds to:

```text
9 malicious clients out of 20
```

The lower PMRs are intended to isolate the subsequent unlearning and recovery stages, while the 0.45 configuration is the larger stress/reproduction condition associated with the CrowdGuard evaluation context.

---

# 9. CrowdGuard Detection

The CrowdGuard detection path is based on:

### HLBIM

Clients compute internal-state/model-distance information between local models and the global model.

The resulting information is used to construct the HLBIM representation.

### Significance testing

The validation process produces votes indicating whether models appear suspicious.

### Stacked clustering

The votes are passed through CrowdGuard's stacked clustering procedure.

The resulting output is a set of detected clients for the current round:

$$
M_t
$$

where \(M_t\) represents the clients detected in round \(t\).

---

# 10. Detection-Only Modification

The original CrowdGuard implementation uses its detection result to remove suspicious models before aggregation.

This project intentionally does **not** do that.

Instead:

```text
                    ┌── CrowdGuard detection
                    │
20 client updates ──┤
                    │
                    └── all updates → FedAvg
```

CrowdGuard therefore acts as a **detector**, rather than the final removal mechanism.

This allows the subsequent Wu stage to operate on the historical contributions of detected clients.

This distinction is important when describing the project:

> CrowdGuard detection is retained, while immediate model filtering is disabled as a project-level modification.

---

# 11. Historical Update Capture

For every client and every round, the submitted update is retained.

The implementation therefore preserves the association:

```text
client ID
    +
round ID
    +
submitted update
```

Conceptually:

$$
\{\Delta_i^t\}
$$

where:

* \(i\) = client,
* \(t\) = FL round,
* \(\Delta_i^t\) = client \(i\)'s submitted update in round \(t\).

This is required because Wu needs to identify and remove the historical contributions belonging to the clients later identified as malicious.

---

# 12. Detection History

The system retains the per-round detection results:

```text
M_0
M_1
M_2
...
M_K
```

The final malicious set is:

$$
M
$$

The exact final-\(M\) decision rule is **not yet finalized**.

The current implementation uses a temporary **union rule** so that the complete pipeline can execute:

$$
M = \bigcup_t M_t
$$

This rule is provisional and should be replaced once the team agrees on the final malicious-client decision rule.

---

# 13. Ground Truth

Attack ground truth is kept separately from the detection output.

This is important because:

```text
ground-truth malicious clients
        ≠
CrowdGuard detected clients
```

The ground truth is used for **offline evaluation** of detection performance.

It is not supplied to the detection mechanism.

---

# 14. Handoff to Wu

The CrowdGuard stage should provide enough information for the subsequent historical-unlearning stage.

The handoff contains the important state required by Wu, including:

```text
θ0
θK
historical client updates {Δi,t}
final malicious set M
per-round detection history
experiment metadata
```

where:

* \(\theta_0\) = initial global model,
* \(\theta_K\) = final global model before unlearning,
* \(\Delta_i^t\) = historical update from client \(i\) in round \(t\),
* \(M\) = detected malicious-client set.

Ground truth is also retained separately for evaluation.

---

# 15. Expected Wu Input

The Wu stage can therefore treat CrowdGuard as producing:

```text
                    CrowdGuard
                        │
          ┌─────────────┼─────────────┐
          │             │             │
         θ0            θK        historical updates
                                      │
                                      │
                                      ▼
                                  client IDs
                                      │
                                      ▼
                                  malicious M
```

Wu can then subtract the historical contributions associated with clients in \(M\), followed by its knowledge-distillation repair stage.

---

# 16. BAERASER Handoff

Pair 2 is independently constructing BAERASER using centralized ML.

They do **not** need the CrowdGuard repository to begin implementing the recovery algorithm.

The eventual integration boundary is:

```text
CrowdGuard
    │
    ▼
Wu historical unlearning
    │
    ▼
Post-unlearning model
    │
    ▼
BAERASER recovery / verification
```

The BAERASER stage receives the post-unlearning model and performs the trigger-recovery/verification procedure defined by the overall architecture.

---

# 17. Post-Unlearning Recovery Context

The overall project does not implement the complete original BAERASER MSA/MI recovery procedure.

Instead, the architecture uses a direct optimization approach for the known trigger geometry.

The verifier searches over:

```text
729 possible 6 × 6 positions
×
10 candidate target labels
```

while optimizing the trigger tensor.

The trigger is represented as a free:

$$
6 \times 6 \times 3
$$

RGB tensor.

This is a deliberate project-level simplification because the trigger is constrained to a small known spatial extent.

---

# 18. Commit–Reveal Layer

Commit–reveal is part of the overall project architecture but is **not yet implemented in the current CrowdGuard repository**.

It should therefore be treated as remaining work.

The intended protocol is:

### Commit

A participant commits to a vote by publishing a hash:

$$
H(vote,\ round,\ context,\ nonce)
$$

### Reveal

The participant later reveals:

```text
vote
nonce
```

The verifier recomputes the commitment hash.

### Verification

Only valid reveals are included in the independent tally.

Conceptually:

```text
Vote
 │
 ▼
Commit hash
 │
 ▼
Reveal vote + nonce
 │
 ▼
Recompute hash
 │
 ├── valid ──► tally
 │
 └── invalid ─► reject
```

This layer provides commitment/auditability.

It does **not** by itself guarantee that participants are honest or that a server is honest.

---

# 19. What Has Been Verified

The following components have been tested independently.

### OpenFL Workflow API

The required OpenFL Workflow imports work with:

```text
OpenFL 1.9
```

The environment successfully imports:

```python
FLSpec
Aggregator
Collaborator
LocalRuntime
```

### Model

Verified:

```text
275,730 parameters
```

Forward pass works.

Internal-state extraction works.

### Trigger

Random 6 × 6 RGB trigger generation works.

### Attack

Both benign and adaptive malicious local updates can be generated.

### CrowdGuard

HLBIM and pruning/detection path works.

### Historical updates

Historical update capture works.

### Detection-only handoff

The system successfully produces the information required for the downstream stage.

### End-to-end standalone pipeline

The local standalone pipeline has successfully executed the sequence:

```text
local training
    ↓
CrowdGuard detection
    ↓
historical update capture
    ↓
detection-only handoff
```

The standalone smoke test has passed.

---

# 20. Smoke-Test Commands

After activating the environment:

```bash
cd ~/Downloads/CrowdGuard
source venv/bin/activate
```

Run:

```bash
python local_smoke_test.py
```

Expected final message:

```text
SMOKE TEST PASSED
```

Then run:

```bash
python local_pipeline_smoke.py
```

Expected final message:

```text
END-TO-END LOCAL PIPELINE SMOKE TEST PASSED
```

These tests validate the core components without requiring the complete 20-client OpenFL experiment.

---

# 21. Environment Setup

The project was tested using:

```text
Ubuntu 22.04
Python 3.10
OpenFL 1.9
```

Create the virtual environment:

```bash
cd ~/Downloads/CrowdGuard
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then install the required dependencies using **one command**:

```bash
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib openfl ray dill "metaflow==2.7.15" "setuptools<81" tabulate nbformat nbdev
```

The pinned Metaflow and setuptools versions are intentional because the OpenFL Workflow environment requires compatibility with the older Metaflow API.

---

# 22. Verify Installation

Check OpenFL:

```bash
python -c "import openfl; print(openfl.__version__)"
```

Expected:

```text
1.9
```

Check the Workflow API:

```bash
python -c "from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator; from openfl.experimental.workflow.runtime import LocalRuntime; print('OpenFL Workflow API OK')"
```

Expected:

```text
OpenFL Workflow API OK
```

Optional PyTorch device check:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 23. Dependency Notes

The final installation command intentionally includes all dependencies discovered during environment setup.

The important compatibility requirements are:

```text
OpenFL       = 1.9
Metaflow     = 2.7.15
setuptools   < 81
```

The remaining packages provide the supporting functionality required by the repository and OpenFL Workflow environment:

```text
torch
torchvision
numpy
scipy
scikit-learn
pandas
matplotlib
ray
dill
tabulate
nbformat
nbdev
```

There should be no need to install these packages individually when setting up a fresh environment.

---

# 24. Running the Main OpenFL Pipeline

The main experiment entry point is:

```bash
python cifar10_crowdguard.py
```

This registers the OpenFL workflow containing the relevant stages:

```text
Aggregator: start
Collaborator: train
Aggregator: collect_models
Collaborator: local_validation
Aggregator: defend
Aggregator: end
```

The exact full-scale experiment configuration should be checked before committing to a long run.

---

# 25. Repository Structure

The original CrowdGuard repository contains:

```text
CrowdGuard/
│
├── director/
│   ├── start_director.sh
│   └── director_config.yaml
│
├── Amsterdam/
├── Bangalore/
├── Chandler/
├── Detroit/
│
├── CrowdGuardClientValidation.py
├── FederatedCrowdGuard.ipynb
├── PoisoningAttackDemo.ipynb
├── PoisoningAttackDemoReduced.ipynb
├── cifar10_crowdguard.py
├── readme.md
└── .gitignore
```

The Amsterdam, Bangalore, Chandler, and Detroit directories are deployment/environment examples associated with the original OpenFL setup.

They are not required to define the project's logical 20-client experiment.

---

# 26. Current Project Status

## Verified

* [x] Ubuntu/Python virtual environment setup
* [x] OpenFL 1.9 installation
* [x] OpenFL Workflow API import
* [x] Lightweight ResNet-18-style model
* [x] Internal-state extraction
* [x] Random 6 × 6 RGB trigger
* [x] Adaptive backdoor attack
* [x] CrowdGuard HLBIM path
* [x] Stacked clustering path
* [x] Historical update capture
* [x] Detection-only aggregation path
* [x] Per-round detection recording
* [x] Standalone end-to-end pipeline smoke test

## Not yet completed

* [ ] Full OpenFL LocalRuntime execution
* [ ] Full 20-client experiment
* [ ] Full-scale GPU experiment
* [ ] Final malicious-set \(M\) decision rule
* [ ] Full experimental evaluation
* [ ] Final detection metrics
* [ ] Full downstream Wu integration
* [ ] Commit–reveal voting/audit layer
* [ ] Final CrowdGuard → Wu → BAERASER integration

---

# 27. Final Malicious-Set Rule

The current code uses:

```text
FINAL_M_RULE = "union"
```

as a temporary mechanism.

Therefore:

$$
M = \bigcup_t M_t
$$

should **not** be treated as the final project methodology yet.

The team still needs to decide how repeated detection across rounds should determine the final malicious-client set.

Possible rules can be evaluated later, but the final choice should be made at the project/architecture level rather than silently fixed inside CrowdGuard.

---

# 28. Experimental Evaluation Still Required

The final experiments should compare the relevant pipeline configurations, including:

```text
No defense
CrowdGuard only
Wu only
CrowdGuard + Wu
Full pipeline
```

Relevant metrics include:

### Detection

* malicious-client detection
* false positives
* false negatives
* precision
* recall
* F1

### Model utility

* clean accuracy
* clean loss
* macro-F1

### Backdoor

* attack success rate (ASR)

### Unlearning

* residual malicious influence
* post-unlearning clean performance
* convergence behavior

### Trigger recovery

* trigger recovery success/failure
* position error
* pattern similarity
* target-label recovery
* recovery runtime

### System-level

* total runtime
* storage overhead
* verifier agreement
* no-recovery/inconclusive cases

---

# 29. Important Interpretation of Recovery Failure

A failed trigger-recovery search should not automatically be interpreted as proof that the model contains no backdoor.

The appropriate interpretation is:

> No trigger was recovered under the specified recovery protocol.

This distinction matters when reporting experimental results.

---

# 30. Reproducibility

For each experiment, record:

```text
dataset
number of clients
client partition
random seed
PMR
malicious client IDs
trigger size
trigger position
trigger pattern
target label
local epochs
batch size
learning rate
attack parameters
CrowdGuard configuration
number of FL rounds
detected sets M_t
final M rule
```

Ground truth trigger/client information should remain separate from the verifier's information during recovery experiments.

---

# 31. Recommended Execution Order

The intended development sequence is:

```text
1. Environment setup
       ↓
2. Standalone smoke tests
       ↓
3. Full OpenFL LocalRuntime validation
       ↓
4. 20-client experiment
       ↓
5. Collect CrowdGuard handoff
       ↓
6. Wu historical subtraction + KD
       ↓
7. BAERASER recovery/verification
       ↓
8. Commit–reveal integration
       ↓
9. Full experimental evaluation
```

For the current CrowdGuard pair, the immediate focus is:

```text
OpenFL full-runtime validation
        ↓
20-client CrowdGuard experiment
        ↓
finalize handoff format
        ↓
pass handoff to Wu/integration
```

Pair 2 can continue developing BAERASER independently.

---

# 32. What the CrowdGuard Pair Should Hand Off

The final handoff should contain, at minimum:

```text
Initial global model θ0
Final pre-unlearning global model θK

Historical updates:
    Δ_i^t for every client i and round t

Per-round detection:
    M_t

Final malicious set:
    M

Experiment metadata

Ground-truth malicious clients
Ground-truth trigger information
```

The ground truth is for evaluation and should not be treated as input to the detection or verification procedure.

---

# 33. Important Methodological Distinctions

When documenting or presenting this project, keep the following distinctions explicit.

### CrowdGuard vs project modification

CrowdGuard's HLBIM and stacked-clustering detection are retained.

Immediate malicious-model filtering is disabled for this project.

### Bagdasaryan vs project attack

The constrain-and-scale formulation is based on the Bagdasaryan attack framework.

The particular adaptive anomaly objective used here is a project implementation choice.

### CrowdGuard paper vs repository

The paper and public repository are not identical implementations.

The 6 × 6 trigger configuration is derived from the repository example and modified for this project.

### Wu

Wu is an integration stage, not a third project pair.

### BAERASER

The BAERASER pair is developing its recovery implementation independently.

### Commit–reveal

Commit–reveal is a project-level auditability contribution and is currently still pending implementation.

---

# 34. Immediate Next Steps

### Step 1 — Finish OpenFL validation

Run:

```bash
python cifar10_crowdguard.py
```

and validate the complete LocalRuntime workflow.

### Step 2 — Move to 20 clients

Use the agreed:

```text
20 clients
2,500 samples/client
10 local epochs
batch size 64
learning rate 0.01
equal-weight FedAvg
```

configuration.

### Step 3 — Run the primary PMR settings

Start with:

```text
PMR = 0.05
PMR = 0.10
```

Then perform:

```text
PMR = 0.45
```

as the stress configuration.

### Step 4 — Preserve the complete handoff

Make sure every round retains:

```text
client ID
update
detection result
round number
```

### Step 5 — Finalize the \(M\) rule

Replace the temporary union rule with the team's agreed malicious-client decision rule.

### Step 6 — Pass the handoff downstream

Provide the CrowdGuard output to the Wu integration stage, while Pair 2 continues BAERASER development independently.

### Step 7 — Add commit–reveal

Implement:

```text
commit
  ↓
reveal
  ↓
hash verification
  ↓
independent tally
```

after the core ML pipeline is stable.

---

# 35. One-Line Project Description

> **CrowdGuard-based federated backdoor detection with HLBIM and stacked clustering, extended with complete historical-update capture and detection-only operation to support subsequent verifiable federated unlearning and trigger recovery.**
