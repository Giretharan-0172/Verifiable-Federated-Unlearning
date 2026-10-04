# Verifiable Federated Unlearning for Backdoor Removal

A research project exploring verifiable backdoor removal in Federated Learning through malicious-client detection, historical-update unlearning, model repair, and empirical verification.

## Overview

Backdoor attacks can cause a Federated Learning model to behave normally on standard inputs while producing attacker-chosen predictions for inputs containing a specific trigger.

This project investigates a pipeline for detecting and removing backdoor behavior from a federated model:

```text
Federated Training
        ↓
CrowdGuard Detection
        ↓
Historical-Update Unlearning
        ↓
Knowledge-Distillation Repair
        ↓
Trigger Recovery / Verification
        ↓
Commit–Reveal Audit
```

The project uses **CIFAR-10** and a multi-client Federated Learning setup to study this pipeline.

## Current Status

The project is under active development.

### Currently Implementing / Testing
- Federated Learning setup using OpenFL
- CIFAR-10 experiments
- Backdoor injection using 6×6 pixel triggers
- CrowdGuard-based malicious-client detection
- HLBIM-based anomaly detection
- Stacked clustering for client identification

### Under Development
- Historical-update unlearning
- Knowledge-distillation repair
- Post-unlearning empirical verification
- Trigger recovery
- Commit–reveal vote integrity mechanism

The current implementation should therefore not be interpreted as a completed end-to-end federated unlearning system.

## Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Federated Clients | 20 |
| Aggregation | Equal-weight FedAvg |
| Samples / Client | 2,500 |
| Local Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Trigger Size | 6×6 pixels |

## CrowdGuard Detection

The CrowdGuard component is used to identify anomalous or potentially malicious clients during federated training.

The detection pipeline uses:

1. Hidden-layer representations from client models
2. Distance-based client comparison
3. HLBIM-based anomaly detection
4. Dimensionality reduction and pruning
5. Stacked clustering
6. Client-level maliciousness votes

The estimated malicious-client set is then intended to be used by the subsequent unlearning stage.

## Project Structure

The repository contains the different components of the proposed pipeline. The current CrowdGuard implementation and experiments are available in the `CrowdGuard` component.

## Research Basis

The project builds on ideas from:

- **CrowdGuard** — malicious-client detection in Federated Learning
- **Wu et al.** — historical-update based federated unlearning and knowledge-distillation repair
- **BAERASER** — trigger recovery / backdoor removal concepts

The project combines and adapts these ideas into a single experimental pipeline. Some components are project-specific adaptations rather than direct reproductions of the original methods.

## Repository

The CrowdGuard implementation currently being developed and tested can be found in the `CrowdGuard` component.

---

**Status:** B.Tech Project — Ongoing
