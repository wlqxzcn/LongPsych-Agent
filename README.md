# LongPsych-Agent

**面向长程心理会话的语义增强检索与智能体系统**

LongPsych-Agent focuses on semantic-enhanced retrieval for long-term
psychological dialogue.

## Research Modules

### 1. ORCA

Training-User-Source Common Directions and Inductive Correction
for Psychological Dialogue Retrieval.

Includes:
- User-source common direction estimation.
- SVD-based projection and inductive correction.
- Dense retrieval and FAISS-based evaluation.
- Competing methods, ablation studies and statistical analysis.

### 2. QR-Future

Future-State Auxiliary Regularization for Retrieval Ranking.

Includes:
- Ranking and future-state auxiliary supervision.
- R-Future and QR-Future architectures.
- Five-fold training and out-of-fold evaluation.
- Matched ablation and component analysis.
- ORCA-integrated retrieval evaluation.

## Repository Structure

```text
LongPsych-Agent/
├── research/
│   ├── orca/
│   │   ├── src/
│   │   ├── commands/
│   │   ├── tests/
│   │   └── README.md
│   └── qr_future/
│       ├── src/
│       ├── manifests/
│       ├── protocols/
│       └── scripts/
├── requirements/
├── docs/
├── scripts/
├── tests/
├── .gitignore
└── README.md
```

## Dependencies

Separate environment specifications are provided:

- `requirements/requirements-orca.txt`
- `requirements/qr-future-environment.txt`

## Data and Model Policy

This repository contains research source code and experiment protocols.

The following artifacts are excluded:
- Original datasets and dialogue corpora.
- Pretrained model weights.
- Embeddings and FAISS indexes.
- Training checkpoints.
- Experiment outputs and logs.
- Private credentials.

These artifacts must be prepared locally before reproducing experiments.

## Reproducibility

The research code preserves historical experimental implementations,
protocols and available manifests.

Some experiments depend on frozen private inputs and trained
checkpoints that are not distributed in this repository.

QR-Future scripts resolve the repository root from their source
locations. Local-only `data/`, `models/` and `results/` directories
can be prepared under the repository root.

## Project Status

Current release: Research code organization for ORCA and QR-Future.

The integrated LongPsych-Agent application layer is under development.
