# LongPsychAgent QR-Future Retrieval Pipeline V1

This is a non-invasive integration layer. It adds new files only and does not
modify the existing Dense / ORCA retriever or the QR-Future scorer.

## Retrieval modes

- `dense`: existing Dense BGE retrieval.
- `orca`: existing global ORCA k=2 retrieval.
- `qr_future`: Dense BGE candidate recall (default top-100) followed by the
  frozen five-fold QR-Future FULL ranking model.

The QR-Future model was trained with a Dense top-100 candidate reservoir, so
V1 preserves Dense candidate recall for the QR-Future mode. ORCA remains
available as an independent retrieval mode. This package does not silently
claim that ORCA-recall -> QR-Future reranking has the same validated candidate
distribution.

## Install

Clone the repository and enter its root directory:

```bash
python -m venv .venv-agent
source .venv-agent/bin/activate
# Install the retrieval/runtime stack. Install a CUDA-compatible PyTorch
# build first if your platform requires a specific wheel.
python -m pip install -r requirements/requirements-orca.txt
python -m pip install -r requirements-agent.txt
python -m pip install -r requirements-mcp.txt
if [ ! -f .env.agent ]; then cp .env.agent.example .env.agent; fi
export PYTHONPATH=agent:.
python -m compileall -q agent/longpsych_agent
python -m pytest -q tests/agent/test_qr_future_pipeline.py
```

## Path configuration

All default filesystem locations are resolved from the repository root, not from the current terminal directory. Copy `.env.agent.example` to `.env.agent` and adjust only the locations that differ on your machine.

- `LONGPSYCH_ROOT`: deployment asset root (default: repository root).
- `LONGPSYCH_MODEL`: BGE model directory. The supplied example uses `../PsychResponse_FSAR/models/BAAI/bge-large-zh-v1___5`.
- `LONGPSYCH_INDEX_DIR`: Dense/ORCA FAISS indexes (default: `data/indexes`).
- `LONGPSYCH_SOURCE_ROOT`: original research data/checkpoint root. The supplied example uses `../PsychResponse_FSAR`.

All filesystem values in `.env.agent.example` are relative to `LONGPSYCH_ROOT` (which defaults to the repository root). If your model or research checkout is stored elsewhere, change these relative values in `.env.agent`; no source-code edits are needed.

The frozen QR-Future model architecture source is included at `src/models/s12c_matched_ablation_models.py`. Model weights, FAISS indexes, datasets, and training checkpoints are not included in the source archive; place or mount them at the configured locations.

## Real integration smoke test

```bash
PYTHONPATH=agent:. python scripts/agent/smoke_qr_future_pipeline.py   --root .   --device cuda   --candidate-k 100   --top-k 5
```

The smoke test loads the existing Dense and ORCA FAISS indexes, BGE encoder,
five QR-Future fold models and their fold-specific Basis files. It then checks
Dense retrieval, ORCA retrieval, and Dense -> QR-Future end-to-end reranking.

## Python API

```python
from longpsych_agent.reranking.qr_future_pipeline import (
    QRFutureRetrievalPipeline,
)
from longpsych_agent.config import settings

pipeline = QRFutureRetrievalPipeline.from_pretrained(
    model_name_or_path=str(settings.model_path),
    index_dir=settings.index_dir,
    qr_future_project_root=settings.root,
    default_candidate_k=100,
    device="cuda",
)

result = pipeline.search(
    "最近总是感到焦虑，不知道该怎么办。",
    top_k=5,
    mode="qr_future",
)
```

`qr_future_score` is a raw ranking score, not a probability. The auxiliary
future head is a training-time regularizer; online reranking uses the model's
`rank_scores()` output.
