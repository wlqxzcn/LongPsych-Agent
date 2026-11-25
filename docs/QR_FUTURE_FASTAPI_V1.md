# LongPsychAgent QR-Future FastAPI V1

This package adds a FastAPI service around the already validated
`QRFutureRetrievalPipeline`. It does not modify the encoder, FAISS indexes,
ORCA projector, QR-Future scorer, or retrieval pipeline.

## Endpoints

- `GET /` service information
- `GET /health` loaded model/index readiness
- `POST /v1/search` single-query or batch retrieval

Supported modes:

- `dense`
- `orca`
- `qr_future` (default)

For QR-Future, candidate recall defaults to Dense Top-100. The scorer applies
the five fold-specific ORCA k=2 projections and averages the five raw ranking
scores. The returned `qr_future_score` is not a probability.

## Install dependencies

From the repository root:

```bash
python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"
```

## Start locally on the server

```bash
bash scripts/agent/run_qr_future_api.sh
```

The service binds to `127.0.0.1:8000` and uses one worker intentionally.
Do not change to multiple workers for this V1: each worker would independently
load the BGE model, five QR-Future fold models, and the response embedding map.

## Verify health

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expected fields include:

```json
{
  "status": "ready",
  "retrieval_modes": ["dense", "orca", "qr_future"],
  "corpus_size": 61501,
  "dimension": 1024,
  "default_candidate_k": 100,
  "qr_future": {
    "status": "ready",
    "condition": "QR_FUTURE_FULL",
    "fold_count": 5,
    "input_space": "ORCA_K2_FOLD_SPECIFIC",
    "device": "cuda",
    "response_assets_loaded": true,
    "row_mapping_loaded": true
  }
}
```

## Verify QR-Future search

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "最近总是感到焦虑，不知道该怎么办。",
    "mode": "qr_future",
    "candidate_k": 100,
    "top_k": 5
  }' | python -m json.tool
```

## Verify existing modes

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"最近总是感到焦虑，不知道该怎么办。","mode":"dense","top_k":5}' \
  | python -m json.tool

curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"最近总是感到焦虑，不知道该怎么办。","mode":"orca","top_k":5}' \
  | python -m json.tool
```

## Batch request

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": [
      "最近总是感到焦虑，不知道该怎么办。",
      "我觉得自己没有信心面对接下来的工作。"
    ],
    "mode": "qr_future",
    "candidate_k": 100,
    "top_k": 5
  }' | python -m json.tool
```

## Remote access

V1 binds to localhost intentionally. For remote access, use an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 root@YOUR_SERVER
```

Then open `http://127.0.0.1:8000/docs` on the local computer.

Do not expose this unauthenticated development API directly to the public
internet. Add authentication, TLS, and a reverse proxy before public deployment.
