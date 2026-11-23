#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation import s8_dev_retrieval_evaluation as s8
from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel, RFutureFullMatchedModel,
    RFutureNoProductMatchedModel, QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

DIM = 1024
MODEL_CODE = ROOT / 'src/models/s12c_matched_ablation_models.py'
EXPECTED_MODEL_SHA = 'be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c'
OUT_DIR = ROOT / 'results/phase_s12c/step_2f5d_fold_scale_audit_v1'
CLASSES = {
    'RANK_ONLY': RankOnlyFullMatchedModel,
    'R_FUTURE_FULL': RFutureFullMatchedModel,
    'R_FUTURE_NO_PRODUCT': RFutureNoProductMatchedModel,
    'QR_FUTURE_FULL': QRFutureFullMatchedModel,
    'QR_FUTURE_NO_PRODUCT': QRFutureNoProductMatchedModel,
}

def checkpoint_path(condition: str, fold: int) -> Path:
    if condition == 'RANK_ONLY':
        return ROOT / f'results/phase_s4/checkpoints/rank_only/rank_only_fold{fold}_best_v1.pt'
    name = condition.lower()
    return ROOT / f'results/phase_s12c/checkpoints/{name}/{name}_fold{fold}_best_v1.pt'

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
        'min': float(x.min()), 'p01': float(q[0]), 'p05': float(q[1]),
        'p50': float(q[2]), 'p95': float(q[3]), 'p99': float(q[4]),
        'max': float(x.max()),
    }

def score_terms(model, zq: torch.Tensor, zr: torch.Tensor) -> dict[str, torch.Tensor]:
    w = model.ranker.rank_head.weight[0]
    if model.ranker.rank_head.bias is not None:
        raise RuntimeError('rank_head bias must be None')
    q_term = (zq * w[:DIM]).sum(-1, keepdim=True)
    r_term = (zr * w[DIM:2*DIM]).sum(-1).unsqueeze(0)
    abs_term = (torch.abs(zq[:, None, :] - zr[None, :, :]) * w[2*DIM:3*DIM]).sum(-1)
    terms = {'query': q_term.expand(-1, zr.shape[0]), 'response': r_term.expand(zq.shape[0], -1), 'absdiff': abs_term}
    if w.numel() == 4 * DIM:
        terms['product'] = torch.matmul(zq * w[3*DIM:4*DIM], zr.transpose(0, 1))
    elif w.numel() != 3 * DIM:
        raise RuntimeError(f'Unexpected rank-head dimension: {w.numel()}')
    terms['total'] = terms['query'] + terms['response'] + terms['absdiff'] + terms.get('product', 0.0)
    return terms

def gold_terms(model, zq: torch.Tensor, zg: torch.Tensor) -> dict[str, torch.Tensor]:
    w = model.ranker.rank_head.weight[0]
    out = {
        'query': (zq * w[:DIM]).sum(-1),
        'response': (zg * w[DIM:2*DIM]).sum(-1),
        'absdiff': (torch.abs(zq - zg) * w[2*DIM:3*DIM]).sum(-1),
    }
    if w.numel() == 4 * DIM:
        out['product'] = ((zq * w[3*DIM:4*DIM]) * zg).sum(-1)
    out['total'] = out['query'] + out['response'] + out['absdiff'] + out.get('product', 0.0)
    return out

def validate_fast_scorer(model, device: torch.device) -> float:
    q = torch.randn(2, DIM, device=device)
    r = torch.randn(3, DIM, device=device)
    with torch.inference_mode():
        zq = model.ranker.adapt_query(q)
        zr = model.ranker.adapt_response(r)
        fast = score_terms(model, zq, zr)['total']
        ref = model.rank_scores(q, r.unsqueeze(0).expand(2, -1, -1))
        diff = float((fast - ref).abs().max().item())
    if diff > 1e-5:
        raise RuntimeError(f'Fast/reference score mismatch: {diff}')
    return diff

def corr_summary(vectors: list[np.ndarray]) -> dict:
    x = np.stack(vectors, axis=0)
    c = np.corrcoef(x)
    upper = c[np.triu_indices(c.shape[0], k=1)]
    stds = x.std(axis=1)
    return {
        'pearson_matrix': c.tolist(),
        'mean_pairwise_pearson': float(upper.mean()),
        'min_pairwise_pearson': float(upper.min()),
        'fold_sample_std': [float(v) for v in stds],
        'std_ratio_max_over_min': float(stds.max() / max(stds.min(), 1e-12)),
        'std_coefficient_of_variation': float(stds.std(ddof=1) / max(stds.mean(), 1e-12)),
    }

def main() -> None:
    if OUT_DIR.exists():
        raise RuntimeError(f'Output collision; refusing overwrite: {OUT_DIR}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this audit')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')

    data = s8.load_and_validate_text_data()
    _, _, dev, long20, _, _, match_idx, coverage = data
    q_path, g_path = s8.QUERY_EMB_OUT, s8.GOLD_EMB_OUT
    r_path = s8.TRAIN_RESPONSE_EMB
    emb_manifest_path = s8.EMBED_MANIFEST_OUT
    for p in (q_path, g_path, r_path, emb_manifest_path, MODEL_CODE):
        if not p.is_file():
            raise RuntimeError(f'Missing required frozen input: {p}')

    emb_manifest = json.loads(emb_manifest_path.read_text(encoding='utf-8'))
    if emb_manifest.get('scope') != 'DEV_ONLY' or emb_manifest.get('test_payload_used') is not False:
        raise RuntimeError('Dev embedding manifest scope/test-payload guard failed')
    if sha256_file(MODEL_CODE) != EXPECTED_MODEL_SHA:
        raise RuntimeError('S12C model source SHA256 mismatch')
    if sha256_file(r_path) != s8.EXPECTED_RESPONSE_EMB_SHA:
        raise RuntimeError('Frozen TRAIN response embedding SHA256 mismatch')

    q_np = np.load(q_path, mmap_mode='r', allow_pickle=False)
    g_np = np.load(g_path, mmap_mode='r', allow_pickle=False)
    r_np = np.load(r_path, mmap_mode='r', allow_pickle=False)
    if emb_manifest.get('query_embedding_sha256') != sha256_file(q_path):
        raise RuntimeError('Dev query embedding/manifest SHA256 mismatch')
    if emb_manifest.get('gold_embedding_sha256') != sha256_file(g_path):
        raise RuntimeError('Dev Gold embedding/manifest SHA256 mismatch')
    if emb_manifest.get('query_embedding_shape') != list(q_np.shape):
        raise RuntimeError('Dev query embedding/manifest shape mismatch')
    if emb_manifest.get('gold_embedding_shape') != list(g_np.shape):
        raise RuntimeError('Dev Gold embedding/manifest shape mismatch')
    n = len(dev)
    if q_np.shape != (n, DIM) or g_np.shape != (n, DIM):
        raise RuntimeError(f'Dev embedding shape mismatch: q={q_np.shape}, gold={g_np.shape}, n={n}')
    if r_np.shape != (61502, DIM):
        raise RuntimeError(f'Train response embedding shape mismatch: {r_np.shape}')
    if q_np.dtype != np.float32 or g_np.dtype != np.float32 or r_np.dtype != np.float32:
        raise RuntimeError('Embedding dtype mismatch; expected float32')
    if len(match_idx) != n or len(dev) != 9070 or len(long20) != 4375:
        raise RuntimeError('Frozen Dev row-count/alignment mismatch')

    rng = np.random.default_rng(20261106)
    q_idx = np.sort(rng.choice(n, size=min(128, n), replace=False))
    r_idx = np.sort(rng.choice(r_np.shape[0], size=min(4096, r_np.shape[0]), replace=False))
    q_sample = torch.from_numpy(np.array(q_np[q_idx], dtype=np.float32, copy=True)).to(device)
    g_sample = torch.from_numpy(np.array(g_np[q_idx], dtype=np.float32, copy=True)).to(device)
    r_sample = torch.from_numpy(np.array(r_np[r_idx], dtype=np.float32, copy=True)).to(device)
    valid = (match_idx[q_idx, None] != r_idx[None, :])
    valid_t = torch.from_numpy(valid).to(device)

    all_results = {}
    checkpoint_hashes = {}
    print('GPU =', torch.cuda.get_device_name(device))
    print('DEV_ROWS =', n, 'DEV_DIALOGUES =', len({str(x['dialogue_id']) for x in dev}))
    print('DEV_LONG20_ROWS =', len(long20))
    print('SAMPLE_QUERIES =', len(q_idx), 'SAMPLE_RESPONSES =', len(r_idx), 'VALID_PAIRS =', int(valid.sum()))
    print('TEST_PAYLOAD_USED = NO')

    for condition, cls in CLASSES.items():
        fold_stats = []
        fold_gold_stats = []
        term_stats_by_fold = []
        vectors_by_term = {}
        max_diffs = []
        print('\nCONDITION =', condition, flush=True)
        for fold in range(5):
            path = checkpoint_path(condition, fold)
            if not path.is_file():
                raise RuntimeError(f'Missing checkpoint: {path}')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            if ck.get('condition') != condition or int(ck.get('fold', -1)) != fold:
                raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
            model = cls()
            model.load_state_dict(ck['model_state_dict'], strict=True)
            model.to(device).eval()
            expected_dim = 3072 if 'NO_PRODUCT' in condition else 4096
            if tuple(model.ranker.rank_head.weight.shape) != (1, expected_dim):
                raise RuntimeError(f'Unexpected rank-head shape: {condition} fold {fold}')
            diff = validate_fast_scorer(model, device)
            max_diffs.append(diff)
            checkpoint_hashes[f'{condition}:fold{fold}'] = {
                'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)
            }
            with torch.inference_mode():
                zq = model.ranker.adapt_query(q_sample)
                zr = model.ranker.adapt_response(r_sample)
                zg = model.ranker.adapt_response(g_sample)
                terms = score_terms(model, zq, zr)
                gold = gold_terms(model, zq, zg)
                fold_stats.append(safe_stats(terms['total'][valid_t].detach().cpu().numpy()))
                fold_gold_stats.append(safe_stats(gold['total'].detach().cpu().numpy()))
                one_fold_terms = {}
                for term in ('query', 'response', 'absdiff', 'product', 'total'):
                    if term not in terms:
                        continue
                    vals = terms[term][valid_t].detach().cpu().numpy()
                    one_fold_terms[term] = safe_stats(vals)
                    vectors_by_term.setdefault(term, []).append(np.asarray(vals, dtype=np.float64))
                term_stats_by_fold.append(one_fold_terms)
            del model, ck, zq, zr, zg, terms, gold
            torch.cuda.empty_cache()
            print(f'  fold={fold} scorer_max_abs_diff={diff:.3g} total_mean={fold_stats[-1]["mean"]:.6g} total_std={fold_stats[-1]["std"]:.6g}', flush=True)

        term_corr = {term: corr_summary(vectors) for term, vectors in vectors_by_term.items()}
        stds = np.array([x['std'] for x in fold_stats], dtype=np.float64)
        all_results[condition] = {
            'fold_total_score_sample_stats': fold_stats,
            'fold_gold_score_stats': fold_gold_stats,
            'fold_component_sample_stats': term_stats_by_fold,
            'cross_fold_pairwise_score_correlations': term_corr,
            'total_score_scale': {
                'std_ratio_max_over_min': float(stds.max() / max(stds.min(), 1e-12)),
                'std_coefficient_of_variation': float(stds.std(ddof=1) / max(stds.mean(), 1e-12)),
                'mean_pairwise_pearson': term_corr['total']['mean_pairwise_pearson'],
                'min_pairwise_pearson': term_corr['total']['min_pairwise_pearson'],
            },
            'max_fast_reference_abs_diff': float(max(max_diffs)),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    summary = {
        'artifact': 'S12C STEP 2F-5D-1 Dev Fold Scale and Score Component Sample Audit V1',
        'status': 'COMPLETED',
        'scope': 'DEV_ONLY_FIXED_PAIR_SCORE_SAMPLE',
        'dev_rows': len(dev),
        'dev_long20_rows': len(long20),
        'dev_dialogues': len({str(x['dialogue_id']) for x in dev}),
        'gold_coverage': coverage,
        'sample': {
            'seed': 20261106, 'query_count': int(len(q_idx)),
            'response_count': int(len(r_idx)), 'valid_pair_count': int(valid.sum()),
            'matched_gold_candidates_excluded': True,
        },
        'conditions': all_results,
        'checkpoint_hashes': checkpoint_hashes,
        'script_sha256': sha256_file(Path(__file__)),
        'source_hashes': {
            's12c_model': sha256_file(MODEL_CODE),
            'dev_query_embeddings': sha256_file(q_path),
            'dev_gold_embeddings': sha256_file(g_path),
            'train_response_embeddings': sha256_file(r_path),
            'dev_embedding_manifest': sha256_file(emb_manifest_path),
        },
        'training_performed': False,
        'checkpoints_modified': False,
        'test_payload_used': False,
        'full_library_ranking_performed': False,
        'interpretation_guard': 'Sample score-scale diagnostics only; no ranking conclusion or success gate is applied.',
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        'S12C STEP 2F-5D-1: DEV FOLD SCALE / COMPONENT SAMPLE AUDIT',
        f'dev_rows={len(dev)}; dev_long20_rows={len(long20)}; dialogues={summary["dev_dialogues"]}',
        f'sample_queries={len(q_idx)}; sample_responses={len(r_idx)}; valid_pairs={int(valid.sum())}',
        'test_payload_used=NO; training=NO; checkpoints_modified=NO; full_library_ranking=NO',
        '',
    ]
    for condition, result in all_results.items():
        s = result['total_score_scale']
        lines.append(f'[{condition}] std_ratio={s["std_ratio_max_over_min"]:.6f} std_cv={s["std_coefficient_of_variation"]:.6f} mean_pairwise_pearson={s["mean_pairwise_pearson"]:.6f} min_pairwise_pearson={s["min_pairwise_pearson"]:.6f}')
        for fold, fs in enumerate(result['fold_total_score_sample_stats']):
            lines.append(f'  fold{fold}: mean={fs["mean"]:.8g} std={fs["std"]:.8g} p01={fs["p01"]:.8g} p50={fs["p50"]:.8g} p99={fs["p99"]:.8g} gold_mean={result["fold_gold_score_stats"][fold]["mean"]:.8g}')
        for term, corr in result['cross_fold_pairwise_score_correlations'].items():
            lines.append(f'  {term}: mean_pairwise_pearson={corr["mean_pairwise_pearson"]:.6f} min_pairwise_pearson={corr["min_pairwise_pearson"]:.6f}')
        lines.append('')
    (OUT_DIR / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nSTEP_2F5D1_STATUS = PASS')
    print('SUMMARY =', OUT_DIR / 'summary.json')
    print('REPORT =', OUT_DIR / 'report.txt')
    print('TEST_PAYLOAD_USED = NO')
    print('TRAINING = NO')
    print('CHECKPOINTS_MODIFIED = NO')
    print('FULL_LIBRARY_RANKING = NO')

if __name__ == '__main__':
    main()
