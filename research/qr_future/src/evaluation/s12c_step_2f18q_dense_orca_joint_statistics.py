#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
N = 62889
N_DIALOGUES = 1666
N_FOLDS = 5
N_CANDIDATES = 101
N_NEGATIVES = 100
N_BOOT = 5000
BOOT_SEED = 43
EXPECTED_ORCA_PROTOCOL_SHA = 'da36ab67fc061baff770ba3eaaed9d52e0b4ce813a69f574e0fd11d6de303a1a'
EXPECTED_CANDIDATE_ORDER_SHA = 'e3dc4ef00cbf001b9e37cfd8369863e745a9b111cea8496ae82a70fccb890e7d'

CONDITIONS = [
    'RANK_ONLY_FULL_MATCHED',
    'RANK_ONLY_NO_PRODUCT_MATCHED',
    'R_FUTURE_FULL',
    'R_FUTURE_NO_PRODUCT',
    'QR_FUTURE_FULL',
    'QR_FUTURE_NO_PRODUCT',
]
FUTURE_CONDITIONS = {
    'R_FUTURE_FULL',
    'R_FUTURE_NO_PRODUCT',
    'QR_FUTURE_FULL',
    'QR_FUTURE_NO_PRODUCT',
}
METRICS = ['MRR', 'R@1', 'R@5', 'R@10', 'R@20']

MASTER = ROOT / 'data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl'
FOLD = ROOT / 'data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy'
RESERVOIR = ROOT / 'cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz'

DENSE_PROTOCOL = ROOT / 'protocols/fsar_s12c_seven_condition_evaluation_v1.json'
DENSE_ARRAYS = ROOT / 'results/phase_s12c/seven_condition_evaluation_v1/s12c_seven_condition_evaluation_arrays_v1.npz'
DENSE_SUMMARY = ROOT / 'results/phase_s12c/seven_condition_evaluation_v1/s12c_seven_condition_evaluation_summary_v1.json'
DENSE_RANK_ONLY_PROTOCOL = ROOT / 'protocols/fsar_s12c_matched_rank_only_training_v1.json'
DENSE_RANK_ONLY_MANIFEST = ROOT / 'results/phase_s12c/rank_only_matched_v1/run_manifest_v1.json'
DENSE_FUTURE_PROTOCOL = ROOT / 'protocols/fsar_s12c_matched_ablation_v1.json'

DENSE_OOF = {
    'RANK_ONLY_FULL_MATCHED': ROOT / 'results/phase_s12c/rank_only_matched_v1/oof/s12c_rank_only_full_matched_oof_scores_v1.npz',
    'RANK_ONLY_NO_PRODUCT_MATCHED': ROOT / 'results/phase_s12c/rank_only_matched_v1/oof/s12c_rank_only_no_product_matched_oof_scores_v1.npz',
    'R_FUTURE_FULL': ROOT / 'results/phase_s12c/s12c_beta8_r_future_full_oof_scores_v1.npz',
    'R_FUTURE_NO_PRODUCT': ROOT / 'results/phase_s12c/s12c_beta8_r_future_no_product_oof_scores_v1.npz',
    'QR_FUTURE_FULL': ROOT / 'results/phase_s12c/s12c_beta8_qr_future_full_oof_scores_v1.npz',
    'QR_FUTURE_NO_PRODUCT': ROOT / 'results/phase_s12c/s12c_beta8_qr_future_no_product_oof_scores_v1.npz',
}
DENSE_SUMMARIES = {
    'RANK_ONLY_FULL_MATCHED': ROOT / 'results/phase_s12c/rank_only_matched_v1/summaries/s12c_rank_only_full_matched_training_summary_v1.json',
    'RANK_ONLY_NO_PRODUCT_MATCHED': ROOT / 'results/phase_s12c/rank_only_matched_v1/summaries/s12c_rank_only_no_product_matched_training_summary_v1.json',
    'R_FUTURE_FULL': ROOT / 'results/phase_s12c/s12c_beta8_r_future_full_training_summary_v1.json',
    'R_FUTURE_NO_PRODUCT': ROOT / 'results/phase_s12c/s12c_beta8_r_future_no_product_training_summary_v1.json',
    'QR_FUTURE_FULL': ROOT / 'results/phase_s12c/s12c_beta8_qr_future_full_training_summary_v1.json',
    'QR_FUTURE_NO_PRODUCT': ROOT / 'results/phase_s12c/s12c_beta8_qr_future_no_product_training_summary_v1.json',
}

ORCA_ROOT = ROOT / 'results/phase_s12c/step_2f18_orca_fsar_training_v1'
ORCA_PROTOCOL = ORCA_ROOT / 'training_protocol_v1.json'
ORCA_MANIFEST = ORCA_ROOT / 'run_manifest.json'
ORCA_OOF = {
    c: ORCA_ROOT / 'conditions' / c.lower() / f'{c.lower()}_oof_scores_v1.npz'
    for c in CONDITIONS
}
ORCA_SUMMARIES = {
    c: ORCA_ROOT / 'conditions' / c.lower() / f'{c.lower()}_training_summary_v1.json'
    for c in CONDITIONS
}

STEP_2F18M_PROTOCOL = ROOT / 'results/phase_s12c/step_2f18m_orca_fsar_oof_evaluation_v1/evaluation_protocol_v1.json'
STEP_2F18M_SUMMARY = ROOT / 'results/phase_s12c/step_2f18m_orca_fsar_oof_evaluation_v1/evaluation_summary_v1.json'

PROTOCOL_OUT = ROOT / 'protocols/fsar_s12c_step_2f18q_dense_orca_joint_statistics_v1.json'
OUT_DIR = ROOT / 'results/phase_s12c/step_2f18q_dense_orca_joint_statistics_v1'


def req(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def load_master_and_candidate_order():
    rows = [json.loads(line) for line in MASTER.read_text(encoding='utf-8').splitlines() if line.strip()]
    req(len(rows) == N, f'master row count mismatch: {len(rows)}')
    qids = np.asarray([str(x['query_id']) for x in rows], dtype=str)
    dids = np.asarray([int(x['dialogue_id']) for x in rows], dtype=np.int64)
    req(np.unique(qids).size == N, 'duplicate query_id in master')
    req(np.unique(dids).size == N_DIALOGUES, f'dialogue count mismatch: {np.unique(dids).size}')

    fold = np.load(FOLD, allow_pickle=False).astype(np.int8, copy=False)
    req(fold.shape == (N,), f'fold shape mismatch: {fold.shape}')
    req(set(np.unique(fold).tolist()) == set(range(N_FOLDS)), 'fold IDs are not 0..4')

    dialogue_fold = {}
    for did, fid in zip(dids.tolist(), fold.tolist()):
        old = dialogue_fold.setdefault(did, fid)
        req(old == fid, f'dialogue split across folds: {did}')

    future_available = np.asarray([bool(x['future_value_available']) for x in rows], dtype=bool)
    future_labels = np.asarray([
        int(x['future_value_label']) if bool(x['future_value_available']) else -2
        for x in rows
    ], dtype=np.int64)
    req(int(future_available.sum()) == 17458, f'future row count mismatch: {int(future_available.sum())}')
    req(len(np.unique(dids[future_available])) == 1653, 'future dialogue count mismatch')

    reservoir = load_npz(RESERVOIR)
    req('candidate_response_row' in reservoir and 'gold_response_row' in reservoir, 'reservoir response-row keys missing')
    req('gold_similarity' in reservoir and 'candidate_similarity' in reservoir, 'reservoir dense-score keys missing')
    neg_rows = np.asarray(reservoir['candidate_response_row'], dtype=np.int32)
    gold_rows = np.asarray(reservoir['gold_response_row'], dtype=np.int32)
    req(neg_rows.shape == (N, N_NEGATIVES), f'negative row shape mismatch: {neg_rows.shape}')
    req(gold_rows.shape == (N,), f'gold row shape mismatch: {gold_rows.shape}')
    req(np.all(neg_rows != gold_rows[:, None]), 'Gold response appears in negative reservoir')
    req(all(np.unique(x).size == N_NEGATIVES for x in neg_rows), 'duplicate negative response in a query')

    candidate_order = np.column_stack([gold_rows, neg_rows]).astype(np.int32, copy=False)
    req(candidate_order.shape == (N, N_CANDIDATES), f'candidate order shape mismatch: {candidate_order.shape}')
    candidate_order_sha = hashlib.sha256(candidate_order.tobytes(order='C')).hexdigest()
    req(candidate_order_sha == EXPECTED_CANDIDATE_ORDER_SHA,
        f'candidate order SHA mismatch: {candidate_order_sha}')

    gold_sim = np.asarray(reservoir['gold_similarity'], dtype=np.float32)
    neg_sim = np.asarray(reservoir['candidate_similarity'], dtype=np.float32)
    req(gold_sim.shape == (N,), f'gold similarity shape mismatch: {gold_sim.shape}')
    req(neg_sim.shape == (N, N_NEGATIVES), f'negative similarity shape mismatch: {neg_sim.shape}')
    req(np.isfinite(gold_sim).all() and np.isfinite(neg_sim).all(), 'non-finite Dense BGE similarities')
    dense_scores = np.column_stack([gold_sim, neg_sim]).astype(np.float32, copy=False)

    return rows, qids, dids, fold, future_available, future_labels, candidate_order_sha, dense_scores


def rank_from_scores(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    req(scores.shape == (N, N_CANDIDATES), f'candidate score shape mismatch: {scores.shape}')
    req(np.isfinite(scores).all(), 'non-finite candidate scores')
    gold = scores[:, 0:1]
    neg = scores[:, 1:]
    strict = 1 + (neg > gold).sum(axis=1)
    ge = 1 + (neg >= gold).sum(axis=1)
    ties = (neg == gold).sum(axis=1)
    req(np.array_equal(ge - strict, ties), 'tie identity check failed')
    req(strict.min() >= 1 and strict.max() <= N_CANDIDATES, 'strict ranks out of range')
    req(ge.min() >= 1 and ge.max() <= N_CANDIDATES, 'GE ranks out of range')
    return strict.astype(np.int16), ge.astype(np.int16), ties.astype(np.int16)


def verify_oof(path: Path, condition: str, fold: np.ndarray,
               future_available: np.ndarray, future_labels: np.ndarray):
    data = load_npz(path)
    required = {'candidate_scores', 'fold_id', 'gold_position'}
    req(required.issubset(data), f'{condition}: missing OOF keys {sorted(required - set(data))}')
    scores = np.asarray(data['candidate_scores'], dtype=np.float32)
    req(scores.shape == (N, N_CANDIDATES), f'{condition}: candidate_scores shape {scores.shape}')
    req(np.isfinite(scores).all(), f'{condition}: non-finite candidate scores')
    req(np.array_equal(data['fold_id'].astype(np.int8), fold), f'{condition}: fold ID mismatch')
    req(data['gold_position'].shape == (N,) and np.all(data['gold_position'] == 0), f'{condition}: Gold position is not all zero')

    if condition in FUTURE_CONDITIONS:
        for key in ('future_available', 'future_value_label'):
            req(key in data, f'{condition}: missing {key}')
        req(np.array_equal(data['future_available'].astype(bool), future_available), f'{condition}: future_available mismatch')
        req(np.array_equal(data['future_value_label'].astype(np.int64), future_labels), f'{condition}: future_value_label mismatch')

    strict, ge, ties = rank_from_scores(scores)
    return strict, ge, ties, sha256_file(path)


def source_paths() -> Dict[str, Path]:
    paths = {
        'master': MASTER,
        'fold': FOLD,
        'reservoir': RESERVOIR,
        'dense_evaluation_protocol': DENSE_PROTOCOL,
        'dense_seven_condition_arrays': DENSE_ARRAYS,
        'dense_seven_condition_summary': DENSE_SUMMARY,
        'dense_rank_only_protocol': DENSE_RANK_ONLY_PROTOCOL,
        'dense_rank_only_run_manifest': DENSE_RANK_ONLY_MANIFEST,
        'dense_future_architecture_protocol': DENSE_FUTURE_PROTOCOL,
        'orca_training_protocol': ORCA_PROTOCOL,
        'orca_training_run_manifest': ORCA_MANIFEST,
        'step_2f18m_evaluation_protocol': STEP_2F18M_PROTOCOL,
        'step_2f18m_evaluation_summary': STEP_2F18M_SUMMARY,
    }
    for c in CONDITIONS:
        paths[f'dense_oof::{c}'] = DENSE_OOF[c]
        paths[f'dense_summary::{c}'] = DENSE_SUMMARIES[c]
        paths[f'orca_oof::{c}'] = ORCA_OOF[c]
        paths[f'orca_summary::{c}'] = ORCA_SUMMARIES[c]
    return paths


def load_and_validate_all():
    rows, qids, dids, fold, future_available, future_labels, candidate_order_sha, dense_scores = load_master_and_candidate_order()
    dense_strict, dense_ge, dense_ties = rank_from_scores(dense_scores)

    dense_eval = load_npz(DENSE_ARRAYS)
    for key in ('retrieval_conditions', 'query_ids', 'dialogue_ids', 'fold_id', 'ranks', 'gold_tie_counts'):
        req(key in dense_eval, f'dense seven-condition array missing {key}')
    dense_names = [str(x) for x in dense_eval['retrieval_conditions'].tolist()]
    req(len(set(dense_names)) == len(dense_names), 'duplicate Dense condition names in reference array')
    dense_index = {name: i for i, name in enumerate(dense_names)}
    req('DENSE_101' in dense_index, f'DENSE_101 missing from Dense reference: {dense_names}')
    for c in CONDITIONS:
        req(c in dense_index, f'{c} missing from Dense reference array: {dense_names}')
    req(np.array_equal(dense_eval['query_ids'].astype(str), qids), 'Dense reference query order mismatch')
    req(np.array_equal(dense_eval['dialogue_ids'].astype(np.int64), dids), 'Dense reference dialogue order mismatch')
    req(np.array_equal(dense_eval['fold_id'].astype(np.int8), fold), 'Dense reference fold mismatch')
    req(dense_eval['ranks'].shape == (len(dense_names), N), f'Dense reference ranks shape mismatch: {dense_eval["ranks"].shape}')
    req(dense_eval['gold_tie_counts'].shape == (len(dense_names), N), 'Dense reference tie shape mismatch')

    ref_dense = dense_index['DENSE_101']
    req(np.array_equal(dense_eval['ranks'][ref_dense].astype(np.int16), dense_strict), 'DENSE_101 raw-score rank reproduction failed')
    req(np.array_equal(dense_eval['gold_tie_counts'][ref_dense].astype(np.int16), dense_ties), 'DENSE_101 tie reproduction failed')

    dense_strict_by_condition = {}
    dense_ge_by_condition = {}
    dense_ties_by_condition = {}
    dense_hashes = {}
    for c in CONDITIONS:
        strict, ge, ties, digest = verify_oof(DENSE_OOF[c], c, fold, future_available, future_labels)
        idx = dense_index[c]
        req(np.array_equal(dense_eval['ranks'][idx].astype(np.int16), strict),
            f'Dense seven-condition rank reproduction failed for {c}')
        req(np.array_equal(dense_eval['gold_tie_counts'][idx].astype(np.int16), ties),
            f'Dense seven-condition tie reproduction failed for {c}')
        dense_strict_by_condition[c] = strict
        dense_ge_by_condition[c] = ge
        dense_ties_by_condition[c] = ties
        dense_hashes[c] = digest

    orca_strict_by_condition = {}
    orca_ge_by_condition = {}
    orca_ties_by_condition = {}
    orca_hashes = {}
    for c in CONDITIONS:
        strict, ge, ties, digest = verify_oof(ORCA_OOF[c], c, fold, future_available, future_labels)
        orca_strict_by_condition[c] = strict
        orca_ge_by_condition[c] = ge
        orca_ties_by_condition[c] = ties
        orca_hashes[c] = digest

    orca_protocol_sha = sha256_file(ORCA_PROTOCOL)
    req(orca_protocol_sha == EXPECTED_ORCA_PROTOCOL_SHA,
        f'ORCA protocol SHA mismatch: {orca_protocol_sha}')
    orca_manifest = json_load(ORCA_MANIFEST)
    req(orca_manifest.get('protocol_sha256') == orca_protocol_sha, 'ORCA run manifest protocol SHA mismatch')
    req(set(orca_manifest.get('condition_oof_sha256', {}).keys()) == set(CONDITIONS), 'ORCA run manifest OOF condition set mismatch')
    req(orca_manifest.get('condition_oof_sha256') == orca_hashes,
        'ORCA run manifest condition_oof_sha256 does not match current OOF files')

    # Dense rank-only run manifest carries explicit OOF file hashes.
    dense_rank_manifest = json_load(DENSE_RANK_ONLY_MANIFEST)
    dense_rank_outputs = dense_rank_manifest.get('outputs', {})
    for c in ('RANK_ONLY_FULL_MATCHED', 'RANK_ONLY_NO_PRODUCT_MATCHED'):
        expected = dense_rank_outputs.get(c, {}).get('oof', {}).get('sha256')
        expected_summary = dense_rank_outputs.get(c, {}).get('summary', {}).get('sha256')
        req(expected is not None, f'Dense rank-only manifest does not expose OOF SHA for {c}')
        req(expected == dense_hashes[c], f'Dense rank-only manifest OOF SHA mismatch for {c}')
        req(expected_summary is not None, f'Dense rank-only manifest does not expose summary SHA for {c}')
        req(expected_summary == sha256_file(DENSE_SUMMARIES[c]), f'Dense rank-only manifest summary SHA mismatch for {c}')

    # Dense future summaries expose their own OOF SHA; verify when present.
    dense_summary_hashes = {}
    for c in CONDITIONS:
        summary = json_load(DENSE_SUMMARIES[c])
        expected = summary.get('oof_sha256')
        if expected is not None:
            req(expected == dense_hashes[c], f'Dense training summary OOF SHA mismatch for {c}')
        dense_summary_hashes[c] = sha256_file(DENSE_SUMMARIES[c])

    # ORCA summaries and OOF hashes must agree with the completed run manifest.
    orca_manifest_summary_hashes = orca_manifest.get('condition_summary_sha256', {})
    for c in CONDITIONS:
        req(sha256_file(ORCA_SUMMARIES[c]) == orca_manifest_summary_hashes.get(c),
            f'ORCA summary SHA mismatch for {c}')
        summary = json_load(ORCA_SUMMARIES[c])
        req(summary.get('oof_sha256') == orca_hashes[c], f'ORCA summary OOF SHA mismatch for {c}')
        req(summary.get('protocol_sha256') == orca_protocol_sha, f'ORCA summary protocol SHA mismatch for {c}')

    # The prior 2F18M evaluation artifacts are provenance inputs only; they are not used to overwrite ranks.
    m_protocol = json_load(STEP_2F18M_PROTOCOL)
    m_summary = json_load(STEP_2F18M_SUMMARY)
    req(isinstance(m_protocol, dict) and isinstance(m_summary, dict), '2F18M protocol/summary JSON structure invalid')

    ranks_strict = [dense_strict]
    ranks_ge = [dense_ge]
    ties_all = [dense_ties]
    labels = ['DENSE_BGE']
    for c in CONDITIONS:
        labels.append(f'DENSE::{c}')
        ranks_strict.append(dense_strict_by_condition[c])
        ranks_ge.append(dense_ge_by_condition[c])
        ties_all.append(dense_ties_by_condition[c])
    for c in CONDITIONS:
        labels.append(f'ORCA::{c}')
        ranks_strict.append(orca_strict_by_condition[c])
        ranks_ge.append(orca_ge_by_condition[c])
        ties_all.append(orca_ties_by_condition[c])

    strict_matrix = np.stack(ranks_strict, axis=0).astype(np.int16, copy=False)
    ge_matrix = np.stack(ranks_ge, axis=0).astype(np.int16, copy=False)
    ties_matrix = np.stack(ties_all, axis=0).astype(np.int16, copy=False)
    req(strict_matrix.shape == (13, N), f'joint strict rank matrix shape mismatch: {strict_matrix.shape}')
    req(np.array_equal(ge_matrix - strict_matrix, ties_matrix), 'joint tie identity failed')

    input_paths = source_paths()
    input_hashes = {}
    for name, path in input_paths.items():
        req(path.is_file(), f'missing input {name}: {path}')
        input_hashes[name] = sha256_file(path)

    audit = {
        'n_queries': N,
        'n_dialogues': N_DIALOGUES,
        'n_folds': N_FOLDS,
        'candidate_shape': [N, N_CANDIDATES],
        'gold_position': 0,
        'negative_count': N_NEGATIVES,
        'candidate_order_sha256': candidate_order_sha,
        'dense_seven_condition_rank_reproduction': 'PASS',
        'dense_rank_only_manifest_oof_sha': 'PASS',
        'orca_run_manifest_oof_sha': 'PASS',
        'orca_protocol_sha256': orca_protocol_sha,
        'dense_oof_hashes': dense_hashes,
        'orca_oof_hashes': orca_hashes,
        'dense_summary_hashes': dense_summary_hashes,
        'dialogue_fold_integrity': 'PASS',
        'future_label_alignment': 'PASS',
        'condition_alignment': 'PASS',
    }

    return {
        'qids': qids,
        'dids': dids,
        'fold': fold,
        'labels': labels,
        'ranks_strict': strict_matrix,
        'ranks_ge': ge_matrix,
        'ties': ties_matrix,
        'audit': audit,
        'input_hashes': input_hashes,
    }


def protocol_payload(validated: dict) -> dict:
    return {
        'artifact': 'FSAR S12C STEP 2F18Q Dense-ORCA Joint Paired Statistics Protocol V1',
        'status': 'FROZEN_BEFORE_STATISTICAL_EVALUATION',
        'scope': 'TRAIN_INTERNAL_5FOLD_OOF_ONLY',
        'purpose': 'Paired comparison of six matched ranking/future conditions between Dense BGE and ORCA k=2 representation spaces.',
        'conditions': CONDITIONS,
        'spaces': ['DENSE', 'ORCA_K2'],
        'reference_baseline': 'DENSE_BGE',
        'data': {
            'queries': N,
            'dialogues': N_DIALOGUES,
            'folds': N_FOLDS,
            'candidate_count': N_CANDIDATES,
            'gold_position': 0,
            'negative_count': N_NEGATIVES,
            'candidate_order_sha256': validated['audit']['candidate_order_sha256'],
        },
        'ranking': {
            'primary_tie_policy': 'STRICT_GREATER',
            'primary_rank_formula': '1 + COUNT(negative_score > gold_score)',
            'sensitivity_tie_policy': 'CONSERVATIVE_GREATER_OR_EQUAL',
            'sensitivity_rank_formula': '1 + COUNT(negative_score >= gold_score)',
            'metrics': METRICS,
            'primary_metric': 'MRR',
            'aggregation': 'Average query metrics within each dialogue, then unweighted mean across dialogues.',
        },
        'statistics': {
            'method': 'paired_dialogue_cluster_bootstrap',
            'replicates': N_BOOT,
            'seed': BOOT_SEED,
            'confidence_level': 0.95,
            'resampling_unit': 'dialogue',
            'shared_resampling_indices_across_all_conditions_and_contrasts': True,
            'bootstrap_chunk_size': 64,
        },
        'planned_contrasts': [
            'ORCA minus Dense for each of six matched conditions',
            'R-Future and QR-Future gains versus the matched Rank-Only baseline within each space',
            'QR-Future minus R-Future within each space',
            'No-Product minus Full ranking ablation within each space',
            'Product Difference-in-Differences within each space',
            'ORCA-by-Future cross-space interaction contrasts',
            'ORCA-by-QR-vs-R cross-space interaction contrasts',
            'ORCA-by-Product-DID cross-space interaction contrast',
        ],
        'input_sha256': validated['input_hashes'],
        'audit': validated['audit'],
        'execution_constraints': {
            'training': False,
            'checkpoint_modification': False,
            'checkpoint_reselection': False,
            'test_payload_used': False,
            'dev_payload_used': False,
            'test_time_fitting': False,
            'parameter_update': False,
            'existing_outputs_overwritten': False,
        },
    }


def dialogue_metrics_from_ranks(ranks: np.ndarray, dialogue_ids: np.ndarray):
    # ranks: [conditions, queries]
    c, n = ranks.shape
    qmetrics = np.empty((n, c, len(METRICS)), dtype=np.float64)
    qmetrics[:, :, 0] = 1.0 / ranks.T
    qmetrics[:, :, 1] = (ranks.T <= 1)
    qmetrics[:, :, 2] = (ranks.T <= 5)
    qmetrics[:, :, 3] = (ranks.T <= 10)
    qmetrics[:, :, 4] = (ranks.T <= 20)

    unique_dids, inverse = np.unique(dialogue_ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_dids)).astype(np.int64)
    req(len(unique_dids) == N_DIALOGUES, f'dialogue metric count mismatch: {len(unique_dids)}')
    sums = np.zeros((len(unique_dids), c, len(METRICS)), dtype=np.float64)
    np.add.at(sums, inverse, qmetrics)
    per_dialogue = sums / counts[:, None, None]
    macro = per_dialogue.mean(axis=0)
    micro = qmetrics.mean(axis=0)
    return unique_dids, counts, qmetrics, per_dialogue, macro, micro


def bootstrap_dialogue_means(per_dialogue: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    b, d = boot_idx.shape
    c, m = per_dialogue.shape[1], per_dialogue.shape[2]
    out = np.empty((b, c, m), dtype=np.float64)
    chunk = 64
    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        out[start:end] = per_dialogue[boot_idx[start:end]].mean(axis=1)
        print(f'BOOTSTRAP_PROGRESS {end}/{b}', flush=True)
    return out


def summarize_models(labels: List[str], macro: np.ndarray, micro: np.ndarray,
                     boot_means: np.ndarray) -> dict:
    lo = np.quantile(boot_means, 0.025, axis=0)
    hi = np.quantile(boot_means, 0.975, axis=0)
    result = {}
    for i, name in enumerate(labels):
        result[name] = {
            'dialogue_macro': {
                metric: {
                    'estimate': float(macro[i, j]),
                    'ci95_lower': float(lo[i, j]),
                    'ci95_upper': float(hi[i, j]),
                }
                for j, metric in enumerate(METRICS)
            },
            'query_micro': {metric: float(micro[i, j]) for j, metric in enumerate(METRICS)},
        }
    return result


def contrast_definitions() -> Dict[str, Dict[str, float]]:
    defs: Dict[str, Dict[str, float]] = {}
    for c in CONDITIONS:
        defs[f'ORCA_MINUS_DENSE::{c}'] = {f'ORCA::{c}': 1.0, f'DENSE::{c}': -1.0}

    for space in ('DENSE', 'ORCA'):
        defs[f'R_FUTURE_GAIN_FULL::{space}'] = {
            f'{space}::R_FUTURE_FULL': 1.0,
            f'{space}::RANK_ONLY_FULL_MATCHED': -1.0,
        }
        defs[f'R_FUTURE_GAIN_NO_PRODUCT::{space}'] = {
            f'{space}::R_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::RANK_ONLY_NO_PRODUCT_MATCHED': -1.0,
        }
        defs[f'QR_FUTURE_GAIN_FULL::{space}'] = {
            f'{space}::QR_FUTURE_FULL': 1.0,
            f'{space}::RANK_ONLY_FULL_MATCHED': -1.0,
        }
        defs[f'QR_FUTURE_GAIN_NO_PRODUCT::{space}'] = {
            f'{space}::QR_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::RANK_ONLY_NO_PRODUCT_MATCHED': -1.0,
        }
        defs[f'QR_MINUS_R_FULL::{space}'] = {
            f'{space}::QR_FUTURE_FULL': 1.0,
            f'{space}::R_FUTURE_FULL': -1.0,
        }
        defs[f'QR_MINUS_R_NO_PRODUCT::{space}'] = {
            f'{space}::QR_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::R_FUTURE_NO_PRODUCT': -1.0,
        }
        defs[f'PRODUCT_ABLATION_R::{space}'] = {
            f'{space}::R_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::R_FUTURE_FULL': -1.0,
        }
        defs[f'PRODUCT_ABLATION_QR::{space}'] = {
            f'{space}::QR_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::QR_FUTURE_FULL': -1.0,
        }
        defs[f'PRODUCT_DID::{space}'] = {
            f'{space}::QR_FUTURE_NO_PRODUCT': 1.0,
            f'{space}::QR_FUTURE_FULL': -1.0,
            f'{space}::R_FUTURE_NO_PRODUCT': -1.0,
            f'{space}::R_FUTURE_FULL': 1.0,
        }

    baseline_for = {
        'R_FUTURE_FULL': 'RANK_ONLY_FULL_MATCHED',
        'R_FUTURE_NO_PRODUCT': 'RANK_ONLY_NO_PRODUCT_MATCHED',
        'QR_FUTURE_FULL': 'RANK_ONLY_FULL_MATCHED',
        'QR_FUTURE_NO_PRODUCT': 'RANK_ONLY_NO_PRODUCT_MATCHED',
    }
    for future_cond, base_cond in baseline_for.items():
        defs[f'ORCA_X_FUTURE_GAIN::{future_cond}'] = {
            f'ORCA::{future_cond}': 1.0,
            f'ORCA::{base_cond}': -1.0,
            f'DENSE::{future_cond}': -1.0,
            f'DENSE::{base_cond}': 1.0,
        }

    for suffix, qr, r in (
        ('FULL', 'QR_FUTURE_FULL', 'R_FUTURE_FULL'),
        ('NO_PRODUCT', 'QR_FUTURE_NO_PRODUCT', 'R_FUTURE_NO_PRODUCT'),
    ):
        defs[f'ORCA_X_QR_MINUS_R::{suffix}'] = {
            f'ORCA::{qr}': 1.0,
            f'ORCA::{r}': -1.0,
            f'DENSE::{qr}': -1.0,
            f'DENSE::{r}': 1.0,
        }

    defs['ORCA_X_PRODUCT_DID'] = {
        'ORCA::QR_FUTURE_NO_PRODUCT': 1.0,
        'ORCA::QR_FUTURE_FULL': -1.0,
        'ORCA::R_FUTURE_NO_PRODUCT': -1.0,
        'ORCA::R_FUTURE_FULL': 1.0,
        'DENSE::QR_FUTURE_NO_PRODUCT': -1.0,
        'DENSE::QR_FUTURE_FULL': 1.0,
        'DENSE::R_FUTURE_NO_PRODUCT': 1.0,
        'DENSE::R_FUTURE_FULL': -1.0,
    }
    return defs


def summarize_contrasts(labels: List[str], macro: np.ndarray,
                        boot_means: np.ndarray) -> dict:
    idx = {name: i for i, name in enumerate(labels)}
    result = {}
    for cname, coeffs in contrast_definitions().items():
        point = np.zeros(len(METRICS), dtype=np.float64)
        boot = np.zeros((boot_means.shape[0], len(METRICS)), dtype=np.float64)
        for label, weight in coeffs.items():
            req(label in idx, f'contrast references unknown label: {label}')
            point += weight * macro[idx[label]]
            boot += weight * boot_means[:, idx[label], :]
        lo = np.quantile(boot, 0.025, axis=0)
        hi = np.quantile(boot, 0.975, axis=0)
        result[cname] = {
            'coefficients': coeffs,
            'metrics': {
                metric: {
                    'delta': float(point[j]),
                    'ci95_lower': float(lo[j]),
                    'ci95_upper': float(hi[j]),
                    'ci_excludes_zero': bool(lo[j] > 0.0 or hi[j] < 0.0),
                }
                for j, metric in enumerate(METRICS)
            },
            'primary_metric': 'MRR',
            'primary_mrr_ci_excludes_zero': bool(lo[0] > 0.0 or hi[0] < 0.0),
        }
    return result


def run_statistics(labels: List[str], ranks: np.ndarray, dids: np.ndarray,
                   boot_idx: np.ndarray) -> dict:
    unique_dids, counts, qmetrics, per_dialogue, macro, micro = dialogue_metrics_from_ranks(ranks, dids)
    boot_means = bootstrap_dialogue_means(per_dialogue, boot_idx)
    return {
        'unique_dialogue_ids': unique_dids,
        'dialogue_query_counts': counts,
        'query_metrics': qmetrics,
        'dialogue_metrics': per_dialogue,
        'dialogue_macro': macro,
        'query_micro': micro,
        'bootstrap_means': boot_means,
        'models_summary': summarize_models(labels, macro, micro, boot_means),
        'contrasts_summary': summarize_contrasts(labels, macro, boot_means),
    }


def write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def prepare() -> None:
    req(not PROTOCOL_OUT.exists(), f'protocol already exists; refusing overwrite: {PROTOCOL_OUT}')
    req(not OUT_DIR.exists(), f'output directory already exists; refusing overwrite: {OUT_DIR}')
    print('=' * 88)
    print('STEP 2F18Q: DENSE x ORCA JOINT PAIRED STATISTICS — PREPARE')
    print('=' * 88)
    validated = load_and_validate_all()
    protocol = protocol_payload(validated)
    protocol['evaluator_script_sha256'] = sha256_file(Path(__file__))
    atomic_json(PROTOCOL_OUT, protocol)
    protocol_sha = sha256_file(PROTOCOL_OUT)

    print('CANDIDATE_ORDER_SHA256 =', validated['audit']['candidate_order_sha256'])
    print('DENSE_SEVEN_CONDITION_RANK_REPRODUCTION = PASS')
    print('DENSE_RANK_ONLY_MANIFEST_OOF_SHA = PASS')
    print('ORCA_RUN_MANIFEST_OOF_SHA = PASS')
    print('ALL_12_OOF_SCORE_FILES = PASS')
    print('DIALOGUE_FOLD_INTEGRITY = PASS')
    print('FUTURE_LABEL_ALIGNMENT = PASS')
    print('INPUT_HASH_COUNT =', len(validated['input_hashes']))
    print('PROTOCOL =', PROTOCOL_OUT)
    print('PROTOCOL_SHA256 =', protocol_sha)
    print('STEP_2F18Q_PREPARE_STATUS = PASS')
    print('TRAINING = NO')
    print('CHECKPOINT_MODIFICATION = NO')
    print('TEST_PAYLOAD_USED = NO')
    print('=' * 88)


def evaluate() -> None:
    req(PROTOCOL_OUT.is_file(), f'frozen protocol missing: {PROTOCOL_OUT}; run --prepare first')
    req(not OUT_DIR.exists(), f'output directory already exists; refusing overwrite: {OUT_DIR}')
    protocol = json_load(PROTOCOL_OUT)
    req(protocol.get('status') == 'FROZEN_BEFORE_STATISTICAL_EVALUATION', 'unexpected joint protocol status')
    req(sha256_file(Path(__file__)) == protocol.get('evaluator_script_sha256'), 'evaluator script changed after protocol freeze')
    req(sha256_file(ORCA_PROTOCOL) == EXPECTED_ORCA_PROTOCOL_SHA, 'ORCA protocol changed after freeze')

    print('=' * 88)
    print('STEP 2F18Q: DENSE x ORCA JOINT PAIRED STATISTICS — EVALUATE')
    print('=' * 88)
    validated = load_and_validate_all()
    req(validated['input_hashes'] == protocol['input_sha256'], 'one or more frozen inputs changed after --prepare')
    req(validated['audit']['candidate_order_sha256'] == protocol['data']['candidate_order_sha256'], 'candidate order changed after --prepare')

    labels = validated['labels']
    dids = validated['dids']
    strict = validated['ranks_strict']
    ge = validated['ranks_ge']
    ties = validated['ties']

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, N_DIALOGUES, size=(N_BOOT, N_DIALOGUES), dtype=np.int32)
    boot_idx_sha = hashlib.sha256(boot_idx.tobytes(order='C')).hexdigest()

    strict_stats = run_statistics(labels, strict, dids, boot_idx)
    ge_stats = run_statistics(labels, ge, dids, boot_idx)

    summary = {
        'artifact': 'FSAR S12C STEP 2F18Q Dense-ORCA Joint Paired Statistics V1',
        'status': 'COMPLETED',
        'protocol_sha256': sha256_file(PROTOCOL_OUT),
        'candidate_order_sha256': validated['audit']['candidate_order_sha256'],
        'data': protocol['data'],
        'ranking': protocol['ranking'],
        'statistics': {
            **protocol['statistics'],
            'bootstrap_indices_sha256': boot_idx_sha,
        },
        'strict_greater': {
            'models': strict_stats['models_summary'],
            'contrasts': strict_stats['contrasts_summary'],
        },
        'greater_equal_sensitivity': {
            'models': ge_stats['models_summary'],
            'contrasts': ge_stats['contrasts_summary'],
        },
        'audit': validated['audit'],
        'training': False,
        'checkpoint_modification': False,
        'checkpoint_reselection': False,
        'dev_payload_used': False,
        'test_payload_used': False,
        'test_time_fitting': False,
        'interpretation_limit': 'Internal train-universe OOF comparison on the frozen 101-candidate Gold-injected set; it is not external test generalization or end-to-end library retrieval coverage.',
    }

    stage_dir = OUT_DIR.with_name(OUT_DIR.name + '.tmp')
    req(not stage_dir.exists(), f'staging directory already exists; inspect before removing: {stage_dir}')
    stage_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = stage_dir / 'predictions_v1.npz'
    arrays_path = stage_dir / 'statistics_arrays_v1.npz'
    summary_path = stage_dir / 'summary_v1.json'
    report_path = stage_dir / 'report_v1.txt'
    manifest_path = stage_dir / 'run_manifest_v1.json'

    write_npz_atomic(
        predictions_path,
        condition_labels=np.asarray(labels, dtype='U64'),
        query_ids=validated['qids'],
        dialogue_ids=validated['dids'],
        fold_id=validated['fold'],
        ranks_strict_greater=strict,
        ranks_greater_equal=ge,
        gold_tie_counts=ties,
    )
    write_npz_atomic(
        arrays_path,
        condition_labels=np.asarray(labels, dtype='U64'),
        metric_names=np.asarray(METRICS, dtype='U16'),
        unique_dialogue_ids=strict_stats['unique_dialogue_ids'],
        dialogue_query_counts=strict_stats['dialogue_query_counts'],
        dialogue_metrics_strict_greater=strict_stats['dialogue_metrics'],
        dialogue_macro_strict_greater=strict_stats['dialogue_macro'],
        query_micro_strict_greater=strict_stats['query_micro'],
        bootstrap_means_strict_greater=strict_stats['bootstrap_means'],
        dialogue_metrics_greater_equal=ge_stats['dialogue_metrics'],
        dialogue_macro_greater_equal=ge_stats['dialogue_macro'],
        query_micro_greater_equal=ge_stats['query_micro'],
        bootstrap_means_greater_equal=ge_stats['bootstrap_means'],
    )
    atomic_json(summary_path, summary)

    lines = [
        'STEP 2F18Q: DENSE x ORCA JOINT PAIRED STATISTICS',
        '=' * 88,
        f"queries={N}; dialogues={N_DIALOGUES}; folds={N_FOLDS}; candidates={N_CANDIDATES}",
        f"candidate_order_sha256={validated['audit']['candidate_order_sha256']}",
        f"bootstrap_replicates={N_BOOT}; seed={BOOT_SEED}; unit=dialogue; shared_indices=YES",
        'PRIMARY_RANK_POLICY=STRICT_GREATER',
        'SENSITIVITY_RANK_POLICY=GREATER_EQUAL',
        '',
        'STRICT_GREATER — DIALOGUE-MACRO METRICS',
    ]
    for name, md in strict_stats['models_summary'].items():
        vals = [f"{m}={md['dialogue_macro'][m]['estimate']:.6f} [{md['dialogue_macro'][m]['ci95_lower']:.6f},{md['dialogue_macro'][m]['ci95_upper']:.6f}]" for m in METRICS]
        lines.append(name + ': ' + ' | '.join(vals))
    lines += ['', 'STRICT_GREATER — PAIRED CONTRASTS (positive means first-side advantage)']
    for cname, cd in strict_stats['contrasts_summary'].items():
        vals = [f"{m}={cd['metrics'][m]['delta']:+.6f} [{cd['metrics'][m]['ci95_lower']:+.6f},{cd['metrics'][m]['ci95_upper']:+.6f}]" for m in METRICS]
        lines.append(cname + ': ' + ' | '.join(vals))
    lines += ['', 'GREATER_EQUAL SENSITIVITY — PRIMARY MRR CONTRASTS']
    for cname, cd in ge_stats['contrasts_summary'].items():
        m = cd['metrics']['MRR']
        lines.append(f"{cname}: MRR={m['delta']:+.6f} [{m['ci95_lower']:+.6f},{m['ci95_upper']:+.6f}] ci_excludes_zero={m['ci_excludes_zero']}")
    lines += [
        '',
        'TRAINING=NO',
        'CHECKPOINT_MODIFICATION=NO',
        'CHECKPOINT_RESELECTION=NO',
        'DEV_PAYLOAD_USED=NO',
        'TEST_PAYLOAD_USED=NO',
        'TEST_TIME_FITTING=NO',
        'MODEL_INFERENCE=NO',
        'RANK_RECOMPUTATION_FROM_FROZEN_OOF_SCORES=YES',
        'STEP_2F18Q_EVALUATION_STATUS=COMPLETED',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    manifest = {
        'artifact': 'FSAR S12C STEP 2F18Q Run Manifest V1',
        'status': 'COMPLETED',
        'protocol_sha256': sha256_file(PROTOCOL_OUT),
        'evaluator_script_sha256': sha256_file(Path(__file__)),
        'input_sha256': protocol['input_sha256'],
        'outputs': {
            'predictions_v1.npz': sha256_file(predictions_path),
            'statistics_arrays_v1.npz': sha256_file(arrays_path),
            'summary_v1.json': sha256_file(summary_path),
            'report_v1.txt': sha256_file(report_path),
        },
        'queries': N,
        'dialogues': N_DIALOGUES,
        'conditions': labels,
        'training': False,
        'checkpoint_modification': False,
        'checkpoint_reselection': False,
        'test_payload_used': False,
    }
    atomic_json(manifest_path, manifest)
    os.replace(stage_dir, OUT_DIR)

    print('STEP_2F18Q_EVALUATION_STATUS = COMPLETED')
    print('OUTPUT_DIR =', OUT_DIR)
    print('SUMMARY =', summary_path)
    print('REPORT =', report_path)
    print('PREDICTIONS =', predictions_path)
    print('STATISTICS_ARRAYS =', arrays_path)
    print('TRAINING = NO')
    print('CHECKPOINT_MODIFICATION = NO')
    print('TEST_PAYLOAD_USED = NO')
    print('=' * 88)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == '__main__':
    main()
