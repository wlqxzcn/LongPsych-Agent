#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import s7_beta8_oof_evaluation as legacy  # frozen S7 future-state metric implementation

PROTO = ROOT / 'protocols/fsar_s12c_matched_ablation_evaluation_v1.json'
BASE_PROTO = ROOT / 'protocols/fsar_s12c_matched_ablation_v1.json'
OUT = ROOT / 'results/phase_s12c/evaluation_v1'
NBOOT = 5000
R_METRICS = ['MRR', 'R@1', 'R@5', 'R@10', 'R@20']
F_METRICS = ['brier', 'ordinal_logloss', 'expected_state_mae', 'threshold_accuracy']
C_METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy', 'weighted_f1', 'ordinal_mae']

# Fixed order: the DID calculation below depends on these four indices.
CONDITIONS = [
    ('R_FUTURE_FULL', 'r_future_full_oof', 'r_future_full_summary'),
    ('R_FUTURE_NO_PRODUCT', 'r_future_no_product_oof', 'r_future_no_product_summary'),
    ('QR_FUTURE_FULL', 'qr_future_full_oof', 'qr_future_full_summary'),
    ('QR_FUTURE_NO_PRODUCT', 'qr_future_no_product_oof', 'qr_future_no_product_summary'),
]
PATHS = {
    'master': 'data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl',
    'fold': 'data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy',
    'future_mask': 'data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz',
    'reservoir': 'cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz',
    's12c_model': 'src/models/s12c_matched_ablation_models.py',
    's12c_architecture_protocol': 'protocols/fsar_s12c_matched_ablation_v1.json',
    's12c_training_script': 'src/training/s12c_train_matched_ablation_beta8_5fold.py',
    'r_future_full_oof': 'results/phase_s12c/s12c_beta8_r_future_full_oof_scores_v1.npz',
    'r_future_full_summary': 'results/phase_s12c/s12c_beta8_r_future_full_training_summary_v1.json',
    'r_future_no_product_oof': 'results/phase_s12c/s12c_beta8_r_future_no_product_oof_scores_v1.npz',
    'r_future_no_product_summary': 'results/phase_s12c/s12c_beta8_r_future_no_product_training_summary_v1.json',
    'qr_future_full_oof': 'results/phase_s12c/s12c_beta8_qr_future_full_oof_scores_v1.npz',
    'qr_future_full_summary': 'results/phase_s12c/s12c_beta8_qr_future_full_training_summary_v1.json',
    'qr_future_no_product_oof': 'results/phase_s12c/s12c_beta8_qr_future_no_product_oof_scores_v1.npz',
    'qr_future_no_product_summary': 'results/phase_s12c/s12c_beta8_qr_future_no_product_training_summary_v1.json',
}
EXPECTED = {
    'master': 'd9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be',
    'fold': 'a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9',
    'future_mask': '84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d',
    'reservoir': 'e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0',
    's12c_model': 'be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c',
    's12c_architecture_protocol': '2e954b30aa028b997d0528fb3657dbe634e8c0043aa0c1c7bda4e819a329bce1',
    's12c_training_script': '5730ccc5fc49f9a91c134a2ffa6bc06c374cb4bb0b3618c12a2a45b9768f908d',
    'r_future_full_oof': 'c2b9cabf149f7517cc6457f81fe0a8defa6b74ec43b83a5b22725213f24732ac',
    'r_future_full_summary': '3a4c8ac31e5e7d6e112e203e996a09dc52682a3392d3a24d3055f2149e714bc5',
    'r_future_no_product_oof': 'e0500cf39d75002569e6bf3fb6269489bf7f303ada7c545bc699d77c78cab55c',
    'r_future_no_product_summary': '2a1a9e879b5b8e0db8f545647704285f68b62aa85d919ac3613ee51cd552058c',
    'qr_future_full_oof': '93e0b536906c6e308e74b07635da2a00eaa774e928bffa3bbeaea7142a46f3fc',
    'qr_future_full_summary': '966a99b14f2ab354bed82e42c0a0ba0a1b4e2dd1e22d25b7cb741e1ad6617011',
    'qr_future_no_product_oof': 'cb90dd863d073a63f5645e227d3a12349e54640807b1cb7372a4d1a8be12d3b9',
    'qr_future_no_product_summary': '23d9d0ef739943ea8673f7f1cfd7d4ca731fce5a152e4173f596bc5aad308036',
}
SHARED_KEYS = ['query_embedding', 'response_embedding', 'master', 'fold', 'future_mask',
               'reservoir', 'training_protocol', 's5_protocol', 's7_protocol', 's4_summary', 's4_oof']
FILES = ['s12c_matched_ablation_evaluation_summary_v1.json',
         's12c_matched_ablation_evaluation_arrays_v1.npz',
         's12c_matched_ablation_evaluation_report_v1.txt']


def req(ok, msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def inventory():
    got = {}
    for name, rel in PATHS.items():
        path = ROOT / rel
        req(path.is_file(), f'Missing input: {name} ({path})')
        got[name] = sha(path)
        req(got[name] == EXPECTED[name],
            f'SHA mismatch {name}: expected={EXPECTED[name]} actual={got[name]}')
        print('SHA_PASS', name, got[name])
    return got


def freeze_protocol():
    req(not PROTO.exists(), f'Evaluation protocol already exists: {PROTO}')
    req(not OUT.exists(), f'Output directory already exists: {OUT}')
    got = inventory()
    base = read_json(BASE_PROTO)
    req(base.get('status') == 'FROZEN_BEFORE_TRAINING', 'S12C base protocol status mismatch')
    req(base['source_artifacts']['s12c_model_sha256'] == got['s12c_model'], 'Model/protocol hash mismatch')
    req(base['data']['ranking_rows'] == 62889 and base['data']['future_available_rows'] == 17458,
        'S12C base protocol data counts mismatch')
    protocol = {
        'project': 'PsychResponse_FSAR',
        'phase': 'S12C_MATCHED_ABLATION_OOF_EVALUATION',
        'artifact': 'FSAR S12C Matched Ranking Interaction Ablation Evaluation V1',
        'status': 'FROZEN_BEFORE_S12C_METRIC_REVEAL',
        'scope': 'TRAIN_INTERNAL_OOF_ONLY',
        'interpretation': 'EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE',
        'evaluation_script': 'src/evaluation/s12c_matched_ablation_oof_evaluation.py',
        'evaluation_script_sha256': sha(__file__),
        'inputs': {k: {'path': PATHS[k], 'sha256': got[k]} for k in PATHS},
        'conditions': [x[0] for x in CONDITIONS],
        'retrieval': {'candidate_count': 101,
            'rank_definition': '1 + COUNT(candidate_score > gold_score)',
            'tie_policy': 'STRICT_GREATER', 'primary_metric': 'DIALOGUE_MACRO_MRR',
            'metrics': R_METRICS,
            'bootstrap': {'unit': 'dialogue_id', 'paired': True, 'replicates': NBOOT,
                          'seed': 42, 'ci': 0.95, 'method': 'PERCENTILE'}},
        'future_state': {'eligible_rows': 17458, 'eligible_dialogues': 1653,
            'classes': [-2, -1, 0, 1, 2], 'thresholds': [-2, -1, 0, 1],
            'classification_metrics': C_METRICS, 'probability_metrics': F_METRICS,
            'bootstrap': {'unit': 'dialogue_id', 'paired': True, 'replicates': NBOOT,
                          'seed': 43, 'ci': 0.95, 'method': 'PERCENTILE'}},
        'planned_contrasts': {
            'R_FUTURE': 'R_FUTURE_NO_PRODUCT - R_FUTURE_FULL',
            'QR_FUTURE': 'QR_FUTURE_NO_PRODUCT - QR_FUTURE_FULL',
            'difference_in_differences': '(QR_FUTURE_NO_PRODUCT - QR_FUTURE_FULL) - (R_FUTURE_NO_PRODUCT - R_FUTURE_FULL)'},
        'outputs': ['results/phase_s12c/evaluation_v1/' + x for x in FILES],
        'dev_payload_used': False, 'test_payload_used': False,
        'no_posthoc_success_gate': True, 'overwrite_policy': 'REFUSE_IF_OUTPUT_DIRECTORY_EXISTS'}
    PROTO.parent.mkdir(parents=True, exist_ok=True)
    PROTO.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\nS12C_EVALUATION_PROTOCOL_FROZEN')
    print('PROTOCOL =', PROTO, '\nPROTOCOL_SHA256 =', sha(PROTO))
    print('SCRIPT_SHA256 =', sha(__file__), '\nMETRIC_COMPUTATION = NO')


def preflight():
    req(PROTO.is_file(), 'Missing frozen S12C evaluation protocol')
    p = read_json(PROTO)
    req(p.get('status') == 'FROZEN_BEFORE_S12C_METRIC_REVEAL', 'Evaluation protocol status mismatch')
    req(p.get('evaluation_script_sha256') == sha(__file__), 'Evaluation script SHA mismatch; do not edit after freeze')
    req(not OUT.exists(), f'Output collision; refusing overwrite: {OUT}')
    actual = {}
    for name, info in p['inputs'].items():
        path = ROOT / info['path']
        req(path.is_file(), f'Missing input: {name}')
        actual[name] = sha(path)
        req(actual[name] == info['sha256'], f'Frozen input SHA mismatch: {name}')
        print('SHA_PASS', name, actual[name])

    base = read_json(BASE_PROTO)
    mp = ROOT / p['inputs']['master']['path']
    with mp.open(encoding='utf-8') as f: master = [json.loads(x) for x in f if x.strip()]
    n = len(master)
    req(n == 62889, f'Unexpected master rows: {n}')
    dids = np.asarray([int(x['dialogue_id']) for x in master], dtype=np.int64)
    qids = np.asarray([str(x['query_id']) for x in master], dtype=str)
    avail = np.asarray([bool(x['future_value_available']) for x in master], dtype=bool)
    y = np.asarray([int(x['future_value_label']) if avail[i] else -2 for i, x in enumerate(master)], dtype=np.int64)
    req(np.unique(dids).size == 1666, 'Dialogue count mismatch')
    req(int(avail.sum()) == 17458, 'Future eligible row count mismatch')
    req(np.unique(dids[avail]).size == 1653, 'Future eligible dialogue count mismatch')
    req(set(np.unique(y[avail]).tolist()).issubset({-2, -1, 0, 1, 2}), 'Unexpected future labels')
    fold = np.load(ROOT / p['inputs']['fold']['path'], allow_pickle=False)
    req(fold.shape == (n,), 'Fold shape mismatch')

    res = load_npz(ROOT / p['inputs']['reservoir']['path'])
    negrows, goldrows = res['candidate_response_row'], res['gold_response_row']
    master_gold = np.asarray([int(x['gold_response_row']) for x in master], dtype=np.int32)
    req(negrows.shape == (n, 100), 'Negative reservoir shape mismatch')
    req(np.array_equal(goldrows, master_gold), 'Reservoir gold response mismatch')
    req(not np.any(negrows == master_gold[:, None]), 'Gold response appears in negative reservoir')

    summaries, loaded = {}, {}
    for name, oof_key, summary_key in CONDITIONS:
        s = read_json(ROOT / p['inputs'][summary_key]['path'])
        req(s.get('status') == 'PASS' and float(s.get('beta')) == 8.0, f'{name}: training status/beta mismatch')
        req(s.get('dev_payload_used') is False and s.get('test_payload_used') is False, f'{name}: dev/test payload used')
        req(s.get('formal_outputs_promoted') is True, f'{name}: formal outputs not promoted')
        req(s.get('ranking_rows') == n and s.get('future_value_rows') == int(avail.sum()), f'{name}: row count mismatch')
        summaries[name] = s
        d = load_npz(ROOT / p['inputs'][oof_key]['path'])
        for key in ['candidate_scores', 'ranking_ce', 'future_logits', 'fold_id', 'gold_position', 'future_available', 'future_value_label']:
            req(key in d, f'{name}: missing OOF array {key}')
        req(d['candidate_scores'].shape == (n, 101) and np.isfinite(d['candidate_scores']).all(), f'{name}: candidate score shape/value error')
        req(d['ranking_ce'].shape == (n,) and np.isfinite(d['ranking_ce']).all(), f'{name}: ranking CE shape/value error')
        req(d['future_logits'].shape == (n, 4) and np.isfinite(d['future_logits']).all(), f'{name}: future logits shape/value error')
        req(np.array_equal(d['fold_id'], fold), f'{name}: fold mismatch')
        req(np.all(d['gold_position'] == 0), f'{name}: gold position mismatch')
        req(np.array_equal(d['future_available'], avail), f'{name}: future availability mismatch')
        req(np.array_equal(d['future_value_label'][avail], y[avail]), f'{name}: future label mismatch')
        loaded[name] = d

    for key in SHARED_KEYS:
        vals = [summaries[name]['input_sha256'][key] for name, _, _ in CONDITIONS]
        req(len(set(vals)) == 1, f'Training summary lineage mismatch: {key}')
        req(vals[0] == base['inherited_training_dependencies'][key], f'Inherited dependency mismatch: {key}')
    ref = summaries[CONDITIONS[0][0]]['input_sha256']
    req(ref['model_code'] == actual['s12c_model'], 'S12C model-code provenance mismatch')
    req(ref['architecture_protocol'] == actual['s12c_architecture_protocol'], 'S12C architecture-protocol provenance mismatch')
    for name, _, _ in CONDITIONS[1:]:
        for key in ['fold_id', 'future_available', 'future_value_label', 'gold_position']:
            req(np.array_equal(loaded[CONDITIONS[0][0]][key], loaded[name][key]), f'Cross-model alignment failure: {key}/{name}')

    print('\nS12C_EVALUATION_PREFLIGHT_PASS')
    print('QUERY_ROWS =', n, '\nDIALOGUES =', np.unique(dids).size)
    print('FUTURE_ROWS =', int(avail.sum()), '\nFUTURE_DIALOGUES =', np.unique(dids[avail]).size)
    print('MODELS = 4\nCANDIDATES_PER_QUERY = 101\nOUTPUT_COLLISION = NONE\nMETRIC_COMPUTATION = NO')
    return p, actual, master, qids, dids, fold, avail, y, loaded


def boot_means(dlg, seed):
    nd, nc, nm = dlg.shape
    rng = np.random.default_rng(seed)
    out = np.empty((NBOOT, nc, nm), dtype=np.float64)
    for b in range(NBOOT): out[b] = dlg[rng.integers(0, nd, size=nd)].mean(axis=0)
    return out


def contrasts(boots, points, metrics, directions):
    specs = [
        ('R_FUTURE_NO_PRODUCT_MINUS_FULL', [(1, 1), (0, -1)]),
        ('QR_FUTURE_NO_PRODUCT_MINUS_FULL', [(3, 1), (2, -1)]),
        ('DIFFERENCE_IN_DIFFERENCES', [(3, 1), (2, -1), (1, -1), (0, 1)]),
    ]
    out = []
    for name, terms in specs:
        bd = sum(coef * boots[:, idx, :] for idx, coef in terms)
        pd = sum(coef * points[idx, :] for idx, coef in terms)
        for j, metric in enumerate(metrics):
            lo, hi = np.quantile(bd[:, j], [0.025, 0.975])
            out.append({'contrast': name, 'metric': metric, 'delta': float(pd[j]),
                        'ci95_lower': float(lo), 'ci95_upper': float(hi),
                        'direction_of_improvement': directions[metric],
                        'status': 'EXPLORATORY_NO_CONFIRMATORY_GATE'})
    return out


def evaluate(data):
    p, actual, master, qids, dids, fold, avail, y, loaded = data
    names = [x[0] for x in CONDITIONS]
    n = len(master)
    ranks = np.empty((4, n), dtype=np.int16)
    ties = np.empty_like(ranks)
    qm = np.empty((n, 4, 5), dtype=np.float64)
    for c, name in enumerate(names):
        s = loaded[name]['candidate_scores']
        ranks[c] = 1 + (s[:, 1:] > s[:, 0:1]).sum(axis=1)
        ties[c] = (s[:, 1:] == s[:, 0:1]).sum(axis=1)
        qm[:, c, 0] = 1.0 / ranks[c]
        for j, k in enumerate([1, 5, 10, 20], 1): qm[:, c, j] = ranks[c] <= k

    ud, inv = np.unique(dids, return_inverse=True)
    counts = np.bincount(inv, minlength=len(ud))
    dlg = np.empty((len(ud), 4, 5), dtype=np.float64)
    for c in range(4):
        for m in range(5): dlg[:, c, m] = np.bincount(inv, weights=qm[:, c, m], minlength=len(ud)) / counts
    rmacro, rmicro = dlg.mean(axis=0), qm.mean(axis=0)
    rb = boot_means(dlg, 42)
    rdirs = {m: 'higher_is_better' for m in R_METRICS}
    rcontr = contrasts(rb, rmacro, R_METRICS, rdirs)
    rsummary, tiesummary = {}, {}
    for i, name in enumerate(names):
        rsummary[name] = {'dialogue_macro': {m: float(rmacro[i, j]) for j, m in enumerate(R_METRICS)},
                          'query_micro': {m: float(rmicro[i, j]) for j, m in enumerate(R_METRICS)}}
        tiesummary[name] = {'queries_with_gold_ties': int((ties[i] > 0).sum()),
                            'total_tied_non_gold_candidates': int(ties[i].sum()),
                            'max_tied_candidates_per_query': int(ties[i].max())}

    freports, fpreds, frows, fdlg, fdlg_ids = {}, [], [], [], None
    for name in names:
        rep, pred, dlgmet, this_ids, rowmet = legacy.future_metrics(
            name, loaded[name]['future_logits'], y, avail, fold, dids)
        if fdlg_ids is None: fdlg_ids = this_ids
        else: req(np.array_equal(fdlg_ids, this_ids), f'Future dialogue universe mismatch: {name}')
        freports[name] = rep
        fpreds.append(pred); fdlg.append(dlgmet); frows.append(rowmet)
    fdall = np.stack(fdlg, axis=1)
    fmacro = fdall.mean(axis=0)
    fb = boot_means(fdall, 43)
    fdirs = {'brier': 'lower_is_better', 'ordinal_logloss': 'lower_is_better',
             'expected_state_mae': 'lower_is_better', 'threshold_accuracy': 'higher_is_better'}
    fcontr = contrasts(fb, fmacro, F_METRICS, fdirs)
    for i, name in enumerate(names):
        freports[name]['dialogue_macro_probability_metrics'] = {m: float(fmacro[i, j]) for j, m in enumerate(F_METRICS)}

    summary = {
        'artifact': 'FSAR S12C Matched Ranking Interaction Ablation Evaluation V1',
        'status': 'PASS', 'scope': 'TRAIN_INTERNAL_OOF_ONLY',
        'interpretation': 'EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE',
        'protocol_sha256': sha(PROTO), 'evaluation_script_sha256': sha(__file__),
        'input_sha256': actual, 'query_rows': int(n), 'dialogues': int(len(ud)),
        'candidate_count': 101, 'future_eligible_rows': int(avail.sum()),
        'future_eligible_dialogues': int(len(fdlg_ids)),
        'retrieval': {'conditions': names, 'primary_metric': 'DIALOGUE_MACRO_MRR',
            'rank_definition': '1 + COUNT(candidate_score > gold_score)', 'tie_policy': 'STRICT_GREATER',
            'condition_metrics': rsummary, 'tie_diagnostics': tiesummary,
            'bootstrap': {'unit': 'dialogue_id', 'paired': True, 'replicates': NBOOT,
                'seed': 42, 'ci': 0.95, 'method': 'PERCENTILE', 'contrasts': rcontr}},
        'future_state': {'conditions': names, 'classification_metrics': freports,
            'probability_metric_names': F_METRICS,
            'probability_metric_dialogue_macro': {
                name: {m: float(fmacro[i, j]) for j, m in enumerate(F_METRICS)} for i, name in enumerate(names)},
            'bootstrap': {'unit': 'dialogue_id', 'paired': True, 'replicates': NBOOT,
                'seed': 43, 'ci': 0.95, 'method': 'PERCENTILE', 'contrasts': fcontr}},
        'planned_contrasts': p['planned_contrasts'], 'dev_payload_used': False,
        'test_payload_used': False, 'no_posthoc_success_gate': True}

    OUT.mkdir(parents=True, exist_ok=False)
    stage = Path(tempfile.mkdtemp(prefix='.stage_', dir=OUT))
    try:
        (stage / FILES[0]).write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        np.savez_compressed(stage / FILES[1],
            retrieval_conditions=np.asarray(names, dtype='U32'), query_ids=qids, dialogue_ids=dids,
            fold_id=fold, ranks=ranks, gold_tie_counts=ties, dialogue_query_counts=counts,
            dialogue_metrics=dlg, dialogue_macro=rmacro, query_micro=rmicro, retrieval_bootstrap_means=rb,
            future_conditions=np.asarray(names, dtype='U32'), future_dialogue_ids=fdlg_ids,
            future_eligible_row_indices=np.flatnonzero(avail), future_available=avail,
            future_value_label=y, future_logits=np.stack([loaded[x]['future_logits'] for x in names]),
            future_predictions=np.asarray(fpreds), future_row_metrics=np.asarray(frows),
            future_dialogue_metrics=fdall, future_dialogue_macro=fmacro, future_bootstrap_means=fb)
        lines = ['FSAR S12C MATCHED RANKING INTERACTION ABLATION', 'Status: PASS',
                 'Scope: TRAIN_INTERNAL_OOF_ONLY', f'Queries={n}; dialogues={len(ud)}; candidates=101',
                 f'Future eligible rows={int(avail.sum())}; future eligible dialogues={len(fdlg_ids)}', '',
                 'RETRIEVAL DIALOGUE-MACRO']
        for name in names: lines.append(name + ': ' + ', '.join(f'{k}={v:.8f}' for k, v in rsummary[name]['dialogue_macro'].items()))
        lines += ['', 'RETRIEVAL CONTRASTS'] + [json.dumps(x, ensure_ascii=False) for x in rcontr]
        lines += ['', 'FUTURE-STATE CLASSIFICATION (ROW-LEVEL)']
        for name in names: lines.append(name + ': ' + json.dumps(freports[name]['metrics'], ensure_ascii=False))
        lines += ['', 'FUTURE-STATE PROBABILITY METRICS (DIALOGUE-MACRO)']
        for name in names: lines.append(name + ': ' + json.dumps(freports[name]['dialogue_macro_probability_metrics'], ensure_ascii=False))
        lines += ['', 'FUTURE-STATE CONTRASTS'] + [json.dumps(x, ensure_ascii=False) for x in fcontr]
        lines += ['', 'Exploratory analysis only.', 'No confirmatory success gate.', 'No dev/test payload used.',
                  'Classification metrics are row-level.', 'Probability metrics are dialogue-macro and bootstrapped by dialogue.']
        (stage / FILES[2]).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        for filename in FILES: os.replace(stage / filename, OUT / filename)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print('\nS12C_MATCHED_ABLATION_EVALUATION_PASS')
    for name in names:
        print(name, rsummary[name]['dialogue_macro'])
        print(name, freports[name]['dialogue_macro_probability_metrics'])
    print('OUTPUT =', OUT)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze-protocol', action='store_true')
    g.add_argument('--preflight-only', action='store_true')
    g.add_argument('--run-evaluation', action='store_true')
    args = ap.parse_args()
    if args.freeze_protocol:
        freeze_protocol(); return
    data = preflight()
    if args.preflight_only:
        print('\nS12C_PREFLIGHT_ONLY_PASS'); return
    if args.run_evaluation: evaluate(data)

if __name__ == '__main__': main()
