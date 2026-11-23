#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / 'protocols/fsar_s12c_seven_condition_evaluation_v1.json'
OUT = ROOT / 'results/phase_s12c/seven_condition_evaluation_v1'
NBOOT = 5000
R_METRICS = ['MRR', 'R@1', 'R@5', 'R@10', 'R@20']
R_KS = [1, 5, 10, 20]
F_METRICS = ['brier', 'ordinal_logloss', 'expected_state_mae', 'threshold_accuracy']
C_METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy', 'weighted_f1', 'ordinal_mae']
CLASSES = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
THRESHOLDS = np.asarray([-2, -1, 0, 1], dtype=np.int64)

# Seven retrieval conditions: S6A Dense + six S12C matched models.
R_CONDITIONS = [
    'DENSE_101', 'RANK_ONLY_FULL_MATCHED', 'RANK_ONLY_NO_PRODUCT_MATCHED',
    'R_FUTURE_FULL', 'R_FUTURE_NO_PRODUCT', 'QR_FUTURE_FULL', 'QR_FUTURE_NO_PRODUCT'
]
F_CONDITIONS = ['R_FUTURE_FULL', 'R_FUTURE_NO_PRODUCT', 'QR_FUTURE_FULL', 'QR_FUTURE_NO_PRODUCT']

PATHS = {
    'master': 'data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl',
    'fold': 'data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy',
    'future_mask': 'data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz',
    'reservoir': 'cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz',
    'query_embedding': 'cache/embeddings/fsar_train_queries_bge_v1.npy',
    'response_embedding': 'cache/embeddings/fsar_train_unique_responses_bge_v1.npy',
    's12c_model': 'src/models/s12c_matched_ablation_models.py',
    's12c_architecture_protocol': 'protocols/fsar_s12c_matched_ablation_v1.json',
    's12c_training_script': 'src/training/s12c_train_matched_ablation_beta8_5fold.py',
    's6a_protocol': 'protocols/fsar_s6a_oof_retrieval_reveal_protocol_v1.json',
    's6a_ranks': 'results/phase_s6/s6a_oof_retrieval_ranks_v1.npz',
    's6a_summary': 'results/phase_s6/s6a_oof_retrieval_reveal_summary_v1.json',
    'rank_only_full_oof': 'results/phase_s12c/rank_only_matched_v1/oof/s12c_rank_only_full_matched_oof_scores_v1.npz',
    'rank_only_full_summary': 'results/phase_s12c/rank_only_matched_v1/summaries/s12c_rank_only_full_matched_training_summary_v1.json',
    'rank_only_no_product_oof': 'results/phase_s12c/rank_only_matched_v1/oof/s12c_rank_only_no_product_matched_oof_scores_v1.npz',
    'rank_only_no_product_summary': 'results/phase_s12c/rank_only_matched_v1/summaries/s12c_rank_only_no_product_matched_training_summary_v1.json',
    'r_future_full_oof': 'results/phase_s12c/s12c_beta8_r_future_full_oof_scores_v1.npz',
    'r_future_full_summary': 'results/phase_s12c/s12c_beta8_r_future_full_training_summary_v1.json',
    'r_future_no_product_oof': 'results/phase_s12c/s12c_beta8_r_future_no_product_oof_scores_v1.npz',
    'r_future_no_product_summary': 'results/phase_s12c/s12c_beta8_r_future_no_product_training_summary_v1.json',
    'qr_future_full_oof': 'results/phase_s12c/s12c_beta8_qr_future_full_oof_scores_v1.npz',
    'qr_future_full_summary': 'results/phase_s12c/s12c_beta8_qr_future_full_training_summary_v1.json',
    'qr_future_no_product_oof': 'results/phase_s12c/s12c_beta8_qr_future_no_product_oof_scores_v1.npz',
    'qr_future_no_product_summary': 'results/phase_s12c/s12c_beta8_qr_future_no_product_training_summary_v1.json',
}

# Fixed hashes are grounded in the frozen S6A/S12C protocols and supplied audits.
# The S6A rank archive and the two new matched Rank-Only OOF arrays are hashed
# at protocol-freeze time because their individual file hashes were not supplied.
EXPECTED = {
    'master': 'd9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be',
    'fold': 'a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9',
    'future_mask': '84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d',
    'reservoir': 'e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0',
    'query_embedding': '18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005',
    'response_embedding': '7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f',
    's12c_model': 'be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c',
    's12c_architecture_protocol': '2e954b30aa028b997d0528fb3657dbe634e8c0043aa0c1c7bda4e819a329bce1',
    's12c_training_script': '5730ccc5fc49f9a91c134a2ffa6bc06c374cb4bb0b3618c12a2a45b9768f908d',
    's6a_protocol': 'd0cd870354e2c4285bf9bbc0a5c26fea016853293d9364db935ab5883b552e7c',
    'rank_only_full_summary': '399a23edcbc61dfbe9a074059ca858c270252dd93e9f3af0f1247365be0d90d0',
    'rank_only_no_product_summary': 'd975eaf5d314f5a14241c1c1357af72744b3b3cfe49ec296ef8843c8013b9d32',
    'r_future_full_oof': 'c2b9cabf149f7517cc6457f81fe0a8defa6b74ec43b83a5b22725213f24732ac',
    'r_future_full_summary': '3a4c8ac31e5e7d6e112e203e996a09dc52682a3392d3a24d3055f2149e714bc5',
    'r_future_no_product_oof': 'e0500cf39d75002569e6bf3fb6269489bf7f303ada7c545bc699d77c78cab55c',
    'r_future_no_product_summary': '2a1a9e879b5b8e0db8f545647704285f68b62aa85d919ac3613ee51cd552058c',
    'qr_future_full_oof': '93e0b536906c6e308e74b07635da2a00eaa774e928bffa3bbeaea7142a46f3fc',
    'qr_future_full_summary': '966a99b14f2ab354bed82e42c0a0ba0a1b4e2dd1e22d25b7cb741e1ad6617011',
    'qr_future_no_product_oof': 'cb90dd863d073a63f5645e227d3a12349e54640807b1cb7372a4d1a8be12d3b9',
    'qr_future_no_product_summary': '23d9d0ef739943ea8673f7f1cfd7d4ca731fce5a152e4173f596bc5aad308036',
}
SHARED_KEYS = ['query_embedding', 'response_embedding', 'master', 'fold', 'reservoir', 'model_code', 'architecture_protocol']
FILES = ['s12c_seven_condition_evaluation_summary_v1.json', 's12c_seven_condition_evaluation_arrays_v1.npz', 's12c_seven_condition_evaluation_report_v1.txt']

# Contrast terms are (condition_index, coefficient); positive delta favors the first named direction.
R_CONTRASTS = [
    ('RANK_ONLY_FULL_MATCHED_MINUS_DENSE_101', [(1,1),(0,-1)]),
    ('RANK_ONLY_NO_PRODUCT_MATCHED_MINUS_DENSE_101', [(2,1),(0,-1)]),
    ('R_FUTURE_FULL_MINUS_RANK_ONLY_FULL_MATCHED', [(3,1),(1,-1)]),
    ('QR_FUTURE_FULL_MINUS_RANK_ONLY_FULL_MATCHED', [(5,1),(1,-1)]),
    ('R_FUTURE_NO_PRODUCT_MINUS_RANK_ONLY_NO_PRODUCT_MATCHED', [(4,1),(2,-1)]),
    ('QR_FUTURE_NO_PRODUCT_MINUS_RANK_ONLY_NO_PRODUCT_MATCHED', [(6,1),(2,-1)]),
    ('R_FUTURE_NO_PRODUCT_MINUS_FULL', [(4,1),(3,-1)]),
    ('QR_FUTURE_NO_PRODUCT_MINUS_FULL', [(6,1),(5,-1)]),
    ('DID_QR_VS_R_NO_PRODUCT_EFFECT', [(6,1),(5,-1),(4,-1),(3,1)]),
]
F_CONTRASTS = [
    ('R_FUTURE_NO_PRODUCT_MINUS_FULL', [(1,1),(0,-1)]),
    ('QR_FUTURE_NO_PRODUCT_MINUS_FULL', [(3,1),(2,-1)]),
    ('QR_FUTURE_FULL_MINUS_R_FUTURE_FULL', [(2,1),(0,-1)]),
    ('QR_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_NO_PRODUCT', [(3,1),(1,-1)]),
    ('DID_QR_VS_R_NO_PRODUCT_EFFECT', [(3,1),(2,-1),(1,-1),(0,1)]),
]

def req(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def inputs_of(s):
    x = s.get('input_sha256', s.get('INPUT_SHA256', {}))
    req(isinstance(x, dict), 'Missing training-summary input_sha256 mapping')
    return x

def inventory():
    got = {}
    for name, rel in PATHS.items():
        path = ROOT / rel
        req(path.is_file(), f'MISSING_INPUT {name}: {path}')
        got[name] = sha(path)
        if name in EXPECTED:
            req(got[name] == EXPECTED[name], f'SHA_MISMATCH {name}: expected={EXPECTED[name]} actual={got[name]}')
        print('SHA_PASS', name, got[name])
    return got

def validate_s6a_protocol():
    p = read_json(ROOT / PATHS['s6a_protocol'])
    req(p.get('status') == 'FROZEN_BEFORE_FIRST_RETRIEVAL_METRIC_COMPUTATION', 'S6A protocol status mismatch')
    req(p.get('conditions') == ['DENSE_101','RANK_ONLY','R_FUTURE','QR_FUTURE'], 'S6A condition order mismatch')
    rd = p.get('rank_definition', {})
    req(rd.get('primary_rank') == '1 + COUNT(candidate_score > gold_score)', 'S6A rank definition mismatch')
    req(rd.get('tie_policy') == 'STRICT_GREATER', 'S6A tie policy mismatch')
    pm = p.get('primary_metric', {})
    req(pm.get('name') == 'MRR' and pm.get('aggregation') == 'DIALOGUE_MACRO', 'S6A primary metric mismatch')
    boot = p.get('bootstrap', {})
    req(boot.get('replicates') == NBOOT and boot.get('seed') == 42, 'S6A bootstrap mismatch')
    req(p.get('future_metrics_computed_in_s6a') is False, 'S6A future-metric flag mismatch')
    return p

def validate_summaries(actual):
    specs = [
        ('rank_only_full_summary','RANK_ONLY_FULL_MATCHED',False),
        ('rank_only_no_product_summary','RANK_ONLY_NO_PRODUCT_MATCHED',False),
        ('r_future_full_summary','R_FUTURE_FULL',True),
        ('r_future_no_product_summary','R_FUTURE_NO_PRODUCT',True),
        ('qr_future_full_summary','QR_FUTURE_FULL',True),
        ('qr_future_no_product_summary','QR_FUTURE_NO_PRODUCT',True),
    ]
    summaries = {}
    for key, name, future in specs:
        s = read_json(ROOT / PATHS[key])
        req(s.get('status') == 'PASS', f'{name}: summary status not PASS')
        req(int(s.get('ranking_rows',-1)) == 62889, f'{name}: ranking row count mismatch')
        req(s.get('dev_payload_used',False) is False and s.get('test_payload_used',False) is False, f'{name}: dev/test payload flag')
        if 'condition' in s: req(s['condition'] == name, f'{name}: summary condition mismatch')
        if future:
            req(float(s.get('beta',-1)) == 8.0, f'{name}: beta mismatch')
            req(int(s.get('future_value_rows',-1)) == 17458, f'{name}: future row count mismatch')
            req(s.get('formal_outputs_promoted') is True, f'{name}: formal output promotion missing')
        summaries[name] = s
    for key in SHARED_KEYS:
        vals = [inputs_of(s).get(key) for s in summaries.values()]
        req(all(v is not None for v in vals) and len(set(vals)) == 1, f'Matched lineage mismatch: {key}')
    ref = inputs_of(next(iter(summaries.values())))
    mapping = {'master':'master','fold':'fold','reservoir':'reservoir','model_code':'s12c_model',
               'architecture_protocol':'s12c_architecture_protocol','query_embedding':'query_embedding',
               'response_embedding':'response_embedding'}
    for k, input_key in mapping.items(): req(ref[k] == actual[input_key], f'Lineage SHA mismatch: {k}')
    for name in F_CONDITIONS:
        req(inputs_of(summaries[name]).get('future_mask') == actual['future_mask'], f'{name}: future-mask lineage mismatch')
    print('SIX_MODEL_SUMMARY_AND_SHARED_LINEAGE = PASS')
    return summaries

def freeze_protocol():
    req(not PROTO.exists(), f'Protocol exists; refusing overwrite: {PROTO}')
    req(not OUT.exists(), f'Output exists; refusing overwrite: {OUT}')
    actual = inventory()
    s6a = validate_s6a_protocol()
    validate_summaries(actual)
    p = {
        'project':'PsychResponse_FSAR', 'phase':'S12C_SEVEN_CONDITION_OOF_EVALUATION',
        'artifact':'FSAR S12C Seven-Condition Matched OOF Evaluation V1',
        'status':'FROZEN_BEFORE_SEVEN_CONDITION_METRIC_REVEAL',
        'scope':'TRAIN_INTERNAL_OOF_ONLY',
        'interpretation':'EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE',
        'evaluation_script':'src/evaluation/s12c_seven_condition_oof_evaluation.py',
        'evaluation_script_sha256':sha(Path(__file__)),
        'inputs':{k:{'path':PATHS[k],'sha256':actual[k]} for k in PATHS},
        'retrieval':{
            'conditions':R_CONDITIONS,'candidate_count':101,
            'rank_definition':'1 + COUNT(candidate_score > gold_score)','tie_policy':'STRICT_GREATER',
            'primary_metric':'DIALOGUE_MACRO_MRR','metrics':R_METRICS,
            'secondary_aggregation':['DIALOGUE_MACRO','QUERY_MICRO_DESCRIPTIVE'],
            'bootstrap':{'unit':'dialogue_id','paired':True,'same_resampled_dialogues_for_all_conditions':True,
                         'replicates':NBOOT,'seed':42,'ci':0.95,'method':'PERCENTILE'},
            'planned_contrasts':[{'name':n,'terms':t} for n,t in R_CONTRASTS]},
        'future_state':{
            'conditions':F_CONDITIONS,'eligible_rows':17458,'eligible_dialogues':1653,
            'classes':CLASSES.tolist(),'thresholds':THRESHOLDS.tolist(),
            'classification_metrics':C_METRICS,'probability_metrics':F_METRICS,
            'classification_aggregation':'ROW_LEVEL','probability_aggregation':'DIALOGUE_MACRO',
            'bootstrap':{'unit':'dialogue_id','paired':True,'same_resampled_dialogues_for_all_conditions':True,
                         'replicates':NBOOT,'seed':43,'ci':0.95,'method':'PERCENTILE'},
            'planned_contrasts':[{'name':n,'terms':t} for n,t in F_CONTRASTS]},
        'multiple_comparison_adjustment':'NONE_EXPLORATORY_CI_ONLY',
        'no_posthoc_success_gate':True,'dev_payload_used':False,'test_payload_used':False,
        'outputs':['results/phase_s12c/seven_condition_evaluation_v1/'+x for x in FILES],
        'overwrite_policy':'REFUSE_IF_PROTOCOL_OR_OUTPUT_EXISTS','s6a_protocol_status':s6a['status']}
    PROTO.parent.mkdir(parents=True, exist_ok=True)
    PROTO.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\nSEVEN_CONDITION_PROTOCOL_FROZEN')
    print('PROTOCOL =',PROTO,'\nPROTOCOL_SHA256 =',sha(PROTO))
    print('SCRIPT_SHA256 =',sha(Path(__file__)),'\nMETRIC_COMPUTATION = NO')

def preflight():
    req(PROTO.is_file(), f'Missing protocol: {PROTO}')
    p = read_json(PROTO)
    req(p.get('status') == 'FROZEN_BEFORE_SEVEN_CONDITION_METRIC_REVEAL','Protocol status mismatch')
    req(p.get('evaluation_script_sha256') == sha(Path(__file__)), 'Script SHA mismatch; do not edit after freeze')
    req(not OUT.exists(), f'Output collision; refusing overwrite: {OUT}')
    actual = {}
    for name, info in p['inputs'].items():
        path = ROOT / info['path']; req(path.is_file(),f'Missing input {name}: {path}')
        actual[name] = sha(path); req(actual[name] == info['sha256'],f'Frozen input SHA mismatch {name}')
        print('SHA_PASS',name,actual[name])
    validate_s6a_protocol(); validate_summaries(actual)
    with (ROOT/PATHS['master']).open(encoding='utf-8') as f: master=[json.loads(x) for x in f if x.strip()]
    n=len(master); req(n==62889,f'Master rows={n}')
    qids=np.asarray([str(x['query_id']) for x in master],dtype=str)
    dids=np.asarray([int(x['dialogue_id']) for x in master],dtype=np.int64)
    if 'ranking_row' in master[0]:
        rr=np.asarray([int(x['ranking_row']) for x in master],dtype=np.int64)
        req(np.array_equal(rr,np.arange(n)),'Master ranking-row order mismatch')
    req(len(np.unique(qids))==n,'Duplicate query IDs')
    req(np.unique(dids).size==1666,'Dialogue count mismatch')
    avail=np.asarray([bool(x['future_value_available']) for x in master],dtype=bool)
    y=np.asarray([int(x['future_value_label']) if avail[i] else -2 for i,x in enumerate(master)],dtype=np.int64)
    req(int(avail.sum())==17458,'Future eligible row count mismatch')
    req(np.unique(dids[avail]).size==1653,'Future dialogue count mismatch')
    req(set(np.unique(y[avail]).tolist()).issubset(set(CLASSES.tolist())),'Unexpected future labels')
    fold=np.load(ROOT/PATHS['fold'],allow_pickle=False); req(fold.shape==(n,),'Fold shape mismatch')
    dense=load_npz(ROOT/PATHS['s6a_ranks'])
    req(dense['conditions'].astype(str).tolist()==['DENSE_101','RANK_ONLY','R_FUTURE','QR_FUTURE'],'S6A condition order mismatch')
    req(np.array_equal(dense['query_ids'].astype(str),qids),'S6A query IDs not aligned')
    req(np.array_equal(dense['dialogue_ids'],dids),'S6A dialogue IDs not aligned')
    req(np.array_equal(dense['fold_id'],fold),'S6A folds not aligned')
    req(dense['ranks'].shape==(4,n) and dense['gold_tie_counts'].shape==(4,n),'S6A rank/tie shape mismatch')
    req(np.all(dense['gold_position']==0),'S6A gold position mismatch')
    req(np.all((dense['ranks']>=1)&(dense['ranks']<=101)),'S6A rank outside [1,101]')
    req(np.all(dense['gold_tie_counts']>=0),'Negative S6A tie count')
    specs=[
        ('RANK_ONLY_FULL_MATCHED','rank_only_full_oof',False),
        ('RANK_ONLY_NO_PRODUCT_MATCHED','rank_only_no_product_oof',False),
        ('R_FUTURE_FULL','r_future_full_oof',True),
        ('R_FUTURE_NO_PRODUCT','r_future_no_product_oof',True),
        ('QR_FUTURE_FULL','qr_future_full_oof',True),
        ('QR_FUTURE_NO_PRODUCT','qr_future_no_product_oof',True)]
    loaded={}
    for name,key,has_future in specs:
        d=load_npz(ROOT/PATHS[key])
        req(d['candidate_scores'].shape==(n,101),f'{name}: candidate score shape mismatch')
        req(np.isfinite(d['candidate_scores']).all(),f'{name}: non-finite scores')
        req(d['ranking_ce'].shape==(n,) and np.isfinite(d['ranking_ce']).all(),f'{name}: ranking CE invalid')
        req(np.array_equal(d['fold_id'],fold),f'{name}: fold mismatch')
        req(np.all(d['gold_position']==0),f'{name}: gold position mismatch')
        if has_future:
            req(d['future_logits'].shape==(n,4) and np.isfinite(d['future_logits']).all(),f'{name}: future logits invalid')
            req(np.array_equal(d['future_available'],avail),f'{name}: availability mismatch')
            req(np.array_equal(d['future_value_label'][avail],y[avail]),f'{name}: future labels mismatch')
        loaded[name]=d
    print('\nSEVEN_CONDITION_PREFLIGHT_PASS')
    print('QUERY_ROWS =',n,'\nDIALOGUES =',np.unique(dids).size)
    print('FUTURE_ROWS =',int(avail.sum()),'\nFUTURE_DIALOGUES =',np.unique(dids[avail]).size)
    print('RETRIEVAL_CONDITIONS =',len(R_CONDITIONS),'\nFUTURE_CONDITIONS =',len(F_CONDITIONS))
    print('OUTPUT_COLLISION = NONE\nMETRIC_COMPUTATION = NO')
    return p,actual,qids,dids,fold,avail,y,dense,loaded

def boot_means(dlg,seed):
    nd,nc,nm=dlg.shape; rng=np.random.default_rng(seed)
    out=np.empty((NBOOT,nc,nm),dtype=np.float64)
    for b in range(NBOOT): out[b]=dlg[rng.integers(0,nd,size=nd)].mean(axis=0)
    return out

def make_contrasts(boots,points,specs,metrics,directions):
    out=[]
    for name,terms in specs:
        bd=np.zeros((boots.shape[0],boots.shape[2]),dtype=np.float64)
        pd=np.zeros((points.shape[1],),dtype=np.float64)
        for idx,coef in terms: bd += coef*boots[:,idx,:]; pd += coef*points[idx,:]
        lo,hi=np.quantile(bd,[0.025,0.975],axis=0)
        for j,m in enumerate(metrics):
            out.append({'contrast':name,'metric':m,'delta':float(pd[j]),
                'ci95_lower':float(lo[j]),'ci95_upper':float(hi[j]),
                'direction_of_improvement':directions[m],
                'status':'EXPLORATORY_NO_CONFIRMATORY_GATE'})
    return out

def sigmoid(x):
    x=np.clip(np.asarray(x,dtype=np.float64),-80,80); return 1/(1+np.exp(-x))

def future_metrics(logits,y,avail,dids,fold):
    z=logits[avail].astype(np.float64); yy=y[avail]; dd=dids[avail]; ff=fold[avail]
    pred=(-2+(z>=0).sum(axis=1)).astype(np.int64)
    pr,re,f1,sup=precision_recall_fscore_support(yy,pred,labels=CLASSES,zero_division=0)
    per_class={str(int(c)):{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(sup[i])} for i,c in enumerate(CLASSES)}
    targets=(yy[:,None]>THRESHOLDS[None,:]).astype(np.int64); per_thr={}
    for j,t in enumerate(THRESHOLDS):
        tar=targets[:,j]; bp=(z[:,j]>=0).astype(np.int64); two=np.unique(tar).size==2
        per_thr[str(int(t))]={'positive_support':int(tar.sum()),'negative_support':int(len(tar)-tar.sum()),
            'accuracy':float(accuracy_score(tar,bp)),'balanced_accuracy':float(balanced_accuracy_score(tar,bp)),
            'f1':float(f1_score(tar,bp,zero_division=0)),
            'roc_auc':float(roc_auc_score(tar,z[:,j])) if two else None,
            'average_precision':float(average_precision_score(tar,z[:,j])) if two else None}
    probs=sigmoid(z); tf=targets.astype(np.float64)
    row=np.column_stack([((probs-tf)**2).mean(axis=1),(np.logaddexp(0,z)-tf*z).mean(axis=1),
        np.abs((-2.0+probs.sum(axis=1))-yy),((z>=0)==tf.astype(bool)).mean(axis=1)])
    cls={'accuracy':float(accuracy_score(yy,pred)),
        'macro_f1':float(f1_score(yy,pred,labels=CLASSES,average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(yy,pred)),
        'weighted_f1':float(f1_score(yy,pred,labels=CLASSES,average='weighted',zero_division=0)),
        'ordinal_mae':float(mean_absolute_error(yy,pred))}
    ud,inv=np.unique(dd,return_inverse=True); cnt=np.bincount(inv,minlength=len(ud)); dlg=np.empty((len(ud),4))
    for j in range(4): dlg[:,j]=np.bincount(inv,weights=row[:,j],minlength=len(ud))/cnt
    viol=np.any(z[:,:-1]<z[:,1:],axis=1)
    report={'n':int(len(yy)),'labeled_dialogues':int(len(ud)),'classification_metrics_row_level':cls,
        'per_class':per_class,'confusion_matrix_labels':CLASSES.tolist(),
        'confusion_matrix':confusion_matrix(yy,pred,labels=CLASSES).tolist(),'per_threshold':per_thr,
        'ordinal_logit_monotonicity':{'violating_rows':int(viol.sum()),'eligible_rows':int(len(z)),
            'violation_rate':float(viol.mean()),'expected_order':'non-increasing across ascending thresholds'},
        'folds_present':sorted(int(x) for x in np.unique(ff))}
    return report,pred,dlg,ud,row

def evaluate(data):
    p,actual,qids,dids,fold,avail,y,dense,loaded=data; n=len(qids)
    ranks=np.empty((7,n),dtype=np.int16); ties=np.empty_like(ranks)
    ranks[0]=dense['ranks'][0]; ties[0]=dense['gold_tie_counts'][0]
    idxs={'RANK_ONLY_FULL_MATCHED':1,'RANK_ONLY_NO_PRODUCT_MATCHED':2,'R_FUTURE_FULL':3,
          'R_FUTURE_NO_PRODUCT':4,'QR_FUTURE_FULL':5,'QR_FUTURE_NO_PRODUCT':6}
    for name,i in idxs.items():
        s=loaded[name]['candidate_scores']; gold=s[:,0:1]; neg=s[:,1:]
        ranks[i]=1+(neg>gold).sum(axis=1); ties[i]=(neg==gold).sum(axis=1)
    req(np.all((ranks>=1)&(ranks<=101)),'Computed rank outside [1,101]')
    qm=np.empty((n,7,5),dtype=np.float64); qm[:,:,0]=1.0/ranks.T
    for j,k in enumerate(R_KS,1): qm[:,:,j]=(ranks<=k).T
    ud,inv=np.unique(dids,return_inverse=True); counts=np.bincount(inv,minlength=len(ud))
    dlg=np.empty((len(ud),7,5),dtype=np.float64)
    for c in range(7):
        for m in range(5): dlg[:,c,m]=np.bincount(inv,weights=qm[:,c,m],minlength=len(ud))/counts
    rmacro=dlg.mean(axis=0); rmicro=qm.mean(axis=0); rb=boot_means(dlg,42)
    rd={m:'higher_is_better' for m in R_METRICS}
    rc=make_contrasts(rb,rmacro,R_CONTRASTS,R_METRICS,rd)
    rsummary={}; tiesummary={}
    for i,name in enumerate(R_CONDITIONS):
        rsummary[name]={'dialogue_macro':{m:float(rmacro[i,j]) for j,m in enumerate(R_METRICS)},
            'query_micro_descriptive':{m:float(rmicro[i,j]) for j,m in enumerate(R_METRICS)}}
        tiesummary[name]={'queries_with_gold_ties':int((ties[i]>0).sum()),
            'total_tied_non_gold_candidates':int(ties[i].sum()),'max_tied_candidates_per_query':int(ties[i].max())}

    fl=np.stack([loaded[x]['future_logits'] for x in F_CONDITIONS],axis=0)
    freports={}; fpreds=[]; fdlg=[]; frows=[]; fids=None
    for i,name in enumerate(F_CONDITIONS):
        rep,pred,dm,ids,row=future_metrics(fl[i],y,avail,dids,fold)
        if fids is None: fids=ids
        else: req(np.array_equal(fids,ids),f'{name}: future dialogue universe mismatch')
        freports[name]=rep; fpreds.append(pred); fdlg.append(dm); frows.append(row)
    fd=np.stack(fdlg,axis=1); fr=np.stack(frows,axis=0); fmacro=fd.mean(axis=0); fb=boot_means(fd,43)
    fdir={'brier':'lower_is_better','ordinal_logloss':'lower_is_better',
          'expected_state_mae':'lower_is_better','threshold_accuracy':'higher_is_better'}
    fc=make_contrasts(fb,fmacro,F_CONTRASTS,F_METRICS,fdir)
    for i,name in enumerate(F_CONDITIONS):
        freports[name]['dialogue_macro_probability_metrics']={m:float(fmacro[i,j]) for j,m in enumerate(F_METRICS)}

    summary={'artifact':'FSAR S12C Seven-Condition Matched OOF Evaluation V1','status':'PASS',
        'scope':'TRAIN_INTERNAL_OOF_ONLY','interpretation':'EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE',
        'protocol_sha256':sha(PROTO),'evaluation_script_sha256':sha(Path(__file__)),
        'input_sha256':actual,'query_rows':int(n),'dialogues':int(len(ud)),'candidate_count':101,
        'retrieval':{'conditions':R_CONDITIONS,'primary_metric':'DIALOGUE_MACRO_MRR',
            'rank_definition':'1 + COUNT(candidate_score > gold_score)','tie_policy':'STRICT_GREATER',
            'condition_metrics':rsummary,'tie_diagnostics':tiesummary,
            'bootstrap':{'unit':'dialogue_id','paired':True,'same_resampled_dialogues_for_all_conditions':True,
                'replicates':NBOOT,'seed':42,'ci':0.95,'method':'PERCENTILE','contrasts':rc}},
        'future_state':{'conditions':F_CONDITIONS,'eligible_rows':int(avail.sum()),
            'eligible_dialogues':int(len(fids)),'class_counts':{str(int(c)):int((y[avail]==c).sum()) for c in CLASSES},
            'classification_metrics':freports,'probability_metric_names':F_METRICS,
            'probability_metric_dialogue_macro':{name:{m:float(fmacro[i,j]) for j,m in enumerate(F_METRICS)} for i,name in enumerate(F_CONDITIONS)},
            'bootstrap':{'unit':'dialogue_id','paired':True,'same_resampled_dialogues_for_all_conditions':True,
                'replicates':NBOOT,'seed':43,'ci':0.95,'method':'PERCENTILE','contrasts':fc}},
        'planned_contrast_interpretation':{
            'rank_only_vs_dense':'Matched Rank-Only relative to frozen Dense BGE.',
            'future_vs_rank_only':'Incremental ranking effect of future-aware objectives under matched inputs.',
            'product_ablation':'Full versus No Product within R-Future and QR-Future.',
            'difference_in_differences':'Whether the No-Product effect differs between QR-Future and R-Future.'},
        'multiple_comparison_adjustment':'NONE_EXPLORATORY_CI_ONLY','no_posthoc_success_gate':True,
        'dev_payload_used':False,'test_payload_used':False,'training_performed':False,'new_model_inference_performed':False}

    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=False)
    stage=Path(tempfile.mkdtemp(prefix='.stage_',dir=OUT))
    try:
        (stage/FILES[0]).write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        np.savez_compressed(stage/FILES[1],retrieval_conditions=np.asarray(R_CONDITIONS,dtype='U40'),
            query_ids=qids,dialogue_ids=dids,fold_id=fold,ranks=ranks,gold_tie_counts=ties,
            query_metrics=qm,dialogue_query_counts=counts,dialogue_metrics=dlg,dialogue_macro=rmacro,
            query_micro=rmicro,retrieval_bootstrap_means=rb,future_conditions=np.asarray(F_CONDITIONS,dtype='U32'),
            future_dialogue_ids=fids,future_eligible_row_indices=np.flatnonzero(avail),future_available=avail,
            future_value_label=y,future_logits=fl,future_predictions=np.asarray(fpreds,dtype=np.int64),
            future_row_metrics=fr,future_dialogue_metrics=fd,future_dialogue_macro=fmacro,future_bootstrap_means=fb)
        lines=['FSAR S12C SEVEN-CONDITION MATCHED OOF EVALUATION V1','Status: PASS',
            'Scope: TRAIN_INTERNAL_OOF_ONLY',f'Queries={n}; dialogues={len(ud)}; candidates/query=101',
            f'Future eligible rows={int(avail.sum())}; dialogues={len(fids)}','','RETRIEVAL DIALOGUE-MACRO / QUERY-MICRO']
        for name in R_CONDITIONS:
            lines.append(name+' | dialogue_macro: '+', '.join(f'{k}={v:.8f}' for k,v in rsummary[name]['dialogue_macro'].items()))
            lines.append(name+' | query_micro: '+', '.join(f'{k}={v:.8f}' for k,v in rsummary[name]['query_micro_descriptive'].items()))
        lines+=['','RETRIEVAL TIE DIAGNOSTICS']+[name+': '+json.dumps(tiesummary[name]) for name in R_CONDITIONS]
        lines+=['','RETRIEVAL PAIRED DIALOGUE BOOTSTRAP CONTRASTS']
        lines += [f"{x['contrast']} / {x['metric']}: delta={x['delta']:+.8f}, CI95=[{x['ci95_lower']:+.8f},{x['ci95_upper']:+.8f}]" for x in rc]
        lines+=['','FUTURE-STATE CLASSIFICATION (ROW-LEVEL)']
        lines += [name+': '+json.dumps(freports[name]['classification_metrics_row_level']) for name in F_CONDITIONS]
        lines+=['','FUTURE-STATE PROBABILITY METRICS (DIALOGUE-MACRO)']
        lines += [name+': '+json.dumps(freports[name]['dialogue_macro_probability_metrics']) for name in F_CONDITIONS]
        lines+=['','FUTURE-STATE PAIRED DIALOGUE BOOTSTRAP CONTRASTS']
        lines += [f"{x['contrast']} / {x['metric']}: delta={x['delta']:+.8f}, CI95=[{x['ci95_lower']:+.8f},{x['ci95_upper']:+.8f}] ({x['direction_of_improvement']})" for x in fc]
        lines+=['','Exploratory analysis only; no confirmatory success gate.','No multiple-comparison adjustment.',
            'No dev/test payload used.','No model training or new model inference performed.',
            'Retrieval primary metric: dialogue-macro MRR; query-micro metrics are descriptive.',
            'Future classification metrics are row-level; probability metrics are dialogue-macro.']
        (stage/FILES[2]).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        for fn in FILES: os.replace(stage/fn,OUT/fn)
    finally: shutil.rmtree(stage,ignore_errors=True)
    print('\nS12C_SEVEN_CONDITION_EVALUATION_PASS')
    print('\nRETRIEVAL DIALOGUE-MACRO')
    for name in R_CONDITIONS: print(name,json.dumps(rsummary[name]['dialogue_macro']))
    print('\nKEY MRR CONTRASTS')
    for x in rc:
        if x['metric']=='MRR': print(json.dumps(x,ensure_ascii=False))
    print('\nFUTURE-STATE METRICS')
    for name in F_CONDITIONS: print(name,json.dumps(freports[name]['classification_metrics_row_level']),json.dumps(freports[name]['dialogue_macro_probability_metrics']))
    print('\nOUTPUT =',OUT)
    for fn in FILES: print(OUT/fn)

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze-protocol',action='store_true')
    g.add_argument('--preflight-only',action='store_true')
    g.add_argument('--run-evaluation',action='store_true')
    a=ap.parse_args()
    if a.freeze_protocol: freeze_protocol(); return
    data=preflight()
    if a.preflight_only: print('\nSEVEN_CONDITION_PREFLIGHT_ONLY_PASS'); return
    if a.run_evaluation: evaluate(data)

if __name__=='__main__': main()
