# ORCA 完整论文实验代码（GTE + BGE）

这是按当前 ORCA 论文整理的完整复现包。上一版误删了独立 RRF 脚本，本版已恢复并改为 GTE+BGE，同时补齐表 10–13 与 AnnoMI 表 15–17 的独立运行代码。

## 模型配置

中文主实验：

- `gte`：`thenlper/gte-large-zh`，直接编码中文原文，不添加 E5 前缀；
- `bge`：`BAAI/bge-large-zh-v1.5`，原 BGE 分支逻辑不改。

英文外部验证：

- `all-mpnet-base-v2`；
- `bge-large-en-v1.5`。

代码中不存在 text2vec 模型分支或默认路径。旧 text2vec 嵌入缓存不能复用。

## 安装

```bash
cd ORCA_GTE_BGE_full_paper_code
python -m pip install -r requirements.txt
```

GPU 环境可根据 CUDA 版本自行将 `faiss-cpu` 替换为兼容的 FAISS GPU 包。

## 目录

```text
src/                 数据清洗、方法实现及各实验 Python 入口
commands/            与论文表格逐项对应的可执行命令
tests/               不依赖模型下载的数学与预处理烟雾测试
EXPERIMENT_MAP.md     表1–17与代码的完整映射
```

## 一次性环境变量

```bash
export ROOT=/path/to/your/orca-workspace
export DATA="$ROOT/data/cleaned/psydt_task_retrieval_final.json"
export GTE_MODEL="$ROOT/models/thenlper/gte-large-zh"
export BGE_MODEL="$ROOT/models/BAAI/bge-large-zh-v1.5"
export DEVICE=cuda

# 表1和表9固定基准源消融需要原始多轮 PsyDT 文件
export RAW_PSYDT=/实际路径/PsyDTCorpus_train_mulit_turn_packing.json

# 表7–9扩展协议需要两份已有原始文件
export PSYDT_TEST_SINGLE=/实际路径/PsyDTCorpus_test_single_turn_split.json
export PSYDT_TRAIN_MULTI=/实际路径/PsyDTCorpus_train_mulit_turn_packing.json

# 表15–17 AnnoMI
export ANNOMI_CSV=/data/AnnoMI/AnnoMI-full.csv
export MPNET_MODEL="$ROOT/models/sentence-transformers/all-mpnet-base-v2"
export BGE_EN_MODEL="$ROOT/models/BAAI/bge-large-en-v1.5"
```

`--train_responses_file` 与 `--train_user_file` 均可指向同一份训练多轮文件；扩展脚本会按 `assistant` 和 `user` 角色分别抽取，不要求手工拆成新文件。

## 逐项运行

```bash
bash commands/02_tables_2_3_main.sh
bash commands/03_table_4_competing_methods.sh
bash commands/04_table_5_rrf.sh
bash commands/05_table_6_paired_significance.sh
bash commands/06_table_7_candidate_scaling.sh
bash commands/07_table_8_runtime_repeats.sh
bash commands/08_table_9_user_sources.sh
bash commands/09_table_10_mean_direction.sh
bash commands/10_table_11_score_decomposition.sh
bash commands/11_table_12_cross_pool.sh
bash commands/12_table_13_source_stability.sh
bash commands/13_table_14_qualitative.sh
bash commands/14_tables_15_17_annomi.sh
```

全部执行：

```bash
bash commands/run_all_paper_experiments.sh
```

完整映射和每项协议见 `EXPERIMENT_MAP.md`。

## RRF 补全说明

`src/run_orca_rrf_fusion_local.py` 会分别输出：

- `bm25_dense_rrf`；
- `bm25_orca_rrf`。

GTE 和 BGE 都会在开发集网格选择 RRF 常数与分支权重，再固定到测试集。它不是 `run_competing_methods_local.py` 中仅融合 BM25 与原始 Dense 的 `hybrid_rrf` 字段。

## 静态检查

```bash
python -m compileall -q src tests
python tests/test_math_smoke.py
python tests/test_annomi_preprocess_smoke.py
```

所有输出默认写入 `results/`，模型嵌入缓存写入 `cache/`，运行日志写入 `logs/`。

### 训练咨询师源同条件对照

`commands/08_table_9_user_sources.sh` 现已通过 `run_user_source_ablation_repeated.py` 同时运行训练咨询师源。该源从 PsyDT 训练划分的 `assistant` 回合中稳定去重后抽取，与训练用户源保持相同 `N=5000`、未中心化 SVD、固定 `k=2`、查询/候选池和随机种子协议。输出中的 `source=train_counselor` 行即为 GTE 与 BGE 的同条件角色源对照。
