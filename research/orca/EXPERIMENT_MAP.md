# 论文实验—代码映射

当前代码包与 `ORCA_v3_GTE_BGE_摘要优化版` 的实验结构一一对应。中文主实验只使用 GTE-large-zh 与 BGE-large-zh-v1.5；AnnoMI 外部验证单独使用 MPNet 与英文 BGE。

| 论文位置 | 实验 | Python 入口 | 可执行命令 |
|---|---|---|---|
| 表1 | PsyDT 清洗、对话级划分、5k 基准与主用户源 | `src/build_psydt_final_dataset.py` | `commands/01_build_psydt_benchmark.sh` |
| 表2–3 | 开发集 k 扫描、Dense、ORCA 主测试 | `src/run_full_experiments_local.py` | `commands/02_tables_2_3_main.sh` |
| 表4 | BM25、规则过滤、候选 SVD/PCA、白化、随机方向、INLP | `src/run_competing_methods_local.py` | `commands/03_table_4_competing_methods.sh` |
| 表5 | BM25+Dense RRF、BM25+ORCA RRF，开发集选融合参数 | `src/run_orca_rrf_fusion_local.py` | `commands/04_table_5_rrf.sh` |
| 表6 | ORCA 与 Dense 的逐查询 Bootstrap、精确 McNemar | `src/run_supplementary_experiments_local.py` | `commands/05_table_6_paired_significance.sh` |
| 表7 | 5k/10k/20k/50k/full 候选池扩展 | `src/orca_bigdata_final_safe.py` | `commands/06_table_7_candidate_scaling.sh` |
| 表8 | 1 次预热 + 5 次正式运行时间 | `src/run_runtime_repeats.py` | `commands/07_table_8_runtime_repeats.sh` |
| 表9 | 主用户源、全部用户、非情绪用户、第一人称非情绪用户、训练咨询师源 | `src/run_user_source_ablation_repeated.py`、`src/orca_bigdata_final_safe.py` | `commands/08_table_9_user_sources.sh` |
| 表10 | 用户均值方向、均值中心化、用户/候选 PCA、随机方向 | `src/run_mean_direction_baselines.py` | `commands/09_table_10_mean_direction.sh` |
| 6.2 测试确认 | v1、v2、k2 与用户均值的冻结测试配对统计 | `src/run_test_mean_direction_ablation.py` | 同上 |
| 表11 | removed/retained 得分分解与恢复/受损组 | `src/run_score_decomposition.py` | `commands/10_table_11_score_decomposition.sh` |
| 表12 | 开发/测试候选池方向与 ORCA 的配对比较 | `src/run_cross_pool_paired_significance.py` | `commands/11_table_12_cross_pool.sh` |
| 表13 | 源规模、R@5、v1/v2 夹角稳定性 | `src/run_source_stability_enhanced.py` | `commands/12_table_13_source_stability.sh` |
| 表14 | Dense/ORCA 定性排序案例导出 | `src/export_dense_orca_qualitative_cases.py` | `commands/13_table_14_qualitative.sh` |
| 表15–17 | AnnoMI 五折、聚类 Bootstrap、方向消融、均值对齐 | `src/run_annomi_fivefold.py` | `commands/14_tables_15_17_annomi.sh` |

## 方法与共享实现

- `src/manifold_decoupler.py`：源子空间拟合、投影与数值诊断。
- `src/dual_track_index.py`：双轨检索索引实现。
- `src/orca_source_lexicons.py`：主用户源及消融源筛选词表。
- `src/paper_experiment_utils.py`：论文分析脚本共享的缓存、SVD/PCA、检索与配对统计工具。
- `src/run_fixed_k_source_ablation_local.py`：被重复源实验复用的 PsyDT 原始对话解析与源筛选函数。

## 重要协议

1. GTE 与 BGE 使用独立嵌入缓存，禁止复用旧 text2vec 缓存。
2. 中文主实验的 k 只在开发集选择；测试集不重新选 k。
3. RRF 参数在开发集选择，然后固定到测试集。
4. 候选池扩展使用论文主用户源 `affective_user`；训练回复和训练用户参数可指向同一份训练多轮 JSON，脚本按角色分别抽取。
5. 表9的固定基准脚本同时从训练划分抽取用户与咨询师话语；训练咨询师源使用相同 N=5,000、未中心化 SVD、固定 k=2 和相同查询—候选协议。
6. AnnoMI 按 transcript 划分，训练折用户回合拟合方向，测试折咨询师回合构成候选池。
