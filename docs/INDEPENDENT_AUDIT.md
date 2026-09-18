# GaCA-73.25-release 独立科研复现审计

完成日期：2026-09-15。审计对象：`D:\EC\GaCA-73.25-release`。

## 结论摘要

**原始结果的保存和重放有直接证据支持，但尚未证明从新下载数据到完整重训的流程可以恢复相同结果。**

本次独立对照了原始重跑目录、CARE 本地源 CSV、原始 feature store、固定划分、日志、checkpoint 和逐蛋白预测，没有把上一轮总结作为通过依据。进行了完整测试集 checkpoint 推理；没有重新训练，没有运行 BLASTp/Foldseek 搜索，没有重新提取 embedding。

正式结论见 A–E。原始探针输出中有几个初步报警，已在 `audit_evidence/final_evidence.json` 的 `resolutions` 中解释；不能把初步报警数量当成最终 FAIL 数量。

## A. PASS：有证据验证

### A1. 数据与划分

| 检查 | 实测结果 |
|---|---|
| 有效训练蛋白 | 13,671 |
| fitting / validation | 12,303 / 1,368 |
| 训练标签 | 3,811，fitting 覆盖全部标签 |
| 两个评估子集 | 243 / 477 |
| 固定划分 CSV | 与原始文件逐行一致 |
| class_index / ec_label mapping | 与原始文件逐行一致 |
| 有效 ID、标签、行序 | 从原始 CSV 与 feature 索引重新求交后完全一致 |
| 重复有效 ID | 无 |
| train/test、test/test ID 交集 | 无 |
| 上述集合的完全相同序列 SHA256 交集 | 无 |
| fitting/validation 完全相同序列 SHA256 交集 | 无 |

CARE 原始训练 CSV 有 28,316 行、134 行重复 ID，其中 131 个 ID 有多个 EC 标签。直接按 ID 索引会展开重复行，导致初步比较不一致。按实际 `sample_training(..., 600, 25, 42)` 的采样和首个 ID 保留规则重建后，原始 13,727 行来源表的 ID、标签、序列和顺序全部一致。被选中的来源 ID 中有 93 个来自多标签 ID；单标签选择规则继承原实验，本轮未改。

21 个训练蛋白和一个 30–50% 测试蛋白的第四字段含 `n` 前缀，是原始来源中的标签字符串。全数字 EC 正则的报警属于审计探针过严，并非新产生的标签错位；这些标签保持原样，仍按四个字段计算层级指标。

以上没有覆盖全部序列成对 identity、近同源泄漏或预训练语料污染，不能推广为所有形式的 data leakage 均已排除。

### A2. 已保存的 embedding 与结构抽查

| store | 蛋白数 | 原始数组 shape |
|---|---:|---|
| seq_train | 13,671 | (4,788,944, 1280) |
| str_train | 13,727 | (4,808,447, 512) |
| seq_lt30 | 312 | (119,017, 1280) |
| str_lt30 | 314 | (120,375, 512) |
| seq_30_50 | 549 | (229,375, 1280) |
| str_30_50 | 549 | (229,375, 512) |

六个索引的 ID 唯一，offset 连续且合法，length 总和等于数组行数，dtype 均为 float16。每个数组抽查 1,000 行，未见非有限值；没有扫描全部数组的每个值。release 的输入 ID manifest 与原始索引一致，结构长度 manifest 一致。

数据读取按 protein ID 查询两个 store，不依赖两个文件的行序相同。独立 pooling 不要求两个模态的残基数相等。

三个原始 PDB 目录分别等间隔抽查 30 个，共 90 个：新提取器按残基组成的 N/CA/C 坐标与旧提取器结果逐值相同，长度与原始 embedding 索引一致。有序列 manifest 的抽查条目，其重建序列哈希也一致。这里只验证了抽样坐标与序列输入，没有重新运行 ESM 编码器。

### A3. 模型和 checkpoint

checkpoint 全部 29 个 state_dict 张量与原文件逐个完全相同，严格加载也覆盖了 BatchNorm buffer，而非仅检查能否 load。

| 模块 | 可训练参数 |
|---|---:|
| seq_pool | 164,097 |
| str_pool | 65,793 |
| seq_proj | 656,896 |
| str_proj | 263,680 |
| gate | 787,456 |
| classifier | 2,218,723 |
| 合计 | **4,156,645** |

projection 为独立 Linear + LayerNorm + Dropout；attention scorer 为 Linear → Tanh → Linear；gate 为 Linear → ReLU → Linear → Sigmoid；融合为 gate × structure + (1 − gate) × sequence。classifier 保留 Linear、BatchNorm、GELU、Dropout、Linear。

release 增加了 `forward_with_gate` 导出接口，没有新增可训练模块。原始模型与 release 在 eval 和固定随机种子的 train 合成测试中，logits 最大绝对差均为 0.0。全部参数在反向测试中获得梯度；改变被 mask 的 padding 数值不改变 eval 输出。

checkpoint SHA256：`cdcd4da798a95322f712c6901ecf862a23e80180736d7f9fe7ac3251928f66ef`。

### A4. 训练配置与指标

GaCA 默认/README 配置是 batch 128、AdamW、learning rate 1e-4、weight decay 5e-3、label smoothing 0.1、hidden dimension 512、dropout 0.5、max epochs 120、patience 30、seed 42、CUDA AMP。没有 scheduler。这些参数实际用于构造器及训练循环，未发现 GaCA 主要训练参数仅声明而未生效。

训练 epoch、validation、DataLoader、划分生成、随机种子函数与原始代码 AST 一致。validation accuracy 严格提高时更新 best epoch；之后重置 seed、重新初始化，在全部 13,671 个训练蛋白上训练所选轮数。GaCA 存档 history 的首次最高验证准确率位于 epoch 75，metrics 记录 75，full-refit history 恰为 75 轮。其余六个神经变体的选轮数与 history 也一致。

固定 split 文件存在时，`--validation-ratio` 和新的 `--seed` 不重新划分该 CSV，这是固定划分设计。`--refit-only-epochs` 会跳过 tuning，不属于主 README 命令。运行环境 JSON 曾被覆盖，因此不能把以上静态配置和日志核对称为原 GaCA 完整 launch record 已恢复。

Level-1/2/3 按前缀、Level-4 按完整标签计算。过滤依据是模态可用性和训练标签空间，未依据测试预测正确与否过滤。NPZ 的 protein_ids、true_indices、argmax 与预测 CSV 一致；probability 行和有效，gate shape 为 (N,512)，数值有限且在 (0,1)。

### A5. 完整 checkpoint 推理与统计复算

使用 release 推理入口、存档 checkpoint、本机原始 feature store，以 batch 128、CUDA AMP 推理全部测试蛋白：

| 子集 | 正确数 / 总数 | Level-4 accuracy | 逐蛋白预测 |
|---|---|---|---|
| <30% | 178 / 243 | 73.25% | 与原始结果逐行一致 |
| 30–50% | 405 / 477 | 84.91% | 与原始结果逐行一致 |

26 份各方法的标准预测 CSV 均使用相同 243 / 477 个 ID，真实标签一致，无重复计数。独立重算 Level-1 至 Level-4、macro-F1、MCC，与最终汇总最大绝对差为 1.11e-16。

LightGBM 的存档 mapping 与固定划分相符。代码按 ID 取 mean-pooled 1280+512 特征，拼接后逐蛋白 L2 normalization，不是在测试集拟合 scaler。500 棵上限和 early stopping 30 确实传入训练；存档 best iteration 为 349。未重新训练 LightGBM。

| GaCA − baseline | 子集 | 准确率差（百分点） | 95% CI（百分点） | exact McNemar p |
|---|---|---:|---|---:|
| BLASTp | <30% | 6.58 | [2.47, 11.11] | 0.00522287935 |
| BLASTp | 30–50% | 3.77 | [1.26, 6.29] | 0.00509764330 |
| Foldseek | <30% | 3.29 | [-0.82, 7.41] | 0.18493334204 |
| Foldseek | 30–50% | 3.56 | [1.05, 6.08] | 0.00947530428 |

独立实现没有调用 release 的 `paired_bootstrap`，使用相同配对抽样序列独立计算 correctness/macro-F1，并以组合数计算 exact McNemar。两张存档表共八行的 difference、CI、p 最大差小于 1e-16。macro-F1 bootstrap p 分别为 0.0016、0.0076、0.0592、0.0132，均与存档一致。

此前“差值 0.0”在论文显示精度上成立，但不是完整实验复现证明。本次对照的是 `reference_results/paper_tables` 存档表，没有重新打开最新版 final.tex 逐格核对，更没有修改论文。

## B. WARNING：未实证验证

1. 未从头安装 environment.yml。本机缺 fair-esm，两个 embedding extractor 的 CLI/推理均未验证。其余 README 中 17 个 Python 模块的 `--help` 成功，仅代表接口导入通过。
2. 未重新下载 CARE、AlphaFold 结构或编码器权重，未重新提取 embedding。90 个 PDB 抽查不能代表全部原子坐标验证。
3. 未完整重训神经网络、LightGBM 或经典 baseline。checkpoint 重放不证明跨平台训练轨迹一致。
4. 未执行 BLASTp/Foldseek 外部搜索。README 输入输出路径、FASTA/PDB 与 TSV 列定义、统计与绘图输入在静态检查中能衔接；外部工具版本和并列 top-hit 行为仍未复验。
5. 多标签源记录的单标签取舍、全部近同源关系、预训练语料覆盖不属于已通过范围。

## C. FAIL：发现的问题及修复

| 问题及影响 | 修复文件与方式 | 是否影响原始 73.25% |
|---|---|---|
| ESM-IF1 依赖遗漏，干净环境可能无法导入 | environment.yml、requirements.txt、README.md：补 PyG，指定建议环境 CUDA 12.1，提供匹配 scatter wheel 和导入检查命令 | 不改 GaCA；安装与 fresh embedding 仍未实测 |
| README 混同服务器/原 GaCA 环境；所谓 GaCA config 实为 cross-attention | README 澄清；gaca/train.py 为新运行按模型保存配置；旧 PIPELINE_AUDIT.md 标明由本报告取代 | 不改结果；历史配置缺失无法凭空恢复 |
| SVM 的 dual 默认随 sklearn 版本改变 | baselines/classical/train.py 显式 dual=True，恢复原 sklearn 1.4.2 的选择 | 不影响 GaCA；避免新 SVM 改用另一求解器，未重训 SVM |
| 仅核对样本数不能阻止换 ID、标签或顺序 | common/protocol.py、preprocessing/export_effective.py、gaca/train.py、gaca/evaluate.py：默认核对存档 ID/标签/行序、mapping、split 标签 | 原始输入通过；错误输入报错，不静默改样本 |
| 源 embedding 更新后可能跳过旧缓存；可整除错误 shape 被强行 reshape | embeddings/build_store.py：检查来源 path/size/mtime/dimension，拒绝错误 shape、空值、非有限或 float16 溢出、规范化后重复 ID | 合法原数组不变；过期缓存需显式重建，不是完整内容哈希保护 |
| 配对 inner join 可静默丢弃 baseline 多余 ID | evaluation/paired_statistics.py：先检查 ID 非空、唯一、集合相同 | 当前配对不变；错误输入报错 |
| standalone evaluate 的 AMP 条件与训练入口不同，CPU 也可能启用 float16 | gaca/evaluate.py：AMP 限定为 CUDA，并验证默认协议 | 原 CUDA 推理不变 |

新增 `tests/test_release_guards.py`：8/8 回归测试通过；主树 40 个 Python 文件语法检查通过。此次不改 GaCA 架构或训练超参数，不改旧目录、权重、存档结果、图或论文。

### 未修复的预处理缺陷

`preprocessing/align_sequences.py::alignment_bounds` 使用无 gap 惩罚的 globalxx，identity 仅计双方非 gap 的列。实际调用 `alignment_bounds('ACDEFG','AEDCFG')` 返回 `(0,6,0,6,1.0)`。这个 1.0 不能可靠证明两个保留区间高度一致；短例仅测试函数，不涉及 main 的最小长度限制。

该规则继承旧流程。改 alignment/scoring/coverage 会改变过滤条件，可能改变有效集合与 73.25%，因此本次没有静默替换实验规则。需要作者决定保留历史模式并说明局限，还是另行验证修正后的过滤流程。90 个原始结构抽查无序列哈希差异，不足以证明所有输入不受影响。

## D. REPRODUCIBILITY RISK

- **原始输入未完整冻结。** 已有 ID、label、序列哈希/截取范围和结构长度，但未分发原始 feature arrays，也未固定全部 PDB 内容哈希和版本。新下载的缺失状态与内容可能不同。加强校验只能报错，不能恢复旧输入。
- **环境与 provenance 不完整。** 原日志支持参数量和 batch，history 支持 epoch 75，但共享环境 JSON 被覆盖。建议 Linux/PyTorch 2.2.2 环境与本地原实验记录的 Windows/PyTorch 2.4.0 不同。固定 seed、deterministic cuDNN 不保证跨版本逐位复现。
- **LightGBM 行采样未启用。** 实测默认 subsample_freq=0，因此 subsample=0.8 不等于训练用了 80% 行采样；原实验同样如此。本轮没有改为正 frequency，否则会改变 baseline 算法。
- **bootstrap 的统计解释仍需谨慎。** macro-F1 的每个 replicate 重新选择其出现的真值类别；支持集合随抽样变化。p 为经验分布的两倍较小尾部概率，并非 exact randomization test。本轮确认配对与数字可复算，未验证该程序的覆盖率/错误率，也未替换检验。
- **检索结果重算不等于检索重跑。** 外部版本、数据库顺序和并列分数处理尚需正式比对。
- **遗留与主流程区分。** 主流程代码未发现旧 GaCA-GitHub 绝对路径或私有环境变量依赖；legacy 和 provenance JSON 的历史路径仍保留。本次外部审计脚本读取原目录是证据对照，不是 release 运行依赖。

## E. FINAL VERDICT

**在保留本报告所列限制的前提下，可以冻结为 historical research-code release；不能标为“从原始下载到完整重训已经验证”的 end-to-end reproduction。**

核心架构、权重、固定划分、checkpoint 推理、存档指标和配对统计有证据支持。不能无条件通过的原因是 fresh extraction/refit 未执行、原始输入/环境未完全冻结，以及对齐过滤的已知缺陷，不是存档 73.25% 算错。

**基于现有证据，我是否相信这套代码能够复现原始 73.25% 结果？基本是但仍需验证。**

最关键的原因：原模型与权重一致且原始 embedding 上 720 个预测完全复现；固定划分和 75 轮选择可核实；重新下载、提取及从头训练的整条链尚未实证，不能用 checkpoint 重放作无条件保证。

## 证据与复核

- `audit_evidence/evidence.json`：首次逐项检查，含初步探针报警。
- `audit_evidence/second_evidence.json`：原始坐标抽查、模型前向、独立指标与统计复算。
- `audit_evidence/final_evidence.json`：最终来源核验、报警解释、完整推理、CLI、语法及修改清单。
- `audit_evidence/changes.diff`：本轮既有文件的精确变更。
- `audit_evidence/inference_artifacts_summary.json`：本次完整 checkpoint 推理摘要。
- 本机额外证据及修改前备份：`C:\Users\a1796\gaca_independent_audit`。

`changes.diff` 由上述可信审计前备份与当前文件重新生成，并通过 `git apply --check` 验证；它不是根据报告人工还原的 diff。其完整性范围限于备份覆盖的既有文件以及本轮新增的测试和审计报告。

轻量回归命令：`python -m unittest discover -s tests -v`，不训练模型。

外部依据仅用于依赖/default 行为核查，不作新实验依据：

- [ESM-IF1 官方环境说明](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding)
- [PyG 安装说明](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html) 与 [PyTorch 2.2 / CUDA 12.1 wheels](https://data.pyg.org/whl/torch-2.2.0+cu121.html)
- [sklearn 1.4.2 LinearSVC 默认值及变更](https://scikit-learn.org/1.4/modules/generated/sklearn.svm.LinearSVC.html)
- [LightGBM 4.6.0 参数说明](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)
