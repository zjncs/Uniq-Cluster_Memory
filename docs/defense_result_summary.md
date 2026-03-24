# 答辩结果总结

## 一句话结论

本工作提出的 **Uniq-Cluster Memory (UCM)** 面向长程医疗对话中的结构化记忆管理问题，通过“事件抽取 + 语义聚类 + 唯一性/冲突管理 + 压缩 + 双通道检索”五阶段 pipeline，将原始多轮对话压缩为可检索、可追踪、可检测冲突的 canonical memory；在主测试集上，UCM 相比无记忆系统在 `Unique-F1` 和 `Recall@5` 上取得了显著提升。

## 研究问题

- 长医疗对话中，同一属性会多次出现、被更新，甚至互相冲突。
- 纯上下文拼接或无结构检索，难以稳定回答“当前最权威的患者状态是什么”。
- 本工作要解决的是：如何把多轮医疗对话转成唯一、时序一致、可检索的长期记忆。

## 方法概述

系统由五个模块组成：

1. `M1 Event Extraction`：从多轮对话中抽取医疗事件。
2. `M2 Clustering`：将语义相近、指向同一事实的事件聚合。
3. `M3 Uniqueness + Conflict Management`：按属性与时间范围维护 canonical memory，并记录冲突历史。
4. `M4 Compression`：去冗余并生成紧凑记忆表示。
5. `M5 Retrieval`：结合结构化 memory 与文本证据进行检索。

## 答辩主结果

### 1. Med-LongMem：唯一记忆与冲突检测

主指标使用 `Med-LongMem` 上的 `Unique-F1 (strict)`，这是最能体现“有记忆 vs 无记忆”差异的指标。

| 系统 | n | Unique-F1 (strict) | Unique-F1 (relaxed) | Conflict-F1 |
| --- | ---: | ---: | ---: | ---: |
| UCM | 20 | **0.8511** | **0.8588** | **0.9562** |
| No Memory | 20 | 0.0000 | 0.5938 | 0.1292 |
| Raw RAG | 20 | 0.0000 | 0.5656 | 0.1292 |

结论：

- UCM 将 `Unique-F1 (strict)` 从 `0.0000` 提升到 `0.8511`。
- UCM 在 `Conflict-F1` 上达到 `0.9562`，显著高于无记忆与原始检索基线。
- 说明系统不仅“能提取信息”，更重要的是“能维护唯一、时序一致的医疗状态表示”。

### 2. LongMemEval：检索与回答能力

主指标使用 `Recall@5`，并补充回答准确率。

| 系统 | n | Recall@5 | QA Accuracy | Mean Quality Score |
| --- | ---: | ---: | ---: | ---: |
| Hybrid RAG (with memory) | 50 | **0.6300** | **0.4000** | **2.66** |
| Raw RAG | 50 | 0.3650 | 0.2800 | 2.18 |
| No Memory | 50 | 0.0000 | 0.0200 | 1.84 |

结论：

- 有 memory 的检索系统在 `Recall@5` 上比 `No Memory` 提升 `+0.6300`。
- 相比 `Raw RAG`，带 memory 的系统也有明显优势，说明提升不只是“用了检索”，而是“用了结构化长期记忆”。
- 回答准确率从 `0.02` 提升到 `0.40`，说明 memory 对最终问答质量有直接帮助。

## 真实世界验证

为了验证方法在自然对话上的可用性，额外使用了官方 `MedDialog` 中文多轮对话做真实世界验证。

当前已完成的正式 batch：

- 样本数：`10`
- 过滤条件：轮次 `>= 10`
- 平均轮次：`11.8`
- 平均预测记忆数：`12.5`
- 平均冲突数：`1.3`
- 含冲突样本数：`7 / 10`
- 失败样本数：`0`

这部分结果的含义：

- 真实医疗对话中确实存在可被 UCM 稳定抽取的长期记忆结构。
- 冲突并不是只在自建对抗集里才出现，真实对话中也会出现多版本状态与更新。
- 这部分结果适合作为 `Real-World Validation / Case Study`，不应作为同版 UCM 的 headline 自证测试集。

## 可以直接放到答辩 PPT 的 3 句话

1. 本工作的核心价值不是简单做信息抽取，而是把长医疗对话转成“唯一、可追踪、可检测冲突”的长期记忆。
2. 在 `Med-LongMem` 上，UCM 将 `Unique-F1` 从 `0.0000` 提升到 `0.8511`，将 `Conflict-F1` 提升到 `0.9562`。
3. 在 `LongMemEval` 上，带 memory 的系统将 `Recall@5` 从 `0.0000` 提升到 `0.6300`，并把问答准确率从 `0.02` 提升到 `0.40`。

## 30 秒口播稿

我的工作针对的是长程医疗对话中的记忆管理问题。传统无记忆或纯文本检索方法，难以稳定维护患者状态的唯一性和时序一致性。为了解决这个问题，我提出了 Uniq-Cluster Memory，把多轮对话经过事件抽取、聚类、唯一性与冲突管理、压缩和双通道检索，转化为 canonical memory。在主测试集 Med-LongMem 上，系统把 Unique-F1 从 0 提升到 0.8511，Conflict-F1 达到 0.9562；在 LongMemEval 上，Recall@5 从 0 提升到 0.63，问答准确率从 0.02 提升到 0.40。说明该方法不仅能记住信息，而且能维护长期、一致、可检索的医疗记忆。

## 创新点表述建议

建议在答辩中把创新点概括为三条：

1. **唯一性记忆建模**：不是简单存储事实片段，而是围绕 `(patient, attribute, time_scope)` 构建 canonical memory。
2. **冲突显式建模**：保留旧值与冲突历史，而不是直接覆盖，支持医疗场景中的状态更新与冲突检测。
3. **结构化记忆驱动检索**：将结构化 memory 与文本证据结合，提升长程问答的召回与答案质量。

## 结果引用口径

答辩中建议优先引用以下结果文件：

- Med-LongMem 主结果：[pipeline_med_longmem_w7.jsonl](/Users/zjn/Desktop/uniq_cluster_memory/results/main_results/pipeline_med_longmem_w7.jsonl)
- Med-LongMem 基线：[med_longmem_v01_eval.json](/Users/zjn/Desktop/uniq_cluster_memory/results/main_results/med_longmem_v01_eval.json)
- LongMemEval with memory：[longmemeval_hybrid_rag_summary.json](/Users/zjn/Desktop/uniq_cluster_memory/results/main_results/longmemeval_hybrid_rag_summary.json)
- LongMemEval no memory：[longmemeval_no_memory_summary.json](/Users/zjn/Desktop/uniq_cluster_memory/results/main_results/longmemeval_no_memory_summary.json)
- 真实世界验证：[summary.json](/Users/zjn/Desktop/uniq_cluster_memory/results/real_world_validation/meddialog_official_zh_test_long_r10_s42/summary.json)

## 答辩时不要拿来做 headline 的材料

- [ablation_summary.json](/Users/zjn/Desktop/uniq_cluster_memory/results/ablation/ablation_summary.json)
  这组消融结果与当前主结果不是同一批配置，适合做补充讨论，不适合做主结论页。
- `MedDialog Real` 的 Silver GT 结果
  这部分适合作为真实世界验证、案例分析和人工审核入口，不适合作为同版系统的 headline 自评分数。

## 工程完成度补充

- 当前测试状态：`58 passed`
- 已补最小 CI：[ci.yml](/Users/zjn/Desktop/uniq_cluster_memory/.github/workflows/ci.yml)
- 真实世界验证支持断点续跑、分批运行与失败落盘

