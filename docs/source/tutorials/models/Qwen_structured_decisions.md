# Qwen 结构化决策（vLLM 0.30）

## 简介

本实现从上游 vLLM `7e3d559565` 移植 `NextTokenStrategy`、问题类型、
标签概率读取及两个完整 HTTP 接口，作为 Ascend 的 endpoint plugin 提供。
参考：[Structured Decisions Models](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)。

| 架构 | 示例模型 | 策略 |
| --- | --- | --- |
| `Qwen3ForCausalLM` | `Qwen/Qwen3-0.6B` | `NextTokenStrategy` |
| `Qwen3_5ForConditionalGeneration` | `Qwen/Qwen3.5-0.8B` | `NextTokenStrategy` |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen/Qwen3.5-35B-A3B` | `NextTokenStrategy` |

这不是 JSON 约束解码。每个问题用一次单 token 生成，读取所有候选标签的全词表
logprob，再在标签间归一化。`label_mass` 是全部标签的全词表概率之和；
`confidence` 是获选标签的全词表概率，并不是经校准的任务准确率。

## 支持范围

- `/v1/systemone`：文本或 JSON `state`，`choice`、`noul`、`score`，概率、置信度和诊断。
- `/v1/decisions`：文本 `input` 或 user 文本消息，`predicate`、`choice`、`score`，
  保留问题顺序、名字、布尔/字符串选项及缓存 token 统计。
- 最多 64 个问题、26 个 choice 选项、10 个 score 等级。`systemone` 的 choice
  可有一个选项，`decisions` 至少两个；score 至少两个等级。
- 保留 cache salt、LoRA 模型名、请求取消、负载控制和 tracing header 转发；
  `systemone` 还支持请求 priority、instructions 和 chat template kwargs。
- 默认关闭 thinking；显式启用返回 400。标签在当前回复位置必须是独立单 token。
- 按上游 Qwen `NextTokenStrategy` 的行为，决策接口仅接受文本/JSON。
  图片、视频、音频、`joint=true` 均返回 400；这不改变模型原有聊天多模态能力。
- 插件只挂载 generate 任务。不支持的架构或 logits 模式无法建立策略，接口返回 501。
  启动日志会说明原因。

## 环境准备

使用已有的 Ascend vLLM 0.30 环境及其匹配的 CANN、PyTorch、torch-npu 版本。
重新安装当前修改后的 vllm-ascend 包，确保 `setup.py` 中的新 entry point 已注册。
不需要修改上游源码或升级 Transformers。

````{tabs}

```{tab} Atlas A2
使用当前已验证可运行 Qwen3/Qwen3.5 的 vLLM Ascend 0.30 A2 容器。
小模型先采用 TP=1；实测硬件显存与容量要求仍待验证。
```

```{tab} Atlas A3
使用当前已验证可运行 Qwen3/Qwen3.5 的 vLLM Ascend 0.30 A3 容器。
35B-A3B 的 TP/EP 配置沿用现有部署，并在本接口上补做验证。
```

````

vLLM 0.30 的 endpoint plugin 必须由 `VLLM_PLUGINS` 显式启用。
该变量也筛选平台和 general plugins，因此不能只填写新增插件名。
下面列出本仓库的 Ascend 插件；如已有其他插件，请保留它们。

## 启动

```bash
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,ascend_structured_decisions
cd /workspace
vllm serve Qwen/Qwen3.5-0.8B \
  --served-model-name decision-model \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --logprobs-mode raw_logprobs \
  --port 8000
```

Qwen3 使用同一命令，将模型替换为 `Qwen/Qwen3-0.6B`。
Qwen3.5 MoE 替换为 `Qwen/Qwen3.5-35B-A3B`，并使用硬件适用的 TP/EP 配置。
支持 `processed_logprobs`；不支持 `raw_logits` 或 `processed_logits`。
排查 ACLGraph 问题时可加 `--enforce-eager`。
如果使用 `VLLM_USE_V2_MODEL_RUNNER=1`，本移植也修复了 runner-v2 的指定标签
logprob 路径及 0.30 的 `compute_topk_scores` 补丁绑定。

升级到原生包含这些接口的 vLLM 后，插件不会覆盖已注册的同名路由。

## 功能验证

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decision-model",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments and refunds", "shipping": "deliveries"}
      },
      "urgent": {"type": "noul", "instructions": "Does this need a refund?"},
      "severity": {
        "type": "score",
        "instructions": "How severe is the problem?",
        "criteria": ["minor", "moderate", "severe"]
      }
    }
  }'
curl --fail-with-body http://localhost:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "decision-model",
    "input": "My card was charged twice for one order.",
    "questions": [
      {"type": "predicate", "name": "refund", "instructions": "Is a refund needed?"},
      {"type": "choice", "instructions": "Which team should handle this ticket?",
       "choices": [{"value": "billing"}, {"value": "shipping"}]}
    ]
  }'
```

```bash
# CPU 单元测试：类型、请求契约、实际 HTTP 路由和模拟 engine。
pytest -q tests/ut/entrypoints/structured_decisions
# 真权重 NPU：Qwen3.5-0.8B 和 Qwen3-0.6B，两接口及独立 completion 概率对照。
pytest -sv tests/e2e/pull_request/one_card/test_structured_decisions.py
# 也应在 runner-v2 上重复。
VLLM_USE_V2_MODEL_RUNNER=1 pytest -sv tests/e2e/pull_request/one_card/test_structured_decisions.py
# NPU kernel：混合普通 top-k/自定义标签、请求重排、多行请求、padding 和 logits 模式。
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_logprobs.py
```

验收应检查 HTTP 200、每个问题都返回概率、概率和为 1、非空响应及日志中无异常。
同时补测并发请求、缓存开关、LoRA（如使用）、真实最大上下文和 MoE 部署。
Dummy 只适合检查启动/算子/接口，不代表真实权重或准确率通过。

## 实现与验证记录

模型计算、权重加载和 runner-v1 沿用现有 Ascend 实现。新增接口代码在
`vllm_ascend/entrypoints/`，通过官方 endpoint plugin 生命周期接入。
问题类型和架构表使用不可变映射，服务状态隔离在各 FastAPI app 内。
后续上游版本原生提供接口后可移除此兼容插件。

runner-v2 原先忽略自定义 `logprob_token_ids`；现在调用已有 Ascend 填充 kernel，
保留每请求标签与有效位 mask，同时支持 0.30 的 logits 模式和 device 上的分段信息。
热路径未增加 `.item()` 或 CPU 回传。

| 验证项目 | 状态 |
| --- | --- |
| CPU 请求/响应、概率处理、错误与插件回归 | vLLM 0.30 源码：125 项通过（隔离 NPU kernel） |
| 真实 tokenizer（3 个模型） | 全部 38 个标签均通过单 token 回复边界检查 |
| Dummy 启动与请求 | 未运行：本机无 NPU |
| 真权重文本推理与标签概率 | 未运行：用户将在 Ascend 环境验证 |
| ACLGraph、runner-v2 NPU kernel | 已提供测试，未运行硬件验证 |
| EP / flashcomm1 | 小型 dense 模型不适用；35B-A3B 待用户验证 |
| MTP | 未验证与决策接口的组合 |
| 决策接口多模态 | 与上游 Qwen next-token 路径一致，返回 400 |
| 128k / batch 16 容量 | 未验证；示例使用 4096 / 16，不代表容量结论 |

## 准确率与性能

未进行真实模型准确率或吞吐评估，因此没有生成虚构评测分数或新增模型准确率配置。
本次是现有模型的接口移植；现有模型的评测配置继续适用。
`confidence` 不是任务准确率，需在实际业务数据上评估。

Qwen3.5 是 hybrid 模型：多问题且共享前缀至少 128 tokens 时，先预热共享前缀再
并发读取各问题。Qwen3 不需要这次 hybrid 预热。接口 usage 按上游口径统计问题
读取的 tokens，不包含额外预热请求。缓存命中和不同 batch 可造成数值差异；
性能测试应分别覆盖单问题、多问题、冷缓存与热缓存，并记录预热开销。
