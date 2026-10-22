# Clef-Flash 9B 文本结构化决策（vLLM 0.30）

## 简介

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) 基于
Qwen3.5-9B，通过独立的 joint schema head 在一次 pooling 请求中联合评分全部问题。
本移植以本地上游 vLLM `7e3d559565` 为基准，复用 Ascend Qwen3.5 模型，
提供文本 `/v1/systemone`。无需修改上游源码或执行模型仓库的 remote code。

模型配置仍声明 `Qwen3_5ForConditionalGeneration`。启动时检测同一 revision 的
`joint_head_config.json`，切换到 `ClefFlashForJointSchema` 和 pooling runner。
加载主干权重后，将 `joint_head.safetensors` 加载到 `pooler.head.*`。
head 使用输出 `lm_head` 的词向量；标准 TP 按词表分片查表并进行 all-reduce。

两个兼容补丁仅用于配置检测与加载：v0.30 的 loader 会用主干索引过滤独立 head，
因此只对明确指定的 `joint_head.safetensors` 绕过索引过滤。上游已有对应能力，
升级到包含这些能力的版本后应删除补丁。无需修改 NPU model runner。

## 支持范围

| 功能 | 当前范围 |
| --- | --- |
| 文本或 JSON state | 已实现；JSON 按 key 排序并紧凑序列化 |
| choice / noul / score | 联合评分，返回每个选项的概率、答案、置信度及诊断 |
| 问题数与选项数 | 最多 64 问、26 个 choice 选项、10 个 score 等级 |
| 请求长度 | 最多 `min(16384, max_model_len)`；只截断 state 尾部，完整保留问题和选项 |
| API | `/v1/systemone`；`/v1/decisions` 返回 501，与上游 pooling 策略一致 |
| instructions / joint | 顶层 instructions 不支持；字段 instructions 支持；不要设置 `joint=true`（该开关属于 diffusion 策略） |
| 图片、视频、音频 | 本次仅移植文本接口，传入媒体返回 400 |
| 精度与并行 | 目标为原始 BF16 权重、标准 TP；PP=1、PCP=1；不支持细粒度 LM-head TP |
| Prefix cache | 必须关闭；head 需要包括 state 在内的完整 token hidden states |
| Chunked prefill | 使用 AllPool 缓存完整 hidden states 后执行 head；CPU 分块一致性已验证 |
| ACLGraph / runner v2 | 沿用现有主干和 pooling 路径，需 NPU 验证；head 不做独立图捕获 |
| EP / flashcomm1 / MTP | 本次不适用：dense 模型的 pooling 任务，无 token 生成 |

模板与 checkpoint 训练格式一致，各字段独立分词并记录相对 prompt 尾部的 span。
choice 按名称排序后输入 head，响应恢复请求顺序。顶层 chat template kwargs 不改变
这个固定模板。一次请求的 `usage.output_tokens` 为 0，input_tokens 只计一次。
`label_mass=1` 表示 head 在候选项内归一化，不代表经校准的任务准确率。

## 环境准备

使用可运行 Qwen3.5 的 vLLM Ascend 0.30 环境，重新安装本分支，不升级 Transformers。
本地模型目录必须同时包含主干分片、索引、tokenizer、`joint_head_config.json`
和 `joint_head.safetensors`，不能仅复制 Qwen3.5 主干。

````{tabs}

```{tab} Atlas A2
使用现有 vLLM Ascend 0.30 A2 容器及配套 CANN、torch-npu。
从单卡 TP=1 开始；实际显存峰值和容量待测。
```

```{tab} Atlas A3
使用现有 vLLM Ascend 0.30 A3 容器及配套 CANN、torch-npu。
可按显存需求使用标准 TP；TP 多卡查表及通信仍需实机验证。
```

````

## 部署

```bash
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,ascend_structured_decisions
cd /workspace
vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --runner pooling \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port 8000
```

`VLLM_PLUGINS` 同时筛选平台、general 和 endpoint plugins；若已有其他插件，
请保留原列表并加入 `ascend_structured_decisions`。可以用完整本地 checkpoint
路径替代 HF 仓库名，或用 `--revision` 固定版本。

首次真权重请求通过后，可将 max-model-len 增加到 16384。验证分块预填充时，
改为 `--enable-chunked-prefill --max-num-batched-tokens 1024`。
若图执行失败，增加 `--enforce-eager` 对比并保留日志。
模型配置的 262144 位置上限不等于该决策模板的 16384 上限；本接口不做 128k 容量承诺。

## 功能验证

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clef-flash",
    "state": {"ticket": "My card was charged twice for the same order. Please refund the duplicate charge."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"shipping": "deliveries", "billing": "payments", "security": "account security"}
      },
      "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
      "severity": {
        "type": "score",
        "instructions": "How severe is the issue?",
        "criteria": ["minor", "moderate", "severe"]
      }
    }
  }'
```

验收：HTTP 200、每题概率和约为 1、choice 属于候选项、noul 位于 `[0,1]`、
score 位于 `[0,2]`、output_tokens 为 0。启动成功本身不足以证明权重加载和推理正确。

在 NPU 容器中运行新增真权重 E2E（自动启动完整模型，分别测试完整和分块预填充）：

```bash
pytest -sv tests/e2e/pull_request/one_card/test_clef_flash.py
```

该测试覆盖三类问题、choice 顺序还原、超长 state 截断、并发请求和不支持的输入。
更换 TP 配置后，还应重复概率检查及同一请求的一致性检查。

## 本地验证与精度评估

本地基于实际 vLLM 0.30 源码运行，169 项特性单元测试通过。
使用 checkpoint revision `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` 的 tokenizer
及已审阅的参考 encoder 验证了 299/16384-token 请求的 token ID 和 span 一致性；
完整 9B head 的 122 个参数张量名称及形状一致（121762820 个参数）。
这没有加载或运行 9B 真权重。

CPU 单元测试位于 `tests/ut/entrypoints/structured_decisions/test_clef_*.py`：

- 实际 vLLM 0.30 HTTP、PoolingParams、PoolingMetadata 和 AllPool 契约。
- 固定模板、负 span、选项重排、截断及请求/引擎错误状态。
- head 与独立逐问题参考公式的一致性；分块与完整 prompt 一致性。
- 词表分片、padding 和新增 token 的词向量重构。
- 本地/Hub revision 配置检测，真实 safetensors 主干及独立 head 的加载和缺失文件报错。
- 回归原有 Qwen3/Qwen3.5 NextTokenStrategy。

| 验证项目 | 状态 |
| --- | --- |
| CPU 逻辑及参考公式 | 已验证；小尺寸随机权重，不等同于完整模型 |
| NPU dummy 启动/请求 | 未运行，无 NPU 环境 |
| NPU 9B 真权重启动/请求 | 未运行，由使用者后续验证 |
| 任务准确率与校准 | 未评估；不填写虚构准确率或评估阈值 |

## 性能

未测量 NPU 延迟、吞吐或显存峰值。head 需要完整 prompt hidden states，分块模式
仍会缓存这些 states。参考实现按请求执行 head；暂未针对 head 做跨请求批处理优化。
本移植不在 NPU 热路径逐 token 调用 `item()`；schema 索引在 CPU 构造后批量异步传输。
请分别记录 TP、prompt 长度、问题数、选项数、并发数、首请求和稳定态延迟，
完成真权重验证后再评估部署容量。

## Native prefill graph（910B / MRV1，待实机验收）

vLLM 0.30 会将 pooling 模型的标准 FULL 模式降级到 PIECEWISE。
旧的 native prefill 校验仍要求标准 FULL，因而在加载权重后抛出
`Native prefill graphs require dense Qwen3.5 with FULL graph support enabled`。
这发生在捕获前，不能据此判断 NPU 算子是否可捕获。

Clef 继承的 Qwen3.5 `forward` 只返回所有 token 的 hidden states。
独立 native registry 可捕获这个完整主干 forward；runner 随后在图外执行
`AllPool`、schema 索引构造、词向量查表和 joint head。当前实现不捕获整个决策流水线，
也不强制绕过上游针对 pooling 的标准 FULL 限制。

本分支允许 Clef pooling 的独立 native registry 与标准 `NONE` / `PIECEWISE`
共存。首次验证建议在上面的部署命令中增加：

```bash
# 设置在 vllm serve 命令之前；保留原 VLLM_PLUGINS。
export VLLM_USE_V2_MODEL_RUNNER=0
# 追加以下 vllm serve 参数：
# --mamba-cache-mode none
# --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}'
# --additional-config '{"native_prefill_graph":true,"native_prefill_graph_max_context":1024,"native_prefill_graph_request_counts":[1,4]}'
```

`NONE` 仅关闭标准 dispatcher 的图；显式启用的 native registry 仍在启动时捕获主干。
不要设置 `--enforce-eager`，它会禁止启动捕获。保持 TP/PP/DP/PCP/DCP=1、
无量化、无 LoRA、无 speculative decoding、无 KV transfer。该 registry 目前只针对 910B。
初始验证保持 `--no-enable-chunked-prefill`；超出 native 容量或续接 prefill 的请求回落原路径。

1024 是 native 默认上限，不是 Clef 的上下文上限。需要 8k 时，可用以下参数替换
上面的容量配置（完整 prompt 包括 state、问题和选项；显存及实机正确性尚待验证）：

```bash
# --max-model-len 8192
# --max-num-batched-tokens 8192
# --no-enable-chunked-prefill
# --additional-config '{"native_prefill_graph":true,"native_prefill_graph_max_context":8192,"native_prefill_graph_request_counts":[1],"native_prefill_graph_context_sizes":[256,512,1024,2048,4096,8192]}'
```

`native_prefill_graph_context_sizes` 独立控制每个请求的预填充长度桶，必须是非空正整数列表，
且不超过 native 上限及最小 request count 对应的 token 预算。自动补入宽度 1 和有效上限。
未指定时沿用标准 capture sizes；若标准列表为空，则使用 2 的幂加上有效上限。
这修复了标准 NONE 下仅有 `[1, max_context]`、短请求被填充到最大长度的问题。
`request_counts:[1]` 先限制捕获内存；并发请求会分组逐图 replay。
后续可测量 `[1,2,4]`，只有 request count × width 不超过 batch token 预算的图才会捕获。
更多长度桶和请求数会增加启动时间及图内存；16384 上限也可按同样方式配置，但不代表已通过容量验收。

不能仅凭延迟或 `Native prefill registry sealed` 判断请求是否 replay：

- `registry sealed`：启动捕获完成。
- `Native prefill first replay`：每个图首次实际 replay 后输出 INFO；包含真实/捕获 token 数和
  `padding_ratio`（捕获 token 数 ÷ 真实 token 数）。后续逐次 hit 使用 DEBUG。
- `Native prefill fallback`：每类原因首次输出 INFO，例如 `context_capacity`（超过长度桶）、
  `cached_context` / `not_fresh_prefill`（缓存或续接）、`batch_capacity`（批次预算）和
  `input_layout`（输入布局不兼容）。这些请求仍走标准 forward。

即使确认 replay，padding、图外输入准备/输出拷贝和 eager joint head 仍可能抵消收益。
比较性能时保持请求、并发、缓存策略和容量一致，仅切换 native 开关。

实际验收命令：

```bash
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_clef_native_prefill.py
```

测试分别覆盖 1024 和 4096 native 容量，并单独断言超过 1024 的请求 replay 计数增加、
概率与 eager 基线一致。尚未在本地执行 NPU 测试。

测试用真实 9B 权重比较 eager 与 native 路径的选项概率，验证不同 schema、单请求、
不等长批次、重复 replay、超长回落及 head 的图外执行；必须同时看到 registry sealed、
replay 计数增长、概率一致性通过。CPU 回归仅验证控制流和 token 行还原，
不能证明 torch-npu/CANN 的实际 capture/replay 可用。当前无 NPU，以上实机测试尚未运行。
