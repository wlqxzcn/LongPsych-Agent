# LongPsychAgent MCP 工具协议 V1

本文档说明 LongPsych Retrieval Skill 1.2.x 所调用的现有 MCP 工具协议。技能包版本升级不代表底层 MCP 接口发生变化；当前仍保留 MCP Tool Contract V1。

## 1. 服务与传输方式

- MCP 协议传输方式：Streamable HTTP。
- 默认 MCP 地址：`http://127.0.0.1:8001/mcp`。
- 底层 FastAPI 地址：`http://127.0.0.1:8000`。
- MCP 服务负责调用已有 FastAPI 服务。
- MCP 进程本身不加载 BGE、FAISS 索引或 QR-Future 模型资产，避免重复占用模型和向量内存。

调用链路：

```text
MCP 客户端 / Agent
    -> Streamable HTTP MCP 服务（127.0.0.1:8001）
        -> 已运行的 FastAPI 服务（127.0.0.1:8000）
            -> LongPsychAgent 检索流水线
                -> BGE 编码与 Dense FAISS
                -> ORCA k=2 检索
                -> 五折 QR-Future FULL 重排
                -> 原始候选回复标识与文本
```

## 2. 工具一：`longpsych_search_dialogue`

用途：根据一个查询检索心理对话候选回复。

### 输入参数

| 参数 | 类型 | 必填 | 默认值 | 约束与说明 |
|---|---|---:|---|---|
| `query` | string | 是 | 无 | 1–4000 个字符 |
| `mode` | string | 否 | `qr_future` | 只能是 `dense`、`orca`、`qr_future` |
| `top_k` | integer | 否 | `5` | 范围 1–100 |
| `candidate_k` | integer 或 null | 否 | `null` | 范围 1–5000；仅 `qr_future` 模式有效，且必须大于或等于 `top_k` |

当 `candidate_k` 未提供时，服务使用 FastAPI 配置的默认候选池大小；当前部署通常为 100。

### 检索模式

- `dense`：现有 BGE Dense 检索。
- `orca`：现有 ORCA k=2 检索。
- `qr_future`：Dense 第一阶段候选召回，再使用冻结的五折 QR-Future FULL 排序模型重排。

QR-Future 的候选召回协议固定为 Dense 第一阶段。不得将其描述为 ORCA 第一阶段召回。

### 返回结构

工具返回 JSON 对象，业务状态字段包括：

- `status`：`ok` 或 `error`。
- `data`：成功时为底层 FastAPI 返回的检索结果。
- `message`：发生错误时提供诊断信息。

成功结果中的候选字段以服务实际返回为准，常见字段包括 `response_uid`、`response_text`、`rank`、`retrieval_score`、`qr_future_score`、`candidate_rank` 和 `faiss_row`。

## 3. 工具二：`longpsych_compare_retrievers`

用途：使用相同查询依次比较 Dense、ORCA 和 QR-Future 三种检索模式。

### 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `query` | string | 是 | 无 | 三种模式使用同一个查询 |
| `top_k` | integer | 否 | `5` | 每种模式返回的候选数量 |
| `candidate_k` | integer | 否 | `100` | 仅作用于 QR-Future 第一阶段候选池 |

该工具按顺序执行三种检索。`candidate_k` 不改变 Dense 和 ORCA 两种独立模式的候选数量设置。

### 使用约束

- 仅在用户明确要求检索模式对比时调用。
- 对比时保持查询文本一致。
- 不得将不同模式的原始分数解释为经过统一校准的概率。
- QR-Future 仍使用 Dense 第一阶段候选召回。

返回对象同样包含 `status`、`data` 或错误诊断信息。

## 4. 工具三：`longpsych_retrieval_health`

用途：检查 LongPsychAgent 底层 FastAPI 检索服务是否就绪。

### 输入参数

无。

### 返回信息

健康检查结果可能包括：

- 服务状态。
- 可用检索模式。
- 语料规模。
- 向量维度。
- 默认候选池大小。
- QR-Future 条件名称。
- 折数。
- 输入空间。
- 运行设备。
- 响应向量资产和行映射是否加载。

健康检查成功不等于某一次检索业务调用必然成功；仍需检查具体工具调用的返回状态。

## 5. QR-Future 候选协议

QR-Future V1 的候选池由 Dense 第一阶段召回生成，不得在没有独立验证的情况下替换为 ORCA 候选池。

五折 QR-Future 排序器分别使用对应折的 ORCA k=2 投影，对候选计算原始排序分数，再对五折分数求平均并确定最终排序。

在线检索使用排序头。未来状态辅助头是训练阶段的辅助目标，不能将 `qr_future_score` 解释为未来状态概率。

## 6. 错误约定

工具级运行失败通过 JSON 对象返回，业务字段为 `status="error"` 时表示失败。

客户端必须：

1. 检查 `status`，不能只根据 MCP 调用是否抛出异常判断业务成功。
2. 不得将错误结果解释为空检索结果。
3. 不得编造候选回复或评分。
4. 必要时调用健康检查工具。
5. QR-Future 调用失败时不得未经说明自动切换检索模式。

## 7. 兼容性说明

本协议描述的是当前 LongPsychAgent MCP 工具接口。技能调用方需要自行配置 MCP 客户端连接；Skill 文件不会自动启动 MCP 服务，也不会自动建立网络连接。
