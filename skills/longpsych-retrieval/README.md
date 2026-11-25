# LongPsych 中文心理对话检索技能包

**版本：V1.2.4**

本技能包为 LongPsychAgent 提供中文 Agent Skill 说明文件，指导兼容的 Agent 通过已有 MCP 工具调用心理对话检索服务。

本次版本主要将技能说明、工具协议、检索流程、结果解释和使用文档完整中文化。底层 FastAPI、MCP Tool、BGE、FAISS、ORCA 和 QR-Future 实现及接口协议均不因本次文档升级而改变。

## 1. 文件结构

```text
longpsych-retrieval/
├── SKILL.md
├── README.md
└── references/
    ├── tool-contract.md
    ├── retrieval-workflow.md
    └── result-interpretation.md
```

## 2. 技能能力

本技能用于从中文心理对话语料中检索并排序候选回复，支持：

- `dense`：BGE Dense 检索。
- `orca`：ORCA k=2 检索。
- `qr_future`：Dense 第一阶段候选召回，随后由五折 QR-Future FULL 模型重排。

默认使用 `qr_future`，默认返回 Top-5。候选池大小通常由 FastAPI 服务设置为 100。

## 3. 系统架构

```text
Agent / MCP 客户端
    |
    v
LongPsych Retrieval Skill
    |
    v
Streamable HTTP MCP 服务（127.0.0.1:8001）
    |
    v
已有 FastAPI 服务（127.0.0.1:8000）
    |
    v
LongPsychAgent 检索流水线
    ├── BGE-large-zh-v1.5 编码器
    ├── Dense FAISS 候选召回
    ├── ORCA k=2 检索
    └── 五折 QR-Future FULL 重排
```

MCP 服务通过 HTTP 调用已有 FastAPI 服务，不会在 MCP 进程中再次加载 BGE、FAISS 索引或 QR-Future 模型资产。

**QR-Future 候选协议：** 第一阶段使用 Dense 候选召回，不得未经验证替换为 ORCA 候选召回。

## 4. MCP 工具

当前 MCP 服务注册三个工具：

| 工具名称 | 用途 |
|---|---|
| `longpsych_search_dialogue` | 常规心理对话候选检索，默认 QR-Future |
| `longpsych_compare_retrievers` | 按相同查询比较 Dense、ORCA 和 QR-Future |
| `longpsych_retrieval_health` | 检查底层检索服务健康状态 |

完整输入参数、返回字段和错误约定见 `references/tool-contract.md`。

## 5. AutoDL 部署

以下命令均在项目根目录中执行。

### 5.1 启动顺序

必须先启动 FastAPI，再启动 MCP 服务。

**终端一：启动或保持 FastAPI 服务**

FastAPI 默认地址：

```text
http://127.0.0.1:8000
```

检查健康状态：

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

应确认返回 `status` 为 `ready`，并确认 `qr_future.status` 为 `ready`。

**终端二：启动 MCP 环境（首次部署时执行）**

```bash
bash scripts/agent/setup_mcp_env.sh
```

**终端二：启动 MCP 服务**

```bash
bash scripts/agent/run_qr_future_mcp.sh
```

默认 MCP 地址：

```text
http://127.0.0.1:8001/mcp
```

保持 MCP 服务进程运行。MCP 服务依赖已经运行的 FastAPI 服务。

### 5.2 远程客户端连接

如果 MCP 客户端运行在本地电脑，可通过 SSH 隧道转发端口：

```bash
ssh -L 8001:127.0.0.1:8001 root@YOUR_SERVER
```

随后在支持 Streamable HTTP 的 MCP 客户端中配置：

```text
http://127.0.0.1:8001/mcp
```

客户端的具体配置格式取决于宿主应用。需要单独配置 MCP 连接；安装 Skill 文件不会自动创建网络连接。

MCP 服务默认绑定本机回环地址。不要在缺少身份验证和 TLS 保护的情况下将服务直接暴露到公网。

## 6. Skill 安装

将本压缩包中的 `longpsych-retrieval/` 目录放入 Agent 宿主所要求的 Skill 目录。

宿主目录结构应能找到：

```text
longpsych-retrieval/SKILL.md
```

同时保留 `references/` 下的三个参考文件。不要只复制 `SKILL.md`，否则技能引用的协议、流程和结果解释文档将缺失。

Skill 安装与 MCP 服务配置是两个独立步骤：

- Skill 提供 Agent 应遵循的检索规则和使用说明。
- MCP 配置提供实际工具连接。
- FastAPI 和 MCP 服务需要在运行环境中正常启动。

## 7. 验收建议

安装后按以下顺序验收：

1. 检查 FastAPI `/health` 返回 `ready`。
2. 确认 MCP 服务可以连接。
3. 调用 `longpsych_retrieval_health`。
4. 调用 `longpsych_search_dialogue`，使用中文查询、`mode="qr_future"`、`top_k=5`。
5. 检查返回业务字段 `status="ok"`，并核对候选文本与排序字段。
6. 只有在需要验证模式差异时，再调用 `longpsych_compare_retrievers`。
7. 在 Agent 宿主中确认 Skill 已加载，并能遵守中文查询构造、QR-Future 默认模式和分数解释规范。

推荐测试查询：

```json
{
  "query": "最近总是感到焦虑，不知道该怎么办。",
  "mode": "qr_future",
  "top_k": 5
}
```

通常可以省略 `candidate_k`，由服务使用默认候选池设置。

## 8. 常见问题

### FastAPI 正常，但 MCP 无法调用

确认 MCP 服务进程仍在运行，并检查客户端配置的地址是否为：

```text
http://127.0.0.1:8001/mcp
```

### MCP 健康检查失败

先确认 FastAPI 的 `http://127.0.0.1:8000/health` 是否正常，再检查 MCP 服务配置的 `LONGPSYCH_API_BASE_URL` 是否指向正确的 FastAPI 地址。

### QR-Future 调用失败

检查工具返回的 `status` 和诊断信息。不要将失败伪装成空结果，也不要静默切换到 Dense 或 ORCA。

### Skill 已安装，但 Agent 找不到工具

Skill 文件不会自动注册 MCP 工具。需要在 Agent 宿主中另外配置 LongPsychAgent MCP 服务，并确认三个工具已经注册。

## 9. 版本说明

V1.2.4 是中文文档与技能指令版本升级：

- 完整中文化五个 Skill 文件。
- 保留 Skill 名称 `longpsych-retrieval`。
- 保留三个 MCP 工具名称及其现有参数。
- 保留 QR-Future Dense 第一阶段候选召回和五折重排协议。
- 明确原始排序分数不是概率。
- 强化当前发言锚定、相关历史筛选、中文查询和心理场景能力边界。

本次不修改底层检索模型、FastAPI 服务、MCP Tool 实现或实验结果。


## LangGraph V3 多轮会话说明（v1.2.4）

v1.2.4 将 Skill 文档与 LongPsychAgent LangGraph V3 的实际多轮运行规则对齐，不改变底层检索算法或 MCP 工具接口。

- LangGraph V3 Agent 负责话题关系判断、历史用户消息筛选、原文来源校验、去重和最终 Query 构造。
- 历史仅来自此前用户原始消息，最多追加两条直接相关且可追溯的内容。
- 当前用户消息完整保留并作为最终 Query 的主体。
- 无关新话题不调用心理检索工具，也不携带旧话题历史。
- 当前部署由 Agent 使用 SQLite 持久化会话状态；恢复同一会话时必须使用相同 `thread_id`。
- MCP 检索服务仍只接收调用方传入的 `query`，不自行读取聊天记录或维护长期记忆。

本次升级只涉及 Skill 文档与发布包，不修改 FastAPI、MCP Server、BGE、FAISS、ORCA、QR-Future 或 LangGraph V3 的运行代码。详细规则见 `SKILL.md` 和 `references/retrieval-workflow.md`。
