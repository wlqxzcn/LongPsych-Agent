# LongPsychAgent V3 安装与验收

## 1. 版本与边界

V3 独立于 V2：
- 不覆盖 `agent_graph_v2.py`。
- 不修改 FastAPI、MCP Tool、BGE、FAISS、ORCA 或 QR-Future 模型。
- 默认检索模式仍为 `qr_future`。
- QR-Future 仍使用现有 Dense 第一阶段候选召回及五折重排协议。
- 会话历史只保存用户原始 Query 和检索元数据，不把候选回复当成用户事实。

## 2. 部署文件

将压缩包内文件复制到项目根目录，保持目录结构：
- `agent/longpsych_agent/graph/agent_graph_v3.py`
- `scripts/agent/run_langgraph_agent_v3.py`
- `scripts/agent/test_agent_v3_multiturn.py`
- `scripts/agent/test_agent_v3_persistence.py`

## 3. 依赖

在 `requirements-agent.txt` 追加：

```text
langgraph-checkpoint-sqlite==3.1.3
aiosqlite==0.22.1
```

当前环境已安装这些依赖，不需要重复安装。

## 4. 静态检查

```bash

PYTHONPATH=agent .venv-agent/bin/python -m py_compile \
  agent/longpsych_agent/graph/agent_graph_v3.py \
  scripts/agent/run_langgraph_agent_v3.py \
  scripts/agent/test_agent_v3_multiturn.py \
  scripts/agent/test_agent_v3_persistence.py
```

## 5. 多轮验收

确保 FastAPI 8000 和 MCP 8001 已启动：

```bash
PYTHONPATH=agent .venv-agent/bin/python \
  scripts/agent/test_agent_v3_multiturn.py
```

## 6. SQLite 持久化验收

```bash
PYTHONPATH=agent .venv-agent/bin/python \
  scripts/agent/test_agent_v3_persistence.py
```

## 7. 交互运行

```bash
PYTHONPATH=agent .venv-agent/bin/python \
  scripts/agent/run_langgraph_agent_v3.py \
  --thread-id summer-demo-001
```

退出后再次运行同一命令，并使用相同数据库路径和 `--thread-id`，即可恢复该会话。

## 8. SQLite 数据文件

默认位置：

`data/agent/longpsych_agent_v3.sqlite`

不要在没有备份的情况下删除该文件。它保存 LangGraph 会话 Checkpoint。
