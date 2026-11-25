#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Build LongPsych Retrieval Skill v1.2.4 on the AutoDL project host.
#
# Run:
#   cd <repository-root>
#   .venv-agent/bin/python scripts/agent/release_skill_v1_2_4.py
#
# The script verifies exact v1.2.3 source hashes, creates a backup, updates
# documentation only, builds the ZIP, and verifies its contents. It does not
# modify FastAPI, MCP server code, ORCA, QR-Future, or LangGraph runtime code.

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "skills" / "longpsych-retrieval"
DIST_DIR = ROOT / "dist"
BACKUP_ROOT = ROOT / "backups" / "skill"

EXPECTED_HASHES = {
    "SKILL.md": "d601463708ad32e35f7991836782c822a7d1fc9730d8f6d96b6161cbbad86b62",
    "README.md": "14ea30ee811cc60d3271ac6409f31dbc90f0eb6ac68d5f18359721b1dff9ab87",
    "references/result-interpretation.md": "2054a1435570751b9c8a5b22a64e4735a8d50776814d1c9efabbe45d126e10cd",
    "references/retrieval-workflow.md": "ad6607f90449faa088c6b959dde6f27892a2c48b6cc85e5dad77988ef950402f",
    "references/tool-contract.md": "18b81c0894b872f3648a836bf53ea904132e6b22900ff7fbce7c76c8843406e5",
}

WORKFLOW_V124 = r'''# 中文心理对话检索工作流程（LangGraph V3）

本文档规定 LongPsychAgent 如何组织中文心理对话检索查询，以及 LangGraph V3、Agent Skill 和 MCP 检索服务之间的职责边界。

## 1. 组件职责

- **LangGraph V3 Agent**：管理当前会话状态，判断话题关系，从历史用户消息中筛选可追溯的上下文，并构造最终查询。
- **SQLite 会话存储**：在当前部署中持久化 Agent 会话状态。恢复同一会话时，应继续使用原来的 `thread_id`。
- **LongPsych Retrieval Skill**：规定检索工具的选择、参数使用和结果解释方式。
- **MCP 检索服务**：接收调用方提供的单个 `query` 并执行检索；MCP 服务本身不读取完整聊天记录，也不独立维护会话历史。

因此，多轮历史管理属于 Agent 运行层，不属于 MCP 检索工具本身。

## 2. V3 多轮上下文流程

每次检索前，LangGraph V3 Context Builder 按以下顺序处理：

1. 读取当前用户消息和此前会话历史。
2. 历史来源仅限此前的用户原始消息，不包含助手回复、检索候选或模型推测。
3. 判断当前消息与历史的关系。允许的关系为：
   - `initial`：首次出现的相关问题。
   - `same_topic`：延续当前正在讨论的话题。
   - `return_to_topic`：用户明确回到较早讨论的话题，可以跨越中间无关轮次。
   - `unrelated`：与心理对话检索无关的新话题。
   - `uncertain`：话题关系不确定。
4. 仅在确实需要历史背景时，选择最多两条相关历史用户原文。
5. 对选择结果执行服务端校验、去重和来源追踪。
6. 构造 `final_query`，并将其作为实际 MCP 检索参数。

## 3. 话题关系与检索规则

### 3.1 首次相关问题

首次出现且属于技能范围的问题可以直接检索。没有必要为了增加上下文而构造历史背景。

```text
当前用户消息：
最近总是感到焦虑，不知道该怎么办。

实际 query：
最近总是感到焦虑，不知道该怎么办。
```

### 3.2 同主题追问

当前问题延续历史话题时，可以追加一至两条直接相关的历史用户原文。

```text
历史用户消息：
最近工作任务不断增加，我担心无法按时完成。

当前用户消息：
今天主管又临时给我加了任务，我压力很大，不知道怎么安排。

实际 query：
今天主管又临时给我加了任务，我压力很大，不知道怎么安排。
相关历史背景：最近工作任务不断增加，我担心无法按时完成。
```

### 3.3 明确返回旧话题

用户明确返回较早话题时，可以跨越中间无关轮次筛选历史，但只能选择与当前问题直接相关的历史用户原文。

```text
当前用户消息：
回到工作这件事，我还是担心自己做不完。

实际 query：
回到工作这件事，我还是担心自己做不完。
相关历史背景：{最多两条可追溯的相关历史用户原文}
```

不得因为某条消息在时间上较近，就优先于真正相关的历史内容。

### 3.4 无关新话题

旅游、购物、天气等与心理对话检索无关的请求不得触发心理检索，也不得携带此前心理话题的历史背景。

当关系判定为 `unrelated` 时，V3 服务端强制设置 `should_search=false` 并清空历史选择。该轮不调用 `longpsych_search_dialogue`。

## 4. 历史原文校验与去重

历史上下文必须满足以下约束：

- 每条历史只能来自此前用户消息。
- 每条历史引用必须是对应历史用户消息中逐字连续出现的原文片段。
- 最多选择两条历史内容。
- 不得把当前消息本身作为自己的历史背景。
- 不得重复注入相同或近似重复的历史表达。
- 不得将助手回复、候选回复或模型推测写成用户事实。

V3 服务端对去重文本执行 NFKC 规范化和 `casefold`，并忽略空白及标点差异。对于足够长且长度相近的文本，还执行近似重复判断。去重后保留可追溯的历史原文，并记录对应的用户轮次。

如果历史引用无法追溯到此前用户消息，服务端应拒绝该引用，而不是将其当作用户事实继续检索。

## 5. 最终查询构造

最终查询必须以当前用户原始消息为主体：

```text
{当前用户原始消息}
相关历史背景：{历史原文1}；{历史原文2}
```

具体约束：

1. 当前用户消息必须完整保留，不得改写、删减或被历史摘要替代。
2. 没有合格历史内容时，`final_query` 就是当前用户原文。
3. 有合格历史内容时，历史背景追加在当前消息之后。
4. `final_query` 必须与实际传给 `longpsych_search_dialogue` 的 `query` 完全一致。
5. MCP `query` 长度上限为 4000 个字符。若追加历史后超限，应舍弃历史背景并保留完整当前消息，不得截断当前用户消息。
6. 不得为了提高检索命中而添加用户没有表达的经历、情绪、原因或结论。

## 6. MCP 工具选择

### 常规检索

默认调用 `longpsych_search_dialogue`，使用：

```json
{
  "query": "当前用户原文，必要时追加经校验的历史背景",
  "mode": "qr_future",
  "top_k": 5
}
```

通常不传 `candidate_k`，由服务采用配置的默认候选池大小。

### 指定基线模式

只有在用户明确指定或任务确实要求基线检索时，才使用 `dense` 或 `orca`。

### 三模式对比

只有在用户明确要求比较时，才调用 `longpsych_compare_retrievers`。三个模式必须使用同一个查询。

### 健康检查

在服务异常、用户要求检查状态或部署验收时，调用 `longpsych_retrieval_health`。不需要在每次成功检索后重复检查。

## 7. SQLite 会话恢复

在当前 LangGraph V3 部署中，Agent 会话状态通过 SQLite 持久化。

- 同一会话恢复时，使用原来的 `thread_id`。
- 使用新的 `thread_id` 表示新的独立会话。
- 持久化的是 Agent 侧会话状态；这不意味着 MCP 检索服务拥有用户长期记忆。
- 如果其他 Agent 宿主没有实现会话存储和历史筛选，就必须由该宿主自行提供这些能力，不能假设 MCP 会自动补齐历史。

## 8. 中文内容与隐私

- 面向中文语料时，优先保留用户原始中文表达。
- 不要为了形式统一而将中文查询翻译成英文。
- 只发送完成检索所需的最少上下文。
- 避免发送姓名、电话、地址、账号标识等不必要的身份信息。
- 不要默认发送完整会话记录。
- 不得从有限上下文中臆测用户未披露的经历或心理状态。

## 9. 检索结果使用

- 保留返回候选的 `response_text` 原文。
- 用 `rank` 表示工具返回的最终名次。
- 如展示 `candidate_rank`，应说明它是 QR-Future 重排前的 Dense 初始名次。
- 如展示 `qr_future_score`，必须说明它是原始排序分数，不是概率。
- 不要仅凭排名或分数断言候选适合某位具体用户。
- 如 Agent 需要生成最终对话回复，应由后续独立、受约束的生成流程完成，并明确区分检索文本与生成文本。
'''

MULTITURN_RULES_V124 = r'''## 多轮检索执行硬约束（LangGraph V3）

本节规定 LongPsychAgent LangGraph V3 运行时的多轮上下文行为。它不改变 MCP Tool Contract V1。

1. **历史来源**：只允许从此前用户原始消息中选择历史内容；助手回复、检索候选和模型推测都不是用户事实。
2. **话题判断**：每次检索前判断 `initial`、`same_topic`、`return_to_topic`、`unrelated` 或 `uncertain`。
3. **历史数量**：最多选择两条与当前问题直接相关的历史用户原文；不得为了增加上下文而选择弱相关信息。
4. **原文可追溯**：每条历史引用必须是此前用户消息中逐字连续出现的片段，并保留来源轮次。
5. **确定性去重**：服务端对历史文本进行规范化去重；相同或近似重复内容不得反复注入。当前问题不得作为自己的历史背景。
6. **无关话题隔离**：`unrelated` 必须设置 `should_search=false`、清空历史选择，并且不得调用心理检索工具。
7. **当前消息优先**：`final_query` 必须以当前用户原始消息完整开头，不得改写、删减或用历史摘要替代。
8. **历史追加**：只有通过来源校验和去重的历史内容才能追加到当前消息之后；没有合格历史时直接使用当前消息原文。
9. **长度保护**：若追加历史会使查询超过 MCP 的 4000 字符限制，应舍弃历史背景并保留完整当前消息，不得截断当前消息。
10. **实际调用一致性**：传给 `longpsych_search_dialogue` 的 `query` 必须与上下文构造器产生的 `final_query` 完全一致。
11. **会话持久化边界**：当前部署由 LangGraph V3 Agent 使用 SQLite 持久化会话状态；恢复同一会话时使用相同 `thread_id`。MCP 检索服务本身仍然是接收单条查询的检索服务，不独立维护聊天历史或用户长期记忆。

对于不运行 LangGraph V3 的其他 MCP 宿主，上述历史管理、持久化和查询构造必须由宿主自行实现；不能仅凭安装本 Skill 就声称已经具备多轮记忆能力。
'''

README_APPENDIX = r'''


## LangGraph V3 多轮会话说明（v1.2.4）

v1.2.4 将 Skill 文档与 LongPsychAgent LangGraph V3 的实际多轮运行规则对齐，不改变底层检索算法或 MCP 工具接口。

- LangGraph V3 Agent 负责话题关系判断、历史用户消息筛选、原文来源校验、去重和最终 Query 构造。
- 历史仅来自此前用户原始消息，最多追加两条直接相关且可追溯的内容。
- 当前用户消息完整保留并作为最终 Query 的主体。
- 无关新话题不调用心理检索工具，也不携带旧话题历史。
- 当前部署由 Agent 使用 SQLite 持久化会话状态；恢复同一会话时必须使用相同 `thread_id`。
- MCP 检索服务仍只接收调用方传入的 `query`，不自行读取聊天记录或维护长期记忆。

本次升级只涉及 Skill 文档与发布包，不修改 FastAPI、MCP Server、BGE、FAISS、ORCA、QR-Future 或 LangGraph V3 的运行代码。详细规则见 `SKILL.md` 和 `references/retrieval-workflow.md`。
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(ROOT.is_dir(), f"Project directory not found: {ROOT}")
    require(SKILL_DIR.is_dir(), f"Skill directory not found: {SKILL_DIR}")

    print("========== 1. VERIFY V1.2.3 SOURCE HASHES ==========")
    originals: dict[str, bytes] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = SKILL_DIR / rel
        require(path.is_file(), f"Missing source file: {path}")
        data = path.read_bytes()
        actual = sha256_bytes(data)
        print(f"{'PASS' if actual == expected else 'FAIL'} {rel} {actual}")
        require(
            actual == expected,
            f"Source hash mismatch for {rel}; refusing to modify an unexpected version.",
        )
        originals[rel] = data

    print("\n========== 2. PREPARE DOCUMENTATION UPDATES ==========")
    skill = originals["SKILL.md"].decode("utf-8")
    readme = originals["README.md"].decode("utf-8")
    tool_contract = originals["references/tool-contract.md"].decode("utf-8")

    skill, count = re.subn(
        r'(?m)^(\s*version:\s*)"1\.2\.3"(\s*)$',
        r'\g<1>"1.2.4"\2',
        skill,
        count=1,
    )
    require(count == 1, "Could not find exactly one Skill metadata version 1.2.3.")

    start_marker = "## 多轮检索执行硬约束（高优先级）"
    start = skill.find(start_marker)
    require(start >= 0, "Could not locate the multi-turn section in SKILL.md.")
    # In the audited v1.2.3 source, the multi-turn hard-constraints section is
    # the final section of SKILL.md, so replace it through EOF and preserve all
    # preceding result-interpretation material unchanged.
    skill = skill[:start] + MULTITURN_RULES_V124 + "\n"

    # Update every standalone v1.2.3 version reference, not just the first
    # README occurrence. Keep an optional v/V prefix and leave unrelated
    # versions (for example v1.2.1) untouched.
    old_version_pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9.])v?1\.2\.3(?![A-Za-z0-9.])"
    )

    def bump_version_reference(match: re.Match[str]) -> str:
        token = match.group(0)
        prefix = token[0] if token[:1].lower() == "v" else ""
        return prefix + "1.2.4"

    skill, skill_ref_count = old_version_pattern.subn(
        bump_version_reference, skill
    )
    readme, readme_ref_count = old_version_pattern.subn(
        bump_version_reference, readme
    )
    require(
        readme_ref_count >= 1,
        "README version marker 1.2.3 was not found; refusing an unversioned release.",
    )
    print(
        "UPDATED VERSION REFERENCES: "
        f"SKILL.md={skill_ref_count}, README.md={readme_ref_count}"
    )
    require(
        "## LangGraph V3 多轮会话说明（v1.2.4）" not in readme,
        "README already contains the v1.2.4 appendix; refusing duplicate insertion.",
    )
    readme = readme.rstrip() + README_APPENDIX + "\n"

    old_contract_intro = (
        "本文档说明 LongPsych Retrieval Skill V1.2.0 所调用的现有 MCP 工具协议。"
        "技能包版本升级不代表底层 MCP 接口发生变化；本版本保留 MCP Tool Contract V1。"
    )
    new_contract_intro = (
        "本文档说明 LongPsych Retrieval Skill 1.2.x 所调用的现有 MCP 工具协议。"
        "技能包版本升级不代表底层 MCP 接口发生变化；当前仍保留 MCP Tool Contract V1。"
    )
    require(old_contract_intro in tool_contract, "Expected tool-contract introduction was not found.")
    tool_contract = tool_contract.replace(old_contract_intro, new_contract_intro, 1)

    updated = {
        "SKILL.md": skill.encode("utf-8"),
        "README.md": readme.encode("utf-8"),
        "references/retrieval-workflow.md": WORKFLOW_V124.encode("utf-8"),
        "references/tool-contract.md": tool_contract.encode("utf-8"),
        # Result interpretation rules were audited and are intentionally unchanged.
        "references/result-interpretation.md": originals["references/result-interpretation.md"],
    }

    require('version: "1.2.4"' in skill, "SKILL metadata did not update to 1.2.4.")
    require(
        re.search(r"(?i)(?<![A-Za-z0-9.])v?1\.2\.4(?![A-Za-z0-9.])", readme)
        is not None,
        "README did not update to version 1.2.4.",
    )
    stale_version_pattern = re.compile(
        r"(?i)(?<![A-Za-z0-9.])v?1\.2\.3(?![A-Za-z0-9.])"
    )
    stale_skill = stale_version_pattern.findall(skill)
    stale_readme = stale_version_pattern.findall(readme)
    require(
        not stale_skill and not stale_readme,
        "Stale v1.2.3 version references remain: "
        f"SKILL.md={stale_skill}, README.md={stale_readme}",
    )
    require("LangGraph V3" in WORKFLOW_V124, "V3 workflow content missing.")

    print("\n========== 3. BACK UP ORIGINAL SOURCE ==========")
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"pre_v1.2.4_{stamp}"
    require(not backup_dir.exists(), f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    for rel, data in originals.items():
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    print(f"BACKUP: {backup_dir}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / "longpsych_retrieval_skill_v1.2.4_zh.zip"
    checksum_path = DIST_DIR / "longpsych_retrieval_skill_v1.2.4_zh.zip.sha256"
    manifest_path = DIST_DIR / "longpsych_retrieval_skill_v1.2.4_zh.manifest.json"

    require(not zip_path.exists(), f"Release ZIP already exists: {zip_path}")
    require(not checksum_path.exists(), f"Checksum file already exists: {checksum_path}")
    require(not manifest_path.exists(), f"Manifest already exists: {manifest_path}")

    print("\n========== 4. WRITE V1.2.4 DOCUMENTATION ==========")
    try:
        for rel, data in updated.items():
            path = SKILL_DIR / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"UPDATED {rel} ({len(data)} bytes)")

        print("\n========== 5. BUILD RELEASE ZIP ==========")
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as zf:
            for rel in sorted(updated):
                arcname = f"longpsych-retrieval/{rel}"
                zf.writestr(arcname, updated[rel])

        expected_members = {f"longpsych-retrieval/{rel}" for rel in updated}
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = set(zf.namelist())
            require(members == expected_members, f"ZIP member mismatch: {members}")
            bad_member = zf.testzip()
            require(bad_member is None, f"ZIP CRC verification failed: {bad_member}")

            for rel, expected_data in updated.items():
                archived = zf.read(f"longpsych-retrieval/{rel}")
                require(archived == expected_data, f"Archive/source mismatch: {rel}")

            archived_skill = zf.read("longpsych-retrieval/SKILL.md").decode("utf-8")
            archived_readme = zf.read("longpsych-retrieval/README.md").decode("utf-8")
            require('version: "1.2.4"' in archived_skill, "ZIP Skill version mismatch.")
            require("V1.2.4" in archived_readme, "ZIP README version mismatch.")

        source_manifest = {
            "skill_version": "1.2.4",
            "mcp_tool_contract": "V1",
            "files": {rel: sha256_bytes(data) for rel, data in sorted(updated.items())},
        }
        manifest_path.write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        zip_hash = sha256_file(zip_path)
        checksum_path.write_text(
            f"{zip_hash}  {zip_path.name}\n",
            encoding="utf-8",
        )

    except Exception:
        # Restore original source files if packaging or verification fails.
        for rel, data in originals.items():
            (SKILL_DIR / rel).write_bytes(data)
        for path in (zip_path, checksum_path, manifest_path):
            if path.exists():
                path.unlink()
        raise

    print("\n========== 6. RELEASE VERIFICATION ==========")
    print("STATUS: PASS")
    print(f"ZIP: {zip_path}")
    print(f"ZIP SIZE: {zip_path.stat().st_size} bytes")
    print(f"SHA256: {zip_hash}")
    print(f"CHECKSUM: {checksum_path}")
    print(f"MANIFEST: {manifest_path}")
    print("\nSOURCE FILE HASHES:")
    for rel in sorted(updated):
        print(f"{sha256_file(SKILL_DIR / rel)}  {rel}")

    print("\nUNCHANGED RUNTIME COMPONENTS:")
    print("- FastAPI")
    print("- MCP Server and MCP Tool Contract V1 interface")
    print("- BGE / FAISS")
    print("- ORCA k=2")
    print("- QR-Future five-fold scorer")
    print("- LangGraph V3 runtime code")
    print("\nNext: run the V3 multi-turn, dedup regression, and SQLite persistence tests.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nRELEASE FAILED: {exc}", file=sys.stderr)
        raise
