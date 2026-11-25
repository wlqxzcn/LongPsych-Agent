import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from longpsych_agent.config import settings
from longpsych_agent.retrieval.unified_retriever import (
    LongPsychRetriever,
)

MODEL = str(settings.model_path)
INDEX_DIR = str(settings.index_dir)

print("========== INITIALIZATION ==========")

retriever = LongPsychRetriever.from_pretrained(
    model_name_or_path=MODEL,
    index_dir=INDEX_DIR,
    default_mode="orca",
    device="cuda",
    batch_size=16,
    show_progress_bar=False,
)

assert retriever.corpus_size == 61501
assert retriever.dimension == 1024

query = "最近总是感到焦虑，不知道该怎么办。"

print("\n========== SINGLE QUERY TEST ==========")

for mode in ["dense", "orca"]:

    result = retriever.search(
        query,
        top_k=5,
        mode=mode,
    )

    assert result["mode"] == mode
    assert len(result["results"]) == 5

    print(f"\nMODE: {mode}")

    for item in result["results"]:
        print(
            item["rank"],
            item["response_uid"],
            f"{item['retrieval_score']:.6f}",
            item["response_text"][:80],
        )

print("\n========== BATCH QUERY TEST ==========")

batch_results = retriever.search(
    [
        "我最近睡眠不好。",
        "我感觉自己没有信心。",
        "我和家人之间经常发生矛盾。",
    ],
    top_k=5,
    mode="orca",
)

assert len(batch_results) == 3

for result in batch_results:
    assert len(result["results"]) == 5

print("BATCH_SEARCH=PASS")

print("\n========== FINAL ASSERTIONS ==========")

assert retriever.dense_retriever.index.ntotal == 61501
assert retriever.orca_retriever.index.ntotal == 61501

assert (
    retriever.dense_retriever.document_ids
    == retriever.orca_retriever.document_ids
)

print("DENSE_INDEX=PASS")
print("ORCA_INDEX=PASS")
print("BASIS_ALIGNMENT=PASS")
print("UID_ALIGNMENT=PASS")
print("SINGLE_QUERY=PASS")
print("BATCH_QUERY=PASS")
print("UNIFIED_RETRIEVER=PASS")
