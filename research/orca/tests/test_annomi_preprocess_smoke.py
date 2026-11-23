#!/usr/bin/env python3
from pathlib import Path
import sys
import tempfile
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_annomi_fivefold import load_and_preprocess, make_stratified_folds

rows = [
    [1, "high", 0, "client", "I want to change."],
    [1, "high", 0, "client", "I want to change."],  # duplicate annotation row
    [1, "high", 1, "client", "But it is hard."],
    [1, "high", 2, "therapist", "What might help?"],
    [2, "low", 0, "client", "I am unsure."],
    [2, "low", 1, "therapist", "Tell me more."],
]
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "annomi.csv"
    pd.DataFrame(rows, columns=["transcript_id", "mi_quality", "utterance_id", "interlocutor", "utterance_text"]).to_csv(path, index=False)
    turns, pairs, stats = load_and_preprocess(str(path))
    assert stats["deduplicated_utterances"] == 5
    assert len(pairs) == 2
    assert pairs[0].query_text == "I want to change. But it is hard."
    folds = make_stratified_folds(turns, 2, 42)
    assert set(folds) == {"1", "2"}
print("AnnoMI preprocessing smoke test passed")
