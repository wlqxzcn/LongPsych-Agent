from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional


class LongPsychTool:
    """
    Agent-facing tool for LongPsych psychological dialogue retrieval.

    The tool delegates all retrieval operations to LongPsychRetriever.
    It does not reimplement encoding, projection, or FAISS search.
    """

    name = "longpsych_search"

    description = (
        "Retrieve relevant psychological dialogue responses "
        "using Dense BGE or ORCA k=2 retrieval."
    )

    def __init__(self, retriever: Any):
        if not callable(getattr(retriever, "search", None)):
            raise TypeError(
                "retriever must provide a callable search method."
            )

        self.retriever = retriever

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        index_dir: str | Path,
        default_mode: str = "orca",
        device: Optional[str] = None,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> "LongPsychTool":

        from ..retrieval.unified_retriever import LongPsychRetriever

        retriever = LongPsychRetriever.from_pretrained(
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            default_mode=default_mode,
            device=device,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        return cls(retriever)

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query must not be empty.")

        return query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        return top_k

    def search(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: int = 5,
    ):

        query = self._validate_query(query)
        top_k = self._validate_top_k(top_k)

        kwargs = {
            "query": query,
            "top_k": top_k,
        }

        if mode is not None:
            kwargs["mode"] = mode

        return self.retriever.search(**kwargs)

    def batch_search(
        self,
        queries: Sequence[str],
        mode: Optional[str] = None,
        top_k: int = 5,
    ):

        if isinstance(queries, (str, bytes)):
            raise TypeError(
                "queries must be a sequence of strings, not a string."
            )

        if not isinstance(queries, Sequence):
            raise TypeError("queries must be a sequence.")

        if len(queries) == 0:
            raise ValueError("queries must not be empty.")

        queries = [
            self._validate_query(q)
            for q in queries
        ]

        top_k = self._validate_top_k(top_k)

        kwargs = {
            "query": queries,
            "top_k": top_k,
        }

        if mode is not None:
            kwargs["mode"] = mode

        return self.retriever.search(**kwargs)

    def __call__(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: int = 5,
    ):
        return self.search(
            query=query,
            mode=mode,
            top_k=top_k,
        )

    @property
    def tool_schema(self):

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Current psychological dialogue "
                                "or user utterance."
                            ),
                        },
                        "mode": {
                            "type": "string",
                            "enum": [
                                "dense",
                                "dense_bge",
                                "orca",
                                "orca_k2",
                            ],
                            "description": (
                                "Retrieval mode. If omitted, "
                                "use the retriever's configured default."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
