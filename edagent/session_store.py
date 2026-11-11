"""Dataset memory and session storage for EDAgent.

Remembers the latest dataset and EDA analysis per user or room to support
the follow-up 'Ask' feature.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Any
import pandas as pd


class SessionData:
    def __init__(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_col: Optional[str],
        result_summary: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ):
        self.raw_df = raw_df
        self.cleaned_df = cleaned_df
        self.target_col = target_col
        self.result_summary = result_summary or {}
        self.filename = filename or "dataset.csv"
        self.updated_at = time.time()


class SessionStore:
    def __init__(self):
        # In-memory dictionary indexed by session key
        self._store: Dict[str, SessionData] = {}

    @staticmethod
    def make_key(room_id: Optional[str], user_id: Optional[str]) -> str:
        """Prefers room_id (shared room memory), falls back to user_id."""
        if room_id and room_id.strip():
            return f"room:{room_id.strip()}"
        if user_id and user_id.strip():
            return f"user:{user_id.strip()}"
        return "global:default"

    def set(
        self,
        key: str,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_col: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> None:
        self._store[key] = SessionData(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            target_col=target_col,
            result_summary=result_summary,
            filename=filename
        )

    def get(self, key: str) -> Optional[SessionData]:
        return self._store.get(key)

    def has(self, key: str) -> bool:
        return key in self._store

    def clear(self, key: str) -> None:
        self._store.pop(key, None)


# Global singleton store
_global_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _global_session_store
