from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from ..exceptions import RAGNavCacheError


@dataclass
class SqliteCacheConfig:
    db_path: str
    ttl_seconds: Optional[int] = None


class SqliteKV:
    """SQLite-backed key/value store with optional TTL enforced at read time."""

    def __init__(self, config: SqliteCacheConfig) -> None:
        self._config = config
        parent = os.path.dirname(os.path.abspath(config.db_path))
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                raise RAGNavCacheError(
                    f"Failed to create cache directory for {config.db_path!r}.\n"
                    f"Create the parent directory manually or choose a path under an existing folder."
                ) from e
        try:
            self._conn = sqlite3.connect(config.db_path)
        except sqlite3.Error as e:
            raise RAGNavCacheError(
                f"Failed to open cache database at {config.db_path!r}.\n"
                f"Check that the directory exists and you have write permission."
            ) from e
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, stored_at REAL NOT NULL, value BLOB NOT NULL)"
        )
        self._conn.commit()

    def _expired(self, stored_at: float) -> bool:
        ttl = self._config.ttl_seconds
        if ttl is None:
            return False
        return (time.time() - stored_at) > ttl

    def get(self, key: str) -> Optional[bytes]:
        try:
            row = self._conn.execute(
                "SELECT stored_at, value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error as e:
            raise RAGNavCacheError(
                f"Cache read failed for key {key!r} in {self._config.db_path!r}.\n"
                f"Try removing the cache file if it may be corrupted."
            ) from e
        if row is None:
            return None
        stored_at, value = row
        if self._expired(float(stored_at)):
            self.delete(key)
            return None
        return bytes(value)

    def set(self, key: str, value: bytes) -> None:
        now = time.time()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, stored_at, value) VALUES (?, ?, ?)",
                (key, now, sqlite3.Binary(value)),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise RAGNavCacheError(
                f"Cache write failed for key {key!r} in {self._config.db_path!r}.\n"
                f"Check disk space and write permission."
            ) from e

    def delete(self, key: str) -> None:
        try:
            self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.commit()
        except sqlite3.Error as e:
            raise RAGNavCacheError(
                f"Cache delete failed for key {key!r} in {self._config.db_path!r}."
            ) from e


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, kv: SqliteKV) -> None:
        self._kv = kv

    def _key(self, model: Optional[str], text: str) -> str:
        m = model or "default"
        return f"emb:{m}:{_text_fingerprint(text)}"

    def get_many(self, *, model: Optional[str], texts: list[str]) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for i, t in enumerate(texts):
            raw = self._kv.get(self._key(model, t))
            if raw is None:
                continue
            try:
                meta_len = int.from_bytes(raw[:4], "big")
                meta_json = raw[4 : 4 + meta_len].decode("utf-8")
                blob = raw[4 + meta_len :]
                meta = json.loads(meta_json)
                shape = tuple(meta["shape"])
                dtype = np.dtype(meta["dtype"])
                arr = np.frombuffer(blob, dtype=dtype).reshape(shape)
                out[i] = arr.astype(np.float64).tolist()
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                raise RAGNavCacheError(
                    "Cached embedding payload is unreadable; delete the cache entry or remove the cache DB.\n"
                    f"Problematic key pattern: emb:{model or 'default'}:<hash>."
                ) from e
        return out

    def set_many(
        self,
        *,
        model: Optional[str],
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(texts) != len(embeddings):
            raise RAGNavCacheError(
                f"Embedding cache set_many: got {len(texts)} texts and {len(embeddings)} embeddings; "
                f"counts must match."
            )
        for t, emb in zip(texts, embeddings):
            arr = np.asarray(emb, dtype=np.float32)
            meta = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}).encode("utf-8")
            payload = len(meta).to_bytes(4, "big") + meta + arr.tobytes()
            self._kv.set(self._key(model, t), payload)


class RetrievalCache:
    def __init__(self, kv: SqliteKV) -> None:
        self._kv = kv

    @staticmethod
    def _payload_key(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ret:{h}"

    def get(self, payload: dict[str, Any]) -> Optional[list[str]]:
        raw = self._kv.get(self._payload_key(payload))
        if raw is None:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, list):
                return None
            return [str(x) for x in data]
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RAGNavCacheError(
                "Cached retrieval payload is not valid JSON; delete the cache file or turn off retrieval caching."
            ) from e

    def set(self, payload: dict[str, Any], block_ids: Iterable[str]) -> None:
        body = json.dumps(list(block_ids), separators=(",", ":")).encode("utf-8")
        self._kv.set(self._payload_key(payload), body)
