from __future__ import annotations

import time

import numpy as np

from ragnav.cache import EmbeddingCache as EmbeddingCacheFromPkg
from ragnav.cache.sqlite_cache import (
    EmbeddingCache,
    RetrievalCache,
    SqliteCacheConfig,
    SqliteKV,
)


def test_cache_package_init_exports_embedding_cache():
    assert EmbeddingCacheFromPkg is EmbeddingCache


def test_sqlite_kv_get_set_delete(tmp_path):
    db = tmp_path / "kv.sqlite3"
    kv = SqliteKV(SqliteCacheConfig(db_path=str(db)))
    assert kv.get("k") is None
    kv.set("k", b"hello")
    assert kv.get("k") == b"hello"
    kv.delete("k")
    assert kv.get("k") is None


def test_embedding_cache_roundtrip(tmp_path):
    db = tmp_path / "e.sqlite3"
    kv = SqliteKV(SqliteCacheConfig(db_path=str(db)))
    ec = EmbeddingCache(kv)
    texts = ["alpha", "beta"]
    vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ec.set_many(model="m", texts=texts, embeddings=vecs)
    got = ec.get_many(model="m", texts=texts)
    assert set(got.keys()) == {0, 1}
    np.testing.assert_allclose(got[0], vecs[0], rtol=1e-5)
    np.testing.assert_allclose(got[1], vecs[1], rtol=1e-5)


def test_retrieval_cache_miss(tmp_path):
    db = tmp_path / "r.sqlite3"
    kv = SqliteKV(SqliteCacheConfig(db_path=str(db)))
    rc = RetrievalCache(kv)
    assert rc.get({"query": "nope", "k": 1}) is None


def test_ttl_expiry(tmp_path):
    db = tmp_path / "ttl.sqlite3"
    kv = SqliteKV(SqliteCacheConfig(db_path=str(db), ttl_seconds=0))
    kv.set("x", b"1")
    time.sleep(0.02)
    assert kv.get("x") is None
