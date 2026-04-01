# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP token store (omlx/mcp/token_store.py).
"""

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.mcp.token_store import TokenData, TokenStore, _keyring_available


class TestTokenData:
    """Tests for TokenData dataclass."""

    def test_basic_creation(self):
        token = TokenData(access_token="abc123")
        assert token.access_token == "abc123"
        assert token.token_type == "Bearer"
        assert token.refresh_token is None
        assert token.expires_at is None
        assert token.scope is None

    def test_not_expired_without_expiry(self):
        token = TokenData(access_token="abc", expires_at=None)
        assert token.is_expired is False

    def test_not_expired_future(self):
        token = TokenData(access_token="abc", expires_at=time.time() + 3600)
        assert token.is_expired is False

    def test_expired_past(self):
        token = TokenData(access_token="abc", expires_at=time.time() - 1)
        assert token.is_expired is True

    def test_expired_within_buffer(self):
        # Within 30-second safety margin → considered expired
        token = TokenData(access_token="abc", expires_at=time.time() + 20)
        assert token.is_expired is True

    def test_to_dict_roundtrip(self):
        token = TokenData(
            access_token="tok",
            token_type="Bearer",
            refresh_token="ref",
            expires_at=9999.0,
            scope="read write",
        )
        d = token.to_dict()
        restored = TokenData.from_dict(d)
        assert restored.access_token == "tok"
        assert restored.refresh_token == "ref"
        assert restored.expires_at == 9999.0
        assert restored.scope == "read write"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"access_token": "x", "unknown_field": "ignored"}
        token = TokenData.from_dict(d)
        assert token.access_token == "x"


class TestTokenStoreFileBacked:
    """Tests for file-backed token persistence."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TokenStore:
        """Return a TokenStore backed by a temp file (no keyring)."""
        store_path = str(tmp_path / "tokens.json")
        s = TokenStore(store_path=store_path)
        s._use_keyring = False
        return s

    def test_load_nonexistent(self, store: TokenStore):
        assert store.load("missing") is None

    def test_save_and_load(self, store: TokenStore):
        token = TokenData(access_token="my-token", refresh_token="refresh")
        store.save("my-server", token)
        loaded = store.load("my-server")
        assert loaded is not None
        assert loaded.access_token == "my-token"
        assert loaded.refresh_token == "refresh"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "tokens.json")
        store = TokenStore(store_path=deep_path)
        store._use_keyring = False
        store.save("srv", TokenData(access_token="t"))
        assert Path(deep_path).exists()

    def test_file_permissions(self, store: TokenStore, tmp_path: Path):
        store.save("srv", TokenData(access_token="t"))
        mode = stat.S_IMODE(os.stat(store._store_path).st_mode)
        assert mode == 0o600

    def test_delete_token(self, store: TokenStore):
        store.save("srv", TokenData(access_token="t"))
        store.delete("srv")
        assert store.load("srv") is None

    def test_delete_nonexistent_is_silent(self, store: TokenStore):
        store.delete("nonexistent")  # should not raise

    def test_multiple_servers(self, store: TokenStore):
        store.save("alpha", TokenData(access_token="a"))
        store.save("beta", TokenData(access_token="b"))
        assert store.load("alpha").access_token == "a"  # type: ignore[union-attr]
        assert store.load("beta").access_token == "b"  # type: ignore[union-attr]

    def test_overwrite(self, store: TokenStore):
        store.save("srv", TokenData(access_token="old"))
        store.save("srv", TokenData(access_token="new"))
        assert store.load("srv").access_token == "new"  # type: ignore[union-attr]

    def test_delete_preserves_other_servers(self, store: TokenStore):
        store.save("a", TokenData(access_token="a"))
        store.save("b", TokenData(access_token="b"))
        store.delete("a")
        assert store.load("a") is None
        assert store.load("b") is not None


class TestTokenStoreKeyringBacked:
    """Tests for keyring-backed token storage."""

    @pytest.fixture
    def store_with_keyring(self, tmp_path: Path):
        """Return a store with a mocked keyring."""
        store = TokenStore(store_path=str(tmp_path / "fallback.json"))
        store._use_keyring = True
        return store

    def test_save_uses_keyring(self, store_with_keyring: TokenStore):
        kr = MagicMock()
        kr.set_password = MagicMock()
        with patch.dict("sys.modules", {"keyring": kr}):
            store_with_keyring.save("srv", TokenData(access_token="t"))
            kr.set_password.assert_called_once()

    def test_load_from_keyring(self, store_with_keyring: TokenStore):
        token = TokenData(access_token="keychain-token")
        kr = MagicMock()
        kr.get_password.return_value = json.dumps(token.to_dict())
        with patch.dict("sys.modules", {"keyring": kr}):
            loaded = store_with_keyring.load("srv")
        assert loaded is not None
        assert loaded.access_token == "keychain-token"

    def test_load_returns_none_when_empty(self, store_with_keyring: TokenStore):
        kr = MagicMock()
        kr.get_password.return_value = None
        with patch.dict("sys.modules", {"keyring": kr}):
            loaded = store_with_keyring.load("srv")
        assert loaded is None

    def test_keyring_save_failure_falls_back_to_file(
        self, store_with_keyring: TokenStore
    ):
        kr = MagicMock()
        kr.set_password.side_effect = Exception("keyring unavailable")
        with patch.dict("sys.modules", {"keyring": kr}):
            # Should not raise; falls back to file store
            store_with_keyring.save("srv", TokenData(access_token="fallback"))
        # File-based fallback should have persisted the token
        store_with_keyring._use_keyring = False
        loaded = store_with_keyring.load("srv")
        assert loaded is not None
        assert loaded.access_token == "fallback"
