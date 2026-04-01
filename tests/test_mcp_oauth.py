# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP OAuth 2.0 support.

Covers:
- Config parsing (MCPAuthConfig)
- Token refresh path
- 401 → auth → retry path in MCPClient
- Failed refresh path
- OAuthManager helpers
"""

import json
import time
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.mcp.oauth import (
    MCPOAuthManager,
    OAuthError,
    _generate_code_challenge,
    _generate_code_verifier,
    _infer_device_auth_url,
    _parse_token_response,
)
from omlx.mcp.token_store import TokenData, TokenStore
from omlx.mcp.types import MCPAuthConfig, MCPServerConfig, MCPTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auth_config(**kwargs) -> MCPAuthConfig:
    defaults = dict(
        client_id="test-client",
        auth_url="https://example.com/oauth/authorize",
        token_url="https://example.com/oauth/token",
    )
    defaults.update(kwargs)
    return MCPAuthConfig(**defaults)


def _make_store_with_token(
    tmp_path: Path,
    server_name: str,
    access_token: str = "live-token",
    expires_at: Optional[float] = None,
    refresh_token: Optional[str] = None,
) -> TokenStore:
    store = TokenStore(store_path=str(tmp_path / "tokens.json"))
    store._use_keyring = False
    token = TokenData(
        access_token=access_token,
        expires_at=expires_at,
        refresh_token=refresh_token,
    )
    store.save(server_name, token)
    return store


# ---------------------------------------------------------------------------
# MCPAuthConfig validation
# ---------------------------------------------------------------------------

class TestMCPAuthConfig:
    def test_valid_config(self):
        cfg = _make_auth_config()
        assert cfg.client_id == "test-client"
        assert cfg.type == "oauth2"

    def test_requires_client_id(self):
        with pytest.raises(ValueError, match="client_id"):
            MCPAuthConfig(client_id="", token_url="https://example.com/token")

    def test_requires_token_url(self):
        with pytest.raises(ValueError, match="token_url"):
            MCPAuthConfig(client_id="id", token_url="")

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported auth type"):
            MCPAuthConfig(
                type="basic",
                client_id="id",
                token_url="https://example.com/token",
            )

    def test_optional_fields(self):
        cfg = _make_auth_config(
            scopes=["read", "write"],
            audience="https://api.example.com",
            device_auth_url="https://example.com/device",
        )
        assert cfg.scopes == ["read", "write"]
        assert cfg.audience == "https://api.example.com"
        assert cfg.device_auth_url == "https://example.com/device"


# ---------------------------------------------------------------------------
# Config parsing (omlx.mcp.config.validate_config)
# ---------------------------------------------------------------------------

class TestAuthConfigParsing:
    def test_server_without_auth_parses_fine(self):
        from omlx.mcp.config import validate_config

        data = {
            "servers": {
                "plain": {
                    "transport": "sse",
                    "url": "http://localhost:3001/sse",
                }
            }
        }
        cfg = validate_config(data)
        assert cfg.servers["plain"].auth is None

    def test_server_with_auth_block(self):
        from omlx.mcp.config import validate_config

        data = {
            "servers": {
                "notion": {
                    "transport": "streamable-http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": {
                        "type": "oauth2",
                        "client_id": "notion-client",
                        "auth_url": "https://api.notion.com/v1/oauth/authorize",
                        "token_url": "https://api.notion.com/v1/oauth/token",
                        "scopes": ["read_content"],
                    },
                }
            }
        }
        cfg = validate_config(data)
        auth = cfg.servers["notion"].auth
        assert auth is not None
        assert auth.client_id == "notion-client"
        assert auth.scopes == ["read_content"]

    def test_invalid_auth_block_raises(self):
        from omlx.mcp.config import validate_config

        data = {
            "servers": {
                "bad": {
                    "transport": "sse",
                    "url": "http://localhost/sse",
                    "auth": "not-a-dict",
                }
            }
        }
        with pytest.raises(ValueError, match="'auth' must be a dictionary"):
            validate_config(data)

    def test_missing_client_id_in_auth_raises(self):
        from omlx.mcp.config import validate_config

        data = {
            "servers": {
                "bad": {
                    "transport": "sse",
                    "url": "http://localhost/sse",
                    "auth": {
                        "type": "oauth2",
                        "client_id": "",
                        "token_url": "https://example.com/token",
                    },
                }
            }
        }
        with pytest.raises(ValueError):
            validate_config(data)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

class TestPKCEHelpers:
    def test_code_verifier_is_urlsafe_base64(self):
        verifier = _generate_code_verifier()
        # Should contain only URL-safe base64 chars
        import re
        assert re.match(r"^[A-Za-z0-9\-_]+$", verifier)
        assert len(verifier) >= 43  # RFC 7636 minimum

    def test_code_challenge_is_deterministic_s256(self):
        import base64
        import hashlib

        verifier = "test-verifier-abc123"
        expected = (
            base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        assert _generate_code_challenge(verifier) == expected

    def test_different_verifiers_give_different_challenges(self):
        v1 = _generate_code_verifier()
        v2 = _generate_code_verifier()
        assert v1 != v2
        assert _generate_code_challenge(v1) != _generate_code_challenge(v2)


# ---------------------------------------------------------------------------
# _infer_device_auth_url
# ---------------------------------------------------------------------------

class TestInferDeviceAuthUrl:
    def test_replaces_authorize(self):
        url = _infer_device_auth_url("https://example.com/oauth/authorize")
        assert url == "https://example.com/oauth/device/code"

    def test_appends_device_code(self):
        url = _infer_device_auth_url("https://example.com/oauth")
        assert url == "https://example.com/oauth/device/code"


# ---------------------------------------------------------------------------
# _parse_token_response
# ---------------------------------------------------------------------------

class TestParseTokenResponse:
    def test_parses_full_response(self):
        data = {
            "access_token": "at",
            "token_type": "Bearer",
            "refresh_token": "rt",
            "expires_in": 3600,
            "scope": "read write",
        }
        token = _parse_token_response(data)
        assert token.access_token == "at"
        assert token.refresh_token == "rt"
        assert token.scope == "read write"
        assert token.expires_at is not None
        assert token.expires_at > time.time() + 3500

    def test_parses_minimal_response(self):
        token = _parse_token_response({"access_token": "x"})
        assert token.access_token == "x"
        assert token.expires_at is None
        assert token.refresh_token is None


# ---------------------------------------------------------------------------
# MCPOAuthManager.get_access_token — happy path and refresh paths
# ---------------------------------------------------------------------------

class TestOAuthManagerGetToken:
    @pytest.fixture
    def auth_config(self) -> MCPAuthConfig:
        return _make_auth_config()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_token_stored(
        self, tmp_path: Path, auth_config: MCPAuthConfig
    ):
        store = TokenStore(store_path=str(tmp_path / "t.json"))
        store._use_keyring = False
        manager = MCPOAuthManager(token_store=store)
        result = await manager.get_access_token("srv", auth_config)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_valid_token(self, tmp_path: Path, auth_config: MCPAuthConfig):
        store = _make_store_with_token(
            tmp_path, "srv", expires_at=time.time() + 3600
        )
        manager = MCPOAuthManager(token_store=store)
        result = await manager.get_access_token("srv", auth_config)
        assert result == "live-token"

    @pytest.mark.asyncio
    async def test_refreshes_expired_token(
        self, tmp_path: Path, auth_config: MCPAuthConfig
    ):
        store = _make_store_with_token(
            tmp_path,
            "srv",
            access_token="old-token",
            expires_at=time.time() - 10,  # expired
            refresh_token="refresh-xyz",
        )
        manager = MCPOAuthManager(token_store=store)

        new_token_data = TokenData(
            access_token="new-token",
            refresh_token="refresh-xyz",
            expires_at=time.time() + 3600,
        )

        with patch.object(
            manager, "_do_refresh", new=AsyncMock(return_value=new_token_data)
        ):
            result = await manager.get_access_token("srv", auth_config)

        assert result == "new-token"
        # Ensure the new token was persisted
        stored = store.load("srv")
        assert stored is not None
        assert stored.access_token == "new-token"

    @pytest.mark.asyncio
    async def test_returns_none_on_failed_refresh(
        self, tmp_path: Path, auth_config: MCPAuthConfig
    ):
        store = _make_store_with_token(
            tmp_path,
            "srv",
            access_token="stale",
            expires_at=time.time() - 10,
            refresh_token="bad-refresh",
        )
        manager = MCPOAuthManager(token_store=store)

        with patch.object(
            manager,
            "_do_refresh",
            new=AsyncMock(side_effect=Exception("refresh rejected")),
        ):
            result = await manager.get_access_token("srv", auth_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_expired_without_refresh_token(
        self, tmp_path: Path, auth_config: MCPAuthConfig
    ):
        store = _make_store_with_token(
            tmp_path,
            "srv",
            expires_at=time.time() - 10,
            refresh_token=None,
        )
        manager = MCPOAuthManager(token_store=store)
        result = await manager.get_access_token("srv", auth_config)
        assert result is None


# ---------------------------------------------------------------------------
# MCPOAuthManager.logout
# ---------------------------------------------------------------------------

class TestOAuthManagerLogout:
    def test_logout_removes_token(self, tmp_path: Path):
        store = _make_store_with_token(tmp_path, "srv")
        manager = MCPOAuthManager(token_store=store)
        assert store.load("srv") is not None
        manager.logout("srv")
        assert store.load("srv") is None

    def test_logout_nonexistent_is_silent(self, tmp_path: Path):
        store = TokenStore(store_path=str(tmp_path / "t.json"))
        store._use_keyring = False
        manager = MCPOAuthManager(token_store=store)
        manager.logout("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# MCPOAuthManager.get_token_info
# ---------------------------------------------------------------------------

class TestOAuthManagerGetTokenInfo:
    def test_returns_none_when_not_authenticated(self, tmp_path: Path):
        store = TokenStore(store_path=str(tmp_path / "t.json"))
        store._use_keyring = False
        manager = MCPOAuthManager(token_store=store)
        assert manager.get_token_info("srv") is None

    def test_returns_metadata_without_raw_tokens(self, tmp_path: Path):
        store = _make_store_with_token(
            tmp_path,
            "srv",
            access_token="secret-at",
            refresh_token="secret-rt",
            expires_at=time.time() + 3600,
        )
        manager = MCPOAuthManager(token_store=store)
        info = manager.get_token_info("srv")
        assert info is not None
        assert info["authenticated"] is True
        assert info["has_refresh_token"] is True
        assert info["is_expired"] is False
        # Raw tokens must NOT be present in the info dict
        assert "access_token" not in info
        assert "refresh_token" not in info

    def test_reflects_expired_state(self, tmp_path: Path):
        store = _make_store_with_token(
            tmp_path, "srv", expires_at=time.time() - 60
        )
        manager = MCPOAuthManager(token_store=store)
        info = manager.get_token_info("srv")
        assert info is not None
        assert info["is_expired"] is True


# ---------------------------------------------------------------------------
# MCPClient — 401 detection and retry
# ---------------------------------------------------------------------------

class TestMCPClientAuthRetry:
    """Tests for the 401 → acquire token → retry path in MCPClient."""

    def _make_oauth_config(self, tmp_path: Path) -> MCPServerConfig:
        auth = _make_auth_config()
        return MCPServerConfig(
            name="notion",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="https://mcp.notion.com/mcp",
            auth=auth,
        )

    @pytest.mark.asyncio
    async def test_connect_injects_token_header(self, tmp_path: Path):
        """Client should inject Bearer header when a valid stored token exists."""
        from omlx.mcp.client import MCPClient

        config = self._make_oauth_config(tmp_path)
        client = MCPClient(config)

        with (
            patch.object(
                client, "_get_oauth_token", new=AsyncMock(return_value="my-access-token")
            ),
            patch.object(client, "_do_connect", new=AsyncMock()),
        ):
            result = await client.connect()

        assert result is True
        assert client.config.headers is not None
        assert client.config.headers.get("Authorization") == "Bearer my-access-token"

    @pytest.mark.asyncio
    async def test_connect_retries_on_401(self, tmp_path: Path):
        """On 401 the client should refresh the token and retry."""
        from omlx.mcp.client import MCPClient

        config = self._make_oauth_config(tmp_path)
        client = MCPClient(config)

        call_count = {"n": 0}

        async def flaky_connect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("HTTP 401 Unauthorized")
            # Second call succeeds
            client._tools = []

        with (
            patch.object(
                client,
                "_get_oauth_token",
                new=AsyncMock(return_value="fresh-token"),
            ),
            patch.object(client, "_do_connect", side_effect=flaky_connect),
        ):
            result = await client.connect()

        assert result is True
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_connect_fails_when_no_token_after_401(self, tmp_path: Path):
        """If no token can be acquired after 401, connection should fail."""
        from omlx.mcp.client import MCPClient

        config = self._make_oauth_config(tmp_path)
        client = MCPClient(config)

        async def always_401():
            raise Exception("HTTP 401 Unauthorized")

        with (
            patch.object(
                client, "_get_oauth_token", new=AsyncMock(return_value=None)
            ),
            patch.object(client, "_do_connect", side_effect=always_401),
        ):
            result = await client.connect()

        assert result is False
        from omlx.mcp.types import MCPServerState
        assert client.state == MCPServerState.ERROR

    @pytest.mark.asyncio
    async def test_connect_non_auth_error_does_not_retry(self, tmp_path: Path):
        """Non-401 errors should not trigger a retry."""
        from omlx.mcp.client import MCPClient

        config = self._make_oauth_config(tmp_path)
        client = MCPClient(config)

        call_count = {"n": 0}

        async def network_error():
            call_count["n"] += 1
            raise ConnectionError("Network unreachable")

        with (
            patch.object(
                client, "_get_oauth_token", new=AsyncMock(return_value="tok")
            ),
            patch.object(client, "_do_connect", side_effect=network_error),
        ):
            result = await client.connect()

        assert result is False
        assert call_count["n"] == 1  # only tried once

    @pytest.mark.asyncio
    async def test_connect_without_auth_config_no_token_injection(self):
        """Servers without an auth config must not have headers injected."""
        from omlx.mcp.client import MCPClient

        config = MCPServerConfig(
            name="plain",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:3001/mcp",
            auth=None,
        )
        client = MCPClient(config)

        with patch.object(client, "_do_connect", new=AsyncMock()):
            result = await client.connect()

        assert result is True
        # No auth header injected
        assert not (client.config.headers or {}).get("Authorization")
