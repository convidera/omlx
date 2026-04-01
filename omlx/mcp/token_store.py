# SPDX-License-Identifier: Apache-2.0
"""
Secure token storage for MCP OAuth 2.0 authentication.

Preference order:
1. OS keychain via ``keyring`` package (when installed).
2. Local JSON file at ``~/.config/omlx/mcp_tokens.json`` with mode 0o600.

Tokens are stored per MCP server name.  Sensitive fields (access_token,
refresh_token) are never logged.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_KEYRING_SERVICE = "omlx-mcp"
_DEFAULT_STORE_PATH = Path("~/.config/omlx/mcp_tokens.json")


@dataclass
class TokenData:
    """OAuth 2.0 token data for a single MCP server."""

    access_token: str
    token_type: str = "Bearer"
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None  # Unix timestamp
    scope: Optional[str] = None
    # Persisted client_id obtained via Dynamic Client Registration (DCR).
    # Set when the server uses RFC 7591 auto-registration (e.g. Notion remote MCP).
    registered_client_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Return True if the access token is expired (with 30-second safety margin)."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at - 30

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (safe to store)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenData":
        """Deserialize from a plain dictionary."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class TokenStore:
    """
    Persistent storage for OAuth tokens.

    Attempts to use the OS keychain (via ``keyring``) and falls back to a
    local JSON file when ``keyring`` is unavailable or fails.
    """

    def __init__(self, store_path: Optional[str] = None):
        """
        Args:
            store_path: Override the default file-based token store path.
        """
        self._store_path = (
            Path(store_path).expanduser()
            if store_path
            else _DEFAULT_STORE_PATH.expanduser()
        )
        self._use_keyring = _keyring_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, server_name: str) -> Optional[TokenData]:
        """Load stored tokens for *server_name*. Returns None if not found."""
        if self._use_keyring:
            token = self._load_keyring(server_name)
            if token is not None:
                return token
        return self._load_file(server_name)

    def save(self, server_name: str, token: TokenData) -> None:
        """Persist *token* for *server_name*."""
        if self._use_keyring:
            self._save_keyring(server_name, token)
        else:
            self._save_file(server_name, token)

    def delete(self, server_name: str) -> None:
        """Remove stored tokens for *server_name*."""
        if self._use_keyring:
            self._delete_keyring(server_name)
        self._delete_file(server_name)

    # ------------------------------------------------------------------
    # Keyring backend
    # ------------------------------------------------------------------

    def _load_keyring(self, server_name: str) -> Optional[TokenData]:
        try:
            import keyring  # type: ignore[import-untyped]
            raw = keyring.get_password(_KEYRING_SERVICE, server_name)
            if raw:
                return TokenData.from_dict(json.loads(raw))
        except Exception as exc:
            logger.debug("Keyring load failed for '%s': %s", server_name, exc)
        return None

    def _save_keyring(self, server_name: str, token: TokenData) -> None:
        try:
            import keyring  # type: ignore[import-untyped]
            keyring.set_password(
                _KEYRING_SERVICE, server_name, json.dumps(token.to_dict())
            )
            return
        except Exception as exc:
            logger.warning(
                "Keyring save failed for '%s': %s — falling back to file store",
                server_name,
                exc,
            )
        self._save_file(server_name, token)

    def _delete_keyring(self, server_name: str) -> None:
        try:
            import keyring  # type: ignore[import-untyped]
            keyring.delete_password(_KEYRING_SERVICE, server_name)
        except Exception as exc:
            logger.debug("Keyring delete failed for '%s': %s", server_name, exc)

    # ------------------------------------------------------------------
    # File-based backend
    # ------------------------------------------------------------------

    def _load_file(self, server_name: str) -> Optional[TokenData]:
        try:
            if not self._store_path.exists():
                return None
            with open(self._store_path) as fh:
                data: Dict[str, Any] = json.load(fh)
            entry = data.get(server_name)
            if entry:
                return TokenData.from_dict(entry)
        except Exception as exc:
            logger.debug("File token load failed for '%s': %s", server_name, exc)
        return None

    def _save_file(self, server_name: str, token: TokenData) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data: Dict[str, Any] = {}
            if self._store_path.exists():
                with open(self._store_path) as fh:
                    data = json.load(fh)
            data[server_name] = token.to_dict()
            with open(self._store_path, "w") as fh:
                json.dump(data, fh, indent=2)
            os.chmod(self._store_path, 0o600)
        except Exception as exc:
            logger.error("File token save failed for '%s': %s", server_name, exc)

    def _delete_file(self, server_name: str) -> None:
        try:
            if not self._store_path.exists():
                return
            with open(self._store_path) as fh:
                data: Dict[str, Any] = json.load(fh)
            if server_name in data:
                del data[server_name]
                with open(self._store_path, "w") as fh:
                    json.dump(data, fh, indent=2)
                os.chmod(self._store_path, 0o600)
        except Exception as exc:
            logger.debug("File token delete failed for '%s': %s", server_name, exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _keyring_available() -> bool:
    """Return True when the ``keyring`` package is importable."""
    try:
        import keyring  # noqa: F401  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False
