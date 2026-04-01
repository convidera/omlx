# SPDX-License-Identifier: Apache-2.0
"""
OAuth 2.0 flows for MCP server authentication.

Supported flows
---------------
- **Authorization Code + PKCE** (``flow="pkce"``) — opens a browser and
  receives the callback on a short-lived local HTTP server.  Preferred for
  desktop / local-app usage.
- **Device Authorization Grant** (``flow="device"``) — prints a user code and
  polls the token endpoint.  Suitable for headless / non-interactive
  environments.

The module is intentionally provider-agnostic: callers supply ``auth_url``,
``token_url``, and (optionally) ``device_auth_url`` via :class:`MCPAuthConfig`.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .token_store import TokenData, TokenStore
from .types import MCPAuthConfig

logger = logging.getLogger(__name__)


class OAuthError(Exception):
    """Raised when an OAuth flow cannot complete successfully."""


class MCPOAuthManager:
    """
    Manages the OAuth 2.0 authentication lifecycle for MCP servers.

    Provides:
    - Token retrieval (with automatic refresh on expiry).
    - First-time interactive login (PKCE or Device Auth Grant).
    - Token revocation / logout.
    - Non-sensitive status reporting (never exposes raw token values).
    """

    def __init__(self, token_store: Optional[TokenStore] = None):
        self._store = token_store or TokenStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_access_token(
        self, server_name: str, auth_config: MCPAuthConfig
    ) -> Optional[str]:
        """
        Return a valid access token, refreshing it if needed.

        Returns ``None`` when no token is stored yet (caller should call
        :meth:`login`).
        """
        token = self._store.load(server_name)
        if token is None:
            return None

        if token.is_expired:
            if token.refresh_token:
                logger.info("Token expired for '%s', attempting refresh…", server_name)
                try:
                    token = await self._do_refresh(auth_config, token.refresh_token)
                    self._store.save(server_name, token)
                    return token.access_token
                except Exception as exc:
                    logger.warning(
                        "Token refresh failed for '%s': %s", server_name, exc
                    )
                    return None
            logger.warning(
                "Token expired for '%s' and no refresh token available", server_name
            )
            return None

        return token.access_token

    async def login(
        self,
        server_name: str,
        auth_config: MCPAuthConfig,
        flow: str = "pkce",
    ) -> TokenData:
        """
        Perform a full interactive OAuth login.

        Args:
            server_name: MCP server name (used as the token store key).
            auth_config: OAuth configuration.
            flow: ``"pkce"`` (default) or ``"device"``.

        Returns:
            The acquired :class:`~omlx.mcp.token_store.TokenData`.
        """
        if flow == "device":
            token = await self._device_code_flow(server_name, auth_config)
        else:
            token = await self._pkce_flow(server_name, auth_config)

        self._store.save(server_name, token)
        logger.info("Successfully authenticated '%s'", server_name)
        return token

    def logout(self, server_name: str) -> None:
        """Remove stored tokens for *server_name*."""
        self._store.delete(server_name)
        logger.info("Logged out from '%s'", server_name)

    def get_token_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Return non-sensitive metadata about the stored token.

        Never includes raw token values.
        """
        token = self._store.load(server_name)
        if token is None:
            return None
        return {
            "authenticated": True,
            "token_type": token.token_type,
            "scope": token.scope,
            "expires_at": token.expires_at,
            "is_expired": token.is_expired,
            "has_refresh_token": token.refresh_token is not None,
        }

    # ------------------------------------------------------------------
    # Authorization Code + PKCE flow
    # ------------------------------------------------------------------

    async def _pkce_flow(
        self, server_name: str, auth_config: MCPAuthConfig
    ) -> TokenData:
        """Run the Authorization Code + PKCE flow."""
        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(16)

        port, code_queue = _start_callback_server()
        redirect_uri = f"http://localhost:{port}/callback"

        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": auth_config.client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if auth_config.scopes:
            params["scope"] = " ".join(auth_config.scopes)
        if auth_config.audience:
            params["audience"] = auth_config.audience

        auth_url = f"{auth_config.auth_url}?{urlencode(params)}"
        print(f"\nOpening browser for MCP authentication ({server_name})…")
        print(f"If the browser does not open, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        try:
            callback_data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, code_queue.get),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            raise OAuthError("Authentication timed out after 120 seconds")

        if "error" in callback_data:
            desc = callback_data.get("error_description", "")
            raise OAuthError(
                f"OAuth error: {callback_data['error']}"
                + (f" — {desc}" if desc else "")
            )

        if callback_data.get("state") != state:
            raise OAuthError("OAuth state mismatch (possible CSRF attack)")

        code = callback_data.get("code")
        if not code:
            raise OAuthError("No authorization code received in callback")

        return await self._exchange_code(
            auth_config, code, redirect_uri, code_verifier
        )

    async def _exchange_code(
        self,
        auth_config: MCPAuthConfig,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> TokenData:
        """Exchange an authorization code for tokens."""
        payload = {
            "grant_type": "authorization_code",
            "client_id": auth_config.client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                auth_config.token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            resp.raise_for_status()
            return _parse_token_response(resp.json())

    # ------------------------------------------------------------------
    # Device Authorization Grant flow
    # ------------------------------------------------------------------

    async def _device_code_flow(
        self, server_name: str, auth_config: MCPAuthConfig
    ) -> TokenData:
        """Run the Device Authorization Grant flow (RFC 8628)."""
        device_auth_url = auth_config.device_auth_url or _infer_device_auth_url(
            auth_config.auth_url
        )

        payload: Dict[str, str] = {"client_id": auth_config.client_id}
        if auth_config.scopes:
            payload["scope"] = " ".join(auth_config.scopes)
        if auth_config.audience:
            payload["audience"] = auth_config.audience

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                device_auth_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            resp.raise_for_status()
            device_data: Dict[str, Any] = resp.json()

        user_code = device_data.get("user_code")
        verification_uri = device_data.get("verification_uri") or device_data.get(
            "verification_url"
        )
        device_code = device_data["device_code"]
        interval: int = device_data.get("interval", 5)
        expires_in: int = device_data.get("expires_in", 300)

        print(f"\nDevice Authorization for '{server_name}'")
        print(f"  Visit:  {verification_uri}")
        print(f"  Code:   {user_code}")
        print()

        deadline = time.time() + expires_in
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                await asyncio.sleep(interval)
                poll_payload = {
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": auth_config.client_id,
                    "device_code": device_code,
                }
                resp = await client.post(
                    auth_config.token_url,
                    data=poll_payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30.0,
                )
                data: Dict[str, Any] = resp.json()
                error = data.get("error")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                if error:
                    desc = data.get("error_description", "")
                    raise OAuthError(
                        f"Device auth error: {error}"
                        + (f" — {desc}" if desc else "")
                    )
                resp.raise_for_status()
                return _parse_token_response(data)

        raise OAuthError("Device authorization code expired")

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    async def _do_refresh(
        self, auth_config: MCPAuthConfig, refresh_token: str
    ) -> TokenData:
        """Refresh an access token using *refresh_token*."""
        payload: Dict[str, str] = {
            "grant_type": "refresh_token",
            "client_id": auth_config.client_id,
            "refresh_token": refresh_token,
        }
        if auth_config.scopes:
            payload["scope"] = " ".join(auth_config.scopes)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                auth_config.token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            resp.raise_for_status()
            token = _parse_token_response(resp.json())

        # Preserve the refresh token if the server did not return a new one
        if not token.refresh_token:
            token.refresh_token = refresh_token
        return token


# ------------------------------------------------------------------
# PKCE helpers (RFC 7636)
# ------------------------------------------------------------------

def _generate_code_verifier() -> str:
    """Generate a cryptographically random PKCE code verifier."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def _generate_code_challenge(verifier: str) -> str:
    """Derive the S256 PKCE code challenge from *verifier*."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ------------------------------------------------------------------
# Local callback HTTP server
# ------------------------------------------------------------------

def _start_callback_server() -> Tuple[int, "Queue[Dict[str, str]]"]:
    """
    Bind a local HTTP server on a free port and return ``(port, queue)``.

    The server handles exactly one GET request, places the query parameters
    in the queue, and terminates.
    """
    code_queue: "Queue[Dict[str, str]]" = Queue()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            flat = {k: v[0] for k, v in params.items()}
            code_queue.put(flat)

            body = (
                b"<html><body>"
                b"<h2>Authentication complete - you may close this window.</h2>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # Suppress access-log noise

    server = HTTPServer(("localhost", 0), _CallbackHandler)
    port = server.server_address[1]
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    return port, code_queue


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_token_response(data: Dict[str, Any]) -> TokenData:
    """Build a :class:`~omlx.mcp.token_store.TokenData` from a token endpoint response."""
    expires_in = data.get("expires_in")
    expires_at = (time.time() + expires_in) if expires_in is not None else None
    return TokenData(
        access_token=data["access_token"],
        token_type=data.get("token_type", "Bearer"),
        refresh_token=data.get("refresh_token"),
        expires_at=expires_at,
        scope=data.get("scope"),
    )


def _infer_device_auth_url(auth_url: str) -> str:
    """
    Attempt to derive a device-authorization URL from *auth_url*.

    Replaces ``/authorize`` with ``/device/code`` when present;
    otherwise appends ``/device/code``.
    """
    parsed = urlparse(auth_url)
    path = parsed.path
    if "/authorize" in path:
        path = path.replace("/authorize", "/device/code")
    else:
        path = path.rstrip("/") + "/device/code"
    return parsed._replace(path=path).geturl()
