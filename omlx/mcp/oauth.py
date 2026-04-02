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
    - Dynamic Client Registration (RFC 7591 / RFC 8414) when the server
      advertises a ``registration_endpoint`` and no ``client_id`` is
      configured (e.g. Notion's remote MCP server).
    - Token revocation / logout.
    - Non-sensitive status reporting (never exposes raw token values).
    """

    def __init__(self, token_store: Optional[TokenStore] = None):
        self._store = token_store or TokenStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_access_token(
        self,
        server_name: str,
        auth_config: MCPAuthConfig,
        server_url: Optional[str] = None,
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
                    token = await self._do_refresh(
                        auth_config, token.refresh_token, stored_token=token
                    )
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
        server_url: Optional[str] = None,
    ) -> TokenData:
        """
        Perform a full interactive OAuth login.

        Args:
            server_name: MCP server name (used as the token store key).
            auth_config: OAuth configuration.
            flow: ``"pkce"`` (default) or ``"device"``.
            server_url: MCP endpoint URL.  Required for servers that use
                Dynamic Client Registration (i.e. when ``auth_config`` has
                no ``client_id``).

        Returns:
            The acquired :class:`~omlx.mcp.token_store.TokenData`.
        """
        if flow == "device":
            token = await self._device_code_flow(server_name, auth_config, server_url)
        else:
            token = await self._pkce_flow(server_name, auth_config, server_url)

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
        self,
        server_name: str,
        auth_config: MCPAuthConfig,
        server_url: Optional[str] = None,
    ) -> TokenData:
        """Run the Authorization Code + PKCE flow.

        When *auth_config* has no ``client_id``, ``auth_url``, or
        ``token_url``, the method first performs OAuth 2.0 Authorization
        Server Metadata discovery (RFC 8414) and then Dynamic Client
        Registration (RFC 7591) to obtain a ``client_id`` automatically.
        """
        auth_url = auth_config.auth_url
        token_url = auth_config.token_url
        client_id = auth_config.client_id

        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(16)

        port, code_queue = _start_callback_server()
        redirect_uri = f"http://localhost:{port}/callback"

        # -- DCR / metadata discovery -----------------------------------
        registered_client_id: Optional[str] = None
        if not auth_url or not token_url or not client_id:
            if not server_url:
                raise OAuthError(
                    f"MCP server '{server_name}' has no OAuth URLs configured "
                    "and no server_url was provided for auto-discovery. "
                    "Either set auth_url/token_url/client_id explicitly, or "
                    "ensure the server URL is provided."
                )
            logger.info(
                "No full OAuth config for '%s'; discovering via RFC 8414…",
                server_name,
            )
            metadata = await _discover_oauth_metadata(server_url)
            if not auth_url:
                auth_url = metadata.get("authorization_endpoint", "")
            if not token_url:
                token_url = metadata.get("token_endpoint", "")
            if not client_id:
                reg_endpoint = metadata.get("registration_endpoint")
                if not reg_endpoint:
                    raise OAuthError(
                        f"OAuth server for '{server_name}' does not advertise a "
                        "'registration_endpoint'. Set 'client_id' explicitly in "
                        "the server's auth config."
                    )
                logger.info("Performing Dynamic Client Registration for '%s'…", server_name)
                client_id = await _register_dynamic_client(reg_endpoint, redirect_uri)
                registered_client_id = client_id
        # ---------------------------------------------------------------

        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if auth_config.scopes:
            params["scope"] = " ".join(auth_config.scopes)
        if auth_config.audience:
            params["audience"] = auth_config.audience

        full_auth_url = f"{auth_url}?{urlencode(params)}"
        print(f"\nOpening browser for MCP authentication ({server_name})…")
        print(f"If the browser does not open, visit:\n  {full_auth_url}\n")
        webbrowser.open(full_auth_url)

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

        # Build a temporary auth_config with the resolved URLs/client_id so
        # that _exchange_code can use them.
        resolved_config = MCPAuthConfig(
            type=auth_config.type,
            client_id=client_id,
            auth_url=auth_url,
            token_url=token_url,
            scopes=auth_config.scopes,
            audience=auth_config.audience,
        )
        token = await self._exchange_code(
            resolved_config, code, redirect_uri, code_verifier
        )
        # Persist the dynamically registered client_id alongside the token
        # so it can be reused for token refresh without re-registering.
        if registered_client_id:
            token.registered_client_id = registered_client_id
        return token

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
        self,
        server_name: str,
        auth_config: MCPAuthConfig,
        server_url: Optional[str] = None,
    ) -> TokenData:
        """Run the Device Authorization Grant flow (RFC 8628)."""
        client_id = auth_config.client_id
        token_url = auth_config.token_url

        if not client_id or not token_url:
            if not server_url:
                raise OAuthError(
                    f"MCP server '{server_name}' has no OAuth URLs configured "
                    "and no server_url was provided for auto-discovery."
                )
            metadata = await _discover_oauth_metadata(server_url)
            if not token_url:
                token_url = metadata.get("token_endpoint", "")
            if not client_id:
                reg_endpoint = metadata.get("registration_endpoint")
                if not reg_endpoint:
                    raise OAuthError(
                        f"OAuth server for '{server_name}' does not advertise a "
                        "'registration_endpoint'. Set 'client_id' explicitly."
                    )
                client_id = await _register_dynamic_client(
                    reg_endpoint, redirect_uri="urn:ietf:wg:oauth:2.0:oob"
                )

        device_auth_url = auth_config.device_auth_url or _infer_device_auth_url(
            auth_config.auth_url or token_url
        )

        payload: Dict[str, str] = {"client_id": client_id}
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
                    "client_id": client_id,
                    "device_code": device_code,
                }
                resp = await client.post(
                    token_url,
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
        self,
        auth_config: MCPAuthConfig,
        refresh_token: str,
        stored_token: Optional[TokenData] = None,
    ) -> TokenData:
        """Refresh an access token using *refresh_token*.

        When the server used Dynamic Client Registration, the ``client_id``
        is taken from *stored_token.registered_client_id* if
        ``auth_config.client_id`` is empty.
        """
        client_id = auth_config.client_id
        if not client_id and stored_token and stored_token.registered_client_id:
            client_id = stored_token.registered_client_id

        if not client_id:
            raise OAuthError(
                "Cannot refresh token: no client_id available. "
                "Re-authenticate with 'omlx mcp login'."
            )

        payload: Dict[str, str] = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        }
        if auth_config.scopes:
            payload["scope"] = " ".join(auth_config.scopes)

        token_url = auth_config.token_url
        if not token_url and stored_token and stored_token.token_url:
            token_url = stored_token.token_url
        if not token_url:
            raise OAuthError(
                "Cannot refresh token: no token_url available. "
                "Re-authenticate with 'omlx mcp login'."
            )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            resp.raise_for_status()
            token = _parse_token_response(resp.json())

        # Preserve the refresh token if the server did not return a new one
        if not token.refresh_token:
            token.refresh_token = refresh_token
        # Propagate the registered client_id so it is persisted on re-save
        if stored_token and stored_token.registered_client_id:
            token.registered_client_id = stored_token.registered_client_id
        # Propagate the discovered token_url so refresh keeps working
        if stored_token and stored_token.token_url:
            token.token_url = stored_token.token_url
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


# ------------------------------------------------------------------
# OAuth 2.0 Authorization Server Metadata discovery (RFC 8414)
# and Dynamic Client Registration (RFC 7591)
# ------------------------------------------------------------------

def _get_discovery_urls(server_url: str) -> list:
    """
    Return candidate OAuth AS metadata discovery URLs for *server_url*.

    Per RFC 8414 §3, when the server URL has a path component the
    discovery document may live at either:
    ``{scheme}://{authority}/.well-known/oauth-authorization-server{path}``
    (preferred) or ``{scheme}://{authority}/.well-known/oauth-authorization-server``.
    OpenID Connect servers additionally publish at
    ``{scheme}://{authority}/.well-known/openid-configuration``.
    """
    parsed = urlparse(server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    candidates = []
    if path and path != "/":
        candidates.append(f"{base}/.well-known/oauth-authorization-server{path}")
    candidates.append(f"{base}/.well-known/oauth-authorization-server")
    candidates.append(f"{base}/.well-known/openid-configuration")
    return candidates


async def _discover_oauth_metadata(server_url: str) -> Dict[str, Any]:
    """
    Fetch OAuth 2.0 Authorization Server Metadata (RFC 8414).

    Tries discovery URLs in order and returns the first successful response.

    Raises:
        OAuthError: When no discovery document can be retrieved.
    """
    urls = _get_discovery_urls(server_url)
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient() as client:
        for url in urls:
            try:
                resp = await client.get(url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    logger.debug("Discovered OAuth metadata at %s", url)
                    return data
            except Exception as exc:
                last_exc = exc
                logger.debug("Discovery failed at %s: %s", url, exc)
    raise OAuthError(
        f"Could not discover OAuth metadata for '{server_url}'. "
        f"Tried: {', '.join(urls)}. "
        "Set auth_url/token_url/client_id explicitly in the server's auth config."
        + (f" Last error: {last_exc}" if last_exc else "")
    )


async def _register_dynamic_client(
    registration_endpoint: str,
    redirect_uri: str,
    client_name: str = "omlx",
) -> str:
    """
    Register a new OAuth 2.0 client via Dynamic Client Registration (RFC 7591).

    Returns the ``client_id`` assigned by the authorization server.

    Raises:
        OAuthError: When registration fails.
    """
    body: Dict[str, Any] = {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",  # public client — no client secret
        "code_challenge_methods_supported": ["S256"],
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            registration_endpoint,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        if not resp.is_success:
            raise OAuthError(
                f"Dynamic Client Registration failed "
                f"(HTTP {resp.status_code}): {resp.text[:200]}"
            )
        data = resp.json()

    client_id = data.get("client_id")
    if not client_id:
        raise OAuthError(
            "Dynamic Client Registration response did not include 'client_id'"
        )
    logger.info("Dynamic Client Registration successful; client_id=%s", client_id)
    return client_id
