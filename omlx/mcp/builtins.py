"""
Built-in tools available without any MCP server configuration.

Currently provides:
  - builtin__fetch       : fetch a URL and return its text content
  - builtin__web_search  : DuckDuckGo web search
"""

import asyncio
import logging
from html.parser import HTMLParser
from typing import Any, Dict, List

from .types import MCPTool, MCPToolResult

logger = logging.getLogger(__name__)

_SERVER_NAME = "builtin"


# ---------------------------------------------------------------------------
# HTML → plain-text helper
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text converter using stdlib only."""

    _SKIP_TAGS = {"script", "style", "head", "noscript", "svg", "iframe"}

    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines
        lines = [ln.rstrip() for ln in raw.splitlines()]
        result, blank_run = [], 0
        for ln in lines:
            if ln:
                blank_run = 0
                result.append(ln)
            else:
                blank_run += 1
                if blank_run <= 2:
                    result.append("")
        return "\n".join(result).strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _sync_fetch(url: str, max_chars: int) -> str:
    """Blocking fetch — run via asyncio.to_thread."""
    import requests

    # Inject the macOS/system trust store so uv-managed Python builds
    # can verify TLS certificates via the system Keychain.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
    except requests.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    content_type = response.headers.get("content-type", "")
    text = _html_to_text(response.text) if "html" in content_type else response.text

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated — {len(text)} chars total]"
    return text


async def _fetch(url: str, max_chars: int = 20_000) -> str:
    """Fetch *url* and return its text content."""
    return await asyncio.to_thread(_sync_fetch, url, max_chars)


async def _web_search(query: str, max_results: int = 8) -> str:
    """Search DuckDuckGo and return formatted results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs is not installed. Run `uv add ddgs`."

    def _sync_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await asyncio.to_thread(_sync_search)

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}")
            lines.append(f"   {r.get('href', '')}")
            body = r.get("body", "").strip()
            if body:
                lines.append(f"   {body}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS: List[MCPTool] = [
    MCPTool(
        server_name=_SERVER_NAME,
        name="fetch",
        description=(
            "Fetch the content of a URL and return it as plain text. "
            "HTML pages are converted to readable text. "
            "Useful for reading documentation, articles, or any web page."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to return (default 20000).",
                    "default": 20000,
                },
            },
            "required": ["url"],
        },
    ),
    MCPTool(
        server_name=_SERVER_NAME,
        name="web_search",
        description=(
            "Search the web using DuckDuckGo and return a list of results "
            "with titles, URLs, and short descriptions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 8, max 20).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    ),
]


class BuiltinToolProvider:
    """Provides built-in tools that work without any MCP server."""

    def get_tools(self) -> List[MCPTool]:
        return list(_TOOLS)

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        if tool_name == "fetch":
            url = arguments.get("url", "")
            if not url:
                return MCPToolResult(
                    tool_name=tool_name,
                    content=None,
                    is_error=True,
                    error_message="Missing required argument: url",
                )
            max_chars = int(arguments.get("max_chars", 20_000))
            result = await _fetch(url, max_chars=max_chars)
            return MCPToolResult(tool_name=tool_name, content=result)

        if tool_name == "web_search":
            query = arguments.get("query", "")
            if not query:
                return MCPToolResult(
                    tool_name=tool_name,
                    content=None,
                    is_error=True,
                    error_message="Missing required argument: query",
                )
            max_results = min(int(arguments.get("max_results", 8)), 20)
            result = await _web_search(query, max_results=max_results)
            return MCPToolResult(tool_name=tool_name, content=result)

        return MCPToolResult(
            tool_name=tool_name,
            content=None,
            is_error=True,
            error_message=f"Unknown built-in tool: {tool_name}",
        )
