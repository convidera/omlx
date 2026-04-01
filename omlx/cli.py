#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models --max-model-memory 32GB

    # With pinned models
    omlx serve --model-dir /path/to/models --max-model-memory 48GB --pin llama-3b,qwen-7b
"""

import argparse
import faulthandler
import sys


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    if hasattr(args, "model_dir") and args.model_dir is not None:
        return True
    if hasattr(args, "port") and args.port is not None:
        return True
    if hasattr(args, "max_model_memory") and args.max_model_memory is not None:
        return True
    if hasattr(args, "max_process_memory") and args.max_process_memory is not None:
        return True
    if hasattr(args, "host") and args.host is not None:
        return True
    if hasattr(args, "log_level") and args.log_level is not None:
        return True
    return False


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from .settings import init_settings, get_settings
    from .logging_config import configure_file_logging, AdminStatsAccessFilter

    try:
        from ._build_info import build_number
    except ImportError:
        build_number = None

    # Print version banner
    print(f"\033[33moMLX - LLM inference, optimized for your Mac\033[0m")
    print(f"\033[33m├─ https://github.com/jundot/omlx\033[0m")
    if build_number:
        print(f"\033[33m├─ Version: {__version__}\033[0m")
        print(f"\033[33m└─ Build: {build_number}\033[0m")
    else:
        print(f"\033[33m└─ Version: {__version__}\033[0m")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set omlx loggers
    for name in ["omlx", "omlx.scheduler", "omlx.paged_ssd_cache",
                 "omlx.memory_monitor", "omlx.paged_cache", "omlx.prefix_cache",
                 "omlx.engine_pool", "omlx.model_discovery"]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive admin stats access logs
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Save CLI args to settings.json if non-default values provided
    if _has_cli_overrides(args):
        try:
            settings.save()
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Import server and config
    from .server import app, init_server
    from .config import parse_size

    model_dirs = settings.model.get_model_dirs(settings.base_path)
    print(f"Base path: {settings.base_path}")
    print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
    print(f"Max model memory: {settings.model.max_model_memory}")
    print(f"Max process memory: {settings.memory.max_process_memory}")

    # Store MCP config path for FastAPI startup
    # Priority: CLI arg > settings.json
    mcp_config = args.mcp_config or settings.mcp.config_path
    if mcp_config:
        print(f"MCP config: {mcp_config}")
        os.environ["OMLX_MCP_CONFIG"] = mcp_config

    # Determine paged SSD cache directory
    # Priority: --no-cache > CLI arg > settings file
    if args.no_cache:
        paged_ssd_cache_dir = None
    elif args.paged_ssd_cache_dir:
        # CLI argument takes precedence
        paged_ssd_cache_dir = args.paged_ssd_cache_dir
    elif settings.cache.enabled:
        # Use settings file value (resolved path or default)
        paged_ssd_cache_dir = str(settings.cache.get_ssd_cache_dir(settings.base_path))
    else:
        # Cache explicitly disabled in settings
        paged_ssd_cache_dir = None

    # Build scheduler config for BatchedEngine
    scheduler_config = settings.to_scheduler_config()
    # Set paged SSD cache options
    scheduler_config.paged_ssd_cache_dir = paged_ssd_cache_dir
    # Determine cache max size: CLI arg > settings (with auto resolution)
    if paged_ssd_cache_dir:
        if args.paged_ssd_cache_max_size:
            # CLI argument specified explicitly
            cache_max_size_bytes = parse_size(args.paged_ssd_cache_max_size)
        else:
            # Use settings value (handles "auto" -> 10% of SSD capacity)
            cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(settings.base_path)
        scheduler_config.paged_ssd_cache_max_size = cache_max_size_bytes
    else:
        scheduler_config.paged_ssd_cache_max_size = 0
        cache_max_size_bytes = 0

    # Hot cache: CLI arg > settings
    if paged_ssd_cache_dir:
        if args.hot_cache_max_size:
            hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
        else:
            hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
        scheduler_config.hot_cache_max_size = hot_cache_max_bytes
    else:
        scheduler_config.hot_cache_max_size = 0

    if args.no_cache:
        print("Mode: Multi-model serving (no oMLX cache, mlx-lm BatchGenerator only)")
    elif paged_ssd_cache_dir:
        print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
        # Format cache size for display
        cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
        print(f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})")
        if scheduler_config.hot_cache_max_size > 0:
            hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
            print(f"Hot cache: {hot_display} (in-memory)")
    else:
        print("Mode: Multi-model serving (continuous batching, no cache)")

    # Set MLX buffer cache limit high to prevent the allocator from
    # immediately releasing Metal buffers when the cache is full.
    # Without this, allocator::free() can call buf->release() while the
    # GPU is still using the buffer, causing kernel panics on M4.
    # With a large cache limit, freed buffers always stay in the pool
    # and are only released via mx.clear_cache() (which we protect
    # with mx.synchronize()). See issue #300.
    import mlx.core as mx
    total_mem = mx.device_info().get("memory_size", 0)
    if total_mem > 0:
        mx.set_cache_limit(total_mem)

    # Initialize server
    # Note: pinned_models and default_model are managed via admin page (model_settings.json)
    # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
    init_server(
        model_dirs=[str(d) for d in model_dirs],
        max_model_memory=settings.model.get_max_model_memory_bytes(),
        scheduler_config=scheduler_config,
        api_key=settings.auth.api_key,
        global_settings=settings,
    )

    # Start server
    print(f"Starting server at http://{settings.server.host}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = "debug" if settings.server.log_level == "trace" else settings.server.log_level
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )


def mcp_command(args):
    """Handle 'omlx mcp' subcommands (login, logout, status)."""
    import asyncio
    import sys

    sub = args.mcp_subcommand

    if sub == "status":
        _mcp_status(args)
    elif sub == "login":
        asyncio.run(_mcp_login(args))
    elif sub == "logout":
        _mcp_logout(args)
    else:
        # No subcommand given — print help.
        print("Usage: omlx mcp {login,logout,status} <server>")
        sys.exit(1)


def _mcp_status(args) -> None:
    """Print OAuth auth status for MCP servers."""
    from .mcp.config import load_mcp_config
    from .mcp.oauth import MCPOAuthManager
    from .settings import GlobalSettings

    settings = GlobalSettings.load()
    mcp_config = args.mcp_config or settings.mcp.config_path
    config = load_mcp_config(mcp_config)
    if not config.servers:
        print("No MCP servers configured.")
        return

    manager = MCPOAuthManager()
    for name, server in config.servers.items():
        if not server.auth:
            print(f"  {name}: no OAuth configured")
            continue
        info = manager.get_token_info(name)
        if info is None:
            print(f"  {name}: not authenticated")
        elif info.get("is_expired"):
            print(f"  {name}: token expired (run 'omlx mcp login {name}' to re-authenticate)")
        else:
            scope = info.get("scope") or ""
            expires_at = info.get("expires_at")
            expiry_note = ""
            if expires_at:
                import time
                remaining = int(expires_at - time.time())
                expiry_note = f", expires in {remaining}s"
            print(f"  {name}: authenticated (scope={scope!r}{expiry_note})")


async def _mcp_login(args) -> None:
    """Perform OAuth login for an MCP server."""
    import sys

    from .mcp.config import load_mcp_config
    from .mcp.oauth import MCPOAuthManager
    from .settings import GlobalSettings

    server_name: str = args.server
    flow: str = args.flow

    settings = GlobalSettings.load()
    mcp_config = args.mcp_config or settings.mcp.config_path
    config = load_mcp_config(mcp_config)

    if server_name not in config.servers:
        print(f"Unknown MCP server: '{server_name}'")
        print(f"Configured servers: {', '.join(config.servers) or '(none)'}")
        sys.exit(1)

    server = config.servers[server_name]
    if not server.auth:
        print(
            f"MCP server '{server_name}' has no OAuth configuration. "
            "Add an 'auth' block to its config entry."
        )
        sys.exit(1)

    manager = MCPOAuthManager()
    try:
        token = await manager.login(server_name, server.auth, flow=flow, server_url=server.url)
        scope = token.scope or ""
        print(f"\nAuthentication successful for '{server_name}' (scope={scope!r})")
    except Exception as exc:
        print(f"\nAuthentication failed for '{server_name}': {exc}")
        sys.exit(1)


def _mcp_logout(args) -> None:
    """Remove stored OAuth tokens for an MCP server."""
    import sys

    from .mcp.config import load_mcp_config
    from .mcp.oauth import MCPOAuthManager
    from .settings import GlobalSettings

    server_name: str = args.server

    settings = GlobalSettings.load()
    mcp_config = args.mcp_config or settings.mcp.config_path
    config = load_mcp_config(mcp_config)
    if server_name not in config.servers:
        print(f"Unknown MCP server: '{server_name}'")
        sys.exit(1)

    manager = MCPOAuthManager()
    manager.logout(server_name)
    print(f"Logged out from '{server_name}': stored tokens removed.")


def launch_command(args):
    """Launch an external tool integrated with oMLX."""
    import requests

    from .integrations import get_integration, list_integrations
    from .settings import GlobalSettings

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    # Resolve host/port: CLI args > env vars > settings.json > defaults
    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    # Check if oMLX server is running
    base_url = f"http://{host}:{port}"
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx serve")
        sys.exit(1)

    # Get API key from CLI args
    api_key = getattr(args, "api_key", None) or ""

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Determine model
    model = args.model
    if not model:
        # Fetch available models from server
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except Exception:
            models = []

        if not models:
            print("No models available. Load a model first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            print("Available models:")
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m}")
            while True:
                try:
                    choice = input("Select model number: ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        model = models[idx]
                        break
                    print(f"Please enter 1-{len(models)}")
                except (ValueError, EOFError):
                    print(f"Please enter 1-{len(models)}")

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # Fetch model limits from server
    context_window = None
    max_tokens = None
    model_type = None
    try:
        resp = requests.get(f"{base_url}/v1/models/status", headers=headers, timeout=5)
        if resp.ok:
            for m in resp.json().get("models", []):
                if m["id"] == model:
                    context_window = m.get("max_context_window")
                    max_tokens = m.get("max_tokens")
                    model_type = m.get("model_type")
                    break
    except Exception:
        pass

    # Launch
    print(f"Launching {integration.display_name} with model {model}...")
    tools_profile = getattr(args, "tools_profile", "coding")
    integration.launch(
        port=port,
        api_key=api_key,
        model=model,
        host=host,
        tools_profile=tools_profile,
        context_window=context_window,
        max_tokens=max_tokens,
        model_type=model_type,
    )


def main():
    parser = argparse.ArgumentParser(
        description="omlx: Production-ready LLM server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  omlx launch codex --model qwen3.5
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    serve_parser.add_argument(
        "--max-model-memory",
        type=str,
        default=None,
        help="Maximum memory for loaded models (e.g., 32GB, 'disabled'). Default: 80%% of system memory.",
    )
    serve_parser.add_argument(
        "--max-process-memory",
        type=str,
        default=None,
        help=(
            "Max total process memory as percentage of system RAM (10-99%%), "
            "'auto' (RAM - 8GB), or 'disabled'. Default: auto."
        ),
    )

    # Server options
    serve_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=None, help="Max concurrent sequences (default: 256)"
    )
    serve_parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=None,
        help="Max sequences for mlx-lm BatchGenerator completion phase (token generation). (default: 32)",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (optional)",
    )

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch an external tool with oMLX integration",
        description="Configure and launch external coding tools (Codex, OpenCode, OpenClaw) "
        "to use the running oMLX server.",
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help="Tool to launch: codex, opencode, openclaw, or 'list' to show available",
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (interactive selection if not specified)",
    )
    launch_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="oMLX server host (default: from settings or 127.0.0.1)",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMLX server port (default: from settings or 8000)",
    )
    launch_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for oMLX server authentication",
    )
    launch_parser.add_argument(
        "--tools-profile",
        type=str,
        default="coding",
        choices=["minimal", "coding", "messaging", "full"],
        help="OpenClaw tools profile (default: coding)",
    )

    # MCP command (login / logout / status)
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP server OAuth authentication",
        description="Authenticate with OAuth-protected MCP servers.",
    )
    mcp_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (default: auto-detected)",
    )
    mcp_subparsers = mcp_parser.add_subparsers(
        dest="mcp_subcommand", help="MCP auth commands"
    )

    # omlx mcp status
    mcp_status_parser = mcp_subparsers.add_parser(
        "status",
        help="Show OAuth authentication status for all configured MCP servers",
    )

    # omlx mcp login <server>
    mcp_login_parser = mcp_subparsers.add_parser(
        "login",
        help="Authenticate with an OAuth-protected MCP server",
    )
    mcp_login_parser.add_argument(
        "server",
        type=str,
        help="MCP server name as defined in the config file",
    )
    mcp_login_parser.add_argument(
        "--flow",
        type=str,
        default="pkce",
        choices=["pkce", "device"],
        help=(
            "OAuth flow to use: 'pkce' opens a browser (default), "
            "'device' prints a code for headless environments"
        ),
    )

    # omlx mcp logout <server>
    mcp_logout_parser = mcp_subparsers.add_parser(
        "logout",
        help="Remove stored OAuth tokens for an MCP server",
    )
    mcp_logout_parser.add_argument(
        "server",
        type=str,
        help="MCP server name as defined in the config file",
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve_command(args)
    elif args.command == "launch":
        launch_command(args)
    elif args.command == "mcp":
        mcp_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
