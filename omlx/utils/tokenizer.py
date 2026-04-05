# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities for oMLX.

This module provides shared tokenizer configuration and fixes that are used
across multiple modules in the codebase.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def unwrap_tokenizer(tokenizer):
    """Unwrap mlx-lm TokenizerWrapper to a HuggingFace PreTrainedTokenizer.

    xgrammar accepts HuggingFace ``PreTrainedTokenizer`` /
    ``PreTrainedTokenizerFast`` but NOT the raw ``tokenizers.Tokenizer``
    nor the mlx-lm ``TokenizerWrapper``.  This helper peels exactly one
    layer of mlx-lm wrapping while keeping the HuggingFace object intact.
    """
    try:
        from transformers import PreTrainedTokenizerBase
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            return tokenizer
    except ImportError:
        pass
    if hasattr(tokenizer, '_tokenizer'):
        inner = tokenizer._tokenizer
        try:
            from transformers import PreTrainedTokenizerBase
            if isinstance(inner, PreTrainedTokenizerBase):
                return inner
        except ImportError:
            pass
        return inner
    return tokenizer


def resolve_vocab_size(model: Any) -> int | None:
    """Extract vocab_size from a model's config/args, handling nested configs.

    Tries ``model.config.vocab_size``, then ``model.args.vocab_size``,
    then ``text_config.vocab_size`` for VLM composite models (e.g. Qwen3.5).

    Args:
        model: An MLX model object (LLM, VLM, or any object with config/args).

    Returns:
        The vocabulary size, or None if it cannot be determined.
    """
    if model is None:
        return None
    for attr in ('config', 'args'):
        config = getattr(model, attr, None)
        if config is None:
            continue
        vs = getattr(config, 'vocab_size', None)
        if isinstance(vs, int):
            return vs
        text_cfg = getattr(config, 'text_config', None)
        if isinstance(text_cfg, dict):
            vs = text_cfg.get('vocab_size')
        elif text_cfg is not None:
            vs = getattr(text_cfg, 'vocab_size', None)
        if isinstance(vs, int):
            return vs
    return None


def is_harmony_model(model_name: str, config: dict[str, Any] | None = None) -> bool:
    """
    Check if the model uses Harmony format.

    Harmony format is used by gpt-oss models with special tokens like
    <|start|>, <|channel|>, <|message|>, <|end|>, <|return|>, <|call|>.

    Detection priority:
    1. model_type == "gpt_oss" in config.json
    2. Fallback: model_name contains "gpt-oss" or "gptoss" (case-insensitive)

    Args:
        model_name: The model name or path.
        config: Optional model config dict (from config.json).

    Returns:
        True if the model uses Harmony format.
    """
    # Primary detection: config.model_type
    if config is not None:
        model_type = config.get("model_type", "")
        if model_type == "gpt_oss":
            logger.debug(f"Harmony model detected via config.model_type: {model_name}")
            return True

    # Fallback detection: model name pattern
    if model_name:
        name_lower = model_name.lower()
        if "gpt-oss" in name_lower or "gptoss" in name_lower:
            logger.debug(f"Harmony model detected via model name pattern: {model_name}")
            return True

    return False


def is_gemma4_model(model_name: str, config: dict[str, Any] | None = None) -> bool:
    """
    Check if the model is a Gemma 4 model.

    Detection priority:
    1. model_type == "gemma4" in config.json
    2. Fallback: model_name contains "gemma-4" or "gemma4" (case-insensitive)
    """
    if config is not None:
        model_type = config.get("model_type", "")
        if model_type == "gemma4":
            logger.debug(f"Gemma 4 model detected via config.model_type: {model_name}")
            return True

    if model_name:
        name_lower = model_name.lower()
        if "gemma-4" in name_lower or "gemma4" in name_lower:
            logger.debug(f"Gemma 4 model detected via model name pattern: {model_name}")
            return True

    return False


def is_qwen3_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen3 model.

    Args:
        model_name: The model name or path.

    Returns:
        True if the model is a Qwen3 model.
    """
    model_lower = model_name.lower()
    return "qwen3" in model_lower or "Qwen3" in model_name


def get_tokenizer_config(
    model_name: str,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """
    Get tokenizer configuration with model-specific fixes.

    This function centralizes tokenizer configuration to ensure consistent
    behavior across different modules.

    Args:
        model_name: The model name or path.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Dictionary of tokenizer configuration options.
    """
    config: dict[str, Any] = {"trust_remote_code": trust_remote_code}

    # Apply Qwen3 fix if needed
    if is_qwen3_model(model_name):
        config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    return config


def inject_tool_calling(tokenizer) -> None:
    """Inject tool calling attributes into a tokenizer that lacks them.

    mlx-lm's TokenizerWrapper sets ``has_tool_calling`` for some models but not
    all (e.g. Gemma 4 reports False even though the model supports tool calls).
    This function uses mlx_vlm.tool_parsers (superset, knows Gemma4) or falls
    back to mlx_lm.tokenizer_utils to detect and inject the right parser.

    Skips injection if the tokenizer already has ``has_tool_calling = True``.
    """
    if getattr(tokenizer, "has_tool_calling", False):
        return

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return

    tool_module = None

    # Prefer mlx_vlm.tool_parsers — superset that knows about Gemma4 etc.
    try:
        from mlx_vlm.tool_parsers import _infer_tool_parser, load_tool_module

        tool_parser_type = _infer_tool_parser(chat_template)
        if tool_parser_type:
            try:
                tool_module = load_tool_module(tool_parser_type)
            except ImportError:
                logger.warning(f"Tool parser module not found: {tool_parser_type}")
    except ImportError:
        pass

    # Fallback: mlx_lm.tokenizer_utils
    if tool_module is None:
        try:
            import importlib
            from mlx_lm.tokenizer_utils import _infer_tool_parser as _mlx_lm_infer

            tool_parser_type = _mlx_lm_infer(chat_template)
            if tool_parser_type:
                try:
                    tool_module = importlib.import_module(
                        f"mlx_lm.tool_parsers.{tool_parser_type}"
                    )
                except ImportError:
                    logger.warning(f"Tool parser module not found: {tool_parser_type}")
        except ImportError:
            pass

    if tool_module is None:
        return

    tool_call_start = getattr(tool_module, "tool_call_start", None)
    tool_call_end = getattr(tool_module, "tool_call_end", None)

    # Validate tokens exist in vocab
    try:
        vocab = tokenizer.get_vocab()
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            logger.warning(
                f"Tool call tokens not in vocab: start={tool_call_start!r} end={tool_call_end!r}"
            )
            return
    except Exception:
        pass

    parse_fn = tool_module.parse_tool_call

    # The mlx_vlm gemma4 parser uses \w+ in its regex which does not match
    # hyphenated tool names (e.g. "notion-search").  Patch the module-level
    # regex in-place so all callers benefit.
    if tool_parser_type == "gemma4" and hasattr(tool_module, "_tool_call_regex"):
        import regex as _re
        tool_module._tool_call_regex = _re.compile(
            r"call:([\w-]+)(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})"
        )

    # mlx-lm TokenizerWrapper exposes has_tool_calling / tool_call_start /
    # tool_call_end / tool_parser as @property backed by _tool_* private attrs.
    # mlx-vlm's wrapper uses plain instance attributes.  We try the private
    # attrs first; fall back to direct assignment for other wrapper types.
    if hasattr(tokenizer, "_tool_call_start"):
        tokenizer._tool_call_start = tool_call_start
        tokenizer._tool_call_end = tool_call_end
        tokenizer._tool_parser = parse_fn
    else:
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tool_call_start
        tokenizer.tool_call_end = tool_call_end
        tokenizer.tool_parser = parse_fn

    logger.info(f"Tool calling enabled via inject: parser={tool_parser_type}")


def apply_qwen3_fix(
    tokenizer_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Apply Qwen3 tokenizer fix to an existing config.

    Qwen3 has a known issue where eos_token changed from <|im_end|> to
    <|endoftext|>, but the chat template still uses <|im_end|>. This
    function applies the fix if needed.

    Args:
        tokenizer_config: Existing tokenizer configuration dict.
        model_name: The model name or path.

    Returns:
        Updated tokenizer configuration with Qwen3 fix applied if needed.
    """
    if is_qwen3_model(model_name):
        tokenizer_config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    return tokenizer_config
