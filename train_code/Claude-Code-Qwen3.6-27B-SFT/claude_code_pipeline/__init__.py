"""Utilities for preparing Claude Code tool traces for Qwen fine-tuning."""

from .converter import (
    ApproxTokenCounter,
    ConversionError,
    HFTokenCounter,
    ParsedSession,
    build_tool_registry,
    chunk_episode,
    find_secrets,
    parse_session_file,
    split_episodes,
    split_name,
    tools_for_messages,
    validate_trace,
)

__all__ = [
    "ApproxTokenCounter",
    "ConversionError",
    "HFTokenCounter",
    "ParsedSession",
    "build_tool_registry",
    "chunk_episode",
    "find_secrets",
    "parse_session_file",
    "split_episodes",
    "split_name",
    "tools_for_messages",
    "validate_trace",
]
