"""YAML -> pydantic config loading, with a content hash for run provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    provider: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0


class LLMConfig(BaseModel):
    fallback_chain: list[ProviderConfig]
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int | None = 512
    max_retries: int = 3
    backoff_base_s: float = 1.0


class AppConfig(BaseModel):
    seed: int = 42
    llm: LLMConfig
    cache_dir: Path = Path(".cache/llm")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(raw)


def config_hash(cfg: AppConfig) -> str:
    """Stable sha256 hex digest of the resolved config, for run_id / provenance."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
