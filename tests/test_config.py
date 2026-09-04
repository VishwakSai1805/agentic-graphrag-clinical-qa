from pathlib import Path

from src.config import config_hash, load_config

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_load_config_parses_fallback_chain():
    cfg = load_config(BASE_CONFIG)
    assert cfg.seed == 42
    assert cfg.llm.fallback_chain[0].provider == "ollama"
    assert cfg.llm.fallback_chain[0].api_key_env is None
    assert cfg.llm.fallback_chain[1].api_key_env == "GROQ_API_KEY"


def test_config_hash_stable_for_same_content():
    cfg1 = load_config(BASE_CONFIG)
    cfg2 = load_config(BASE_CONFIG)
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_changes_with_content():
    cfg = load_config(BASE_CONFIG)
    other = cfg.model_copy(deep=True)
    other.seed = 7
    assert config_hash(cfg) != config_hash(other)
