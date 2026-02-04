"""
Pipeline configuration loader

Reads config/pipeline_config.yaml for centralized seeds, paths, and thresholds.
"""

from pathlib import Path
from typing import Any


def load_pipeline_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load pipeline configuration from YAML.
    
    Args:
        config_path: Path to config file. Default: config/pipeline_config.yaml
        
    Returns:
        Config dict. Empty dict if file not found or invalid.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "pipeline_config.yaml"
    config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_random_seed(config: dict | None = None) -> int:
    """Get random seed from config. Default: 42."""
    if config is None:
        config = load_pipeline_config()
    return config.get("pipeline", {}).get("random_seed", 42)


def get_default_output_dir(config: dict | None = None) -> str:
    """Get default output directory from config. Default: out2."""
    if config is None:
        config = load_pipeline_config()
    return config.get("pipeline", {}).get("default_output_dir", "out2")
