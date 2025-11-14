import yaml
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'


class Settings:
    """
        Class-based loader that reads all YAML files in `config/`.

        - Per-file dicts are available as attributes named `<stem>_config`.
        - Merged keys (later files override earlier) are available in `settings.config`.
    """

    def __init__(self, config_files: list[str] = None):
        if config_files is None:
            self.config_files = CONFIG_DIR.glob('*.yaml')
        else:
            self.config_files = [CONFIG_DIR / file for file in config_files]

        self.config = {}
        for p in self.config_files:
            val = self._load_yaml(p)
            globals()[f"{p.stem}_config"] = val

            if isinstance(val, dict):
                self.config.update(val)
            else:
                self.config[p.stem] = val

    @staticmethod
    def _load_yaml(p: Path) -> Any:
        try:
            d = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        except Exception:
            return {}
        return d.get(p.stem, d) if isinstance(d, dict) else d


settings = Settings()

__all__ = ['settings', *[name for name in globals().keys()
                         if name.endswith("_config")]]
