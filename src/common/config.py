"""Configuration management"""
import yaml
from typing import Dict, Any, List
from pathlib import Path


class Config:
    """Configuration manager for the multi-Mac workflow"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def management_host(self) -> str:
        return self.get('management.host', '0.0.0.0')

    @property
    def management_port(self) -> int:
        return self.get('management.port', 5000)

    @property
    def max_workers(self) -> int:
        return self.get('management.max_workers', 10)

    @property
    def result_dir(self) -> str:
        return self.get('management.result_dir', './results')

    @property
    def workers(self) -> List[Dict[str, Any]]:
        return self.get('workers', [])

    @property
    def worktree_base(self) -> str:
        return self.get('git.worktree_base', './worktrees')

    @property
    def keep_best_n(self) -> int:
        return self.get('git.keep_best_n', 1)

    @property
    def monitoring_interval(self) -> int:
        return self.get('monitoring.interval', 5)

    @property
    def tmux_session_prefix(self) -> str:
        return self.get('tmux.session_prefix', 'ml-task')
