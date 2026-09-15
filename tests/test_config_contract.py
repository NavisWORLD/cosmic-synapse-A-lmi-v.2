from pathlib import Path

from a_lmi.config import load_config


def test_load_config_accepts_mapping_without_reopening_it():
    config = {"logging": {"level": "INFO"}, "infrastructure": {}}
    assert load_config(config) is config


def test_load_config_accepts_yaml_path(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded["logging"]["level"] == "DEBUG"
