import ast
from pathlib import Path
import subprocess


ACTIVE_PYTHON_FILES = (
    "a_lmi/cli.py",
    "a_lmi/config.py",
    "a_lmi/continuity.py",
    "a_lmi/providers.py",
    "a_lmi/runtime.py",
    "a_lmi/security/encryption.py",
    "a_lmi/services/multimodal_encoder.py",
    "a_lmi/memory/object_storage_client.py",
    "a_lmi/memory/vector_db_client.py",
    "a_lmi/memory/tkg_client.py",
    "cosmic_synapse/ipc/bridge.py",
    "cosmic_synapse/ipc/schema.py",
)

SECRET_VARIABLES = (
    "A_LMI_MINIO_ACCESS_KEY",
    "A_LMI_MINIO_SECRET_KEY",
    "A_LMI_NEO4J_PASSWORD",
    "MILVUS_MINIO_ACCESS_KEY",
    "MILVUS_MINIO_SECRET_KEY",
)


def _attribute_name(node: ast.AST) -> tuple[str | None, str | None]:
    if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
        return None, None
    return node.value.id, node.attr


def test_active_product_python_has_no_direct_dynamic_execution_shortcuts():
    for relative in ACTIVE_PYTHON_FILES:
        tree = ast.parse(Path(relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                raise AssertionError(f"{relative} calls builtin {node.func.id}()")

            owner, method = _attribute_name(node.func)
            if (owner, method) in {("os", "system"), ("pickle", "load"), ("yaml", "load")}:
                raise AssertionError(f"{relative} calls {owner}.{method}()")

            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    raise AssertionError(f"{relative} invokes a call with shell=True")


def test_root_env_file_is_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "root .env must remain untracked"


def test_env_example_contains_placeholders_not_live_secret_values():
    values = {}
    for raw_line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    for key in SECRET_VARIABLES:
        assert key in values
        assert values[key].startswith("replace-with-"), f"{key} must remain a placeholder"


def test_active_config_does_not_embed_secret_fallbacks():
    config = Path("infrastructure/config.yaml").read_text(encoding="utf-8")
    assert 'access_key: "${A_LMI_MINIO_ACCESS_KEY:-}"' in config
    assert 'secret_key: "${A_LMI_MINIO_SECRET_KEY:-}"' in config
    assert 'password: "${A_LMI_NEO4J_PASSWORD:-}"' in config


def test_compose_published_ports_remain_loopback_bound():
    compose = Path("infrastructure/docker-compose.yml").read_text(encoding="utf-8")
    published = [
        line.strip().removeprefix('- "').removesuffix('"')
        for line in compose.splitlines()
        if line.strip().startswith('- "') and line.strip().endswith('"') and ":" in line
    ]
    assert published
    for mapping in published:
        assert mapping.startswith("127.0.0.1:"), f"non-loopback published port: {mapping}"
