"""Every documented uvx line must pin Python, or it breaks on an Intel Mac.

torch stopped at 2.2.2 on x86_64 macOS, with no wheels for Python 3.13+. An unpinned
`uvx --with 'sqlite-graph-memory[embeddings]' ...` resolves on uv's default Python and
fails with "No solution found" as soon as that default moves to 3.13 or later (measured
2026-11-02, MacPro192: the same README line worked the day before).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "examples/README.md", "llms.txt"]
EXTRA = "sqlite-graph-memory[embeddings]"


def unpinned_uvx_lines(text):
    """uvx command lines that install the embeddings extra without --python."""
    return [
        line for line in text.splitlines()
        if "uvx " in line and EXTRA in line and "--python" not in line
    ]


def unpinned_json_args(text):
    """Client-config JSON blocks whose uvx args lack --python."""
    bad = []
    for body in re.findall(r"```json\n(.*?)```", text, re.S):
        for section in ("mcpServers", "servers"):
            for name, server in json.loads(body).get(section, {}).items():
                if server.get("command") == "uvx" and "--python" not in server.get("args", []):
                    bad.append(name)
    return bad


def test_detector_catches_the_old_lines():
    assert unpinned_uvx_lines(f"uvx --with '{EXTRA}' sqlite-graph-memory")
    old = '```json\n{"mcpServers": {"x": {"command": "uvx", "args": ["--with", "y"]}}}\n```'
    assert unpinned_json_args(old) == ["x"]


def test_docs_pin_python():
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not unpinned_uvx_lines(text), name
        assert not unpinned_json_args(text), name


def test_toml_block_pins_python():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    toml_args = re.findall(r"^args = \[.*$", text, re.M)
    assert toml_args and all("--python" in line for line in toml_args)


def test_server_json_pins_python():
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    args = server["packages"][0]["runtimeArguments"]
    assert {"name": "--python", "value": "3.12"}.items() <= next(
        a for a in args if a["name"] == "--python"
    ).items()
