"""ARCHITECTURE.md §4 must list every package directory under src/so101_rl/ (plan: docs sync rule)."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARCH = (ROOT / "docs" / "ARCHITECTURE.md").read_text()


def _section(n: int) -> str:
    m = re.search(rf"^## {n}\. .*?(?=^## \d+\. |\Z)", ARCH, flags=re.DOTALL | re.MULTILINE)
    assert m, f"section {n} missing"
    return m.group(0)


def test_every_package_dir_is_listed_in_section_4():
    section = _section(4)
    pkgs = sorted(
        p.name for p in (ROOT / "src" / "so101_rl").iterdir() if p.is_dir() and (p / "__init__.py").exists()
    )
    assert pkgs, "no packages found"
    missing = [p for p in pkgs if not re.search(rf"\b{re.escape(p)}/", section)]
    assert not missing, f"not listed in ARCHITECTURE.md §4: {missing}"


def test_version_table_has_latest_row():
    rows = re.findall(r"^\| (\d+\.\d+) \|", _section(10), flags=re.MULTILINE)
    header = re.search(r"\*\*Status:\*\* v(\d+\.\d+)", ARCH)
    assert rows and header and rows[-1] == header.group(1)
