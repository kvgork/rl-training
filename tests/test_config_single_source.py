"""L4: limits live in src/so101_rl/config/ only. A numeric literal assigned to a known limit name anywhere
else in src/ is a second source of truth."""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "so101_rl"
EXCLUDED_DIRS = {"config", "_vendor"}
NAMES = ("max_delta_ticks", "tracking_error_ticks", "trip_c", "resume_c", "stale_cmd_ms", "bound_s")
# name [: annotation] (= | :) [ ( or [ ] number
PATTERN = re.compile(
    r"""\b(?:"""
    + "|".join(NAMES)
    + r""")\b["']?\s*(?::\s*[\w\[\], .|]+?)?\s*[:=]\s*[\[\(]?\s*-?\d"""
)


def _offenders():
    hits = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts[0] in EXCLUDED_DIRS:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if PATTERN.search(code):
                hits.append(f"{rel}:{i}: {line.strip()}")
    return hits


def test_no_limit_literals_outside_config():
    assert _offenders() == []


@pytest.mark.parametrize(
    "line",
    [
        "trip_c = 60.0",
        "trip_c: float = 60",
        "cfg = dict(max_delta_ticks=[45, 45])",
        "x = {'tracking_error_ticks': (120, 400)}",
        "resume_c=50",
        "bound_s: float = 5.0",
        "stale_cmd_ms = 150",
    ],
)
def test_pattern_detects_offenders(line):
    assert PATTERN.search(line)


@pytest.mark.parametrize("line", ["trip_c = cfg.thermal.trip_c", "max_delta_ticks=cfg.action.max_delta_ticks", "bound = 5.0"])
def test_pattern_ignores_config_reads(line):
    assert not PATTERN.search(line)
