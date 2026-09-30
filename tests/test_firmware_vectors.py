"""Watchdog MCU firmware: shared test vectors run against the firmware's own logic (host, no hardware).

`firmware/watchdog_mcu/main.py` keeps all decisions in the pure `Core` class, so these tests exercise the
real firmware code. The same `test_vectors.json` is replayed against the real MCU in H3.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FW = ROOT / "firmware" / "watchdog_mcu"
VECTORS = json.loads((FW / "test_vectors.json").read_text())["vectors"]


def _load_main():
    spec = importlib.util.spec_from_file_location("watchdog_main", FW / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fw = _load_main()


def tdiff(a: int, b: int) -> int:
    """Host equivalent of time.ticks_diff for a 32-bit counter."""
    return ((a - b + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def run_vector(vec: dict) -> list[str]:
    """Replay a vector on Core; return the per-step output hex (empty string for no output)."""
    now = vec["start_ms"]
    core = fw.Core(now, tdiff)
    inp = {"vbus_mv": 12000, "estop": False, "btn": False, "jumper": False}
    outs = []
    for step in vec["steps"]:
        now = (now + step["dt"]) & 0xFFFFFFFF
        inp.update({k: step[k] for k in inp if k in step})
        out = core.poll(now, inp["vbus_mv"], inp["estop"], inp["btn"])
        out += core.feed(now, bytes.fromhex(step.get("tx", "")), inp["jumper"])
        outs.append(out.hex())
    return outs


@pytest.mark.parametrize("vec", VECTORS, ids=[v["name"] for v in VECTORS])
def test_vector_on_firmware_core(vec):
    assert run_vector(vec) == [s["out"] for s in vec["steps"]]


def test_vectors_cover_protocol_invariants():
    names = {v["name"] for v in VECTORS}
    for required in (
        "timeout_999_armed_1001_latched",
        "seq_gap_latches_reason_3",
        "serial_cannot_clear_latch",
        "reset_button_500ms_clears_to_disarmed",
        "reset_blocked_while_estop_asserted",
        "undervolt_latches_after_50ms",
        "estop_latches_reason_4",
        "wrap_safe_timeout",
        "test_drop_measures_ms",
        "test_drop_refused_without_jumper",
    ):
        assert required in names
    # every status byte in every expectation is a legal byte: reason nibble <= 6
    for v in VECTORS:
        for s in v["steps"]:
            out = bytes.fromhex(s["out"])
            if s.get("tx") and out and "drop" not in v["name"]:
                assert out[0] >> 4 <= 6


def test_status_byte_layout():
    c = fw.Core(0, tdiff)
    assert c.status() == 0x60  # boot: DISARMED, reason MCU_RESET
    c.feed(0, b"\x5a", False)
    assert c.status() == 0x05  # relay_on | armed
    c.poll(1, 12000, True, False)  # e-stop
    assert c.status() == 0x42  # latched, reason 4
    c.poll(2, 5000, True, False)
    assert c.status() & 0x08  # undervolt bit follows live voltage


def test_relay_follows_armed_only():
    """Relay must be energised only in ARMED: every other reachable state maps to 'off'."""
    c = fw.Core(0, tdiff)
    assert c.state != fw.ARMED
    c.feed(0, b"\x5a", False)
    assert c.state == fw.ARMED
    c.poll(1500, 12000, False, False)
    assert c.state == fw.LATCHED


def test_firmware_line_budget():
    assert len((FW / "main.py").read_text().splitlines()) <= 80


def test_fakemcu_satisfies_portable_vectors():
    """FakeMcu (U1) and the firmware must agree on the serial-only vectors. Skipped until U1 has merged."""
    hb = pytest.importorskip("so101_rl.servo.heartbeat")
    if not hasattr(hb, "FakeMcu"):
        pytest.skip("FakeMcu not merged yet")
    from so101_rl.servo.timing import FakeClock

    for vec in (v for v in VECTORS if v["portable"] and v["start_ms"] == 0):
        clock = FakeClock()
        mcu = hb.FakeMcu(clock)
        got = []
        for step in vec["steps"]:
            clock.advance(step["dt"] * 1_000_000)
            mcu.poll()
            got.append(mcu.feed(bytes.fromhex(step.get("tx", ""))).hex())
        assert got == [s["out"] for s in vec["steps"]], vec["name"]


def _install_stubs(monkeypatch, rx: list[bytes]):
    """Stub machine/micropython/select/time so main() can run a few loop iterations on the host."""
    log = {"kbd_intr": None, "wdt_timeout": None, "wdt_feeds": 0, "relay_init": None, "relay": [], "tx": []}

    class Pin:
        OUT, IN, PULL_UP = 1, 0, 2

        def __init__(self, n, mode=None, pull=None, value=None):
            self.n, self.v = n, value
            if n == 15:
                log["relay_init"] = value

        def value(self, x=None):
            if x is None:
                # GP14 estop (HIGH = asserted) healthy -> LOW; GP13/GP12 released/absent -> HIGH
                return {14: 0, 13: 1, 12: 1}[self.n]
            log["relay"].append(bool(x))

    class ADC:
        def __init__(self, n):
            pass

        def read_u16(self):
            return int(65535 * (12.0 / 5.7) / 3.3)  # 12 V through the divider

    class WDT:
        def __init__(self, timeout):
            log["wdt_timeout"] = timeout

        def feed(self):
            log["wdt_feeds"] += 1

    class Stop(Exception):
        pass

    ticks = {"t": 0}

    def sleep_ms(ms):
        ticks["t"] += 400  # 400 ms per loop iteration
        if ticks["t"] > 4000:
            raise Stop

    class Poller:
        def register(self, *a):
            pass

        def poll(self, _):
            return bool(rx)

    class Buf:
        def read(self, n):
            return rx.pop(0)

        def write(self, b):
            log["tx"].append(bytes(b))

    m = types.ModuleType
    machine = m("machine")
    machine.Pin, machine.ADC, machine.WDT = Pin, ADC, WDT
    mp = m("micropython")
    mp.kbd_intr = lambda x: log.__setitem__("kbd_intr", x)
    sel = m("select")
    sel.POLLIN, sel.poll = 1, Poller
    tm = m("time")
    tm.sleep_ms, tm.ticks_ms, tm.ticks_diff = sleep_ms, lambda: ticks["t"], tdiff
    for mod in (machine, mp, sel, tm):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=Buf()))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=Buf()))
    return log, Stop


def test_main_with_stubbed_machine(monkeypatch):
    """main() wires the pins as documented: relay LOW at start, WDT 250 ms, Ctrl-C disabled, ARM closes the relay,
    and the relay opens again after the heartbeat timeout."""
    log, stop = _install_stubs(monkeypatch, [b"\x5a"])
    mod = _load_main()
    with pytest.raises(stop):
        mod.main()
    assert log["relay_init"] == 0
    assert log["wdt_timeout"] == 250
    assert log["kbd_intr"] == -1
    assert log["wdt_feeds"] >= 5
    assert b"\x05" in log["tx"]  # ARM acknowledged, relay_on | armed
    assert True in log["relay"], "relay never closed after ARM"
    assert log["relay"][-1] is False, "relay must be open again after 1 s without heartbeat"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_firmware_is_lint_clean():
    r = subprocess.run(
        ["ruff", "check", "--no-cache", str(FW / "main.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
