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


def tdiff_bits(bits: int):
    """Host equivalent of time.ticks_diff for a counter that wraps at 2**bits (RP2040 MicroPython: 30)."""
    mod, half = 1 << bits, 1 << (bits - 1)
    return lambda a, b: ((a - b + half) & (mod - 1)) - half


tdiff = tdiff_bits(32)
INPUT_KEYS = {"dt", "tx", "out", "vbus_mv", "servo_mv", "estop", "btn", "jumper"}


def run_vector(vec: dict) -> list[str]:
    """Replay a vector on Core; return the per-step output hex (empty string for no output)."""
    bits = vec.get("wrap_bits", 32)
    now = vec["start_ms"] & ((1 << bits) - 1)
    core = fw.Core(now, tdiff_bits(bits))
    inp = {"vbus_mv": 12000, "servo_mv": None, "estop": False, "btn": False, "jumper": False}
    outs = []
    for step in vec["steps"]:
        assert set(step) <= INPUT_KEYS, step
        now = (now + step["dt"]) & ((1 << bits) - 1)
        inp.update({k: step[k] for k in inp if k in step})
        # servo_mv absent/None: the servo side follows the relay (PSU voltage while the previous step ended ARMED)
        smv = (
            inp["servo_mv"]
            if inp["servo_mv"] is not None
            else (inp["vbus_mv"] if core.state == fw.ARMED else 0)
        )
        out = core.poll(now, inp["vbus_mv"], inp["estop"], inp["btn"], smv)
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
        "boot_is_latched_reason_mcu_reset",
        "boot_serial_arm_refused",
        "boot_button_exactly_500ms_clears",
        "boot_button_blocked_while_estop",
        "timeout_exactly_1000_latched",
        "heartbeat_at_exactly_1000_cannot_rescue",
        "undervolt_exactly_50ms_still_armed",
        "reset_button_exactly_500ms_clears",
        "wrap_safe_timeout_30bit",
        "test_drop_with_jumper_but_disarmed_refused",
        "test_drop_with_jumper_but_latched_refused",
        "arm_uses_psu_side_not_servo_side",
        "arm_refused_while_servo_side_live",
        "relay_weld_latches_reason_7",
        "test_drop_then_weld",
        "partial_frame_discarded_after_100ms",
        "partial_frame_99ms_completes",
    ):
        assert required in names
    # every status byte in every expectation is a legal byte: reason nibble <= 7
    for v in VECTORS:
        for s in v["steps"]:
            out = bytes.fromhex(s["out"])
            if s.get("tx") and out and "drop" not in v["name"]:
                assert out[0] >> 4 <= 7


def clear_boot_latch(core, t0: int) -> int:
    """Hold the (simulated) reset button for exactly 500 ms starting at t0; returns the time afterwards."""
    core.poll(t0, 12000, False, True)
    core.poll(t0 + 500, 12000, False, True)
    core.poll(t0 + 500, 12000, False, False)
    return t0 + 500


def test_status_byte_layout():
    c = fw.Core(0, tdiff)
    assert c.status() == 0x62  # boot: LATCHED, reason MCU_RESET
    c.feed(0, b"\x5a", False)
    assert c.status() == 0x62  # serial ARM refused while latched
    clear_boot_latch(c, 0)
    assert c.status() == 0x60  # DISARMED, reason MCU_RESET kept
    c.feed(1000, b"\x5a", False)
    assert c.status() == 0x05  # relay_on | armed
    c.poll(1001, 12000, True, False)  # e-stop
    assert c.status() == 0x42  # latched, reason 4
    c.poll(1002, 5000, True, False)
    assert c.status() & 0x08  # undervolt bit follows live voltage


def test_relay_follows_armed_only():
    """Relay must be energised only in ARMED: every other reachable state maps to 'off'."""
    c = fw.Core(0, tdiff)
    assert c.state == fw.LATCHED
    t = clear_boot_latch(c, 0)
    assert c.state == fw.DISARMED
    c.feed(t, b"\x5a", False)
    assert c.state == fw.ARMED
    c.poll(t + 1500, 12000, False, False, 12000)
    assert c.state == fw.LATCHED


def test_firmware_line_budget():
    """Budget 80 -> 95 (wave-1 fix B): the second ADC (PSU-side vs servo-side divider, B4), the MCU_RESET boot latch
    (S7), relay-weld detection (B4) and the 100 ms partial-frame timeout (NIT) are real behaviour that the host
    vectors pin down; the extra lines are logic, not comments. Keep it reviewable."""
    assert len((FW / "main.py").read_text().splitlines()) <= 95


def run_vector_fake(vec: dict) -> list[str]:
    """Replay a vector through FakeMcu's public API (no private attributes)."""
    hb = pytest.importorskip("so101_rl.servo.heartbeat")
    from so101_rl.servo.timing import FakeClock

    clock = FakeClock(
        vec["start_ms"] * 1_000_000
    )  # unwrapped: wrap-safety is the firmware's concern, outputs must match
    mcu = hb.FakeMcu(clock)
    mcu.drop_time_ms = 0  # vectors force or follow the servo side explicitly
    outs = []
    for step in vec["steps"]:
        clock.advance(step["dt"] * 1_000_000)
        if "estop" in step:
            mcu.set_estop(step["estop"])
        if "vbus_mv" in step:
            mcu.set_bus_voltage(step["vbus_mv"] / 1000)
        if "servo_mv" in step:
            mcu.set_servo_voltage(None if step["servo_mv"] is None else step["servo_mv"] / 1000)
        if "btn" in step:
            mcu.set_button(step["btn"])
        if "jumper" in step:
            mcu.test_jumper = step["jumper"]
        out = mcu.poll()
        out += mcu.feed(bytes.fromhex(step.get("tx", "")))
        outs.append(out.hex())
    return outs


@pytest.mark.parametrize("vec", VECTORS, ids=[v["name"] for v in VECTORS])
def test_fakemcu_satisfies_every_vector(vec):
    """FakeMcu must produce byte-identical output to the firmware for EVERY vector (not only the portable ones),
    so drift between the two models (timeout `>=`, TEST_DROP refusal, reset reason, boot latch, weld) is caught."""
    assert run_vector_fake(vec) == [s["out"] for s in vec["steps"]]


def _install_stubs(monkeypatch, rx: list[bytes]):
    """Stub machine/micropython/select/time so main() can run a few loop iterations on the host."""
    log = {"kbd_intr": None, "wdt_timeout": None, "wdt_feeds": 0, "relay_init": None, "relay": [], "tx": []}
    ticks = {"t": 0}

    class Pin:
        OUT, IN, PULL_UP = 1, 0, 2

        def __init__(self, n, mode=None, pull=None, value=None):
            self.n, self.v = n, value
            if n == 15:
                log["relay_init"] = value

        def value(self, x=None):
            if x is None:
                # GP14 estop (HIGH = asserted) healthy -> LOW; GP12 absent -> HIGH;
                # GP13 reset button: pressed (LOW) for the first 900 ms to clear the boot latch
                return {14: 0, 13: 0 if ticks["t"] <= 900 else 1, 12: 1}[self.n]
            log["relay"].append(bool(x))

    class ADC:
        def __init__(self, n):
            self.n = n

        def read_u16(self):
            volts = 12.0  # GP26: PSU side, always 12 V. GP27: servo side, live only while the relay is closed
            if self.n == 27 and not (log["relay"] and log["relay"][-1]):
                volts = 0.0
            return int(65535 * (volts / 5.7) / 3.3)

    class WDT:
        def __init__(self, timeout):
            log["wdt_timeout"] = timeout

        def feed(self):
            log["wdt_feeds"] += 1

    class Stop(Exception):
        pass

    def sleep_ms(ms):
        ticks["t"] += 400  # 400 ms per loop iteration
        if ticks["t"] > 4000:
            raise Stop

    class Poller:
        def register(self, *a):
            pass

        def poll(self, _):
            return (
                bool(rx) and ticks["t"] >= 1200
            )  # the host sends ARM only after the button cleared the latch

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
    """main() wires the pins as documented: relay LOW at start, WDT 250 ms, Ctrl-C disabled, the boot latch needs the
    button, ARM then closes the relay (servo-side ADC reads live, no false weld), and the relay opens again after the
    heartbeat timeout."""
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
