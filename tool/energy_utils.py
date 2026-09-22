"""Cross-platform energy measurement, shared by every managed-system script.

pyRAPL only works on Linux with a readable /sys/class/powercap/intel-rapl
interface - it hard-crashes at `pyRAPL.setup()`/import time everywhere else
(Windows, macOS, containers/VMs without RAPL access), which is what forced a
Windows-only pyRAPL stub into this project's dev venv previously. This module
makes that a runtime fallback instead of a crash: real RAPL readings on Linux
when available, otherwise a CPU-utilization-based estimate so `energy_uJ`
stays populated (and responsive to actual load) on every platform.

The estimate is: energy_J = TDP_watts * (process CPU% since begin() / cpu_count)
* elapsed_seconds. That's the same fallback strategy tools like CodeCarbon use
when a hardware energy counter isn't available - it is an ESTIMATE, not a
hardware measurement, and won't reproduce the paper's exact Joule figures
(those were measured via real RAPL on Linux). It exists so the tool runs
end-to-end for local development/testing on Mac and Windows.

Every call site keeps using the exact `pyRAPL.setup()` / `pyRAPL.Measurement
(label)` / `.begin()` / `.end()` / `.result.pkg[0]` interface - only the
import line changes (`import energy_utils as pyRAPL`), since this module
mirrors that interface directly.
"""
import logging
import os
import platform
import time

import psutil

_DEFAULT_TDP_WATTS = float(os.environ.get("HARMONE_CPU_TDP_WATTS", "15"))

# Scheduler tick. Process CPU-time counters cannot resolve anything shorter,
# so a zero delta below this is "unmeasurable", not "idle".
_CPU_COUNTER_TICK_S = 0.0157


def _try_setup_rapl():
    if platform.system() != "Linux":
        return None
    try:
        import pyRAPL
        pyRAPL.setup()
        return pyRAPL
    except Exception:
        return None


_pyRAPL = _try_setup_rapl()
RAPL_AVAILABLE = _pyRAPL is not None

if not RAPL_AVAILABLE:
    logging.warning(
        "[energy_utils] Intel RAPL not available on this platform (%s) - "
        "falling back to a CPU-utilization-based energy ESTIMATE, not a "
        "hardware measurement. Set HARMONE_CPU_TDP_WATTS to override the "
        "assumed %.0fW package TDP used for the estimate.",
        platform.system(), _DEFAULT_TDP_WATTS,
    )


def setup():
    """Compatibility shim for `pyRAPL.setup()` call sites. RAPL availability
    is probed once at import time above, so this is intentionally a no-op."""
    pass


class _Result:
    __slots__ = ("pkg",)

    def __init__(self, pkg=None):
        self.pkg = pkg


class EnergyMeter:
    """Drop-in replacement for pyRAPL.Measurement(label): same begin()/end()/
    result.pkg[0] interface, backed by real RAPL where available and a CPU
    utilization estimate everywhere else.

    Not safe to nest or overlap two meters on the same process - both share
    the process's single CPU% sampler - but nothing in this codebase does
    that today (each call site runs one meter at a time, sequentially).
    """

    def __init__(self, label, tdp_watts=None):
        self.label = label
        self.tdp_watts = tdp_watts if tdp_watts is not None else _DEFAULT_TDP_WATTS
        self.result = _Result()
        self._rapl_meter = _pyRAPL.Measurement(label) if _pyRAPL else None
        self._process = psutil.Process(os.getpid())
        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._start_time = None
        self._start_cpu = None
        self._estimated = False

    def begin(self):
        # Recorded unconditionally, even when RAPL is about to be used, so a
        # CPU-estimate fallback is always available with a correct elapsed
        # duration if RAPL fails later in end() (a "begin() succeeded, end()
        # didn't" split failure - see below).
        self._start_cpu = self._cpu_seconds()
        self._start_time = time.perf_counter()
        if self._rapl_meter is not None:
            try:
                self._rapl_meter.begin()
            except Exception:
                logging.warning("[energy_utils] RAPL begin() failed mid-run, switching to the CPU estimate for '%s'", self.label)
                self._rapl_meter = None

    def end(self):
        if self._rapl_meter is not None:
            try:
                self._rapl_meter.end()
                pkg = self._rapl_meter.result.pkg
                # pyRAPL sets pkg to None (not an empty list) when the sensor
                # read produced no valid value - that's a documented, real
                # outcome, not just an error path, so fall through to the CPU
                # estimate for this call rather than a crash or a fake 0.0.
                if pkg:
                    self.result.pkg = pkg
                    return
                logging.warning("[energy_utils] RAPL returned no valid reading for '%s', falling back to the CPU estimate", self.label)
                self._rapl_meter = None
            except Exception:
                logging.warning("[energy_utils] RAPL end() failed for '%s', falling back to the CPU estimate", self.label)
                self._rapl_meter = None

        duration_s = max(0.0, time.perf_counter() - (self._start_time or time.perf_counter()))
        cpu_s = max(0.0, self._cpu_seconds() - (self._start_cpu or 0.0))

        # Process CPU time comes from GetProcessTimes on Windows (and jiffies on
        # Linux), so its granularity is a scheduler tick - about 15.6ms. A single
        # inference takes ~2.5ms, so the measured delta is exactly 0.0 and the
        # estimate collapsed to zero: 475 of 500 logged rows read 0.0 uJ.
        #
        # Below one tick, charge the window as one core busy for its wall-clock
        # duration. Above a tick the counter CAN resolve the work, so a zero
        # delta there means the process really was idle - do not floor it, or
        # every MLflow network call inside a measured region would be billed as
        # full-core compute. This is a granularity floor, not a measurement.
        if cpu_s <= 0.0 and 0.0 < duration_s < _CPU_COUNTER_TICK_S:
            cpu_s = duration_s
            self._estimated = True

        cpu_s = min(cpu_s, duration_s * self._cpu_count)  # cannot exceed all cores
        energy_uJ = self.tdp_watts * (cpu_s / self._cpu_count) * 1e6
        self.result.pkg = [energy_uJ]

    def _cpu_seconds(self):
        try:
            t = self._process.cpu_times()
            return t.user + t.system
        except Exception:
            return 0.0


# Alias so `import pyRAPL` call sites need only change to
# `import energy_utils as pyRAPL` - everything else (pyRAPL.setup(),
# pyRAPL.Measurement(...)) then resolves unchanged.
Measurement = EnergyMeter
