"""UTF-8 console setup, needed by every process that prints.

This tool prints emoji and arrows throughout (🆕, ⚡, 🔹, →). A Windows console
defaults to cp1252, which cannot encode them, so those prints raise
UnicodeEncodeError - which in practice killed inference.py outright and made
monitor_mape() throw on every telemetry tick, leaving the dashboard empty.

Call force_utf8_console() at the top of any entry point. Setting the env var
covers child processes (Popen inherits os.environ, so spawning entry points
only strictly need this themselves); reconfigure() covers the current
process's already-open streams, which PYTHONIOENCODING cannot fix
retroactively since it is only read at interpreter startup.
"""
import os
import sys


def force_utf8_console():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Not a real console (redirected to a pipe/file that's already
            # byte-oriented, or a stream without reconfigure) - the env var
            # still applies to anything spawned from here.
            pass
