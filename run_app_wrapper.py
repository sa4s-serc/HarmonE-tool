"""
Local dev-only launcher: runs tool/app.py with the correct working directory
(app.py uses paths relative to tool/) and prepends the project's venv Scripts
dir to PATH so app.py's own `subprocess.Popen(["python3", "run_managed_system.py"])`
call resolves to this venv instead of a system python3 alias.

Not part of the HarmonE tool itself - just a shim for previewing it from this
session's browser pane.
"""
import os
import runpy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(PROJECT_ROOT, "tool")
VENV_SCRIPTS = os.path.join(TOOL_DIR, "harmone_env", "Scripts")

os.environ["PATH"] = VENV_SCRIPTS + os.pathsep + os.environ.get("PATH", "")
# The tool prints emoji/unicode symbols throughout, fine on the Linux target's
# UTF-8 console but not Windows' default cp1252 - without this, subprocesses
# spawned by app.py (run_managed_system.py, inference.py, manage.py) crash on
# their first such print. Setting it here propagates to the whole process tree.
os.environ["PYTHONIOENCODING"] = "utf-8"
os.chdir(TOOL_DIR)

runpy.run_path("app.py", run_name="__main__")
