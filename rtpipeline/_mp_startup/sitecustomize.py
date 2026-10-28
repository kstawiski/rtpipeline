"""Apply socket setup in external Python CLIs while retaining site customisation."""
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_startup_dir = Path(__file__).resolve().parent
# Spawn workers receive the parent's process config after site initialization.
# The resource tracker and fork server need no socket directory of their own.
_multiprocessing_child = "--multiprocessing-fork" in sys.argv or (
    sys.argv[:1] == ["-c"]
    and any(arg.startswith((
        "from multiprocessing.resource_tracker import main;",
        "from multiprocessing.forkserver import main;",
    )) for arg in sys.orig_argv)
)

# Normal import semantics let the chained module refer to itself in sys.modules
# and ensure later imports return that same module. Keep PYTHONPATH in the
# environment so external subprocesses still receive this hook.
sys.modules.pop("sitecustomize", None)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _startup_dir]
try:
    import_module("sitecustomize")
finally:
    if not _multiprocessing_child:
        _spec = spec_from_file_location(
            "_rtpipeline_multiprocessing_utils", _startup_dir.parent / "multiprocessing_utils.py"
        )
        _helper = module_from_spec(_spec)
        _spec.loader.exec_module(_helper)
        _helper.configure_multiprocessing_tempdir()
# A missing chained module is handled by site.py. Other chained errors also
# propagate to site.py for its normal startup report, after socket setup.
