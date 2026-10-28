"""Apply socket setup in external Python CLIs while retaining site customisation."""
from importlib.machinery import PathFinder
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_startup_dir = Path(__file__).resolve().parent
# Preserve an existing sitecustomize supplied by the tool's environment.
_other_paths = [p for p in sys.path if Path(p).resolve() != _startup_dir]
_previous = PathFinder.find_spec("sitecustomize", _other_paths)
if _previous is not None and _previous.loader is not None:
    _previous.loader.exec_module(module_from_spec(_previous))

_spec = spec_from_file_location(
    "_rtpipeline_multiprocessing_utils", _startup_dir.parent / "multiprocessing_utils.py"
)
_helper = module_from_spec(_spec)
_spec.loader.exec_module(_helper)
_helper.configure_multiprocessing_tempdir()
