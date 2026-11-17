"""Exact replacement byte pins for the ROI identity implementation."""
import hashlib
import json
from pathlib import Path

PINS = json.loads(Path(__file__).with_suffix('.json').read_text())

def assert_source_bytes(relative, current, historical):
    if relative in PINS:
        assert hashlib.sha256(current).hexdigest() == PINS[relative], relative
    else:
        assert current == historical, relative
