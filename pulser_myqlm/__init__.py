"""An extension to execute MyQLM jobs on Pasqal devices."""
import json
from pathlib import Path

from pulser_myqlm._version import __version__
from pulser_myqlm.pulserAQPU import FresnelQPU, IsingAQPU

with open(Path(__file__).parent / "temp_device.json", "r", encoding="utf-8") as f:
    TEMP_DEVICE = json.load(f)
