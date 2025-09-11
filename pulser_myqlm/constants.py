"""Defines common constants."""

import os
from pathlib import Path
from typing import cast

from pulser.devices._device_datacls import Device
from pulser.json.abstract_repr.deserializer import deserialize_device

DEFAULT_NUMBER_OF_SHOTS = int(os.getenv("DEFAULT_NUMBER_OF_SHOTS", 2000))
QPU_POLLING_INTERVAL_SECONDS = int(os.getenv("QPU_POLLING_INTERVAL_SECONDS", 5))
JOB_POLLING_INTERVAL_SECONDS = int(os.getenv("JOB_POLLING_INTERVAL_SECONDS", 1))
QPU_POLLING_TIMEOUT_SECONDS = int(os.getenv("QPU_POLLING_TIMEOUT_SECONDS", -1))
JOB_POLLING_TIMEOUT_SECONDS = int(os.getenv("JOB_POLLING_TIMEOUT_SECONDS", -1))
MAX_CONNECTION_ATTEMPTS_QLM = int(os.getenv("MAX_CONNECTION_ATTEMPS_QLM", 5))
MAX_CONNECTION_ATTEMPTS_QPU = int(os.getenv("MAX_CONNECTION_ATTEMPTS_QPU", 5))

with open(Path(__file__).parent / "temp_device.json", "r", encoding="utf-8") as f:
    TEMP_DEVICE = cast(Device, deserialize_device(f.read()))
