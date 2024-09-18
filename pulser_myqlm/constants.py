"""Defines common constants."""

from pathlib import Path
from typing import cast

from pulser.devices._device_datacls import Device
from pulser.json.abstract_repr.deserializer import deserialize_device

DEFAULT_NUMBER_OF_SHOTS = 2000
QPU_POLLING_INTERVAL_SECONDS = 5
JOB_POLLING_INTERVAL_SECONDS = 1
JOB_POLLING_MAX_RETRIES = 10

with open(Path(__file__).parent / "temp_device.json", "r", encoding="utf-8") as f:
    TEMP_DEVICE = cast(Device, deserialize_device(f.read()))
