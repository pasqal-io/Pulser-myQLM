"""Defines simulate_seq method."""

from __future__ import annotations

from collections import Counter

from pulser import Sequence
from pulser_simulation import QutipEmulator

from pulser_myqlm.constants import DEFAULT_NUMBER_OF_SHOTS


def simulate_seq(seq: Sequence, modulation: bool, nbshots: int | None) -> Counter:
    """Simulates a Sequence using pulser-simulation.

    Args:
        seq: A Pulser Sequence to simulate.
        modulation: If True, uses modulated samples of the Sequence to perform
            the simulation.
        nbshots: The number of shots to perform (if None or 0,
            defaults to pulser_myqlm.constants.DEFAULT_NUMBER_OF_SHOTS).

    Returns:
        A Counter object, output of pulser-simulation.
    """
    emulator = QutipEmulator.from_sequence(
        seq,
        with_modulation=modulation,
    )
    return emulator.run().sample_final_state(
        DEFAULT_NUMBER_OF_SHOTS if not nbshots else nbshots
    )
