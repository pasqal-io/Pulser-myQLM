import pytest

def test_legacy_imports():
    """Backwards compatibility test.

    Test that we can still import FresnelQPU and IsingAQPU from the old
    PulserAQPU module with a deprecation warning.
    """
    with pytest.warns(DeprecationWarning):
        from pulser_myqlm.PulserAQPU import FresnelQPU, IsingAQPU  # noqa F401
