from __future__ import annotations


def compare_results_raw_data(results1: list, results2: list[tuple]) -> None:
    """Check that two lists of samples (as Result.raw_data) are the same."""
    for i, sample1 in enumerate(results1):
        res_sample1 = (sample1.probability, sample1._state, sample1.state.__str__())
        res_sample2 = (
            results2[i][0].probability,
            results2[i][0]._state,
            results2[i][1],
        )
        assert res_sample1 == res_sample2
