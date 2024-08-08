"""Defines deserialize_other method."""

from __future__ import annotations

import json

from pulser import Sequence


def deserialize_other(other_bytestr: bytes | None) -> dict:
    """Deserialize MyQLM Job.schedule._other.

    Job.schedule._other must contain a string in bytes containing a dict serialized with
    json.dumps. The dict must contain at least one key: "abstr_seq". The value
    associated with that key must be a serialized Pulser Sequence.

    Arg:
        other_bytestr: The content of Job.schedule._other, a string encoded in utf-8.

    Returns:
        A dictionary containing a serialized Sequence (under the key "abstr_seq"),
        and its associated Sequence (under the key "seq").
    """
    if not isinstance(other_bytestr, bytes):
        raise ValueError("job.schedule._other must be a string encoded in bytes.")
    other_dict = json.loads(other_bytestr.decode(encoding="utf-8"))
    if not (isinstance(other_dict, dict) and "abstr_seq" in other_dict):
        raise ValueError(
            "An abstract representation of the Sequence must be associated with the"
            " 'abstr_seq' key of the dictionary serialized in job.schedule._other."
        )
    # Validate that value associated to abstr_seq is a serialized Sequence
    try:
        seq = Sequence.from_abstract_repr(other_dict["abstr_seq"])
    except Exception as e:
        raise ValueError(
            "Failed to deserialize the value associated to 'abstr_seq' in "
            "job.schedule._other as a Pulser Sequence."
        ) from e
    other_dict["seq"] = seq
    return other_dict
