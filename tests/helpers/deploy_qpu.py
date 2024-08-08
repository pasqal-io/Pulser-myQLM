from __future__ import annotations

from time import sleep

from qat.core.qpu import QPUHandler
from qat.qpus import RemoteQPU
from thrift.transport.TTransport import TTransportException


def deploy_qpu(qpu: QPUHandler, port: int) -> None:
    """Deploys the QPU on a server on a port at IP 127.0.0.1."""
    qpu.serve(port, "127.0.0.1")


def get_remote_qpu(port: int) -> RemoteQPU:
    tries = 0
    while tries < 10:
        try:
            return RemoteQPU(port, "localhost")
        except TTransportException as e:
            tries += 1
            sleep(1)
            error = e
    raise error
