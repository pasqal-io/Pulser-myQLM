from fresnel_qpu_server import create_parser
from pulser_myqlm.constants import DEFAULT_SERVER_PORT


def test_server_port_defaults_to_1234():
    args = create_parser().parse_args(["--local"])

    assert args.server_port == DEFAULT_SERVER_PORT == 1234


def test_server_port_can_be_overridden():
    args = create_parser().parse_args(["--local", "--server-port", "4321"])

    assert args.server_port == 4321
