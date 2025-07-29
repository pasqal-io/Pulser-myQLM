#!/usr/bin/env python3

"""Connects to a Pasqal QPU and deploys a MyQLM server using FresnelQPU."""
import argparse
import logging
from pulser_myqlm import FresnelQPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MyQLM server for FresnelQPU",
        description="Connects to a QPU on IP --qpu-ip and port --qpu-port using a "
        "FresnelQPU. Runs the created FresnelQPU inside a MyQLM server published on"
        " IP --server-ip and port --server-port. It is possible to emulate the "
        "connection to the QPU by passing the argument --local.",
    )
    parser.add_argument(
        "--qpu-port", help="Port of the target QPU.", type=int, default=None
    )
    parser.add_argument(
        "--qpu-ip", help="IP of the target QPU.", type=str, default=None
    )
    parser.add_argument(
        "--server-port", help="Port of the MyQLM server.", type=int, required=True
    )
    parser.add_argument(
        "--server-ip",
        help="IP of the MyQLM server.",
        default="127.0.0.1",
        type=str,
    )
    parser.add_argument(
        "--local",
        help="If passed as an argument, emulates the connection to the QPU "
        "using pulser-simulation.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        help="Sets the logging level in the default logging config",
        default="INFO",
        choices=logging.getLevelNamesMapping().keys()
    )
    parser.add_argument(
        "--log-file",
        help="Sets the logging file in the default logging config",
        default="logs_fresnel_qpu_server.txt",
    )
    parser.add_argument(
        "--log-config-file",
        help="Overrides the whole logging config from file",
        default=None,
    )
    args = parser.parse_args()

    if args.log_config_file is None:
        handlers = [logging.StreamHandler()]
        if args.log_file:
            handlers.append(logging.FileHandler(args.log_file))
        logging.basicConfig(
            format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.getLevelNamesMapping()[args.log_level],
            handlers=handlers
        )
    else:
        print("Setting log configuration from file %s" % args.log_config_file)
        logging.config.fileConfig(args.log_config_file)

    logger = logging.getLogger(__name__)

    if args.local:
        # Emulate the connection with a FresnelQPU using pulser-simulation
        logger.info(
            "Deploying a FresnelQPU using pulser-simulation as backend on IP:"
            f"{args.server_ip}, PORT:{args.server_port}."
        )
        FresnelQPU(None).serve(args.server_port, args.server_ip)
    elif args.qpu_ip is None or args.qpu_port is None:
        raise ValueError(
            "--qpu-ip and --qpu-port must be provided if --local is not passed "
            "as an argument."
        )
    # a FresnelQPU connected to a remote VM
    logger.info("Connecting to IP:", args.qpu_ip, ", PORT:", args.qpu_port)
    fresnel_qpu = FresnelQPU(f"http://{args.qpu_ip}:{args.qpu_port}/api", version="v1")
    logger.info("Connected. QPU is operational:", fresnel_qpu.is_operational)

    # Deploy the QPU on a port and ip
    logger.info("Creating a server on IP: ", args.server_ip, ", PORT:", args.server_port)
    fresnel_qpu.serve(args.server_port, args.server_ip)

    # Connect to this server remotely with RemoteQPU(SERVER_PORT, SERVER_IP)
