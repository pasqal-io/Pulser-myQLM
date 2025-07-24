# Pulser-MyQLM

Pulser-MyQLM is an extension of [Pulser](https://pulser.readthedocs.io/en/stable/index.html) and [myQLM](https://qlm.bull.com/bin/view/Main/) for the integration of the Pulser framework and Pasqal devices within the myQLM framework.

## Installation

To install the ``pulser-myqlm`` package, simply clone this repository, go to its root folder and run

```bash
pip install -e .
```
or 

```bash
python setup.py install
```

This will install ``pulser`` and ``myqlm``. If you also wish to install the development requirements (optional), follow up with:

```bash
pip install -r dev_requirements.txt
```

or

```bash
python setup.py install easy_install "pulser-myqlm[dev]"
```

If, among the development requirements, you only wish to install the test requirements, do:

```bash
python setup.py install easy_install "pulser-myqlm[test_dev]"
```

## Documentation

Pulser-MyQLM enables to submit quantum programs written with Pulser to Pasqal's QPUs via a QLM. It then provides tools for users of the QLM to submit their quantum programs and simulate them, as well as tools for internal developpers to connect Pasqal QPUs with their QLM. All these features are documented with notebooks in the [tutorials folder](./tutorials/).

### Submitting a Pulser Sequence to the QLM

#### A Pulser Remote Connection to submit to the QPU

Pulser is an open-source Python software package that provides libraries for designing and simulating pulse sequences that act on programmable arrays of neutral atoms. The quantum programs are then written with a Pulser `Sequence`.

Sequences are submitted to remote QPUs using the [QPUBackend of Pulser](https://pulser.readthedocs.io/en/stable/tutorials/backends.html). This backend needs a remote connection to communicate with the QPU. Pulser-MyQLM provides a `QLMConnection` to communicate with the QPU, i.e, to submit Sequences, to find the available QPUs on a given QLM, etc. . See an example of the submission of a Sequence to a QPU via a `QLMConnection` in [this tutorial](./tutorials/Submitting%20AFM%20state%20prep%20to%20QPU.ipynb).


#### Converting a Pulser Sequence into a MyQLM Job

The QLM takes as input myQLM Jobs. An alternative approach to using the `QLMConnection` to submit `Sequences` to the QPU is to convert them into a `Job` using `IsingAQPU.convert_sequence_to_job`. An example is given in [this tutorial](./tutorials/Submitting%20AFM%20state%20prep%20to%20QPU.ipynb).

The obtained Job is called ["Analog"](https://myqlm.github.io/02_user_guide/01_write/02_analog_schedule/03_an_jobs.html): it contains the time-dependent Hamiltonian associated with the Sequence. It can be simulated using the simulators of the QLM accepting analog jobs.

A thorough presentation of the conversion of Sequence into a Job is presented in [this tutorial](./tutorials/pulser-myqlm.ipynb). The reverse conversion is not implemented, its limitations are presented in [this other tutorial](./tutorials/pulser_schedule_creation.ipynb).

### Connecting a QPU with a QLM

Internal developpers wanting to connect a Pasqal QPU with their QLM have to use a `RemoteQPU` in their QLM, connected to a `FresnelQPU`, itself connected to the QPU. 

To ease the development, `pulser-myqlm` provides a python executable [fresnel_qpu_server.py](./fresnel_qpu_server.py), that takes as input the IP address of the QPU and its port, as well as the IP address and the port on which to create a `FresnelQPU` server. By using a `RemoteQPU` pointing to the IP address and the port of this server, any Sequence that will be submitted to this `RemoteQPU` will be executed on the QPU (see an example in [this tutorial](./tutorials/pulser-myqlm.ipynb)).

## Testing

Unitary tests are developed in the ./tests folder.
Non-regression tests are provided in the ./tutorials folder as python scripts. Here is how to check the correct deloyment of a QPU in a Qaptiva Stack using pulser-myqlm:
- AFM_direct.py: Assume you have a QPU installed accessible at a given IP and PORT. Use this script to submit a Sequence preparing an AFM state.
- AFM_remote.py: Assume you have deployed a FresnelQPU server, that is able to submit myQLM Jobs to the QPU, at a given IP and port (for instance, using `fresnel_qpu_server.py`). Specify the IP and port of this server in this script to submit a Sequence preparing an AFM state.
- AFM_qaptiva.py: Assume you have installed in your Qaptiva a QLMaaSQPU named PasqalQPU, that is connected with the QPU (either via a FresnelQPU or via a RemoteQPU connected to a FresnelQPU server for instance). Use this script to submit a Sequence preparing an AFM state to the QPU.
- AFM_PulserQLMConnection.py: Assume the same installation as in the AFM_qaptiva.py. Test the correct submission of a Pulser Sequence preparing an AFM state to the QPU, using the PulserQLMConnection. 

## Continuous Integration Requirements

We enforce some continuous integration standards. Make sure you follow them, otherwise your pull requests will be blocked until you fix them. To check if your changes pass all CI tests before you make the PR, you'll need additional packages, which you can install by running

```shell
pip install -r dev_requirements.txt
```

or

```bash
python setup.py install easy_install "pulser-myqlm[dev]"
```

- **Tests**: We use [`pytest`](https://docs.pytest.org/en/latest/) to run unit tests on our code. If your changes break existing tests, you'll have to update these tests accordingly. Additionally, we aim for 100% coverage over our code. Try to cover all the new lines of code with simple tests, which should be placed in the `tests/` folder. To run all tests and check coverage, run:

    ```bash
    pytest --cov .
    ```

    All lines that are not meant to be tested must be tagged with `# pragma: no cover`. Use it sparingly,
    every decision to leave a line uncovered must be well justified.

- **Style**: We use [`flake8`](https://flake8.pycqa.org/en/latest/) and the `flake8-docstrings` extension to enforce PEP8 style guidelines. To lint your code with `flake8`, simply run:

    ```bash
    flake8 .
    ```

    To help you keep your code compliant with PEP8 guidelines effortlessly, we suggest you look into installing a linter for your text editor of choice.

- **Format**: We use the [`black`](https://black.readthedocs.io/en/stable/index.html) auto-formatter to enforce a consistent style throughout the entire code base, including the Jupyter notebooks (so make sure to install `black[jupyter]`). It will also ensure your code is compliant with the formatting enforced by `flake8` for you. To automatically format your code with black, just run:

    ```bash
    black .
    ```

    Note that some IDE's and text editors support plug-ins which auto-format your code with `black` upon saving, so you don't have to worry about code format at all.

- **Import sorting**: We use [`isort`](https://pycqa.github.io/isort/) to automatically sort all library imports. You can do the same by running:

    ```bash
    isort .
    ```

- **Type hints**: We use [`mypy`](http://mypy-lang.org/) to type check the code. Your code should have type
annotations and pass the type checks from running:

    ```bash
    mypy
    ```

    In case `mypy` produces a false positive, you can ignore the respective line by adding the `# type: ignore` annotation.

    **Note**: Type hints for `numpy` have only been added in version 1.20. Make sure you have `numpy >= 1.20`
    installed before running the type checks.


## License

Copyright 2023 Pasqal Quantum Solutions / Pulser Development Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
