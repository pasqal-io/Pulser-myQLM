# Pulser-MyQLM

Pulser-MyQLM is an extension of Pulser and myqlm qat.core for the integration of the Pulser framework and Pasqal devices within the Atos MyQLM framework.

## Installation

To install ``pulser`` and ``myqlm``, simply clone this repository, go to its root folder and run

```bash
pip install -e .
```

If you also wish to install the development requirements (optional), follow up with:

```bash
pip install -r dev_requirements.txt
```

## Continuous Integration Requirements

We enforce some continuous integration standards. Make sure you follow them, otherwise your pull requests will be blocked until you fix them. To check if your changes pass all CI tests before you make the PR, you'll need additional packages, which you can install by running

```shell
pip install -r dev_requirements.txt
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
