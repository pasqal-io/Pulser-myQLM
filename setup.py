#    Copyright 2023 Pasqal Quantum Solutions / Pulser Development Team

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from pathlib import Path

from setuptools import find_packages, setup

distribution_name = "pulser-myqlm"
package_name = "pulser_myqlm"
current_directory = Path(__file__).parent

# Reads the version from the VERSION.txt file
with open(current_directory / "VERSION.txt", "r") as f:
    __version__ = f.read().strip()

# Changes to the directory where setup.py is
os.chdir(current_directory)

# Stashes the source code for the local version file
local_version_fpath = Path(package_name) / "_version.py"
with open(local_version_fpath, "r") as f:
    stashed_version_source = f.read()

# Overwrites the _version.py for the source distribution (reverted at the end)
with open(local_version_fpath, "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dev_requirements.txt") as f:
    dev_requirements = f.read().splitlines()
test_requirements = [req for req in dev_requirements if "pytest" in req]

description = "An extension to interface MyQLM with Pulser."

setup(
    name=distribution_name,
    version=__version__,
    description=description,
    install_requires=requirements,
    extras_require={"dev": dev_requirements, "test_dev": test_requirements},
    packages=find_packages(),
    include_package_data=True,
    author="Pasqal Quantum Solutions / Pulser Development Team",
    python_requires=">=3.8.0",
    license="Apache 2.0",
)

# Restores the original source code of _version.py
with open(local_version_fpath, "w") as f:
    f.write(stashed_version_source)
