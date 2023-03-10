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

description = "An extension to execute MyQLM jobs on Pasqal devices."

setup(
    name=distribution_name,
    version=__version__,
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8.0",
)

# Restores the original source code of _version.py
with open(local_version_fpath, "w") as f:
    f.write(stashed_version_source)
