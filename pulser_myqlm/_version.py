from pathlib import PurePath

# Sets the version to the same as 'pulser'.
version_file_path = PurePath(__file__).parent.parent / "VERSION.txt"

with open(version_file_path, "r") as f:
    __version__ = f.read().strip()
