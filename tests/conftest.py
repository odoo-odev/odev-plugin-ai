"""Make the odev framework importable, plugins have no meaning without it."""

import shutil
import sys
from importlib.util import find_spec
from pathlib import Path


if find_spec("odev") is None and (odev_executable := shutil.which("odev")):
    sys.path.insert(0, Path(odev_executable).resolve().parent.as_posix())
