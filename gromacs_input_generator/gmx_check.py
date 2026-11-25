"""Optional ``gmx grompp`` check of the generated files (only when gmx is on PATH)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

GMX_NAMES = ("gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d")


def find_gmx() -> Optional[str]:
    for name in GMX_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def grompp_command(gmx: str, stage: str, conf: str, index: str = "") -> list:
    command = [gmx, "grompp", "-f", f"{stage}.mdp", "-c", conf, "-p", "topol.top",
               "-o", "check.tpr", "-maxwarn", "0"]
    if index:
        command += ["-n", index]
    return command


def run_grompp(files: dict, conf: str, stage: str = "em", gmx: Optional[str] = None,
               timeout: float = 120.0) -> Tuple[bool, str]:
    """Write the files to a temporary directory and run grompp there.

    Blocking; call it from a worker thread.  Returns (success, combined output).
    """
    gmx = gmx or find_gmx()
    if not gmx:
        return False, "gmx was not found on PATH."
    if f"{stage}.mdp" not in files:
        return False, f"{stage}.mdp is not part of the generated files."
    index = "index.ndx" if "index.ndx" in files else ""
    with tempfile.TemporaryDirectory(prefix="moleditpy_gmx_") as tmp:
        for name, text in files.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        command = grompp_command(gmx, stage, conf, index)
        try:
            proc = subprocess.run(command, cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"Could not run {' '.join(command)}: {exc}"
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, f"$ {' '.join(command)}\n\n{output}"
