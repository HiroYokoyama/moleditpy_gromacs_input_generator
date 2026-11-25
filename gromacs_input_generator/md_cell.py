"""Simulation box handling: the suite's ``moleditpy_cell`` property and GROMACS boxes.

GROMACS wants a lower-triangular box (a along x, b in the xy plane) with
positive diagonal and |b_x| <= a_x/2, |c_x| <= a_x/2, |c_y| <= b_y/2.  Any
right-handed cell is rotated into that setting, its coordinates rotated with
it, and the off-diagonal elements reduced by adding lattice vectors (which
describes the same periodic lattice).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np

from .cell_model import cell_vectors, lattice_parameters

CELL_PROPERTY = "moleditpy_cell"


class CellError(ValueError):
    """Raised for an unusable cell description."""


def parse_cell_json(text) -> dict:
    """Parse and validate a ``moleditpy_cell`` JSON string (or dict).

    Returns ``{"lattice": 3x3 ndarray (Angstrom, rows = vectors), "origin":
    ndarray(3), "pbc": [bool]*3}``.
    """
    if isinstance(text, (str, bytes)):
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise CellError(f"moleditpy_cell is not valid JSON: {exc}") from exc
    else:
        data = text
    if not isinstance(data, dict):
        raise CellError("moleditpy_cell must be a JSON object.")
    try:
        lattice = np.asarray(data["lattice"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise CellError("moleditpy_cell has no usable 'lattice'.") from exc
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise CellError("moleditpy_cell 'lattice' must be three finite 3-vectors.")
    origin = data.get("origin", [0.0, 0.0, 0.0])
    try:
        origin = np.asarray(origin, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise CellError("moleditpy_cell 'origin' must be three numbers.") from exc
    pbc = data.get("pbc", [True, True, True])
    if not isinstance(pbc, (list, tuple)) or len(pbc) != 3:
        pbc = [True, True, True]
    volume = float(np.linalg.det(lattice))
    if abs(volume) < 1e-6:
        raise CellError("moleditpy_cell lattice is singular (zero volume).")
    return {"lattice": lattice, "origin": origin, "pbc": [bool(v) for v in pbc]}


def cell_to_json(lattice, origin=(0.0, 0.0, 0.0), pbc=(True, True, True)) -> str:
    lattice = np.asarray(lattice, dtype=float)
    return json.dumps(
        {
            "lattice": [[float(v) for v in row] for row in lattice],
            "origin": [float(v) for v in origin],
            "pbc": [bool(v) for v in pbc],
        }
    )


def read_cell_property(mol) -> dict | None:
    """The molecule's ``moleditpy_cell`` as a parsed dict, or None if absent."""
    if mol is None:
        return None
    try:
        if not mol.HasProp(CELL_PROPERTY):
            return None
        return parse_cell_json(mol.GetProp(CELL_PROPERTY))
    except CellError:
        raise
    except (AttributeError, RuntimeError):  # pragma: no cover - RDKit-like guard
        return None


def box_from_lengths(lengths: Sequence[float], angles: Sequence[float] = (90.0, 90.0, 90.0)):
    """Lattice (Angstrom) from a, b, c and alpha, beta, gamma."""
    try:
        return cell_vectors(lengths, angles)
    except ValueError as exc:
        raise CellError(str(exc)) from exc


def box_from_padding(coords_angstrom, padding: float, cubic: bool = False):
    """Orthorhombic box around the coordinates; returns (lattice, origin)."""
    coords = np.asarray(coords_angstrom, dtype=float).reshape(-1, 3)
    if coords.size == 0:
        raise CellError("No atoms to put in a box.")
    if padding < 0:
        raise CellError("Padding must not be negative.")
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.maximum(high - low + 2.0 * padding, 1.0)
    if cubic:
        span = np.full(3, span.max())
    centre = (low + high) / 2.0
    origin = centre - span / 2.0
    return np.diag(span), origin


def gromacs_setting(lattice):
    """Rotate a right-handed cell into the GROMACS setting.

    Returns ``(box, rotation)`` where ``box`` is the lower-triangular lattice
    (rows) in Angstrom and ``rotation`` maps row-vector coordinates:
    ``coords @ rotation``.
    """
    lattice = np.asarray(lattice, dtype=float)
    if np.linalg.det(lattice) <= 0:
        raise CellError(
            "The cell is left-handed (a x b points away from c); GROMACS needs a "
            "right-handed cell. Swap two cell vectors."
        )
    lengths, angles = lattice_parameters(lattice)
    standard = cell_vectors(lengths, angles)
    rotation = np.linalg.solve(lattice, standard)
    # Clean numerical noise: a pure rotation is orthogonal.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    box = reduce_box(standard)
    return box, rotation


def reduce_box(box):
    """Apply GROMACS' triclinic restrictions by adding lattice vectors."""
    box = np.array(box, dtype=float)
    a, b, c = box
    c = c - round(c[1] / b[1]) * b if b[1] else c
    c = c - round(c[0] / a[0]) * a
    b = b - round(b[0] / a[0]) * a
    box = np.vstack([a, b, c])
    box[np.abs(box) < 1e-10] = 0.0
    return box


def is_rectangular(box, tol: float = 1e-6) -> bool:
    box = np.asarray(box, dtype=float)
    return bool(np.all(np.abs(box - np.diag(np.diag(box))) < tol))


def gro_box_line(box_angstrom) -> str:
    """The last line of a .gro file (nm)."""
    box = np.asarray(box_angstrom, dtype=float) / 10.0
    diag = [box[0, 0], box[1, 1], box[2, 2]]
    if is_rectangular(box):
        values = diag
    else:
        values = diag + [box[0, 1], box[0, 2], box[1, 0], box[1, 2], box[2, 0], box[2, 1]]
    return "".join(f"{v:10.5f}" for v in values)


def minimum_box_width(box_angstrom) -> float:
    """Smallest perpendicular width of the cell (Angstrom), for cutoff checks."""
    box = np.asarray(box_angstrom, dtype=float)
    volume = abs(float(np.linalg.det(box)))
    a, b, c = box
    areas = [np.linalg.norm(np.cross(b, c)), np.linalg.norm(np.cross(c, a)), np.linalg.norm(np.cross(a, b))]
    return min(volume / area for area in areas if area > 0)


def describe(box_angstrom) -> str:
    lengths, angles = lattice_parameters(box_angstrom)
    return (
        "a={:.3f} b={:.3f} c={:.3f} A, alpha={:.2f} beta={:.2f} gamma={:.2f} deg".format(
            *lengths, *angles
        )
    )


def angle_ok(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value < 180.0
