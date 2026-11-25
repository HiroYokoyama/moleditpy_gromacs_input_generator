"""Small hand-built systems for the tests (no RDKit needed)."""

import os

import numpy as np

from gromacs_input_generator.system_model import build_system

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def water(offset=(0.0, 0.0, 0.0), order="OHH"):
    o = np.array(offset, float)
    oxygen, h1, h2 = o, o + [0.9572, 0.0, 0.0], o + [-0.2400, 0.9266, 0.0]
    if order == "OHH":
        return ["O", "H", "H"], [oxygen, h1, h2], [(0, 1), (0, 2)]
    return ["H", "O", "H"], [h1, oxygen, h2], [(1, 0), (1, 2)]


def ethanol(offset=(0.0, 0.0, 0.0)):
    """RDKit AddHs order for CCO: C C O H H H H H H."""
    o = np.array(offset, float)
    coords = [
        [0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.0, 1.35, 0.0],
        [-0.36, 1.03, 0.0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
        [1.88, -0.51, 0.89], [1.88, -0.51, -0.89], [2.95, 1.30, 0.0],
    ]
    bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8)]
    return list("CCOHHHHHH"), [np.array(c) + o for c in coords], bonds


def ion(element, offset):
    return [element], [np.array(offset, float)], []


def assemble(parts, charges=None, **kwargs):
    elements, coords, bonds = [], [], []
    for el, xyz, bd in parts:
        base = len(elements)
        elements += el
        coords += list(xyz)
        bonds += [(i + base, j + base) for i, j in bd]
    return build_system(elements, np.array(coords), bonds, formal_charges=charges, **kwargs)


def example_system():
    """Ethanol + 2 water + Na+ + Cl-, matching examples/ethanol_water_nacl.mdff.json."""
    parts = [ethanol(), water((6, 0, 0)), water((0, 6, 0)), ion("Na", (6, 6, 0)), ion("Cl", (-6, 0, 0))]
    return assemble(parts, [0] * 15 + [1, -1])


def example_mdff_path(name="ethanol_water_nacl.mdff.json"):
    return os.path.join(EXAMPLES, name)
