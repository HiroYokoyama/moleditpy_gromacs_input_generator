"""A plain-data view of the host molecule: atoms, bonds, fragments, residues.

Everything downstream (topology, .gro, index groups) works on this model, so
it can be tested without RDKit.  :func:`from_rdkit` is the only adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .cell_model import normalize_element as normalize_symbol
from .elements import atomic_mass, bond_cutoff
from .ff_tables import (
    ION_ELEMENTS,
    ION_RESIDUE_NAMES,
    PROTEIN_RESIDUE_NAMES,
    WATER_RESIDUE_NAMES,
)


@dataclass
class Atom:
    element: str
    xyz: tuple[float, float, float]  # Angstrom
    name: str = ""
    resname: str = ""
    resnum: int = 0
    formal_charge: int = 0

    @property
    def mass(self) -> float:
        return atomic_mass(self.element)


@dataclass
class Fragment:
    """One connected molecule, with its atoms in host order."""

    atoms: list[int]
    kind: str = "other"  # "water", "ion", "protein", "other"
    signature: tuple = ()

    @property
    def size(self) -> int:
        return len(self.atoms)


@dataclass
class SystemModel:
    atoms: list[Atom]
    bonds: list[tuple[int, int]] = field(default_factory=list)
    fragments: list[Fragment] = field(default_factory=list)
    has_residue_info: bool = False

    @property
    def coords(self) -> np.ndarray:
        return np.array([a.xyz for a in self.atoms], dtype=float).reshape(-1, 3)

    @property
    def elements(self) -> list[str]:
        return [a.element for a in self.atoms]

    @property
    def net_charge(self) -> int:
        return sum(a.formal_charge for a in self.atoms)

    def neighbours(self) -> dict[int, list[int]]:
        table: dict[int, list[int]] = {i: [] for i in range(len(self.atoms))}
        for i, j in self.bonds:
            table[i].append(j)
            table[j].append(i)
        return table


def build_system(
    elements: Sequence[str],
    coords,
    bonds: Sequence[tuple[int, int]] = (),
    names: Sequence[str] | None = None,
    resnames: Sequence[str] | None = None,
    resnums: Sequence[int] | None = None,
    formal_charges: Sequence[int] | None = None,
    infer_bonds: bool = False,
) -> SystemModel:
    """Make a :class:`SystemModel` and detect its fragments."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(elements) != len(coords):
        raise ValueError("elements and coords differ in length")
    atoms = []
    for i, element in enumerate(elements):
        atoms.append(
            Atom(
                element=normalize_symbol(element),
                xyz=tuple(float(v) for v in coords[i]),
                name=(names[i] if names else "") or "",
                resname=(resnames[i] if resnames else "") or "",
                resnum=int(resnums[i]) if resnums else 0,
                formal_charge=int(formal_charges[i]) if formal_charges else 0,
            )
        )
    bond_list = sorted({(min(i, j), max(i, j)) for i, j in bonds if i != j})
    if infer_bonds and not bond_list:
        bond_list = guess_bonds(atoms)
    model = SystemModel(
        atoms=atoms,
        bonds=bond_list,
        has_residue_info=bool(resnames) and any(r.strip() for r in resnames),
    )
    model.fragments = find_fragments(model)
    return model


def guess_bonds(atoms: Sequence[Atom], tolerance: float = 0.45) -> list[tuple[int, int]]:
    """Distance-based bonds (cell list), used only when the molecule has none."""
    coords = np.array([a.xyz for a in atoms], dtype=float)
    if len(coords) < 2:
        return []
    cell = 3.0
    keys = np.floor(coords / cell).astype(int)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in enumerate(map(tuple, keys)):
        grid.setdefault(key, []).append(idx)
    bonds = []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i, key in enumerate(map(tuple, keys)):
        for dx, dy, dz in offsets:
            for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                if j <= i:
                    continue
                cutoff = bond_cutoff(atoms[i].element, atoms[j].element, tolerance)
                if np.sum((coords[i] - coords[j]) ** 2) <= cutoff * cutoff:
                    bonds.append((i, j))
    return sorted(bonds)


def find_fragments(model: SystemModel) -> list[Fragment]:
    """Connected components, ordered by their first atom; classify each."""
    n = len(model.atoms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in model.bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    fragments = [Fragment(atoms=sorted(members)) for members in groups.values()]
    fragments.sort(key=lambda f: f.atoms[0])
    for fragment in fragments:
        fragment.kind = classify(model, fragment)
        fragment.signature = signature(model, fragment)
    return fragments


def classify(model: SystemModel, fragment: Fragment) -> str:
    atoms = [model.atoms[i] for i in fragment.atoms]
    elements = sorted(a.element for a in atoms)
    resnames = {a.resname.strip().upper() for a in atoms if a.resname.strip()}
    if elements == ["H", "H", "O"]:
        bonded = {frozenset(b) for b in model.bonds if b[0] in fragment.atoms}
        oxygen = next(i for i in fragment.atoms if model.atoms[i].element == "O")
        hydrogens = [i for i in fragment.atoms if model.atoms[i].element == "H"]
        if all(frozenset((oxygen, h)) in bonded for h in hydrogens):
            return "water"
        if resnames and resnames <= WATER_RESIDUE_NAMES:
            return "water"
    if len(atoms) == 1 and atoms[0].element in ION_ELEMENTS:
        if atoms[0].formal_charge != 0 or (resnames and resnames <= ION_RESIDUE_NAMES):
            return "ion"
        # A lone alkali/halide atom in an MD system is an ion in practice.
        return "ion"
    if resnames and resnames & PROTEIN_RESIDUE_NAMES:
        return "protein"
    return "other"


def signature(model: SystemModel, fragment: Fragment) -> tuple:
    """Identity of a fragment for counting copies: elements in order, local
    bond graph, residue names and formal charges.  Two fragments with the same
    signature can share one [ moleculetype ]."""
    local = {atom: k for k, atom in enumerate(fragment.atoms)}
    members = set(fragment.atoms)
    elements = tuple(model.atoms[i].element for i in fragment.atoms)
    charges = tuple(model.atoms[i].formal_charge for i in fragment.atoms)
    resnames = tuple(model.atoms[i].resname.strip() for i in fragment.atoms)
    bonds = tuple(
        sorted(
            (local[i], local[j])
            for i, j in model.bonds
            if i in members and j in members
        )
    )
    return (elements, bonds, charges, resnames)


def water_order(model: SystemModel, fragment: Fragment) -> list[int]:
    """Host indices of a water fragment in O, H, H order (SETTLE needs it)."""
    oxygen = [i for i in fragment.atoms if model.atoms[i].element == "O"]
    hydrogens = [i for i in fragment.atoms if model.atoms[i].element == "H"]
    return oxygen + hydrogens


def from_rdkit(mol, infer_bonds: bool = True) -> SystemModel:
    """Read the host RDKit molecule (needs a conformer)."""
    if mol is None:
        raise ValueError("No molecule is loaded.")
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule has no 3D coordinates. Generate 3D first.")
    conf = mol.GetConformer()
    elements, coords, names, resnames, resnums, charges = [], [], [], [], [], []
    has_info = False
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if atom.HasProp("custom_symbol"):
            custom = atom.GetProp("custom_symbol").strip()
            if custom and custom[:2].isalpha():
                symbol = normalize_symbol(custom)
        if atom.GetAtomicNum() == 0:
            raise ValueError(
                f"Atom {idx + 1} is a dummy/ghost atom; remove it before writing an MD input."
            )
        pos = conf.GetAtomPosition(idx)
        elements.append(symbol)
        coords.append((pos.x, pos.y, pos.z))
        charges.append(atom.GetFormalCharge())
        info = atom.GetPDBResidueInfo() if hasattr(atom, "GetPDBResidueInfo") else None
        if info is not None:
            has_info = True
            names.append(info.GetName().strip())
            resnames.append(info.GetResidueName().strip())
            resnums.append(info.GetResidueNumber())
        else:
            names.append("")
            resnames.append("")
            resnums.append(0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return build_system(
        elements,
        coords,
        bonds,
        names=names,
        resnames=resnames if has_info else None,
        resnums=resnums if has_info else None,
        formal_charges=charges,
        infer_bonds=infer_bonds,
    )
