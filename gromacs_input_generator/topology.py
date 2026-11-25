"""Assign molecule types to the host atoms and write .top / .itp text.

Sources, in order of preference:

1. a ``moleditpy-mdff`` document (atom spans from its ``system`` list),
2. built-in water models and monatomic ions (``ff_tables``),
3. a skeleton ``[ moleculetype ]`` with TODO markers for anything else.

GROMACS needs each molecule's atoms contiguous and in topology order, so the
output atom order (``Topology.order``) can differ from the host order; the
writer uses it for the .gro and the index file.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import ff_tables
from .elements import atomic_number
from .mdff import coverage, element_mismatches
from .system_model import SystemModel, water_order


@dataclass
class MolType:
    name: str
    source: str  # "mdff", "water", "ion", "skeleton"
    atoms: list[dict]  # name, type, charge, mass, resnr, resname, element
    nrexcl: int = 3
    bonds: list[tuple[int, int, float, float]] = field(default_factory=list)
    angles: list[tuple[int, int, int, float, float]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int, int, float, float, int]] = field(default_factory=list)
    constraints: list[tuple[int, int, float]] = field(default_factory=list)
    settles: tuple[int, float, float] | None = None
    flexible_water: dict | None = None
    connectivity: list[tuple[int, int]] = field(default_factory=list)  # skeleton only
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.atoms)

    @property
    def charge(self) -> float:
        return sum(a["charge"] for a in self.atoms)

    @property
    def is_todo(self) -> bool:
        return self.source == "skeleton"


@dataclass
class Block:
    """One molecule instance: its type and host atom indices in topology order."""

    moltype: str
    atoms: list[int]


@dataclass
class Topology:
    moltypes: dict[str, MolType]
    blocks: list[Block]
    atomtypes: dict[str, dict]
    combination_rule: int = 2
    fudge_lj: float = 0.5
    fudge_qq: float = 0.8333
    gen_pairs: bool = True
    warnings: list[str] = field(default_factory=list)
    source_note: str = ""

    @property
    def order(self) -> list[int]:
        return [i for block in self.blocks for i in block.atoms]

    @property
    def reordered(self) -> bool:
        order = self.order
        return order != sorted(order)

    def molecules(self) -> list[tuple[str, int]]:
        """[ molecules ] rows: consecutive copies collapsed."""
        rows: list[list] = []
        for block in self.blocks:
            if rows and rows[-1][0] == block.moltype:
                rows[-1][1] += 1
            else:
                rows.append([block.moltype, 1])
        return [(name, count) for name, count in rows]

    @property
    def total_charge(self) -> float:
        return sum(self.moltypes[b.moltype].charge for b in self.blocks)

    @property
    def has_todo(self) -> bool:
        return any(self.moltypes[b.moltype].is_todo for b in self.blocks)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^A-Za-z0-9_+\-]")


def _clean_name(text: str, fallback: str) -> str:
    text = _NAME_RE.sub("", text or "")
    return text or fallback


def _unique(name: str, taken) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}{k}" in taken:
        k += 1
    return f"{name}{k}"


def _residue_fields(model: SystemModel, indices: Sequence[int], default_resname: str):
    """Relative residue numbers and names for one instance (first copy)."""
    if not model.has_residue_info:
        return [1] * len(indices), [default_resname[:5]] * len(indices)
    numbers, names = [], []
    mapping: dict[tuple[int, str], int] = {}
    for i in indices:
        atom = model.atoms[i]
        key = (atom.resnum, atom.resname)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        numbers.append(mapping[key])
        names.append((atom.resname or default_resname)[:5])
    return numbers, names


def _atom_names(model: SystemModel, indices: Sequence[int]) -> list[str]:
    names = []
    counts: dict[str, int] = {}
    for i in indices:
        atom = model.atoms[i]
        if atom.name:
            names.append(atom.name[:5])
            continue
        counts[atom.element] = counts.get(atom.element, 0) + 1
        names.append(f"{atom.element}{counts[atom.element]}"[:5])
    return names


def _add_atomtype(atomtypes: dict[str, dict], name: str, element: str, mass: float,
                  sigma: float, epsilon: float, warnings: list[str]) -> str:
    """Register an atom type; rename on a parameter clash."""
    entry = {"element": element, "mass": mass, "sigma": sigma, "epsilon": epsilon}
    current = atomtypes.get(name)
    if current is None:
        atomtypes[name] = entry
        return name
    if (abs(current["sigma"] - sigma) < 1e-9 and abs(current["epsilon"] - epsilon) < 1e-9):
        return name
    k = 2
    while f"{name}_{k}" in atomtypes:
        other = atomtypes[f"{name}_{k}"]
        if abs(other["sigma"] - sigma) < 1e-9 and abs(other["epsilon"] - epsilon) < 1e-9:
            return f"{name}_{k}"
        k += 1
    new_name = f"{name}_{k}"
    atomtypes[new_name] = entry
    warnings.append(
        f"Atom type {name} appears with two different LJ parameter sets; the second "
        f"was renamed {new_name}."
    )
    return new_name


def _moltype_from_mdff(mt: dict, model: SystemModel, start: int, atomtypes, warnings) -> MolType:
    indices = list(range(start, start + len(mt["atoms"])))
    resnrs, resnames = _residue_fields(model, indices, mt["name"])
    atoms = []
    for k, atom in enumerate(mt["atoms"]):
        element = atom["element"].capitalize() or model.atoms[start + k].element
        mass = atom["mass"] or model.atoms[start + k].mass
        type_name = _add_atomtype(
            atomtypes, atom["type"], element, mass, atom["sigma"], atom["epsilon"], warnings
        )
        atoms.append(
            {
                "name": atom["name"][:5],
                "type": type_name,
                "charge": atom["charge"],
                "mass": mass,
                "resnr": resnrs[k],
                "resname": resnames[k],
                "element": element,
            }
        )
    result = MolType(name=mt["name"], source="mdff", atoms=atoms, nrexcl=mt["exclusions_nrexcl"])
    result.bonds = [(b["i"], b["j"], b["r0"], b["k"]) for b in mt["bonds"]]
    result.angles = [(a["i"], a["j"], a["k_atom"], a["theta0"], a["k"]) for a in mt["angles"]]
    for t in mt["torsions"]:
        funct = 4 if t["improper"] else 9
        result.dihedrals.append(
            (t["i"], t["j"], t["k_atom"], t["l"], funct, t["phase"], t["k"], t["periodicity"])
        )
    bonded = {frozenset((b[0], b[1])) for b in result.bonds}
    result.constraints = [
        (c["i"], c["j"], c["r0"]) for c in mt["constraints"]
        if frozenset((c["i"], c["j"])) not in bonded
    ]
    if mt["settles"]:
        s = mt["settles"][0]
        result.settles = (s["i"], s["doh"], s["dhh"])
    return result


def _water_moltype(model_name: str, atomtypes, warnings) -> MolType:
    w = ff_tables.WATER_MODELS[model_name]
    o_type = _add_atomtype(atomtypes, w["o_type"], "O", w["o_mass"], w["o_sigma"], w["o_epsilon"], warnings)
    h_type = _add_atomtype(atomtypes, w["h_type"], "H", w["h_mass"], 0.0, 0.0, warnings)
    res = ff_tables.WATER_RESNAME
    atoms = [
        {"name": "OW", "type": o_type, "charge": w["o_charge"], "mass": w["o_mass"],
         "resnr": 1, "resname": res, "element": "O"},
        {"name": "HW1", "type": h_type, "charge": w["h_charge"], "mass": w["h_mass"],
         "resnr": 1, "resname": res, "element": "H"},
        {"name": "HW2", "type": h_type, "charge": w["h_charge"], "mass": w["h_mass"],
         "resnr": 1, "resname": res, "element": "H"},
    ]
    mt = MolType(name=res, source="water", atoms=atoms, nrexcl=2)
    mt.settles = (0, w["doh"], w["dhh"])
    mt.flexible_water = w
    mt.notes.append(f"{w['label']} water (rigid, SETTLE); define -DFLEXIBLE for a flexible model.")
    return mt


def _ion_moltype(element: str, water_model: str, atomtypes, warnings) -> MolType | None:
    params = ff_tables.ion_parameters(element, water_model)
    if params is None:
        return None
    type_name = _add_atomtype(
        atomtypes, params["type"], element, params["mass"], params["sigma"], params["epsilon"], warnings
    )
    atoms = [{"name": params["name"], "type": type_name, "charge": params["charge"],
              "mass": params["mass"], "resnr": 1, "resname": params["name"], "element": element}]
    mt = MolType(name=params["name"], source="ion", atoms=atoms, nrexcl=1)
    mt.notes.append(f"Joung-Cheatham/Li-Merz ion parameters for {water_model} water.")
    return mt


def _skeleton_moltype(name: str, model: SystemModel, indices: Sequence[int]) -> MolType:
    resnrs, resnames = _residue_fields(model, indices, name)
    names = _atom_names(model, indices)
    local = {host: k for k, host in enumerate(indices)}
    atoms = []
    for k, host in enumerate(indices):
        atom = model.atoms[host]
        atoms.append({
            "name": names[k], "type": f"TODO_{atom.element}", "charge": float(atom.formal_charge),
            "mass": atom.mass, "resnr": resnrs[k], "resname": resnames[k], "element": atom.element,
        })
    mt = MolType(name=name, source="skeleton", atoms=atoms, nrexcl=3)
    mt.connectivity = sorted(
        (local[i], local[j]) for i, j in model.bonds if i in local and j in local
    )
    return mt


def build_topology(model: SystemModel, mdff_doc: dict | None = None,
                   water_model: str = "TIP3P") -> Topology:
    warnings: list[str] = []
    atomtypes: dict[str, dict] = {}
    moltypes: dict[str, MolType] = {}
    blocks: list[Block] = []
    natoms = len(model.atoms)
    covered = [False] * natoms
    rule, fudge_lj, fudge_qq = 2, 0.5, 0.8333
    source_note = "built-in tables"

    if mdff_doc:
        rule = mdff_doc["combination_rule"]
        fudge_lj, fudge_qq = mdff_doc["fudge_lj"], mdff_doc["fudge_qq"]
        source_note = f"moleditpy-mdff ({mdff_doc.get('source') or 'unknown source'})"
        spans, cov_warnings = coverage(mdff_doc, natoms)
        warnings.extend(cov_warnings)
        warnings.extend(element_mismatches(mdff_doc, spans, model.elements))
        for start, end, type_name in spans:
            if type_name not in moltypes:
                moltypes[type_name] = _moltype_from_mdff(
                    mdff_doc["types_by_name"][type_name], model, start, atomtypes, warnings
                )
            blocks.append(Block(type_name, list(range(start, end))))
            for i in range(start, end):
                covered[i] = True

    by_signature: dict[tuple, str] = {}
    skeleton_count = 0
    for fragment in model.fragments:
        todo = [i for i in fragment.atoms if not covered[i]]
        if not todo:
            continue
        partial = len(todo) != fragment.size
        if partial:
            warnings.append(
                f"Atoms {todo[0] + 1}..{todo[-1] + 1} are bonded to atoms the force-field file "
                "covers but are not in it themselves; they get a TODO skeleton."
            )
        if not partial and fragment.kind == "water":
            name = ff_tables.WATER_RESNAME
            if name in moltypes and moltypes[name].source != "water":
                name = _unique("SOL", moltypes)
            if name not in moltypes:
                moltypes[name] = _water_moltype(water_model, atomtypes, warnings)
                moltypes[name].name = name
            blocks.append(Block(name, water_order(model, fragment)))
            continue
        if not partial and fragment.kind == "ion":
            element = model.atoms[fragment.atoms[0]].element
            ion = _ion_moltype(element, water_model, atomtypes, warnings)
            if ion is not None:
                if ion.name not in moltypes:
                    moltypes[ion.name] = ion
                formal = model.atoms[fragment.atoms[0]].formal_charge
                if formal and abs(formal - ion.charge) > 1e-6:
                    warnings.append(
                        f"Atom {fragment.atoms[0] + 1} ({element}) has formal charge {formal:+d} "
                        f"but the built-in ion carries {ion.charge:+.0f}."
                    )
                blocks.append(Block(ion.name, list(fragment.atoms)))
                continue
        key = fragment.signature if not partial else ("partial", tuple(todo))
        if key in by_signature and not partial:
            blocks.append(Block(by_signature[key], list(todo)))
            continue
        resnames = {model.atoms[i].resname for i in todo if model.atoms[i].resname}
        base = _clean_name(next(iter(resnames)) if len(resnames) == 1 else "", "MOL")
        skeleton_count += 1
        name = _unique(base, moltypes)
        moltypes[name] = _skeleton_moltype(name, model, todo)
        by_signature[key] = name
        blocks.append(Block(name, list(todo)))

    blocks.sort(key=lambda b: min(b.atoms))
    topo = Topology(
        moltypes=moltypes, blocks=blocks, atomtypes=atomtypes, combination_rule=rule,
        fudge_lj=fudge_lj, fudge_qq=fudge_qq, warnings=warnings, source_note=source_note,
    )
    if topo.reordered:
        warnings.append(
            "Atoms were reordered so each molecule is contiguous (GROMACS needs this); "
            "conf.gro and index.ndx use the new order."
        )
    if skeleton_count:
        names = sorted(n for n, m in moltypes.items() if m.is_todo)
        warnings.append(
            "No parameters for: " + ", ".join(names) + ". Their .itp files are skeletons "
            "with TODO markers; build parameters with the Force Field Builder plugin "
            "(moleditpy-mdff JSON) and load that file here."
        )
    if not topo.has_todo and abs(topo.total_charge) > 1e-3:
        warnings.append(f"The system is not neutral (net charge {topo.total_charge:+.4f} e); PME will add a uniform background.")
    return topo


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _lj_columns(sigma: float, epsilon: float, rule: int) -> tuple[float, float]:
    if rule == 1:
        return 4.0 * epsilon * sigma ** 6, 4.0 * epsilon * sigma ** 12
    return sigma, epsilon


def pairs_14(n: int, edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Atom pairs exactly three bonds apart (and not closer)."""
    neighbours: dict[int, set] = {i: set() for i in range(n)}
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)
    pairs = set()
    for a in range(n):
        dist = {a: 0}
        frontier = [a]
        for depth in (1, 2, 3):
            nxt = []
            for x in frontier:
                for y in neighbours[x]:
                    if y not in dist:
                        dist[y] = depth
                        nxt.append(y)
            frontier = nxt
        for b, d in dist.items():
            if d == 3 and a < b:
                pairs.add((a, b))
    return sorted(pairs)


def itp_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".itp"


def write_itp(mt: MolType, gen_pairs: bool = True) -> str:
    lines = []
    if mt.is_todo:
        lines += [
            f"; TODO: {mt.name} has NO force-field parameters yet.",
            "; This skeleton lists atoms and connectivity only. grompp will stop on",
            "; the undefined TODO_* atom types and parameterless bonds until you",
            "; replace it: build parameters with the Force Field Builder plugin,",
            "; load its .mdff.json file in the GROMACS Input Generator and write",
            "; again, or paste an .itp made by another tool here.",
            "",
        ]
    for note in mt.notes:
        lines.append(f"; {note}")
    lines += ["[ moleculetype ]", "; name  nrexcl", f"{mt.name:<10s} {mt.nrexcl}", "", "[ atoms ]",
              ";   nr  type        resnr residue  atom   cgnr    charge       mass"]
    for k, atom in enumerate(mt.atoms, start=1):
        lines.append(
            f"{k:6d}  {atom['type']:<11s} {atom['resnr']:5d} {atom['resname']:<7s} "
            f"{atom['name']:<6s} {k:5d} {atom['charge']:10.6f} {atom['mass']:10.5f}"
        )
    if mt.source == "water" and mt.settles is not None:
        w = mt.flexible_water
        lines += ["", "#ifndef FLEXIBLE", "", "[ settles ]", "; OW  funct  doh  dhh",
                  f"1  1  {mt.settles[1]:.5f}  {mt.settles[2]:.5f}", "", "[ exclusions ]",
                  "1  2  3", "2  1  3", "3  1  2", "", "#else", "", "[ bonds ]",
                  f"1  2  1  {w['doh']:.5f}  {w['flexible_kb']:.1f}",
                  f"1  3  1  {w['doh']:.5f}  {w['flexible_kb']:.1f}", "", "[ angles ]",
                  f"2  1  3  1  {w['hoh_angle']:.2f}  {w['flexible_ka']:.1f}", "", "#endif"]
        return "\n".join(lines) + "\n"
    if mt.is_todo:
        if mt.connectivity:
            lines += ["", "[ bonds ]", "; TODO: add funct, b0 (nm) and kb (kJ/mol/nm^2) to every line"]
            lines += [f"{i + 1:5d} {j + 1:5d}" for i, j in mt.connectivity]
        return "\n".join(lines) + "\n"
    if mt.settles is not None:
        o = mt.settles[0]
        others = [k for k in range(mt.size) if k != o]
        lines += ["", "#ifndef FLEXIBLE", "", "[ settles ]", "; OW  funct  doh  dhh",
                  f"{o + 1}  1  {mt.settles[1]:.5f}  {mt.settles[2]:.5f}", "", "[ exclusions ]"]
        members = [o] + others
        for a in members:
            lines.append("  ".join(str(x + 1) for x in [a] + [m for m in members if m != a]))
        lines += ["", "#else"]
    if mt.bonds:
        lines += ["", "[ bonds ]", ";  ai    aj funct   b0 (nm)       kb (kJ/mol/nm^2)"]
        lines += [f"{i + 1:5d} {j + 1:5d}     1 {r0:12.6f} {k:16.4f}" for i, j, r0, k in mt.bonds]
    if mt.constraints:
        lines += ["", "[ constraints ]", ";  ai    aj funct   b0 (nm)"]
        lines += [f"{i + 1:5d} {j + 1:5d}     1 {r0:12.6f}" for i, j, r0 in mt.constraints]
    edges = [(b[0], b[1]) for b in mt.bonds] + [(c[0], c[1]) for c in mt.constraints]
    if gen_pairs and mt.nrexcl >= 3:
        pairs = pairs_14(mt.size, edges)
        if pairs:
            lines += ["", "[ pairs ]", ";  ai    aj funct"]
            lines += [f"{i + 1:5d} {j + 1:5d}     1" for i, j in pairs]
    if mt.angles:
        lines += ["", "[ angles ]", ";  ai    aj    ak funct   theta0 (deg)   k (kJ/mol/rad^2)"]
        lines += [f"{i + 1:5d} {j + 1:5d} {k + 1:5d}     1 {t0:12.4f} {kk:14.4f}"
                  for i, j, k, t0, kk in mt.angles]
    propers = [d for d in mt.dihedrals if d[4] == 9]
    impropers = [d for d in mt.dihedrals if d[4] == 4]
    for title, rows in (("proper (periodic, funct 9)", propers), ("improper (periodic, funct 4)", impropers)):
        if rows:
            lines += ["", "[ dihedrals ]", f"; {title}: ai aj ak al funct phase(deg) k(kJ/mol) mult"]
            lines += [f"{i + 1:5d} {j + 1:5d} {k + 1:5d} {m + 1:5d} {f:5d} {ph:10.4f} {kk:12.6f} {n:3d}"
                      for i, j, k, m, f, ph, kk, n in rows]
    if mt.settles is not None:
        lines += ["", "#endif"]
    return "\n".join(lines) + "\n"


def write_top(topo: Topology, title: str = "MoleditPy system", header: str = "") -> str:
    lines = []
    if header:
        lines += [f"; {h}" for h in header.splitlines()]
    lines += [f"; Parameters: {topo.source_note}", ""]
    if topo.has_todo:
        lines += ["; TODO: this topology includes skeleton molecule types without",
                  "; parameters (see the TODO_* .itp files). grompp will fail until",
                  "; they are replaced, e.g. with Force Field Builder output.", ""]
    lines += ["[ defaults ]", "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
              (f"  1       {topo.combination_rule}          {'yes' if topo.gen_pairs else 'no'}"
               f"        {topo.fudge_lj:g}      {topo.fudge_qq:g}"), "", "[ atomtypes ]"]
    if topo.combination_rule == 1:
        lines.append("; name        at.num   mass       charge  ptype  c6              c12")
    else:
        lines.append("; name        at.num   mass       charge  ptype  sigma (nm)      epsilon (kJ/mol)")
    for name, entry in topo.atomtypes.items():
        v, w = _lj_columns(entry["sigma"], entry["epsilon"], topo.combination_rule)
        number = atomic_number(entry["element"]) or 0
        lines.append(f"{name:<13s} {number:4d}  {entry['mass']:10.5f}  0.0000  A  {v:16.8e} {w:16.8e}")
    lines.append("")
    used = []
    for block in topo.blocks:
        if block.moltype not in used:
            used.append(block.moltype)
    for name in used:
        mt = topo.moltypes[name]
        if mt.is_todo:
            lines.append(f"; TODO: {itp_filename(name)} is a skeleton without parameters")
        lines.append(f'#include "{itp_filename(name)}"')
    lines += ["", "[ system ]", title or "MoleditPy system", "", "[ molecules ]", "; name      count"]
    lines += [f"{name:<11s} {count}" for name, count in topo.molecules()]
    return "\n".join(lines) + "\n"
