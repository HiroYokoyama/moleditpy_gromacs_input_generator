"""Reader for the suite's force-field hand-off file (``*.mdff.json``, version 1).

Units are GROMACS units: nm, kJ/mol, degrees, e, amu.  Atom indices are 0-based
within a molecule type.  ``system`` lists the types in the host molecule's atom
order: ``{"type": name, "count": n, "atom_offset": k}``.
"""

from __future__ import annotations

import json
import math

FORMAT_NAME = "moleditpy-mdff"
SUPPORTED_VERSION = 1


class MdffError(ValueError):
    """The file is not a usable moleditpy-mdff document."""


def _num(value, what: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MdffError(f"{what} is not a number: {value!r}") from exc
    if not math.isfinite(result):
        raise MdffError(f"{what} is not finite")
    return result


def _idx(value, natoms: int, what: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise MdffError(f"{what} is not an atom index: {value!r}") from exc
    if not 0 <= index < natoms:
        raise MdffError(f"{what} = {index} is outside 0..{natoms - 1}")
    return index


def _settles(raw, natoms: int, name: str) -> list[dict]:
    """Accept {i, doh, dhh}, {oxygen, doh, dhh} or a list of those."""
    if raw in (None, [], {}):
        return []
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise MdffError(f"{name}: settles entries must be objects")
        oxygen = item.get("i", item.get("oxygen", 0))
        out.append(
            {
                "i": _idx(oxygen, natoms, f"{name} settles oxygen"),
                "doh": _num(item.get("doh"), f"{name} settles doh"),
                "dhh": _num(item.get("dhh"), f"{name} settles dhh"),
            }
        )
    return out


def _molecule_type(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise MdffError("molecule_types entries must be objects")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise MdffError("A molecule type has no name")
    atoms_raw = raw.get("atoms")
    if not isinstance(atoms_raw, list) or not atoms_raw:
        raise MdffError(f"{name}: 'atoms' must be a non-empty list")
    atoms = []
    for k, atom in enumerate(atoms_raw):
        if not isinstance(atom, dict):
            raise MdffError(f"{name}: atom {k} is not an object")
        label = f"{name} atom {k}"
        atoms.append(
            {
                "name": str(atom.get("name") or f"{atom.get('element', 'X')}{k + 1}"),
                "element": str(atom.get("element", "")).strip(),
                "type": str(atom.get("type") or atom.get("element") or "X").strip(),
                "charge": _num(atom.get("charge", 0.0), f"{label} charge"),
                "mass": _num(atom.get("mass", 0.0), f"{label} mass"),
                "sigma": _num(atom.get("sigma", 0.0), f"{label} sigma"),
                "epsilon": _num(atom.get("epsilon", 0.0), f"{label} epsilon"),
            }
        )
    n = len(atoms)
    bonds = [
        {
            "i": _idx(b.get("i"), n, f"{name} bond i"),
            "j": _idx(b.get("j"), n, f"{name} bond j"),
            "r0": _num(b.get("r0"), f"{name} bond r0"),
            "k": _num(b.get("k"), f"{name} bond k"),
        }
        for b in raw.get("bonds", []) or []
    ]
    angles = [
        {
            "i": _idx(a.get("i"), n, f"{name} angle i"),
            "j": _idx(a.get("j"), n, f"{name} angle j"),
            "k_atom": _idx(a.get("k_atom"), n, f"{name} angle k_atom"),
            "theta0": _num(a.get("theta0"), f"{name} angle theta0"),
            "k": _num(a.get("k"), f"{name} angle k"),
        }
        for a in raw.get("angles", []) or []
    ]
    torsions = []
    for t in raw.get("torsions", []) or []:
        torsions.append(
            {
                "i": _idx(t.get("i"), n, f"{name} torsion i"),
                "j": _idx(t.get("j"), n, f"{name} torsion j"),
                "k_atom": _idx(t.get("k_atom"), n, f"{name} torsion k_atom"),
                "l": _idx(t.get("l"), n, f"{name} torsion l"),
                "periodicity": int(_num(t.get("periodicity", 1), f"{name} torsion periodicity")),
                "phase": _num(t.get("phase", 0.0), f"{name} torsion phase"),
                "k": _num(t.get("k", 0.0), f"{name} torsion k"),
                "improper": bool(t.get("improper", False)),
            }
        )
    constraints = []
    for c in raw.get("constraints", []) or []:
        length = c.get("r0", c.get("length", c.get("b0")))
        constraints.append(
            {
                "i": _idx(c.get("i"), n, f"{name} constraint i"),
                "j": _idx(c.get("j"), n, f"{name} constraint j"),
                "r0": _num(length, f"{name} constraint length"),
            }
        )
    nrexcl = raw.get("exclusions_nrexcl", 3)
    try:
        nrexcl = int(nrexcl)
    except (TypeError, ValueError):
        raise MdffError(f"{name}: exclusions_nrexcl must be an integer") from None
    return {
        "name": name,
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
        "torsions": torsions,
        "constraints": constraints,
        "settles": _settles(raw.get("settles"), n, name),
        "exclusions_nrexcl": nrexcl,
    }


def parse_mdff(data) -> dict:
    """Validate a parsed JSON document and return a normalised copy."""
    if not isinstance(data, dict):
        raise MdffError("The file is not a JSON object")
    if data.get("format") != FORMAT_NAME:
        raise MdffError(f"'format' must be {FORMAT_NAME!r} (got {data.get('format')!r})")
    try:
        version = int(data.get("version"))
    except (TypeError, ValueError):
        raise MdffError("'version' is missing") from None
    if version != SUPPORTED_VERSION:
        raise MdffError(f"Unsupported moleditpy-mdff version {version}; this plugin reads version 1")
    rule = int(_num(data.get("combination_rule", 2), "combination_rule"))
    if rule not in (1, 2, 3):
        raise MdffError("combination_rule must be 1, 2 or 3")
    types_raw = data.get("molecule_types")
    if not isinstance(types_raw, list) or not types_raw:
        raise MdffError("'molecule_types' must be a non-empty list")
    types = [_molecule_type(t) for t in types_raw]
    by_name: dict[str, dict] = {}
    for mt in types:
        if mt["name"] in by_name:
            raise MdffError(f"Molecule type {mt['name']!r} is defined twice")
        by_name[mt["name"]] = mt
    system = []
    cursor = 0
    for entry in data.get("system", []) or []:
        if not isinstance(entry, dict):
            raise MdffError("'system' entries must be objects")
        type_name = str(entry.get("type", ""))
        if type_name not in by_name:
            raise MdffError(f"system refers to unknown molecule type {type_name!r}")
        count = int(_num(entry.get("count", 1), "system count"))
        if count < 1:
            raise MdffError("system count must be at least 1")
        offset = entry.get("atom_offset")
        offset = cursor if offset is None else int(_num(offset, "system atom_offset"))
        if offset < cursor:
            raise MdffError(
                f"system entry {type_name!r} at atom_offset {offset} overlaps the previous entry"
            )
        system.append({"type": type_name, "count": count, "atom_offset": offset})
        cursor = offset + count * len(by_name[type_name]["atoms"])
    return {
        "format": FORMAT_NAME,
        "version": version,
        "source": str(data.get("source", "")),
        "combination_rule": rule,
        "fudge_lj": _num(data.get("fudge_lj", 0.5), "fudge_lj"),
        "fudge_qq": _num(data.get("fudge_qq", 0.8333), "fudge_qq"),
        "molecule_types": types,
        "types_by_name": by_name,
        "system": system,
    }


def load_mdff(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise MdffError(f"Cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise MdffError(f"{path} is not valid JSON: {exc}") from exc
    return parse_mdff(data)


def coverage(doc: dict, natoms: int) -> tuple[list[tuple[int, int, str]], list[str]]:
    """Atom spans the document covers: [(start, end_exclusive, type)], warnings."""
    spans = []
    warnings: list[str] = []
    for entry in doc["system"]:
        size = len(doc["types_by_name"][entry["type"]]["atoms"])
        for copy in range(entry["count"]):
            start = entry["atom_offset"] + copy * size
            spans.append((start, start + size, entry["type"]))
    if spans and spans[-1][1] > natoms:
        warnings.append(
            f"The force-field file describes {spans[-1][1]} atoms but the molecule has {natoms}."
        )
        spans = [s for s in spans if s[1] <= natoms]
    return spans, warnings


def element_mismatches(doc: dict, spans, elements) -> list[str]:
    """Compare mdff atom elements with the host atoms they are mapped onto."""
    problems = []
    for start, _end, type_name in spans:
        mt = doc["types_by_name"][type_name]
        for k, atom in enumerate(mt["atoms"]):
            want = atom["element"].capitalize()
            have = elements[start + k]
            if want and want != have:
                problems.append(
                    f"{type_name} atom {k} ({atom['name']}) is {want} in the force-field "
                    f"file but host atom {start + k + 1} is {have}"
                )
                if len(problems) >= 10:
                    return problems
    return problems


def find_type_for(doc: dict | None, name: str) -> dict | None:
    if not doc:
        return None
    return doc["types_by_name"].get(name)
