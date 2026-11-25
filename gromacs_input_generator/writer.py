"""Assemble the full set of GROMACS files from a system, a box and settings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import md_cell, mdp, run_scripts
from .ff_tables import PROTEIN_RESIDUE_NAMES
from .topology import Topology, build_topology, itp_filename, write_itp, write_top

POSRES_FORCE = 1000.0  # kJ/mol/nm^2


def default_settings() -> dict:
    return {
        "title": "MoleditPy system",
        "basename": "conf",
        "mdff_path": "",
        "water_model": "TIP3P",
        "box_source": "auto",  # auto, cell, cif, padding, lengths
        "padding": 1.0,  # nm
        "cubic": False,
        "lengths": [3.0, 3.0, 3.0],  # nm
        "angles": [90.0, 90.0, 90.0],
        "write_index": True,
        "write_scripts": True,
        "mdp": mdp.default_mdp_settings(),
    }


def merge_settings(base: dict, update: dict) -> dict:
    """Recursive update that keeps unknown keys out and nested dicts whole."""
    result = dict(base)
    for key, value in (update or {}).items():
        if key not in base:
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            result[key] = merge_settings(base[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Box:
    lattice: np.ndarray  # Angstrom, rows, any orientation
    origin: np.ndarray
    source: str


@dataclass
class Bundle:
    files: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    topology: Topology | None = None
    box: np.ndarray | None = None  # GROMACS-setting lattice, Angstrom

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------
# box
# --------------------------------------------------------------------------


def resolve_box(settings: dict, coords, cell_prop=None, cif_lattice=None) -> Box:
    """Pick the box from the settings and what is available.

    ``auto`` = the molecule's moleditpy_cell, else the CIF Viewer cell, else padding.
    """
    source = settings.get("box_source", "auto")
    if source in ("auto", "cell") and cell_prop is not None:
        return Box(np.asarray(cell_prop["lattice"], float), np.asarray(cell_prop["origin"], float),
                   "moleditpy_cell")
    if source == "cell":
        raise md_cell.CellError("The molecule has no moleditpy_cell property.")
    if source in ("auto", "cif") and cif_lattice is not None:
        return Box(np.asarray(cif_lattice, float), np.zeros(3), "CIF Viewer cell")
    if source == "cif":
        raise md_cell.CellError("The CIF Viewer panel has no cell loaded.")
    if source == "lengths":
        lattice = md_cell.box_from_lengths([10.0 * float(v) for v in settings["lengths"]],
                                           settings["angles"])
        coords = np.asarray(coords, float).reshape(-1, 3)
        centre = coords.mean(axis=0) if len(coords) else np.zeros(3)
        origin = centre - 0.5 * lattice.sum(axis=0)
        return Box(lattice, origin, "explicit lengths")
    lattice, origin = md_cell.box_from_padding(coords, 10.0 * float(settings["padding"]),
                                               bool(settings.get("cubic")))
    return Box(lattice, origin, "bounding box + padding")


def place_in_box(coords, box: Box):
    """Coordinates shifted to the cell origin and rotated to the GROMACS setting."""
    gmx_box, rotation = md_cell.gromacs_setting(box.lattice)
    shifted = np.asarray(coords, float).reshape(-1, 3) - box.origin
    return shifted @ rotation, gmx_box


# --------------------------------------------------------------------------
# .gro and .ndx
# --------------------------------------------------------------------------


def write_gro(topo: Topology, coords_angstrom, box_angstrom, title: str) -> str:
    coords = np.asarray(coords_angstrom, float) / 10.0
    lines = [title or "MoleditPy system", f"{len(topo.order):5d}"]
    serial = 0
    residue = 0
    for block in topo.blocks:
        mt = topo.moltypes[block.moltype]
        last = None
        for k, host in enumerate(block.atoms):
            atom = mt.atoms[k]
            if atom["resnr"] != last:
                residue += 1
                last = atom["resnr"]
            serial += 1
            x, y, z = coords[host]
            lines.append(
                f"{residue % 100000:5d}{atom['resname'][:5]:<5s}{atom['name'][:5]:>5s}"
                f"{serial % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
            )
    lines.append(md_cell.gro_box_line(box_angstrom))
    return "\n".join(lines) + "\n"


def index_groups(topo: Topology) -> dict[str, list[int]]:
    """1-based serials in topology order: System, Protein, Water_and_ions, Other, ..."""
    groups: dict[str, list[int]] = {"System": [], "Protein": [], "Water": [], "Ion": [],
                                    "Water_and_ions": [], "Other": [], "non-Water": []}
    serial = 0
    for block in topo.blocks:
        mt = topo.moltypes[block.moltype]
        for k in range(len(block.atoms)):
            serial += 1
            groups["System"].append(serial)
            resname = mt.atoms[k]["resname"].upper()
            if mt.source == "water":
                kind = "Water"
            elif mt.source == "ion":
                kind = "Ion"
            elif resname in PROTEIN_RESIDUE_NAMES:
                kind = "Protein"
            elif mt.size == 3 and resname in ("SOL", "HOH", "WAT", "TIP3"):
                kind = "Water"
            elif mt.size == 1 and mt.atoms[0]["element"] in ("Na", "Cl", "K", "Li", "Mg", "Ca", "Br", "I", "F", "Rb", "Cs", "Zn", "Sr", "Ba"):
                kind = "Ion"
            else:
                kind = "Other"
            groups[kind].append(serial)
            if kind in ("Water", "Ion"):
                groups["Water_and_ions"].append(serial)
            if kind != "Water":
                groups["non-Water"].append(serial)
    return {name: members for name, members in groups.items() if members}


def write_ndx(groups: dict[str, list[int]]) -> str:
    lines = []
    for name, members in groups.items():
        lines.append(f"[ {name} ]")
        for start in range(0, len(members), 15):
            lines.append(" ".join(f"{m:6d}" for m in members[start:start + 15]))
    return "\n".join(lines) + "\n"


def write_posre(mt) -> str:
    lines = [f"; Position restraints on the heavy atoms of {mt.name}", "[ position_restraints ]",
             ";  ai  funct       fcx       fcy       fcz"]
    for k, atom in enumerate(mt.atoms, start=1):
        if atom["element"] != "H":
            lines.append(f"{k:5d}      1 {POSRES_FORCE:9.1f} {POSRES_FORCE:9.1f} {POSRES_FORCE:9.1f}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------


def build_bundle(model, settings: dict, mdff_doc=None, cell_prop=None, cif_lattice=None,
                 header: str = "") -> Bundle:
    settings = merge_settings(default_settings(), settings or {})
    files: dict[str, str] = {}
    errors: list[str] = []
    try:
        box = resolve_box(settings, model.coords, cell_prop, cif_lattice)
        coords, gmx_box = place_in_box(model.coords, box)
    except (md_cell.CellError, ValueError) as exc:
        return Bundle(files={}, errors=[str(exc)])

    topo = build_topology(model, mdff_doc, settings["water_model"])
    warnings = list(topo.warnings)
    warnings.insert(0, f"Box: {box.source}; {md_cell.describe(gmx_box)}")

    base = (settings.get("basename") or "conf").strip() or "conf"
    conf = f"{base}.gro"
    files[conf] = write_gro(topo, coords, gmx_box, settings.get("title", ""))

    mdp_settings = settings["mdp"]
    posres = bool(mdp_settings.get("posres"))
    files["topol.top"] = write_top(topo, settings.get("title", ""), header)
    for name, mt in topo.moltypes.items():
        text = write_itp(mt, topo.gen_pairs)
        if posres and mt.source == "mdff" and mt.settles is None and mt.size > 1:
            posre_name = f"posre_{itp_filename(name)}"
            files[posre_name] = write_posre(mt)
            text += f'\n#ifdef POSRES\n#include "{posre_name}"\n#endif\n'
        files[itp_filename(name)] = text

    groups = index_groups(topo)
    groups_present = {name: True for name in groups}
    index_name = ""
    if settings.get("write_index") and len(groups) > 1:
        index_name = "index.ndx"
        files[index_name] = write_ndx(groups)
    elif mdp_settings.get("tc_grps_split"):
        warnings.append("Separate thermostat groups need the index file; using tc-grps = System.")

    stages = mdp.enabled_stages(mdp_settings)
    first_dyn = next((s for s in stages if s != "em"), None)
    for stage in stages:
        files[f"{stage}.mdp"] = mdp.stage_mdp(
            stage, mdp_settings, groups_present if index_name else {}, header,
            first_dynamics=(stage == first_dyn),
        )
    if settings.get("write_scripts") and stages:
        files["run.sh"] = run_scripts.run_sh(stages, conf, "topol.top", index_name, posres)
        files["run.bat"] = run_scripts.run_bat(stages, conf, "topol.top", index_name, posres)

    width_nm = md_cell.minimum_box_width(gmx_box) / 10.0
    if stages:
        warnings.extend(mdp.check_settings(mdp_settings, width_nm))
    return Bundle(files=files, warnings=warnings, errors=errors, topology=topo, box=gmx_box)


def save_bundle(bundle: Bundle, directory: str) -> list[str]:
    import os

    os.makedirs(directory, exist_ok=True)
    written = []
    for name, text in bundle.files.items():
        path = os.path.join(directory, name)
        newline = "\r\n" if name.endswith(".bat") else "\n"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text.replace("\r\n", "\n").replace("\n", newline))
        if name.endswith(".sh"):
            try:
                os.chmod(path, 0o755)
            except OSError:  # pragma: no cover - Windows without POSIX modes
                pass
        written.append(path)
    return written
