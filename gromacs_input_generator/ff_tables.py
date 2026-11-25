"""Built-in water models and monatomic ions (GROMACS units: nm, kJ/mol, e, amu).

Water and ion Lennard-Jones parameters are copied digit for digit from OpenMM's
``amber14/tip3p.xml`` and ``amber14/spce.xml`` (the Joung-Cheatham 2008
monovalent ions and the Li/Merz 12-6 "standard" divalent ions that Amber ships
for each water model).  Ion parameters are tied to the water model, so the
table is keyed by model.

Water geometry is rigid (SETTLE): TIP3P d(OH) = 0.09572 nm, angle 104.52 deg;
SPC/E d(OH) = 0.1 nm, angle 109.47 deg.  ``dhh`` is derived from those.
"""

from __future__ import annotations

import math

WATER_RESNAME = "SOL"


def _dhh(doh: float, angle_deg: float) -> float:
    return 2.0 * doh * math.sin(math.radians(angle_deg) / 2.0)


WATER_MODELS = {
    "TIP3P": {
        "label": "TIP3P",
        "o_type": "OW_tip3p",
        "h_type": "HW_tip3p",
        "o_charge": -0.834,
        "h_charge": 0.417,
        "o_sigma": 0.31507524065751241,
        "o_epsilon": 0.635968,
        "o_mass": 15.99943,
        "h_mass": 1.007947,
        "doh": 0.09572,
        "hoh_angle": 104.52,
        "dhh": _dhh(0.09572, 104.52),
        "flexible_kb": 462750.4,
        "flexible_ka": 836.8,
    },
    "SPC/E": {
        "label": "SPC/E",
        "o_type": "OW_spce",
        "h_type": "HW_spce",
        "o_charge": -0.8476,
        "h_charge": 0.4238,
        "o_sigma": 0.31657195050398818,
        "o_epsilon": 0.6497752,
        "o_mass": 15.99943,
        "h_mass": 1.007947,
        "doh": 0.1,
        "hoh_angle": 109.47,
        "dhh": _dhh(0.1, 109.47),
        "flexible_kb": 462750.4,
        "flexible_ka": 836.8,
    },
}

# element -> (charge, mass, sigma nm, epsilon kJ/mol) per water model.
_ION_MASSES = {
    "Li": 6.94, "Na": 22.99, "K": 39.1, "Rb": 85.47, "Cs": 132.91,
    "F": 19.0, "Cl": 35.45, "Br": 79.9, "I": 126.9,
    "Mg": 24.305, "Ca": 40.08, "Sr": 87.62, "Ba": 137.33, "Zn": 65.4,
}

_ION_LJ = {
    "TIP3P": {
        "Li": (1.0, 0.18263423721876956, 0.1171084864),
        "Na": (1.0, 0.2439280690268249, 0.3658460312),
        "K": (1.0, 0.3037964628858557, 0.8103692536),
        "Rb": (1.0, 0.32303987519768707, 1.3716068296000001),
        "Cs": (1.0, 0.3520831734090621, 1.7009608496),
        "F": (-1.0, 0.4103479495754403, 0.014074976),
        "Cl": (-1.0, 0.4477656957373345, 0.14891274399999999),
        "Br": (-1.0, 0.46469277138200105, 0.24541419360000002),
        "I": (-1.0, 0.5095940667762741, 0.22460381440000002),
        "Mg": (2.0, 0.2423244513341723, 0.042686716080000006),
        "Ca": (2.0, 0.29381839724268394, 0.4432056808),
        "Sr": (2.0, 0.32250533596680286, 0.92601852816),
        "Ba": (2.0, 0.35974490238506907, 1.7014071987200001),
        "Zn": (2.0, 0.22646645415127425, 0.01381916624),
    },
    "SPC/E": {
        "Li": (1.0, 0.14094017720980168, 1.4088967296000001),
        "Na": (1.0, 0.21595384927721822, 1.4754532912),
        "K": (1.0, 0.2838403315995121, 1.7978873936000002),
        "Rb": (1.0, 0.3094982146819539, 1.8623134624),
        "Cs": (1.0, 0.36010126187232516, 0.37595959600000006),
        "F": (-1.0, 0.4021516813685492, 0.030963692),
        "Cl": (-1.0, 0.48304528497569194, 0.05349244),
        "Br": (-1.0, 0.4901724747208147, 0.1127947824),
        "I": (-1.0, 0.5201066716503301, 0.179010348),
        "Mg": (2.0, 0.2423244513341723, 0.042686716080000006),
        "Ca": (2.0, 0.2913238808318909, 0.40953067312),
        "Sr": (2.0, 0.31965446006875375, 0.8713768270400001),
        "Ba": (2.0, 0.35279589238357434, 1.55336865968),
        "Zn": (2.0, 0.2273573528694146, 0.014823368080000001),
    },
}

# GROMACS-style molecule/residue names (amber ports use these).
ION_NAMES = {
    "Li": "LI", "Na": "NA", "K": "K", "Rb": "RB", "Cs": "CS",
    "F": "F", "Cl": "CL", "Br": "BR", "I": "I",
    "Mg": "MG", "Ca": "CA", "Sr": "SR", "Ba": "BA", "Zn": "ZN",
}

ION_ELEMENTS = tuple(ION_NAMES)


def ion_parameters(element: str, water_model: str):
    """Return a dict for a built-in ion, or None if the table has no entry."""
    table = _ION_LJ.get(water_model) or _ION_LJ["TIP3P"]
    entry = table.get(element)
    if entry is None:
        return None
    charge, sigma, epsilon = entry
    tag = "tip3p" if water_model == "TIP3P" else "spce"
    return {
        "name": ION_NAMES[element],
        "type": f"{ION_NAMES[element]}_{tag}",
        "element": element,
        "charge": charge,
        "mass": _ION_MASSES[element],
        "sigma": sigma,
        "epsilon": epsilon,
    }


WATER_RESIDUE_NAMES = frozenset({"SOL", "HOH", "WAT", "TIP3", "TIP3P", "SPC", "SPCE", "H2O", "T3P"})
ION_RESIDUE_NAMES = frozenset(
    {"NA", "NA+", "SOD", "CL", "CL-", "CLA", "K", "K+", "POT", "LI", "RB", "CS", "CES",
     "F", "BR", "I", "MG", "CA", "CAL", "SR", "BA", "ZN", "ZN2"}
)
PROTEIN_RESIDUE_NAMES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
        "LYN", "ACE", "NME", "NMA", "SEC", "PYL", "MSE",
    }
)
