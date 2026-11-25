"""GROMACS .mdp files for the EM -> NVT -> NPT -> production chain."""

from __future__ import annotations

STAGES = ("em", "nvt", "npt", "md")
STAGE_TITLES = {
    "em": "Energy minimisation",
    "nvt": "NVT equilibration",
    "npt": "NPT equilibration",
    "md": "Production",
}

THERMOSTATS = ("V-rescale", "Nose-Hoover", "Berendsen")
BAROSTATS = ("C-rescale", "Parrinello-Rahman", "Berendsen")
CONSTRAINTS = ("h-bonds", "none", "all-bonds", "h-angles")
EM_INTEGRATORS = ("steep", "cg", "l-bfgs")
MD_INTEGRATORS = ("md", "sd")
COULOMB = ("PME", "Reaction-Field", "Cut-off")
PCOUPLTYPES = ("isotropic", "semiisotropic", "anisotropic")


def default_mdp_settings() -> dict:
    return {
        "stages": {"em": True, "nvt": True, "npt": True, "md": True},
        "temperature": 300.0,
        "pressure": 1.0,
        "dt": 0.002,
        "em_integrator": "steep",
        "em_steps": 50000,
        "emtol": 1000.0,
        "emstep": 0.01,
        "nvt_steps": 50000,
        "npt_steps": 50000,
        "md_steps": 500000,
        "md_integrator": "md",
        "thermostat": "V-rescale",
        "tau_t": 0.1,
        "barostat": "C-rescale",
        "tau_p": 2.0,
        "pcoupltype": "isotropic",
        "compressibility": 4.5e-5,
        "constraints": "h-bonds",
        "coulombtype": "PME",
        "rcoulomb": 1.0,
        "rvdw": 1.0,
        "dispcorr": True,
        "nstxout_compressed": 5000,
        "nstenergy": 1000,
        "nstlog": 1000,
        "tc_grps_split": False,
        "posres": False,
        "extra": {"em": "", "nvt": "", "npt": "", "md": ""},
    }


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _rows(rows: list[tuple[str, object, str]]) -> list[str]:
    out = []
    for key, value, comment in rows:
        if key.startswith(";"):
            out.append(key)
            continue
        text = f"{key:<24s}= {_fmt(value)}"
        if comment:
            text = f"{text:<44s}; {comment}"
        out.append(text)
    return out


def _tc_groups(settings: dict, groups_present: dict[str, bool]) -> tuple[str, int]:
    if settings.get("tc_grps_split") and groups_present.get("Water_and_ions") and groups_present.get("non-Water"):
        return "Water_and_ions non-Water", 2
    return "System", 1


def _nonbonded(settings: dict) -> list[tuple[str, object, str]]:
    rows = [
        ("; Neighbour searching and non-bonded interactions", None, ""),
        ("cutoff-scheme", "Verlet", ""),
        ("nstlist", 10, "Verlet buffer adjusts this automatically"),
        ("pbc", "xyz", ""),
        ("coulombtype", settings["coulombtype"], ""),
        ("rcoulomb", float(settings["rcoulomb"]), "nm"),
        ("vdwtype", "Cut-off", ""),
        ("vdw-modifier", "Potential-shift", ""),
        ("rvdw", float(settings["rvdw"]), "nm"),
    ]
    if settings["coulombtype"] == "PME":
        rows += [("fourierspacing", 0.12, "nm"), ("pme-order", 4, "")]
    if settings["dispcorr"]:
        rows.append(("DispCorr", "EnerPres", "long-range dispersion correction"))
    return rows


def stage_mdp(stage: str, settings: dict, groups_present=None, header: str = "",
              first_dynamics: bool | None = None) -> str:
    """Text of one .mdp file.

    ``first_dynamics`` marks the first MD stage of the chain, which generates
    velocities; later stages continue from the previous checkpoint.
    """
    if first_dynamics is None:
        first_dynamics = stage == "nvt"
    groups_present = groups_present or {}
    s = settings
    rows: list[tuple[str, object, str]] = []
    title = STAGE_TITLES[stage]
    lines = [f"; {title}"]
    if header:
        lines += [f"; {h}" for h in header.splitlines()]
    if stage == "em":
        rows += [
            ("integrator", s["em_integrator"], ""),
            ("nsteps", int(s["em_steps"]), "maximum number of minimisation steps"),
            ("emtol", float(s["emtol"]), "kJ/mol/nm, stop when Fmax < emtol"),
            ("emstep", float(s["emstep"]), "nm, initial step size"),
            ("nstenergy", 500, ""),
            ("nstlog", 500, ""),
        ]
        rows += _nonbonded(s)
        rows += [("constraints", "none", "unconstrained minimisation")]
    else:
        nsteps = int(s[f"{stage}_steps"])
        dt = float(s["dt"])
        tc, ngroups = _tc_groups(s, groups_present)
        integrator = s["md_integrator"]
        rows += [
            ("integrator", integrator, ""),
            ("nsteps", nsteps, f"{nsteps * dt / 1000.0:g} ns"),
            ("dt", dt, "ps"),
        ]
        if s.get("posres") and stage in ("nvt", "npt"):
            rows.insert(0, ("define", "-DPOSRES", "position restraints from posre.itp"))
        rows += [
            ("; Output", None, ""),
            ("nstxout-compressed", int(s["nstxout_compressed"]), "xtc frames"),
            ("nstenergy", int(s["nstenergy"]), ""),
            ("nstlog", int(s["nstlog"]), ""),
            ("nstcalcenergy", 100, ""),
        ]
        rows += _nonbonded(s)
        rows += [
            ("; Bonds", None, ""),
            ("constraints", s["constraints"], ""),
            ("constraint-algorithm", "lincs", ""),
            ("lincs-order", 4, ""),
            ("lincs-iter", 1, ""),
            ("; Temperature coupling", None, ""),
        ]
        if integrator == "sd":
            rows += [("tc-grps", tc, ""), ("tau-t", " ".join(["1.0"] * ngroups), "ps, sd friction"),
                     ("ref-t", " ".join([f"{float(s['temperature']):g}"] * ngroups), "K")]
        else:
            rows += [("tcoupl", s["thermostat"], ""), ("tc-grps", tc, ""),
                     ("tau-t", " ".join([f"{float(s['tau_t']):g}"] * ngroups), "ps"),
                     ("ref-t", " ".join([f"{float(s['temperature']):g}"] * ngroups), "K")]
        rows.append(("; Pressure coupling", None, ""))
        if stage == "nvt":
            rows.append(("pcoupl", "no", "constant volume"))
        else:
            barostat = s["barostat"]
            if stage == "npt" and barostat == "Parrinello-Rahman":
                barostat = "C-rescale"
                rows.append(("; Parrinello-Rahman is unstable far from equilibrium; C-rescale used here", None, ""))
            rows += [
                ("pcoupl", barostat, ""),
                ("pcoupltype", s["pcoupltype"], ""),
                ("tau-p", float(s["tau_p"]), "ps"),
                ("ref-p", _pressures(s), "bar"),
                ("compressibility", _compress(s), "1/bar (water)"),
                ("refcoord-scaling", "com", ""),
            ]
        rows.append(("; Velocities", None, ""))
        if first_dynamics:
            rows += [("continuation", "no", "first dynamics run"),
                     ("gen-vel", "yes", ""), ("gen-temp", float(s["temperature"]), "K"),
                     ("gen-seed", -1, "random seed")]
        else:
            rows += [("continuation", "yes", "continue from the previous stage"),
                     ("gen-vel", "no", "")]
    lines += _rows([r for r in rows])
    extra = (s.get("extra") or {}).get(stage, "").strip()
    if extra:
        lines += ["", "; User additions", extra]
    return "\n".join(lines) + "\n"


def _pressures(s) -> str:
    count = {"isotropic": 1, "semiisotropic": 2, "anisotropic": 6}[s["pcoupltype"]]
    if s["pcoupltype"] == "anisotropic":
        return " ".join([f"{float(s['pressure']):g}"] * 3 + ["0", "0", "0"])
    return " ".join([f"{float(s['pressure']):g}"] * count)


def _compress(s) -> str:
    value = f"{float(s['compressibility']):g}"
    if s["pcoupltype"] == "anisotropic":
        return " ".join([value] * 3 + ["0", "0", "0"])
    if s["pcoupltype"] == "semiisotropic":
        return f"{value} {value}"
    return value


def enabled_stages(settings: dict) -> list[str]:
    flags = settings.get("stages") or {}
    return [stage for stage in STAGES if flags.get(stage, True)]


def check_settings(settings: dict, min_box_width_nm: float = 0.0) -> list[str]:
    """Problems grompp would complain about, found early."""
    problems = []
    cutoff = max(float(settings["rcoulomb"]), float(settings["rvdw"]))
    if min_box_width_nm and cutoff * 2.0 > min_box_width_nm:
        problems.append(
            f"The cut-off ({cutoff:g} nm) is more than half the smallest box width "
            f"({min_box_width_nm:.3f} nm); grompp will refuse it. Enlarge the box or reduce the cut-off."
        )
    if settings["coulombtype"] == "PME" and abs(float(settings["rcoulomb"]) - float(settings["rvdw"])) > 1e-9:
        problems.append("With the Verlet scheme and PME, rcoulomb and rvdw should be equal.")
    if float(settings["dt"]) > 0.0021 and settings["constraints"] == "none":
        problems.append("A 2 fs time step needs constraints = h-bonds (or a smaller dt).")
    if float(settings["dt"]) > 0.0041:
        problems.append("Time steps above 4 fs are unstable without hydrogen mass repartitioning.")
    return problems
