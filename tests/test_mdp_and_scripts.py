import re

import pytest

from gromacs_input_generator import mdp, run_scripts


def _value(text, key):
    match = re.search(rf"^{re.escape(key)}\s*=\s*([^;\n]*)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def test_em_stage():
    text = mdp.stage_mdp("em", mdp.default_mdp_settings())
    assert _value(text, "integrator") == "steep"
    assert _value(text, "nsteps") == "50000"
    assert _value(text, "constraints") == "none"
    assert _value(text, "tcoupl") is None


def test_first_dynamics_stage_generates_velocities_and_later_ones_continue():
    s = mdp.default_mdp_settings()
    nvt = mdp.stage_mdp("nvt", s)
    assert _value(nvt, "gen-vel") == "yes" and _value(nvt, "continuation") == "no"
    assert _value(nvt, "pcoupl") == "no"
    npt = mdp.stage_mdp("npt", s)
    assert _value(npt, "gen-vel") == "no" and _value(npt, "continuation") == "yes"
    assert _value(npt, "pcoupl") == "C-rescale"
    # without an NVT stage the NPT stage is the first dynamics stage
    npt_first = mdp.stage_mdp("npt", s, first_dynamics=True)
    assert _value(npt_first, "gen-vel") == "yes"


def test_parrinello_rahman_only_in_production():
    s = mdp.default_mdp_settings()
    s["barostat"] = "Parrinello-Rahman"
    assert _value(mdp.stage_mdp("npt", s), "pcoupl") == "C-rescale"
    assert _value(mdp.stage_mdp("md", s), "pcoupl") == "Parrinello-Rahman"


def test_pressure_coupling_types():
    s = mdp.default_mdp_settings()
    s["pcoupltype"] = "semiisotropic"
    text = mdp.stage_mdp("md", s)
    assert _value(text, "ref-p") == "1 1"
    assert _value(text, "compressibility") == "4.5e-05 4.5e-05"
    s["pcoupltype"] = "anisotropic"
    text = mdp.stage_mdp("md", s)
    assert len(_value(text, "ref-p").split()) == 6
    assert len(_value(text, "compressibility").split()) == 6


def test_split_thermostat_groups_and_sd():
    s = mdp.default_mdp_settings()
    s["tc_grps_split"] = True
    text = mdp.stage_mdp("nvt", s, {"Water_and_ions": True, "non-Water": True})
    assert _value(text, "tc-grps") == "Water_and_ions non-Water"
    assert _value(text, "ref-t") == "300 300"
    assert _value(mdp.stage_mdp("nvt", s, {}), "tc-grps") == "System"
    s["md_integrator"] = "sd"
    text = mdp.stage_mdp("md", s)
    assert _value(text, "tcoupl") is None and _value(text, "integrator") == "sd"


def test_posres_define_and_extra_lines():
    s = mdp.default_mdp_settings()
    s["posres"] = True
    s["extra"]["md"] = "nstxout = 0"
    assert _value(mdp.stage_mdp("nvt", s), "define") == "-DPOSRES"
    assert _value(mdp.stage_mdp("md", s), "define") is None
    assert mdp.stage_mdp("md", s).rstrip().endswith("nstxout = 0")


def test_reaction_field_has_no_pme_keys():
    s = mdp.default_mdp_settings()
    s["coulombtype"] = "Reaction-Field"
    s["dispcorr"] = False
    text = mdp.stage_mdp("md", s)
    assert _value(text, "fourierspacing") is None and _value(text, "DispCorr") is None


def test_check_settings():
    s = mdp.default_mdp_settings()
    assert mdp.check_settings(s, 3.0) == []
    assert "half the smallest box width" in mdp.check_settings(s, 1.5)[0]
    s["rvdw"] = 1.2
    assert any("rcoulomb and rvdw" in p for p in mdp.check_settings(s))
    s["constraints"] = "none"
    s["dt"] = 0.002
    s["rvdw"] = 1.0
    assert mdp.check_settings(s) == []
    s["dt"] = 0.003
    assert any("h-bonds" in p for p in mdp.check_settings(s))
    s["dt"] = 0.005
    assert any("4 fs" in p for p in mdp.check_settings(s))


def test_enabled_stages():
    s = mdp.default_mdp_settings()
    s["stages"]["nvt"] = False
    assert mdp.enabled_stages(s) == ["em", "npt", "md"]


def test_stage_commands_chain():
    commands = dict(run_scripts.stage_commands(["em", "nvt", "npt", "md"], "conf.gro", "topol.top",
                                               "index.ndx", posres=True))
    assert commands["em"] == "grompp -f em.mdp -c conf.gro -p topol.top -o em.tpr -n index.ndx"
    assert "-c em.gro" in commands["nvt"] and "-r em.gro" in commands["nvt"]
    assert "-t" not in commands["nvt"]
    assert "-t nvt.cpt" in commands["npt"] and "-r nvt.gro" in commands["npt"]
    assert "-t npt.cpt" in commands["md"] and "-r" not in commands["md"]


@pytest.mark.parametrize("writer", [run_scripts.run_sh, run_scripts.run_bat])
def test_scripts_run_every_stage(writer):
    text = writer(["em", "md"], "conf.gro", "topol.top")
    assert text.count("grompp") == 2 and text.count("mdrun -deffnm") == 2
    assert "-maxwarn 0" in text


def test_bat_uses_crlf_and_stops_on_error():
    text = run_scripts.run_bat(["em"], "conf.gro", "topol.top")
    assert "\r\n" in text and "goto :error" in text
