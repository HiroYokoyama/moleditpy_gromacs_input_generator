import numpy as np
import pytest
from helpers import assemble, example_mdff_path, example_system, ion, water

from gromacs_input_generator import md_cell, mdff, writer


def _gro_atoms(text):
    lines = text.splitlines()
    count = int(lines[1])
    return lines[2:2 + count], lines[2 + count]


def test_default_bundle_files_and_gro_format():
    bundle = writer.build_bundle(example_system(), {"padding": 1.0})
    assert bundle.ok
    for name in ("conf.gro", "topol.top", "MOL.itp", "SOL.itp", "NA.itp", "CL.itp", "index.ndx",
                 "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp", "run.sh", "run.bat"):
        assert name in bundle.files
    atoms, box_line = _gro_atoms(bundle.files["conf.gro"])
    assert len(atoms) == 17
    for line in atoms:
        assert len(line) == 44  # %5d%-5s%5s%5d + 3 x %8.3f
    assert atoms[9][5:15] == "SOL     OW"
    assert atoms[15][5:10].strip() == "NA"
    assert len(box_line.split()) == 3


def test_padding_box_contains_every_atom():
    model = example_system()
    bundle = writer.build_bundle(model, {"padding": 1.0})
    atoms, _ = _gro_atoms(bundle.files["conf.gro"])
    coords = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atoms])
    lengths = np.diag(bundle.box) / 10.0
    assert np.all(coords >= 0.099) and np.all(coords <= lengths - 0.099)


def test_mdff_bundle_uses_force_field_names():
    doc = mdff.load_mdff(example_mdff_path())
    bundle = writer.build_bundle(example_system(), {}, doc)
    assert "ETH.itp" in bundle.files and "MOL.itp" not in bundle.files
    atoms, _ = _gro_atoms(bundle.files["conf.gro"])
    assert atoms[0][5:15] == "ETH     C1"
    assert atoms[8][10:15].strip() == "HO"


def test_cell_property_triclinic_box_is_rotated():
    lattice = md_cell.box_from_lengths([30.0, 30.0, 30.0], [90.0, 90.0, 120.0])
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cell = {"lattice": lattice @ rotation, "origin": np.array([1.0, 2.0, 3.0]), "pbc": [True] * 3}
    model = assemble([water((5, 5, 5)), water((10, 10, 10))])
    bundle = writer.build_bundle(model, {}, cell_prop=cell)
    assert bundle.ok
    _, box_line = _gro_atoms(bundle.files["conf.gro"])
    values = [float(v) for v in box_line.split()]
    assert len(values) == 9
    assert values[5] == pytest.approx(-1.5, abs=1e-4)  # v2(x) after reduction
    # intramolecular distances survive the rotation
    atoms, _ = _gro_atoms(bundle.files["conf.gro"])
    xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atoms])
    assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(0.0957, abs=1e-3)
    assert "moleditpy_cell" in bundle.warnings[0]


def test_box_source_errors_and_cif_fallback():
    model = assemble([water()])
    bundle = writer.build_bundle(model, {"box_source": "cell"})
    assert not bundle.ok and "moleditpy_cell" in bundle.errors[0]
    bundle = writer.build_bundle(model, {"box_source": "cif"})
    assert not bundle.ok
    bundle = writer.build_bundle(model, {"box_source": "auto"}, cif_lattice=np.diag([25.0, 25.0, 25.0]))
    assert "CIF Viewer" in bundle.warnings[0]


def test_explicit_lengths_centre_the_molecule():
    model = assemble([water()])
    bundle = writer.build_bundle(model, {"box_source": "lengths", "lengths": [3.0, 3.0, 3.0]})
    atoms, box_line = _gro_atoms(bundle.files["conf.gro"])
    assert box_line.split() == ["3.00000", "3.00000", "3.00000"]
    xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atoms])
    assert np.allclose(xyz.mean(axis=0), 1.5, atol=1e-3)


def test_small_box_cutoff_warning():
    bundle = writer.build_bundle(assemble([water()]), {"padding": 0.3})
    assert any("half the smallest box width" in w for w in bundle.warnings)


def test_index_groups():
    bundle = writer.build_bundle(example_system(), {})
    ndx = bundle.files["index.ndx"]
    groups = writer.index_groups(bundle.topology)
    assert list(groups) == ["System", "Water", "Ion", "Water_and_ions", "Other", "non-Water"]
    assert groups["Water_and_ions"] == list(range(10, 18))
    assert groups["Other"] == list(range(1, 10))
    assert "[ Water_and_ions ]" in ndx


def test_protein_group_from_residue_names():
    from gromacs_input_generator.system_model import build_system

    model = build_system(["N", "C", "O", "H", "H"], [[0, 0, 0], [1.4, 0, 0], [9, 9, 9], [9.96, 9, 9],
                                                      [8.76, 9.93, 9]],
                         [(0, 1), (2, 3), (2, 4)], resnames=["ALA", "ALA", "SOL", "SOL", "SOL"],
                         resnums=[1, 1, 2, 2, 2])
    groups = writer.index_groups(writer.build_bundle(model, {}).topology)
    assert groups["Protein"] == [1, 2]
    assert groups["Water_and_ions"] == [3, 4, 5]


def test_options_drop_index_scripts_and_stages():
    settings = {"write_index": False, "write_scripts": False,
                "mdp": {"stages": {"em": True, "nvt": False, "npt": False, "md": True},
                        "tc_grps_split": True}}
    bundle = writer.build_bundle(example_system(), settings)
    assert "index.ndx" not in bundle.files and "run.sh" not in bundle.files
    assert sorted(n for n in bundle.files if n.endswith(".mdp")) == ["em.mdp", "md.mdp"]
    assert "gen-vel                 = yes" in bundle.files["md.mdp"]
    assert any("thermostat groups need the index" in w for w in bundle.warnings)


def test_posres_files_for_parameterised_solutes():
    doc = mdff.load_mdff(example_mdff_path())
    bundle = writer.build_bundle(example_system(), {"mdp": {"posres": True}}, doc)
    assert "posre_ETH.itp" in bundle.files
    assert '#include "posre_ETH.itp"' in bundle.files["ETH.itp"]
    posre = bundle.files["posre_ETH.itp"]
    assert posre.count("1000.0") == 9  # 3 heavy atoms x 3 force constants
    assert "-r em.gro" in bundle.files["run.sh"]


def test_save_bundle(tmp_path):
    bundle = writer.build_bundle(assemble([water(), ion("Na", (5, 0, 0))], [0, 0, 0, 1]), {})
    written = writer.save_bundle(bundle, str(tmp_path / "out"))
    assert len(written) == len(bundle.files)
    assert b"\r\n" in (tmp_path / "out" / "run.bat").read_bytes()
    assert b"\r\n" not in (tmp_path / "out" / "topol.top").read_bytes()


def test_merge_settings_ignores_unknown_keys():
    merged = writer.merge_settings(writer.default_settings(), {"bogus": 1, "mdp": {"temperature": 310}})
    assert "bogus" not in merged
    assert merged["mdp"]["temperature"] == 310
    assert merged["mdp"]["pressure"] == 1.0
