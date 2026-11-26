"""Real-PySide6 dialog tests (offscreen)."""

import os

import pytest

pytest.importorskip("PySide6")
Chem = pytest.importorskip("rdkit.Chem")
from conftest import flush_qt
from rdkit.Chem import AllChem

import gromacs_input_generator as plugin
from gromacs_input_generator import main_dialog
from gromacs_input_generator.md_cell import cell_to_json

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples",
                       "ethanol_water_nacl.mdff.json")


def _example_mol():
    parts = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in ("CCO", "O", "O")]
    parts += [Chem.MolFromSmiles("[Na+]"), Chem.MolFromSmiles("[Cl-]")]
    mol = parts[0]
    for part in parts[1:]:
        mol = Chem.CombineMols(mol, part)
    AllChem.EmbedMolecule(mol, randomSeed=11)
    return mol


@pytest.fixture
def presets_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(main_dialog, "SETTINGS_FILE", str(path))
    return path


@pytest.fixture
def dialog(qapp, presets_file):
    mol = _example_mol()
    settings = plugin.get_default_settings()
    marks = []
    dlg = main_dialog.GromacsInputDialog(persistent_settings=settings, get_molecule=lambda: mol,
                                         mark_modified=lambda: marks.append(1))
    dlg.marks = marks
    dlg.mol = mol
    yield dlg
    plugin.remembered_cell.clear()
    flush_qt(qapp)


def _files(dlg):
    return [dlg.file_combo.itemText(i) for i in range(dlg.file_combo.count())]


def test_opens_with_a_preview_and_does_not_mark_the_project(dialog):
    assert "conf.gro" in _files(dialog) and "topol.top" in _files(dialog)
    assert "17 atoms" in dialog.system_label.text()
    assert dialog.mol_table.rowCount() == 4
    assert dialog.marks == []
    dialog.file_combo.setCurrentIndex(_files(dialog).index("topol.top"))
    assert "[ molecules ]" in dialog.preview.toPlainText()


def test_settings_round_trip(dialog):
    settings = dialog.read_settings()
    settings["water_model"] = "SPC/E"
    settings["box_source"] = "lengths"
    settings["lengths"] = [4.0, 4.5, 5.0]
    settings["mdp"]["temperature"] = 310.0
    settings["mdp"]["extra"]["md"] = "nstxout = 0"
    dialog.apply_settings(settings)
    again = dialog.read_settings()
    assert again["water_model"] == "SPC/E" and again["box_source"] == "lengths"
    assert again["lengths"] == [4.0, 4.5, 5.0]
    assert again["mdp"]["temperature"] == 310.0
    assert again["mdp"]["extra"]["md"] == "nstxout = 0"


def test_user_change_updates_persistent_settings_and_marks_modified(dialog):
    dialog.temp_spin.setValue(320.0)
    dialog.update_preview()
    assert dialog.persistent_settings["mdp"]["temperature"] == 320.0
    assert dialog.marks
    assert "ref-t                   = 320" in dialog.bundle.files["nvt.mdp"]


def test_mdff_file_is_used(dialog):
    dialog.mdff_edit.setText(EXAMPLE)
    dialog.update_preview()
    assert "ETH.itp" in _files(dialog)
    assert "Types: ETH, SOL, NA, CL" in dialog.mdff_label.text()
    dialog.mdff_edit.setText(EXAMPLE + ".missing")
    dialog.update_preview()
    assert "not found" in dialog.messages.toPlainText()


def test_cell_property_and_remembered_cell(dialog):
    dialog.mol.SetProp("moleditpy_cell", cell_to_json([[30, 0, 0], [0, 31, 0], [0, 0, 32]]))
    dialog.update_preview()
    assert "cell from moleditpy_cell" in dialog.system_label.text()
    assert dialog.bundle.files["conf.gro"].splitlines()[-1].split() == ["3.00000", "3.10000", "3.20000"]
    # an undo snapshot drops the property: the remembered cell takes over
    dialog.mol.ClearProp("moleditpy_cell")
    dialog.update_preview()
    assert "remembered" in dialog.messages.toPlainText()


def test_no_molecule(qapp, presets_file):
    dlg = main_dialog.GromacsInputDialog(get_molecule=lambda: None)
    assert dlg.system_label.text() == "No molecule is loaded."
    assert not dlg.save_btn.isEnabled() or dlg.bundle is None
    dlg.close()


def test_presets(dialog, presets_file, monkeypatch):
    dialog.temp_spin.setValue(350.0)
    monkeypatch.setattr(main_dialog.QInputDialog, "getText", lambda *a, **k: ("hot", True))
    dialog.save_preset()
    assert presets_file.exists()
    assert main_dialog.load_presets()["hot"]["mdp"]["temperature"] == 350.0
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("builtin:Default"))
    dialog.apply_preset()
    assert dialog.temp_spin.value() == 300.0
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("user:hot"))
    dialog.apply_preset()
    assert dialog.temp_spin.value() == 350.0
    dialog.delete_preset()
    assert "hot" not in main_dialog.load_presets()


def test_quick_test_preset_keeps_the_force_field_file(dialog):
    dialog.mdff_edit.setText(EXAMPLE)
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("builtin:Quick test (10 ps per stage)"))
    dialog.apply_preset()
    assert dialog.md_steps_spin.value() == 5000
    assert dialog.mdff_edit.text() == EXAMPLE


def test_save_files(dialog, tmp_path):
    written = dialog.save_files(str(tmp_path))
    names = sorted(os.path.basename(p) for p in written)
    assert "conf.gro" in names and "run.sh" in names
    assert (tmp_path / "topol.top").read_text(encoding="utf-8").startswith("; Generated by MoleditPy")


def test_extra_lines_per_stage(dialog):
    dialog.extra_stage_combo.setCurrentIndex(3)  # md
    dialog.extra_edit.setPlainText("nstxout = 0")
    dialog.extra_stage_combo.setCurrentIndex(0)  # em
    assert dialog.extra_edit.toPlainText() == ""
    dialog.update_preview()
    assert dialog.bundle.files["md.mdp"].rstrip().endswith("nstxout = 0")
