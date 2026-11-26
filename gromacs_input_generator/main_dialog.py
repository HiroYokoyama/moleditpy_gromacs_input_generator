"""The GROMACS Input Generator dialog: options on the left, live file preview on the right."""

from __future__ import annotations

import copy
import json
import logging
import os

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import PLUGIN_NAME, PLUGIN_VERSION, SETTINGS_FILE, gmx_check, md_cell, mdp
from .mdff import MdffError, load_mdff
from .system_model import from_rdkit
from .wheel_guard import install_wheel_guard
from .writer import build_bundle, default_settings, merge_settings, save_bundle

PREVIEW_LINE_LIMIT = 3000

BOX_SOURCES = [
    ("auto", "Automatic (cell property, then CIF Viewer, then padding)"),
    ("cell", "Molecule cell (moleditpy_cell)"),
    ("cif", "CIF Viewer cell"),
    ("padding", "Bounding box + padding"),
    ("lengths", "Explicit lengths and angles"),
]

_log = logging.getLogger(__name__)

BUILTIN_PRESETS = {
    "Default": {},
    "Quick test (10 ps per stage)": {
        "mdp": {"em_steps": 5000, "nvt_steps": 5000, "npt_steps": 5000, "md_steps": 5000,
                "nstxout_compressed": 500, "nstenergy": 100, "nstlog": 100},
    },
    "Production 100 ns (Parrinello-Rahman)": {
        "mdp": {"md_steps": 50000000, "barostat": "Parrinello-Rahman", "nstxout_compressed": 50000,
                "nstenergy": 5000, "nstlog": 5000},
    },
}


def load_presets() -> dict:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        presets = data.get("presets", {})
        return presets if isinstance(presets, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def save_presets(presets: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"presets": presets}, handle, indent=2)


class GromppWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, files, conf, stage, parent=None):
        super().__init__(parent)
        self.files, self.conf, self.stage = dict(files), conf, stage

    def run(self):  # pragma: no cover - needs gmx
        ok, text = gmx_check.run_grompp(self.files, self.conf, self.stage)
        self.done.emit(ok, text)


def _dspin(lo, hi, decimals, step, suffix=""):
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _ispin(lo, hi, step=1):
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setGroupSeparatorShown(True)
    return spin


def _combo(items):
    combo = QComboBox()
    combo.addItems(list(items))
    return combo



def _scrolled(page):
    """A settings page that scrolls, so a long tab never makes the dialog taller than the
    screen and pushes its buttons out of reach."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(page)
    # Scrolling over a field must not change it: fields take the wheel once clicked.
    install_wheel_guard(page)
    return area


def fit_to_screen(dialog, fraction=0.9):
    """Shrink the dialog's first size to the screen it opens on (small laptop screens)."""
    parent = dialog.parentWidget()
    screen = parent.screen() if parent is not None else dialog.screen()
    if screen is None:
        return
    area = screen.availableGeometry()
    dialog.resize(
        min(dialog.width(), int(area.width() * fraction)),
        min(dialog.height(), int(area.height() * fraction)),
    )


class GromacsInputDialog(QDialog):
    def __init__(self, parent=None, persistent_settings=None, get_molecule=None,
                 mark_modified=None, get_cif_viewer=None, context=None):
        super().__init__(parent)
        self.setWindowTitle(f"{PLUGIN_NAME} {PLUGIN_VERSION}")
        self.resize(1180, 760)
        fit_to_screen(self)
        self.persistent_settings = persistent_settings if persistent_settings is not None else {}
        self.get_molecule = get_molecule or (lambda: None)
        self.mark_modified = mark_modified
        self.get_cif_viewer = get_cif_viewer or (lambda: None)
        self.context = context
        self._updating = False
        self._user_changed = False
        self._model_key = None
        self._model = None
        self._model_error = ""
        self._mdff_cache = (None, None, None)  # (path, mtime, doc)
        self.bundle = None
        self._worker = None
        self._extra = {stage: "" for stage in mdp.STAGES}
        self._extra_stage = "em"

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.update_preview)

        root = QVBoxLayout(self)
        root.addLayout(self._build_preset_row())
        splitter = QSplitter()
        self.tabs = QTabWidget()
        self.tabs.addTab(_scrolled(self._build_system_tab()), "System")
        self.tabs.addTab(_scrolled(self._build_ff_tab()), "Force field")
        self.tabs.addTab(_scrolled(self._build_box_tab()), "Box")
        self.tabs.addTab(_scrolled(self._build_md_tab()), "MD stages")
        self.tabs.addTab(_scrolled(self._build_output_tab()), "Output")
        splitter.addWidget(self.tabs)
        splitter.addWidget(self._build_preview_panel())
        splitter.setSizes([520, 660])
        root.addWidget(splitter, 1)
        root.addLayout(self._build_buttons())

        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_files)
        self.apply_settings(merge_settings(default_settings(), self.persistent_settings))
        self._connect_changes()
        self.refresh_molecule()

    # ------------------------------------------------------------------ UI
    def _build_preset_row(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self._fill_presets()
        row.addWidget(self.preset_combo, 1)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_preset)
        save_btn = QPushButton("Save as...")
        save_btn.clicked.connect(self.save_preset)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self.delete_preset)
        for btn in (apply_btn, save_btn, del_btn):
            row.addWidget(btn)
        return row

    def _build_system_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.title_edit = QLineEdit()
        form.addRow("System title:", self.title_edit)
        layout.addLayout(form)
        self.system_label = QLabel("-")
        self.system_label.setWordWrap(True)
        layout.addWidget(self.system_label)
        self.mol_table = QTableWidget(0, 5)
        self.mol_table.setHorizontalHeaderLabels(["Molecule type", "Parameters", "Copies", "Atoms", "Charge"])
        self.mol_table.verticalHeader().setVisible(False)
        self.mol_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mol_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.mol_table, 1)
        refresh = QPushButton("Re-read the current molecule")
        refresh.clicked.connect(self.refresh_molecule)
        layout.addWidget(refresh)
        return widget

    def _build_ff_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        group = QGroupBox("Force-field file (moleditpy-mdff JSON from the Force Field Builder)")
        g = QVBoxLayout(group)
        row = QHBoxLayout()
        self.mdff_edit = QLineEdit()
        self.mdff_edit.setPlaceholderText("none: built-in water/ion tables, skeletons for the rest")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_mdff)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.mdff_edit.setText(""))
        row.addWidget(self.mdff_edit, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        g.addLayout(row)
        self.mdff_label = QLabel("")
        self.mdff_label.setWordWrap(True)
        g.addWidget(self.mdff_label)
        layout.addWidget(group)
        fallback = QGroupBox("Built-in fallback")
        f = QFormLayout(fallback)
        self.water_combo = _combo(["TIP3P", "SPC/E"])
        f.addRow("Water model (and matching ions):", self.water_combo)
        note = QLabel(
            "Molecules the force-field file does not cover: water (O + 2 H) and monatomic "
            "ions (Li, Na, K, Rb, Cs, F, Cl, Br, I, Mg, Ca, Sr, Ba, Zn) use built-in "
            "parameters (Joung-Cheatham / Li-Merz ions for the chosen water). Anything "
            "else gets a skeleton .itp marked TODO, which grompp will reject until it is "
            "parameterised with the Force Field Builder."
        )
        note.setWordWrap(True)
        f.addRow(note)
        layout.addWidget(fallback)
        layout.addStretch(1)
        return widget

    def _build_box_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.box_combo = QComboBox()
        for key, label in BOX_SOURCES:
            self.box_combo.addItem(label, key)
        form.addRow("Box from:", self.box_combo)
        self.padding_spin = _dspin(0.0, 20.0, 3, 0.1, " nm")
        form.addRow("Padding:", self.padding_spin)
        self.cubic_check = QCheckBox("Cubic box")
        form.addRow("", self.cubic_check)
        self.length_spins = [_dspin(0.1, 1000.0, 3, 0.1, " nm") for _ in range(3)]
        lengths = QHBoxLayout()
        for spin in self.length_spins:
            lengths.addWidget(spin)
        form.addRow("a, b, c:", lengths)
        self.angle_spins = [_dspin(10.0, 170.0, 3, 1.0, " deg") for _ in range(3)]
        angles = QHBoxLayout()
        for spin in self.angle_spins:
            angles.addWidget(spin)
        form.addRow("alpha, beta, gamma:", angles)
        layout.addLayout(form)
        self.box_label = QLabel("-")
        self.box_label.setWordWrap(True)
        layout.addWidget(self.box_label)
        layout.addStretch(1)
        return widget

    def _build_md_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        stages = QHBoxLayout()
        self.stage_checks = {}
        for stage in mdp.STAGES:
            check = QCheckBox(mdp.STAGE_TITLES[stage])
            self.stage_checks[stage] = check
            stages.addWidget(check)
        layout.addLayout(stages)

        common = QGroupBox("Common")
        c = QFormLayout(common)
        self.temp_spin = _dspin(1.0, 5000.0, 2, 10.0, " K")
        self.pressure_spin = _dspin(0.0, 1e5, 3, 0.1, " bar")
        self.dt_spin = _dspin(0.0001, 0.01, 4, 0.0005, " ps")
        self.constraints_combo = _combo(mdp.CONSTRAINTS)
        self.thermostat_combo = _combo(mdp.THERMOSTATS)
        self.tau_t_spin = _dspin(0.001, 100.0, 3, 0.1, " ps")
        self.barostat_combo = _combo(mdp.BAROSTATS)
        self.tau_p_spin = _dspin(0.01, 100.0, 3, 0.5, " ps")
        self.pcoupltype_combo = _combo(mdp.PCOUPLTYPES)
        self.compress_spin = _dspin(0.0, 1e-3, 8, 1e-6, " 1/bar")
        self.coulomb_combo = _combo(mdp.COULOMB)
        self.rc_spin = _dspin(0.5, 5.0, 3, 0.1, " nm")
        self.rvdw_spin = _dspin(0.5, 5.0, 3, 0.1, " nm")
        self.dispcorr_check = QCheckBox("Dispersion correction (EnerPres)")
        self.tcsplit_check = QCheckBox("Separate thermostat groups (Water_and_ions / non-Water)")
        self.posres_check = QCheckBox("Restrain solute heavy atoms in NVT/NPT (-DPOSRES)")
        for label, w in (("Temperature:", self.temp_spin), ("Pressure:", self.pressure_spin),
                         ("Time step:", self.dt_spin), ("Constraints:", self.constraints_combo),
                         ("Thermostat:", self.thermostat_combo), ("tau-t:", self.tau_t_spin),
                         ("Barostat:", self.barostat_combo), ("tau-p:", self.tau_p_spin),
                         ("Pressure coupling type:", self.pcoupltype_combo),
                         ("Compressibility:", self.compress_spin), ("Electrostatics:", self.coulomb_combo),
                         ("rcoulomb:", self.rc_spin), ("rvdw:", self.rvdw_spin)):
            c.addRow(label, w)
        c.addRow("", self.dispcorr_check)
        c.addRow("", self.tcsplit_check)
        c.addRow("", self.posres_check)
        layout.addWidget(common)

        steps = QGroupBox("Stages")
        s = QFormLayout(steps)
        self.em_integrator_combo = _combo(mdp.EM_INTEGRATORS)
        self.em_steps_spin = _ispin(0, 10_000_000, 1000)
        self.emtol_spin = _dspin(0.1, 1e6, 1, 100.0, " kJ/mol/nm")
        self.nvt_steps_spin = _ispin(0, 2_000_000_000, 1000)
        self.npt_steps_spin = _ispin(0, 2_000_000_000, 1000)
        self.md_steps_spin = _ispin(0, 2_000_000_000, 10000)
        self.md_integrator_combo = _combo(mdp.MD_INTEGRATORS)
        self.xtc_spin = _ispin(0, 100_000_000, 500)
        self.nstenergy_spin = _ispin(0, 100_000_000, 100)
        self.nstlog_spin = _ispin(0, 100_000_000, 100)
        for label, w in (("EM algorithm:", self.em_integrator_combo), ("EM max steps:", self.em_steps_spin),
                         ("EM tolerance:", self.emtol_spin), ("NVT steps:", self.nvt_steps_spin),
                         ("NPT steps:", self.npt_steps_spin), ("Production steps:", self.md_steps_spin),
                         ("Dynamics integrator:", self.md_integrator_combo),
                         ("xtc every (steps):", self.xtc_spin), ("Energies every:", self.nstenergy_spin),
                         ("Log every:", self.nstlog_spin)):
            s.addRow(label, w)
        layout.addWidget(steps)

        extra = QGroupBox("Extra .mdp lines (appended verbatim)")
        e = QVBoxLayout(extra)
        self.extra_stage_combo = QComboBox()
        for stage in mdp.STAGES:
            self.extra_stage_combo.addItem(mdp.STAGE_TITLES[stage], stage)
        self.extra_stage_combo.currentIndexChanged.connect(self._extra_stage_changed)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("e.g. nstxout = 0")
        self.extra_edit.setMaximumHeight(90)
        e.addWidget(self.extra_stage_combo)
        e.addWidget(self.extra_edit)
        layout.addWidget(extra)
        layout.addStretch(1)
        return widget

    def _build_output_tab(self):
        widget = QWidget()
        form = QFormLayout(widget)
        self.basename_edit = QLineEdit()
        form.addRow("Coordinate file name:", self.basename_edit)
        self.index_check = QCheckBox("Write index.ndx (System / Protein / Water_and_ions / Other ...)")
        self.scripts_check = QCheckBox("Write run.sh and run.bat (grompp + mdrun chain)")
        form.addRow("", self.index_check)
        form.addRow("", self.scripts_check)
        return widget

    def _build_preview_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        row.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._show_file)
        row.addWidget(self.file_combo, 1)
        layout.addLayout(row)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.preview.setFont(font)
        layout.addWidget(self.preview, 3)
        layout.addWidget(QLabel("Messages:"))
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setMaximumHeight(150)
        layout.addWidget(self.messages, 1)
        return widget

    def _build_buttons(self):
        row = QHBoxLayout()
        self.gmx_path = gmx_check.find_gmx()
        self.check_btn = QPushButton("Check with gmx grompp")
        self.check_btn.setEnabled(bool(self.gmx_path))
        self.check_btn.setToolTip(self.gmx_path or "gmx was not found on PATH")
        self.check_btn.clicked.connect(self.run_check)
        self.save_btn = QPushButton("Save files...")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_files)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(self.check_btn)
        row.addStretch(1)
        row.addWidget(self.save_btn)
        row.addWidget(close)
        return row

    def _connect_changes(self):
        for widget in self.findChildren(QLineEdit) + self.findChildren(QPlainTextEdit):
            if widget in (self.preview, self.messages):
                continue
            widget.textChanged.connect(self._changed)
        for widget in self.findChildren(QDoubleSpinBox) + self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._changed)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._changed)
        for widget in self.findChildren(QComboBox):
            if widget in (self.preset_combo, self.file_combo, self.extra_stage_combo):
                continue
            widget.currentIndexChanged.connect(self._changed)

    # ------------------------------------------------------------ settings
    def read_settings(self) -> dict:
        self._extra[self._extra_stage] = self.extra_edit.toPlainText()
        m = {
            "stages": {stage: check.isChecked() for stage, check in self.stage_checks.items()},
            "temperature": self.temp_spin.value(),
            "pressure": self.pressure_spin.value(),
            "dt": self.dt_spin.value(),
            "em_integrator": self.em_integrator_combo.currentText(),
            "em_steps": self.em_steps_spin.value(),
            "emtol": self.emtol_spin.value(),
            "emstep": 0.01,
            "nvt_steps": self.nvt_steps_spin.value(),
            "npt_steps": self.npt_steps_spin.value(),
            "md_steps": self.md_steps_spin.value(),
            "md_integrator": self.md_integrator_combo.currentText(),
            "thermostat": self.thermostat_combo.currentText(),
            "tau_t": self.tau_t_spin.value(),
            "barostat": self.barostat_combo.currentText(),
            "tau_p": self.tau_p_spin.value(),
            "pcoupltype": self.pcoupltype_combo.currentText(),
            "compressibility": self.compress_spin.value(),
            "constraints": self.constraints_combo.currentText(),
            "coulombtype": self.coulomb_combo.currentText(),
            "rcoulomb": self.rc_spin.value(),
            "rvdw": self.rvdw_spin.value(),
            "dispcorr": self.dispcorr_check.isChecked(),
            "nstxout_compressed": self.xtc_spin.value(),
            "nstenergy": self.nstenergy_spin.value(),
            "nstlog": self.nstlog_spin.value(),
            "tc_grps_split": self.tcsplit_check.isChecked(),
            "posres": self.posres_check.isChecked(),
            "extra": dict(self._extra),
        }
        return {
            "title": self.title_edit.text(),
            "basename": self.basename_edit.text(),
            "mdff_path": self.mdff_edit.text().strip(),
            "water_model": self.water_combo.currentText(),
            "box_source": self.box_combo.currentData(),
            "padding": self.padding_spin.value(),
            "cubic": self.cubic_check.isChecked(),
            "lengths": [spin.value() for spin in self.length_spins],
            "angles": [spin.value() for spin in self.angle_spins],
            "write_index": self.index_check.isChecked(),
            "write_scripts": self.scripts_check.isChecked(),
            "mdp": m,
        }

    def apply_settings(self, settings: dict) -> None:
        s = merge_settings(default_settings(), settings or {})
        m = s["mdp"]
        self._updating = True
        try:
            self.title_edit.setText(s["title"])
            self.basename_edit.setText(s["basename"])
            self.mdff_edit.setText(s["mdff_path"])
            self._set_combo(self.water_combo, s["water_model"])
            index = self.box_combo.findData(s["box_source"])
            self.box_combo.setCurrentIndex(max(index, 0))
            self.padding_spin.setValue(float(s["padding"]))
            self.cubic_check.setChecked(bool(s["cubic"]))
            for spin, value in zip(self.length_spins, s["lengths"]):
                spin.setValue(float(value))
            for spin, value in zip(self.angle_spins, s["angles"]):
                spin.setValue(float(value))
            self.index_check.setChecked(bool(s["write_index"]))
            self.scripts_check.setChecked(bool(s["write_scripts"]))
            for stage, check in self.stage_checks.items():
                check.setChecked(bool(m["stages"].get(stage, True)))
            self.temp_spin.setValue(float(m["temperature"]))
            self.pressure_spin.setValue(float(m["pressure"]))
            self.dt_spin.setValue(float(m["dt"]))
            self._set_combo(self.em_integrator_combo, m["em_integrator"])
            self.em_steps_spin.setValue(int(m["em_steps"]))
            self.emtol_spin.setValue(float(m["emtol"]))
            self.nvt_steps_spin.setValue(int(m["nvt_steps"]))
            self.npt_steps_spin.setValue(int(m["npt_steps"]))
            self.md_steps_spin.setValue(int(m["md_steps"]))
            self._set_combo(self.md_integrator_combo, m["md_integrator"])
            self._set_combo(self.thermostat_combo, m["thermostat"])
            self.tau_t_spin.setValue(float(m["tau_t"]))
            self._set_combo(self.barostat_combo, m["barostat"])
            self.tau_p_spin.setValue(float(m["tau_p"]))
            self._set_combo(self.pcoupltype_combo, m["pcoupltype"])
            self.compress_spin.setValue(float(m["compressibility"]))
            self._set_combo(self.constraints_combo, m["constraints"])
            self._set_combo(self.coulomb_combo, m["coulombtype"])
            self.rc_spin.setValue(float(m["rcoulomb"]))
            self.rvdw_spin.setValue(float(m["rvdw"]))
            self.dispcorr_check.setChecked(bool(m["dispcorr"]))
            self.xtc_spin.setValue(int(m["nstxout_compressed"]))
            self.nstenergy_spin.setValue(int(m["nstenergy"]))
            self.nstlog_spin.setValue(int(m["nstlog"]))
            self.tcsplit_check.setChecked(bool(m["tc_grps_split"]))
            self.posres_check.setChecked(bool(m["posres"]))
            self._extra = {stage: str((m.get("extra") or {}).get(stage, "")) for stage in mdp.STAGES}
            self.extra_edit.setPlainText(self._extra.get(self._extra_stage, ""))
        finally:
            self._updating = False
        self._update_box_widgets()
        self._timer.start()

    @staticmethod
    def _set_combo(combo, text):
        index = combo.findText(str(text))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _extra_stage_changed(self, _index):
        self._extra[self._extra_stage] = self.extra_edit.toPlainText()
        self._extra_stage = self.extra_stage_combo.currentData()
        self._updating = True
        try:
            self.extra_edit.setPlainText(self._extra.get(self._extra_stage, ""))
        finally:
            self._updating = False

    def _update_box_widgets(self):
        source = self.box_combo.currentData()
        self.padding_spin.setEnabled(source in ("auto", "padding"))
        self.cubic_check.setEnabled(source in ("auto", "padding"))
        for spin in self.length_spins + self.angle_spins:
            spin.setEnabled(source == "lengths")

    def _changed(self, *_args):
        if self._updating:
            return
        self._update_box_widgets()
        self._user_changed = True
        self._timer.start()

    # -------------------------------------------------------------- presets
    def _fill_presets(self):
        self.preset_combo.clear()
        for name in BUILTIN_PRESETS:
            self.preset_combo.addItem(name, f"builtin:{name}")
        for name in sorted(load_presets()):
            self.preset_combo.addItem(f"{name} (saved)", f"user:{name}")

    def apply_preset(self):
        kind, _, name = (self.preset_combo.currentData() or "builtin:Default").partition(":")
        if kind == "builtin":
            update = copy.deepcopy(BUILTIN_PRESETS.get(name, {}))
            base = default_settings()
            current = self.read_settings()
            base["mdff_path"] = current["mdff_path"]
            base["title"] = current["title"]
            self.apply_settings(merge_settings(base, update))
        else:
            preset = load_presets().get(name)
            if isinstance(preset, dict):
                self.apply_settings(preset)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        presets = load_presets()
        settings = self.read_settings()
        settings["mdff_path"] = ""  # a preset is about MD options, not one system
        presets[name] = settings
        try:
            save_presets(presets)
        except OSError as exc:
            QMessageBox.warning(self, "Save preset", f"Could not write {SETTINGS_FILE}: {exc}")
            return
        self._fill_presets()
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(f"user:{name}"))

    def delete_preset(self):
        kind, _, name = (self.preset_combo.currentData() or "builtin:").partition(":")
        if kind != "user":
            QMessageBox.information(self, "Delete preset", "Built-in presets cannot be deleted.")
            return
        presets = load_presets()
        presets.pop(name, None)
        try:
            save_presets(presets)
        except OSError as exc:  # pragma: no cover - file system
            QMessageBox.warning(self, "Delete preset", str(exc))
        self._fill_presets()

    # --------------------------------------------------------------- model
    def refresh_molecule(self):
        self._model_key = None
        self.update_preview()

    def _current_model(self):
        mol = self.get_molecule()
        if mol is None:
            return None, None, "No molecule is loaded."
        key = (id(mol), mol.GetNumAtoms(), mol.GetNumBonds(), mol.GetNumConformers())
        if key != self._model_key:
            self._model_key = key
            try:
                self._model = from_rdkit(mol)
                self._model_error = ""
            except ValueError as exc:
                self._model, self._model_error = None, str(exc)
        return mol, self._model, self._model_error

    def _cell_property(self, mol):
        from . import remember_cell, remembered_cell

        if remember_cell(mol):
            return md_cell.parse_cell_json(remembered_cell["json"]), "moleditpy_cell"
        if remembered_cell and mol is not None and mol.GetNumAtoms() == remembered_cell.get("natoms"):
            return md_cell.parse_cell_json(remembered_cell["json"]), "remembered moleditpy_cell"
        return None, ""

    def _cif_lattice(self):
        try:
            holder = self.get_cif_viewer()
            structure = getattr(holder, "structure", None) if holder is not None else None
            lattice = getattr(structure, "lattice", None) if structure is not None else None
            if lattice is not None:
                array = np.asarray(lattice, dtype=float)
                if array.shape == (3, 3):
                    return array
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - other plugin's internals
            _log.debug("CIF Viewer cell lookup failed: %s", exc)
        return None

    def _mdff(self, path):
        if not path:
            self.mdff_label.setText("No force-field file: built-in tables only.")
            return None, ""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self.mdff_label.setText("File not found.")
            return None, f"Force-field file not found: {path}"
        cached_path, cached_mtime, doc = self._mdff_cache
        if cached_path != path or cached_mtime != mtime:
            try:
                doc = load_mdff(path)
            except MdffError as exc:
                self._mdff_cache = (None, None, None)
                self.mdff_label.setText(f"Not usable: {exc}")
                return None, str(exc)
            self._mdff_cache = (path, mtime, doc)
        types = ", ".join(mt["name"] for mt in doc["molecule_types"])
        self.mdff_label.setText(
            f"{doc['source'] or 'moleditpy-mdff'}\nTypes: {types}; combination rule "
            f"{doc['combination_rule']}, fudgeLJ {doc['fudge_lj']:g}, fudgeQQ {doc['fudge_qq']:g}"
        )
        return doc, ""

    # -------------------------------------------------------------- preview
    def update_preview(self):
        settings = self.read_settings()
        self.persistent_settings.clear()
        self.persistent_settings.update(settings)
        if self.mark_modified and self._user_changed:
            self._user_changed = False
            try:
                self.mark_modified()
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - host guard
                pass
        mol, model, error = self._current_model()
        doc, mdff_error = self._mdff(settings["mdff_path"])
        if model is None:
            self.bundle = None
            self._set_files({})
            self.system_label.setText(error)
            self.messages.setPlainText(error)
            return
        try:
            cell, cell_source = self._cell_property(mol)
        except md_cell.CellError as exc:
            cell, cell_source = None, ""
            mdff_error = (mdff_error + "\n" if mdff_error else "") + str(exc)
        header = f"Generated by MoleditPy {PLUGIN_NAME} {PLUGIN_VERSION}"
        self.bundle = build_bundle(model, settings, doc, cell, self._cif_lattice(), header)
        messages = list(self.bundle.errors)
        if mdff_error:
            messages.append(mdff_error)
        if cell_source == "remembered moleditpy_cell":
            messages.append("The molecule lost its moleditpy_cell (undo or reload); using the remembered cell.")
        messages += self.bundle.warnings
        self.messages.setPlainText("\n".join(messages))
        self._update_system_summary(model, cell_source)
        if self.bundle.box is not None:
            self.box_label.setText(md_cell.describe(self.bundle.box)
                                   + f"\nVolume {abs(np.linalg.det(self.bundle.box)) / 1000:.3f} nm^3")
        else:
            self.box_label.setText("; ".join(self.bundle.errors))
        self._set_files(self.bundle.files)
        self.save_btn.setEnabled(self.bundle.ok and bool(self.bundle.files))
        self.check_btn.setEnabled(bool(self.gmx_path) and self.bundle.ok)

    def _update_system_summary(self, model, cell_source):
        topo = self.bundle.topology if self.bundle else None
        kinds = {}
        for fragment in model.fragments:
            kinds[fragment.kind] = kinds.get(fragment.kind, 0) + 1
        text = (f"{len(model.atoms)} atoms, {len(model.fragments)} molecules ("
                + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + ")")
        if cell_source:
            text += f"; cell from {cell_source}"
        if topo is not None:
            text += f"; net charge {topo.total_charge:+.3f} e"
        self.system_label.setText(text)
        self.mol_table.setRowCount(0)
        if topo is None:
            return
        counts = {}
        for name, count in topo.molecules():
            counts[name] = counts.get(name, 0) + count
        labels = {"mdff": "force-field file", "water": "built-in water", "ion": "built-in ion",
                  "skeleton": "TODO: none"}
        for name, count in counts.items():
            mt = topo.moltypes[name]
            row = self.mol_table.rowCount()
            self.mol_table.insertRow(row)
            for col, value in enumerate((name, labels[mt.source], str(count), str(mt.size),
                                         f"{mt.charge:+.3f}")):
                self.mol_table.setItem(row, col, QTableWidgetItem(value))
        self.mol_table.resizeColumnsToContents()

    def _set_files(self, files):
        current = self.file_combo.currentText()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for name in files:
            self.file_combo.addItem(name)
        index = self.file_combo.findText(current)
        self.file_combo.setCurrentIndex(max(index, 0))
        self.file_combo.blockSignals(False)
        self._show_file()

    def _show_file(self, *_args):
        name = self.file_combo.currentText()
        text = (self.bundle.files.get(name, "") if self.bundle else "")
        lines = text.splitlines()
        if len(lines) > PREVIEW_LINE_LIMIT:
            text = "\n".join(lines[:PREVIEW_LINE_LIMIT]) + (
                f"\n... ({len(lines) - PREVIEW_LINE_LIMIT} more lines; the saved file is complete)"
            )
        self.preview.setPlainText(text)

    def _browse_mdff(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open force-field file", os.path.dirname(self.mdff_edit.text()),
            "moleditpy-mdff (*.mdff.json *.json);;All files (*)")
        if path:
            self.mdff_edit.setText(path)

    # -------------------------------------------------------------- actions
    def save_files(self, directory=None):
        self.update_preview()
        if not self.bundle or not self.bundle.ok:
            QMessageBox.warning(self, PLUGIN_NAME, "Nothing to save:\n" + self.messages.toPlainText())
            return []
        if not directory:
            directory = QFileDialog.getExistingDirectory(self, "Save GROMACS files to folder")
        if not directory:
            return []
        existing = [n for n in self.bundle.files if os.path.exists(os.path.join(directory, n))]
        if existing:
            answer = QMessageBox.question(
                self, PLUGIN_NAME,
                f"{len(existing)} file(s) already exist in this folder and will be replaced:\n"
                + ", ".join(existing[:12]) + (" ..." if len(existing) > 12 else ""))
            if answer != QMessageBox.StandardButton.Yes:
                return []
        try:
            written = save_bundle(self.bundle, directory)
        except OSError as exc:
            QMessageBox.critical(self, PLUGIN_NAME, f"Could not write the files: {exc}")
            return []
        message = f"Wrote {len(written)} files to {directory}"
        self.messages.appendPlainText(message)
        if self.context is not None:
            try:
                self.context.show_status_message(message, 5000)
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - host guard
                pass
        return written

    def run_check(self):
        if not self.bundle or not self.bundle.ok or not self.gmx_path:
            return
        stages = mdp.enabled_stages(self.read_settings()["mdp"])
        if not stages:
            return
        conf = next(n for n in self.bundle.files if n.endswith(".gro"))
        self.check_btn.setEnabled(False)
        self.messages.appendPlainText(f"Running gmx grompp on {stages[0]}.mdp ...")
        self._worker = GromppWorker(self.bundle.files, conf, stages[0], self)
        self._worker.done.connect(self._check_done)
        self._worker.start()

    def _check_done(self, ok, text):  # pragma: no cover - needs gmx
        self.check_btn.setEnabled(True)
        self.messages.appendPlainText(("grompp succeeded." if ok else "grompp failed.") + "\n" + text[-4000:])

    def closeEvent(self, event):
        self._timer.stop()
        if self._worker is not None and self._worker.isRunning():  # pragma: no cover
            self._worker.wait(5000)
        self.persistent_settings.update(self.read_settings())
        super().closeEvent(event)
