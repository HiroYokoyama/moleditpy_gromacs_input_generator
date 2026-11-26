"""GROMACS Input Generator plugin for MoleditPy.

Writes conf.gro, topol.top + .itp files, EM/NVT/NPT/production .mdp files,
an index file and run.sh/run.bat from the current molecule.
"""

import logging
import os

_log = logging.getLogger(__name__)

PLUGIN_NAME = "GROMACS Input Generator"
PLUGIN_VERSION = "0.1.2"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Write GROMACS inputs (.gro with triclinic boxes, .top/.itp from a moleditpy-mdff "
    "force-field file or built-in water/ion tables, EM/NVT/NPT/MD .mdp sets, index "
    "groups and a grompp/mdrun run script) from the current molecule or MD system."
)
PLUGIN_CATEGORY = "Input Generator"
PLUGIN_TAGS = ["Generator", "MD"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = []
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_LICENSE = "LGPL-3.0-or-later"

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
WINDOW_ID = "gromacs_input_generator_dialog"

from .writer import default_settings

_context = None
_dialog_opened = False

# Last cell seen on the molecule: {"json": str, "natoms": int}.  The host's
# undo snapshots and project files drop molecule-level RDKit properties, so the
# plugin keeps a copy and restores it.
remembered_cell = {}


def get_default_settings():
    return default_settings()


current_settings = get_default_settings()


def remember_cell(mol):
    """Record the molecule's moleditpy_cell (if any) for later restoration."""
    from .md_cell import CELL_PROPERTY

    try:
        if mol is not None and mol.HasProp(CELL_PROPERTY):
            remembered_cell.clear()
            remembered_cell.update({"json": mol.GetProp(CELL_PROPERTY), "natoms": mol.GetNumAtoms()})
            return True
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - RDKit-like guard
        _log.debug("%s: could not read the cell property: %s", PLUGIN_NAME, exc)
    return False


def restore_cell(mol):
    """Put the remembered cell back on a molecule that lost it (same atom count)."""
    from .md_cell import CELL_PROPERTY

    if mol is None or not remembered_cell:
        return False
    try:
        if mol.HasProp(CELL_PROPERTY):
            return False
        if mol.GetNumAtoms() != remembered_cell.get("natoms"):
            return False
        mol.SetProp(CELL_PROPERTY, remembered_cell["json"])
        return True
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - RDKit-like guard
        _log.debug("%s: could not restore the cell property: %s", PLUGIN_NAME, exc)
        return False


def _current_molecule(mw=None):
    try:
        if _context is not None:
            return _context.current_molecule
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
        _log.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
    return getattr(mw, "current_mol", None)


def _open_dialog(mw):
    """Open the dialog (deliberately not named ``run``; see the template)."""
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

    from .main_dialog import GromacsInputDialog
    from .structure_panel import find_cif_viewer_widget

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - host API guard
                pass

    _dialog_opened = True
    dlg = GromacsInputDialog(
        parent=mw,
        persistent_settings=current_settings,
        get_molecule=lambda: _current_molecule(mw),
        mark_modified=_mark_modified,
        get_cif_viewer=lambda: find_cif_viewer_widget(mw),
        context=_context,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()


def initialize(context):
    """Entry point for the plugin API, called once at start-up."""
    global _context
    _context = context

    def show_dialog():
        _open_dialog(context.get_main_window())

    context.add_export_action("GROMACS Input (.gro/.top/.mdp)...", show_dialog)

    def save_state():
        state = {}
        remember_cell(_current_molecule())
        if remembered_cell:
            state["cell"] = dict(remembered_cell)
        if _dialog_opened:
            state["settings"] = dict(current_settings)
        return state

    def load_state(data):
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if isinstance(saved, dict):
            current_settings.update(saved)
            dlg = context.get_window(WINDOW_ID)
            if dlg is not None:
                try:
                    dlg.apply_settings(current_settings)
                except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
                    _log.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)
        cell = data.get("cell")
        if isinstance(cell, dict) and isinstance(cell.get("json"), str):
            remembered_cell.clear()
            remembered_cell.update({"json": cell["json"], "natoms": int(cell.get("natoms") or 0)})
            # Load handlers run before the host restores the molecule: re-apply
            # the cell once the event loop has turned.
            try:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda: restore_cell(_current_molecule()))
            except Exception as exc:  # noqa: BLE001 - pragma: no cover - headless guard
                _log.debug("%s: could not schedule the cell restore: %s", PLUGIN_NAME, exc)

    def handle_reset():
        global _dialog_opened
        remembered_cell.clear()
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            return
        current_settings.clear()
        current_settings.update(get_default_settings())
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
