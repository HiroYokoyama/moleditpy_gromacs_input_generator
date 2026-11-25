"""Metadata, registration and session-handler tests (template contract)."""

import re

import pytest

import gromacs_input_generator as plugin


class FakeContext:
    def __init__(self, main_window=None):
        self.main_window = main_window
        self.menu_actions, self.export_actions, self.analysis_tools = [], [], []
        self.save_handlers, self.load_handlers, self.reset_handlers = [], [], []
        self.windows = {}
        self.current_molecule = None
        self.modified = 0

    def add_menu_action(self, path, callback):
        self.menu_actions.append((path, callback))

    def add_export_action(self, label, callback):
        self.export_actions.append((label, callback))

    def add_analysis_tool(self, label, callback):
        self.analysis_tools.append((label, callback))

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def register_window(self, window_id, window):
        self.windows[window_id] = window

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def get_main_window(self):
        return self.main_window

    def mark_project_modified(self):
        self.modified += 1


class FakeMol:
    def __init__(self, natoms=3, props=None):
        self.natoms = natoms
        self.props = dict(props or {})

    def HasProp(self, key):
        return key in self.props

    def GetProp(self, key):
        return self.props[key]

    def SetProp(self, key, value):
        self.props[key] = value

    def GetNumAtoms(self):
        return self.natoms


CELL = '{"lattice": [[10,0,0],[0,10,0],[0,0,10]], "origin": [0,0,0], "pbc": [true,true,true]}'


@pytest.fixture
def context():
    original = dict(plugin.current_settings)
    ctx = FakeContext()
    plugin.initialize(ctx)
    yield ctx
    plugin._context = None
    plugin._dialog_opened = False
    plugin.remembered_cell.clear()
    plugin.current_settings.clear()
    plugin.current_settings.update(original)


def test_required_metadata():
    for name in ("PLUGIN_NAME", "PLUGIN_VERSION", "PLUGIN_AUTHOR", "PLUGIN_DESCRIPTION",
                 "PLUGIN_SUPPORTED_MOLEDITPY_VERSION", "PLUGIN_LICENSE", "PLUGIN_CATEGORY"):
        assert getattr(plugin, name, "").strip(), f"{name} must be set"
    assert plugin.PLUGIN_VERSION == "0.1.0"
    assert plugin.PLUGIN_AUTHOR == "HiroYokoyama"
    assert plugin.PLUGIN_LICENSE == "LGPL-3.0-or-later"
    assert plugin.PLUGIN_CATEGORY == "Input Generator"
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)


def test_no_extra_dependencies():
    assert plugin.PLUGIN_DEPENDENCIES == []
    assert isinstance(plugin.PLUGIN_OPTIONAL_DEPENDENCIES, list)
    assert 1 <= len(plugin.PLUGIN_TAGS) <= 3


def test_supported_version_range():
    from packaging.specifiers import SpecifierSet

    supported = SpecifierSet(plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION)
    assert supported.contains("5.0.0") and supported.contains("5.3.2")
    assert not supported.contains("4.9.9") and not supported.contains("6.0.0")


def test_no_run_or_autorun():
    assert not hasattr(plugin, "run")
    assert not hasattr(plugin, "autorun")


def test_initialize_registers_one_export_entry(context):
    assert len(context.export_actions) == 1
    assert context.menu_actions == [] and context.analysis_tools == []
    assert "GROMACS" in context.export_actions[0][0]
    assert len(context.save_handlers) == len(context.load_handlers) == len(context.reset_handlers) == 1


def test_save_handler_is_silent_until_used(context):
    assert context.save_handlers[0]() == {}


def test_save_handler_emits_settings_after_use(context):
    plugin._dialog_opened = True
    plugin.current_settings["water_model"] = "SPC/E"
    assert context.save_handlers[0]()["settings"]["water_model"] == "SPC/E"


def test_save_handler_persists_the_cell(context):
    context.current_molecule = FakeMol(3, {"moleditpy_cell": CELL})
    state = context.save_handlers[0]()
    assert state["cell"] == {"json": CELL, "natoms": 3}
    assert "settings" not in state


def test_load_handler_restores_settings_and_remembers_the_cell(context):
    context.load_handlers[0]({"settings": {"water_model": "SPC/E"}, "cell": {"json": CELL, "natoms": 3}})
    assert plugin.current_settings["water_model"] == "SPC/E"
    assert plugin.remembered_cell == {"json": CELL, "natoms": 3}


def test_restore_cell_only_for_matching_atom_count(context):
    plugin.remembered_cell.update({"json": CELL, "natoms": 3})
    other = FakeMol(4)
    assert plugin.restore_cell(other) is False and not other.HasProp("moleditpy_cell")
    same = FakeMol(3)
    assert plugin.restore_cell(same) is True and same.GetProp("moleditpy_cell") == CELL
    assert plugin.restore_cell(same) is False  # already has it
    assert plugin.restore_cell(None) is False


def test_load_handler_ignores_junk(context):
    before = dict(plugin.current_settings)
    for junk in (None, {}, {"settings": "x"}, {"cell": {"json": 5}}):
        context.load_handlers[0](junk)
    assert plugin.current_settings == before
    assert plugin.remembered_cell == {}


def test_reset_handler(context):
    plugin._dialog_opened = True
    plugin.remembered_cell.update({"json": CELL, "natoms": 3})
    plugin.current_settings["water_model"] = "SPC/E"
    context.reset_handlers[0]()
    assert plugin.current_settings == plugin.get_default_settings()
    assert plugin._dialog_opened is False and plugin.remembered_cell == {}


def test_reset_handler_leaves_an_open_dialog_alone(context):
    class _Dialog:
        def isVisible(self):
            return True

    context.windows[plugin.WINDOW_ID] = _Dialog()
    plugin.current_settings["water_model"] = "SPC/E"
    context.reset_handlers[0]()
    assert plugin.current_settings["water_model"] == "SPC/E"


def test_load_handler_survives_a_deleted_dialog(context):
    class _Dialog:
        def apply_settings(self, settings):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    context.windows[plugin.WINDOW_ID] = _Dialog()
    context.load_handlers[0]({"settings": {"water_model": "SPC/E"}})
    assert plugin.current_settings["water_model"] == "SPC/E"
