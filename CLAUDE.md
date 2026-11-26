# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`moleditpy_gromacs_input_generator`, a MoleditPy plugin (package and registry id
`gromacs_input_generator`) that writes .gro/.top/.itp/.mdp/.ndx files and run scripts.
Part of the MD plugin suite (LAMMPS / GROMACS / OpenMM generators, MD System Builder,
Trajectory Analysis, Force Field Builder) that shares the `moleditpy_cell` property and
the `moleditpy-mdff` JSON force-field format described in README.md.

Modules: `system_model.py` (plain-data view of the host molecule), `md_cell.py` (box
handling), `mdff.py` (reader), `topology.py` (type assignment, .top/.itp), `mdp.py`,
`run_scripts.py`, `writer.py` (bundle), `gmx_check.py`, `main_dialog.py`, `__init__.py`.

## Shared modules: do NOT edit them here

`cell_model.py`, `elements.py`, `cell_preview.py`, `structure_panel.py` are
materialized from the `_periodic_shared` submodule (pinned v1.1.1) and gitignored:

```bash
git submodule update --init
python scripts/materialize_shared.py
```

## Testing

```bash
python -m pytest tests/ -v
python -m ruff check .
```

Tests use real numpy/PySide6 (offscreen) and RDKit where needed (CI installs all three).
No `gmx` is available on the development machine; the generated files were cross-checked
by loading them with OpenMM's `GromacsTopFile`.

## Conventions

- No module-level `run()` (the host would add a duplicate Plugins-menu entry).
- No pip dependencies beyond the host.
- Never call `Chem.SetDefaultPickleProperties`; the cell is persisted through the
  plugin's own save/load handlers instead.

## Copied maintenance scripts

See [scripts/COMMON_TOOLS.md](scripts/COMMON_TOOLS.md).
