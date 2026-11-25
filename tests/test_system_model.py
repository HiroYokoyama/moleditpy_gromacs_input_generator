import numpy as np
import pytest
from helpers import assemble, ethanol, example_system, ion, water

from gromacs_input_generator import system_model


def test_fragments_and_kinds():
    model = example_system()
    kinds = [(f.kind, f.size) for f in model.fragments]
    assert kinds == [("other", 9), ("water", 3), ("water", 3), ("ion", 1), ("ion", 1)]
    assert model.net_charge == 0


def test_identical_fragments_share_a_signature():
    model = assemble([ethanol(), ethanol((5, 0, 0)), water((10, 0, 0))])
    a, b, w = model.fragments
    assert a.signature == b.signature
    assert a.signature != w.signature


def test_water_order_puts_oxygen_first():
    model = assemble([water(order="HOH")])
    fragment = model.fragments[0]
    assert fragment.kind == "water"
    order = system_model.water_order(model, fragment)
    assert [model.atoms[i].element for i in order] == ["O", "H", "H"]


def test_bonds_are_guessed_only_when_asked_and_missing():
    elements, coords, _ = water()
    model = system_model.build_system(elements, coords, infer_bonds=True)
    assert model.bonds == [(0, 1), (0, 2)]
    assert model.fragments[0].kind == "water"
    unbonded = system_model.build_system(elements, coords, infer_bonds=False)
    assert len(unbonded.fragments) == 3


def test_interleaved_fragment_atoms():
    model = system_model.build_system(["C", "Na", "O"], [[0, 0, 0], [5, 5, 5], [1.2, 0, 0]], [(0, 2)])
    assert [f.atoms for f in model.fragments] == [[0, 2], [1]]


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        system_model.build_system(["C"], np.zeros((2, 3)))


def test_lone_ion_is_classified_as_ion():
    model = assemble([ion("K", (0, 0, 0))])
    assert model.fragments[0].kind == "ion"


def test_unbonded_water_with_water_residue_name():
    elements, coords, _ = water()
    model = system_model.build_system(elements, coords, [(0, 1), (1, 2)], resnames=["HOH"] * 3,
                                      resnums=[1] * 3)
    assert model.fragments[0].kind == "water"


def test_protein_residue_names_mark_protein():
    model = system_model.build_system(["N", "C"], [[0, 0, 0], [1.4, 0, 0]], [(0, 1)],
                                      resnames=["ALA", "ALA"], resnums=[1, 1])
    assert model.has_residue_info
    assert model.fragments[0].kind == "protein"


def test_from_rdkit_reads_residue_info_and_charges():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("O.[Na+]"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("NA" if atom.GetSymbol() == "Na" else "SOL")
        info.SetResidueNumber(2 if atom.GetSymbol() == "Na" else 1)
        info.SetName(atom.GetSymbol())
        atom.SetMonomerInfo(info)
    model = system_model.from_rdkit(mol)
    assert model.has_residue_info
    assert sorted(f.kind for f in model.fragments) == ["ion", "water"]
    assert sum(a.formal_charge for a in model.atoms) == 1


def test_from_rdkit_needs_coordinates():
    Chem = pytest.importorskip("rdkit.Chem")
    with pytest.raises(ValueError, match="3D"):
        system_model.from_rdkit(Chem.MolFromSmiles("C"))
    with pytest.raises(ValueError):
        system_model.from_rdkit(None)


def test_from_rdkit_rejects_dummy_atoms():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    editable = Chem.RWMol(mol)
    editable.GetAtomWithIdx(1).SetAtomicNum(0)
    with pytest.raises(ValueError, match="dummy"):
        system_model.from_rdkit(editable.GetMol())
