import copy
import json

import pytest
from helpers import example_mdff_path

from gromacs_input_generator import mdff


def _raw(name="ethanol_water_nacl.mdff.json"):
    with open(example_mdff_path(name), encoding="utf-8") as handle:
        return json.load(handle)


def test_examples_load():
    doc = mdff.load_mdff(example_mdff_path())
    assert [mt["name"] for mt in doc["molecule_types"]] == ["ETH", "SOL", "NA", "CL"]
    assert doc["combination_rule"] == 2
    sol = doc["types_by_name"]["SOL"]
    assert sol["settles"] == [{"i": 0, "doh": 0.09572, "dhh": 0.15139}]
    eth = doc["types_by_name"]["ETH"]
    assert abs(sum(a["charge"] for a in eth["atoms"])) < 1e-9
    assert mdff.load_mdff(example_mdff_path("water.mdff.json"))["system"][0]["count"] == 2


def test_coverage_and_element_check():
    doc = mdff.load_mdff(example_mdff_path())
    spans, warnings = mdff.coverage(doc, 17)
    assert spans[0] == (0, 9, "ETH") and spans[-1] == (16, 17, "CL")
    assert warnings == []
    elements = list("CCOHHHHHH") + ["O", "H", "H"] * 2 + ["Na", "Cl"]
    assert mdff.element_mismatches(doc, spans, elements) == []
    elements[0] = "N"
    assert "host atom 1 is N" in mdff.element_mismatches(doc, spans, elements)[0]


def test_coverage_beyond_the_molecule_is_reported():
    doc = mdff.load_mdff(example_mdff_path())
    spans, warnings = mdff.coverage(doc, 12)
    assert spans[-1][1] <= 12
    assert "describes 17 atoms" in warnings[0]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(format="other"), "format"),
        (lambda d: d.update(version=2), "version"),
        (lambda d: d.pop("version"), "version"),
        (lambda d: d.update(combination_rule=4), "combination_rule"),
        (lambda d: d.update(molecule_types=[]), "molecule_types"),
        (lambda d: d["molecule_types"].append(copy.deepcopy(d["molecule_types"][0])), "twice"),
        (lambda d: d["system"].append({"type": "XYZ", "count": 1}), "unknown molecule type"),
        (lambda d: d["system"][1].update(atom_offset=3), "overlaps"),
        (lambda d: d["molecule_types"][0]["bonds"][0].update(j=99), "outside"),
        (lambda d: d["molecule_types"][0]["atoms"][0].update(charge="x"), "not a number"),
        (lambda d: d["molecule_types"][0]["atoms"][0].update(sigma=float("nan")), "not finite"),
        (lambda d: d["molecule_types"][0].update(name=""), "no name"),
        (lambda d: d["system"][0].update(count=0), "at least 1"),
    ],
)
def test_invalid_documents_are_rejected(mutate, message):
    data = _raw()
    mutate(data)
    with pytest.raises(mdff.MdffError, match=message):
        mdff.parse_mdff(data)


def test_settles_and_constraints_variants():
    data = _raw()
    sol = data["molecule_types"][1]
    sol["settles"] = [{"oxygen": 0, "doh": 0.1, "dhh": 0.1633}]
    sol["constraints"] = [{"i": 0, "j": 1, "length": 0.1}]
    doc = mdff.parse_mdff(data)
    assert doc["types_by_name"]["SOL"]["settles"][0]["doh"] == 0.1
    assert doc["types_by_name"]["SOL"]["constraints"][0]["r0"] == 0.1


def test_offsets_default_to_consecutive():
    data = _raw()
    for entry in data["system"]:
        entry.pop("atom_offset")
    doc = mdff.parse_mdff(data)
    assert [e["atom_offset"] for e in doc["system"]] == [0, 9, 15, 16]


def test_unreadable_files(tmp_path):
    with pytest.raises(mdff.MdffError, match="Cannot read"):
        mdff.load_mdff(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(mdff.MdffError, match="not valid JSON"):
        mdff.load_mdff(str(bad))
