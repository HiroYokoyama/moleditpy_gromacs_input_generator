import json

import numpy as np
import pytest

from gromacs_input_generator import md_cell


def test_parse_cell_json_round_trip():
    text = md_cell.cell_to_json(np.diag([10.0, 11.0, 12.0]), origin=(1, 2, 3))
    cell = md_cell.parse_cell_json(text)
    assert np.allclose(cell["lattice"], np.diag([10, 11, 12]))
    assert np.allclose(cell["origin"], [1, 2, 3])
    assert cell["pbc"] == [True, True, True]


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"lattice": [[1, 0], [0, 1]]}),
        json.dumps({"lattice": [[1, 0, 0], [2, 0, 0], [0, 0, 1]]}),
        json.dumps({"lattice": np.eye(3).tolist(), "origin": "x"}),
    ],
)
def test_parse_cell_json_rejects_bad_input(text):
    with pytest.raises(md_cell.CellError):
        md_cell.parse_cell_json(text)


def test_origin_defaults_to_zero():
    cell = md_cell.parse_cell_json({"lattice": np.eye(3).tolist()})
    assert np.allclose(cell["origin"], 0)


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("seed", range(5))
def test_gromacs_setting_is_lower_triangular_and_preserves_the_lattice(seed):
    lattice = md_cell.box_from_lengths([20.0, 25.0, 30.0], [70.0, 80.0, 100.0])
    rotated = lattice @ _random_rotation(seed)
    box, rotation = md_cell.gromacs_setting(rotated)
    assert abs(box[0, 1]) < 1e-9 and abs(box[0, 2]) < 1e-9 and abs(box[1, 2]) < 1e-9
    assert box[0, 0] > 0 and box[1, 1] > 0 and box[2, 2] > 0
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    # The rotated input cell and the reduced box describe the same lattice.
    integer = (rotated @ rotation) @ np.linalg.inv(box)
    assert np.allclose(integer, np.round(integer), atol=1e-6)
    assert abs(np.linalg.det(box) - abs(np.linalg.det(lattice))) < 1e-6


def test_reduce_box_meets_gromacs_restrictions():
    box = np.array([[10.0, 0, 0], [8.0, 10.0, 0], [-7.0, 9.0, 10.0]])
    a, b, c = md_cell.reduce_box(box)
    assert abs(b[0]) <= a[0] / 2 + 1e-9
    assert abs(c[0]) <= a[0] / 2 + 1e-9
    assert abs(c[1]) <= b[1] / 2 + 1e-9
    assert np.isclose(np.linalg.det(np.vstack([a, b, c])), np.linalg.det(box))


def test_left_handed_cell_is_rejected():
    with pytest.raises(md_cell.CellError, match="left-handed"):
        md_cell.gromacs_setting(np.diag([10.0, 10.0, -10.0]))


def test_gro_box_line_rectangular_and_triclinic():
    assert md_cell.gro_box_line(np.diag([30.0, 40.0, 50.0])) == "   3.00000   4.00000   5.00000"
    tric = np.array([[30.0, 0, 0], [10.0, 40.0, 0], [5.0, 6.0, 50.0]])
    values = [float(v) for v in md_cell.gro_box_line(tric).split()]
    # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    assert values == [3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.6]


def test_box_from_padding():
    lattice, origin = md_cell.box_from_padding([[0, 0, 0], [10, 2, 4]], 5.0)
    assert np.allclose(np.diag(lattice), [20, 12, 14])
    assert np.allclose(origin, [-5, -5, -5])
    cubic, _ = md_cell.box_from_padding([[0, 0, 0], [10, 2, 4]], 5.0, cubic=True)
    assert np.allclose(np.diag(cubic), [20, 20, 20])
    with pytest.raises(md_cell.CellError):
        md_cell.box_from_padding([], 1.0)
    with pytest.raises(md_cell.CellError):
        md_cell.box_from_padding([[0, 0, 0]], -1.0)


def test_minimum_box_width():
    assert md_cell.minimum_box_width(np.diag([30.0, 20.0, 40.0])) == pytest.approx(20.0)
    hexagonal = md_cell.box_from_lengths([30, 30, 30], [90, 90, 120])
    assert md_cell.minimum_box_width(hexagonal) == pytest.approx(30 * np.sin(np.radians(60)))


def test_box_from_lengths_rejects_impossible_angles():
    with pytest.raises(md_cell.CellError):
        md_cell.box_from_lengths([10, 10, 10], [10, 10, 170])


class _Mol:
    def __init__(self, props):
        self.props = props

    def HasProp(self, key):
        return key in self.props

    def GetProp(self, key):
        return self.props[key]


def test_read_cell_property_from_rdkit_like_object():
    assert md_cell.read_cell_property(None) is None
    assert md_cell.read_cell_property(_Mol({})) is None
    cell = md_cell.read_cell_property(_Mol({"moleditpy_cell": md_cell.cell_to_json(np.eye(3) * 5)}))
    assert np.allclose(cell["lattice"], np.eye(3) * 5)
    with pytest.raises(md_cell.CellError):
        md_cell.read_cell_property(_Mol({"moleditpy_cell": "{"}))
