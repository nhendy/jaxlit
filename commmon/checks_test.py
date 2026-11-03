import eincheck
import numpy as np
import pytest

from commmon.checks import set_shape_checks, shape_checks_enabled


@eincheck.check_func("n 2 -> n")
def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


@pytest.fixture(autouse=True)
def restore() -> object:
    yield
    set_shape_checks(True)


def test_off_skips_checks_and_on_restores_them() -> None:
    bad = np.zeros((4, 3))
    with pytest.raises(ValueError):
        _norm(bad)
    set_shape_checks(False)
    assert not shape_checks_enabled()
    _norm(bad)  # not checked
    set_shape_checks(True)
    with pytest.raises(ValueError):
        _norm(bad)


def test_setting_the_same_value_twice_is_a_no_op() -> None:
    set_shape_checks(False)
    set_shape_checks(False)
    set_shape_checks(True)
    with pytest.raises(ValueError):
        _norm(np.zeros((4, 3)))
