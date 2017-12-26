import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath('..'), 'src'))

# if I can't find it, I can't test it
# TODO: the path is missing
from cube_root import cube_root

cubes = (
    (0, 0),
    (1, 1),
    (8, 2),
    (27, 2),
)


@pytest.mark.parametrize('n,expected', cubes)
def test_cube_root(n, expected):
    assert cube_root(n) == expected
