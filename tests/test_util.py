import unittest
import numpy.testing as npt


class TestFoo(unittest.TestCase):
    def test_foo(self):
        npt.assert_equal(True, True)
