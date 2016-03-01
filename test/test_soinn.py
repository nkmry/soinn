import unittest
import numpy as np
from soinn import Soinn


class TestSoinn(unittest.TestCase):
    def setUp(self):
        self.soinn = Soinn()

    def test_input_signal(self):
        pass

    def test_check_signal(self):
        signal=[0, 1, 2]
        self.assertRaises(TypeError, self.soinn._Soinn__check_signal, signal)
        signal=np.arange(6).reshape(2, 3)
        self.assertRaises(TypeError, self.soinn._Soinn__check_signal, signal)
        d = 6
        signal = np.arange(d)
        self.soinn._Soinn__check_signal(signal)
        self.assertTrue(hasattr(self.soinn, 'dim'))
        self.assertEqual(self.soinn.dim, d)
        signal = np.arange(d + 1)
        self.assertRaises(TypeError, self.soinn._Soinn__check_signal, signal)

if __name__ == '__main__':
    unittest.main()
