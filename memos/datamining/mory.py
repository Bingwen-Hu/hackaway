# dense coding
import numpy as np

def densecoding(labels):
    """input labels and return dense coding"""
    num_of_class = len(set(labels))
    for i in range(1, 100):
        if 2**i >= num_of_class:
            break
    vec = np.zeros(i)
    return vec

import unittest
class TestMory(unittest.TestCase):
    def test_densecoding(self):
        labels = ['A', 'B', 'C']
        self.assertEqual(densecoding(labels), 2)
        labels.extend(['C', 'D', 'E', 'A', 'B'])
        self.assertEqual(densecoding(labels), 3)
        labels.extend(['1', '2,', '4'])
        self.assertEqual(densecoding(labels), 3)
        labels.append('[Mory')
        self.assertEqual(densecoding(labels), 4)

def runtest():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMory))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))