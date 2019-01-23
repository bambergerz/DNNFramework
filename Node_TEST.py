import unittest
import Node
import numpy as np


class PlusTest(unittest.TestCase):
    
    def __init__(self):
        super(PlusTest, self).__init__()
        self.node = Node.Plus()
        self.a = np.array([1, 2, 3])
        self.b = np.array([4, 5, 6])
        self.result = np.array([5, 7, 9])

    def test_forward(self):
        self.assertEqual(self.result,
                         self.node.forward(self.a, self.b))


if __name__ == '__main__':
    unittest.main()
