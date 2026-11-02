import unittest
import numpy as np
from geometry import football_anchors

class TestFootballAnchors(unittest.TestCase):
    def test_ball_center_and_all_human_feet(self):
        boxes = [[10,20,30,80]]*4
        np.testing.assert_allclose(football_anchors(boxes, [0,1,2,3]),
            [[20,50],[20,80],[20,80],[20,80]])
    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            football_anchors([[0,0,1,1]], [4])
    def test_empty_result(self):
        self.assertEqual(football_anchors([], []).shape, (0,2))

if __name__ == '__main__': unittest.main()
