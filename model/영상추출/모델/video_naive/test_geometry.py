import unittest
import numpy as np
from geometry import box_anchors, image_to_pitch


class GeometryTests(unittest.TestCase):
    def test_person_and_ball_anchors(self):
        anchors = box_anchors([[10, 20, 30, 80], [100, 100, 110, 110]], [0, 32])
        np.testing.assert_allclose(anchors, [[20, 80], [105, 105]])

    def test_projective_roundtrip(self):
        matrix = np.array([[1000, 200, 10], [20, 500, 30], [.1, -.2, 1.]])
        original = np.array([[.1, .2], [.8, .7]])
        homogeneous = np.column_stack((original, np.ones(2))) @ matrix.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
        result = image_to_pitch(matrix, pixels)
        np.testing.assert_allclose(result['xy_normalized'], original)
        np.testing.assert_allclose(result['xy_metres'], original * [105.3, 68])
        self.assertTrue(result['projection_valid'].all())

    def test_invalid_and_singular_matrices(self):
        for matrix, valid in [(np.eye(3), False), (np.zeros((3, 3)), True), (np.full((3, 3), np.nan), True)]:
            result = image_to_pitch(matrix, [[10, 10]], valid=valid)
            self.assertFalse(result['projection_valid'].any())
            self.assertTrue(np.isnan(result['xy_normalized']).all())

    def test_outside_pitch_is_flagged_not_clamped(self):
        result = image_to_pitch(np.eye(3), [[-0.1, .5]])
        self.assertTrue(result['projection_valid'][0])
        self.assertFalse(result['in_pitch'][0])
        self.assertEqual(result['xy_normalized'][0, 0], -.1)


if __name__ == '__main__':
    unittest.main()
