import unittest
from possession_share import summarize_labels


class PossessionShareTests(unittest.TestCase):
    def test_denominator_excludes_dead_and_abstain(self):
        row = summarize_labels([0, 0, 0, 1, 2, 2, 3], "scope", "model")
        self.assertEqual((row["H_frames"], row["A_frames"], row["D_frames"], row["X_frames"]), (3, 1, 2, 1))
        self.assertEqual((row["H_share_percent"], row["A_share_percent"]), (75.0, 25.0))
        self.assertEqual(row["H_A_denominator_frames"], 4)
        self.assertAlmostEqual(row["H_seconds"], .12)

    def test_undefined_when_no_alive(self):
        row = summarize_labels([2, 3, 2], "scope", "model")
        self.assertIsNone(row["H_share_percent"])
        self.assertIsNone(row["A_share_percent"])

    def test_invalid_code(self):
        with self.assertRaises(ValueError):
            summarize_labels([0, 4], "scope", "model")


if __name__ == "__main__":
    unittest.main()
