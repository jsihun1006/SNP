"""합성 데이터로 dead_ball 제약 디코더를 검사한다(대회 데이터 불필요).

    python model/점유율계산/계산알고리즘/model1_dead_ball/test_model.py
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import possession as ps  # noqa: E402


class TestConstraint(unittest.TestCase):
    def test_dead_frames_become_d_and_alive_frames_never_d(self):
        n = 400
        costs = np.zeros((n, 3))
        costs[:, ps.D] = 0.0        # 관측만 보면 D가 공짜(긴 누락)
        costs[:, ps.A] = 0.08       # H 쪽 관측
        graded = np.ones(n, dtype=bool)
        dead = np.zeros(n, dtype=bool)
        dead[100:200] = True
        labels = ps.viterbi(ps.constrained_costs(costs, graded, dead), 1.2, 0.8)
        self.assertTrue((labels[100:200] == ps.D).all())
        self.assertFalse((labels[np.r_[0:100, 200:n]] == ps.D).any())
        self.assertTrue((labels[:100] == ps.H).all())

    def test_ungraded_frames_are_unconstrained(self):
        n = 50
        costs = np.zeros((n, 3))
        costs[:, ps.H] = costs[:, ps.A] = 0.1
        graded = np.zeros(n, dtype=bool)
        labels = ps.viterbi(ps.constrained_costs(costs, graded, np.zeros(n, dtype=bool)), 1.2, 0.8)
        self.assertTrue((labels == ps.D).all())


if __name__ == "__main__":
    unittest.main()
