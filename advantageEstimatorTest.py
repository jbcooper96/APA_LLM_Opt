import unittest
from advantageEstimator import AdvantageEstimator
from timeStamp import TimeStamp
import torch

def value_func(state):
    return state

class AdvantageEstimatorTest(unittest.TestCase):
    def test_calc_bellman_residuals(self):
        TIME_STAMP_COUNT = 5
        time_stamps = []
        """
        states =  0 1 0 1 0
        rewards = 0 0 0 0 1
        r       = .5 -1 .5 -1 1
        """
        for i in range(TIME_STAMP_COUNT):
            reward = 0 if i < TIME_STAMP_COUNT - 1 else 1
            done = i == TIME_STAMP_COUNT - 1 
            time_stamps.append(TimeStamp(i % 2, 1, reward, i % 2, done))

        advantageEstimator = AdvantageEstimator(.5, .5)
        b_residuals = advantageEstimator.calc_bellman_residuals(time_stamps)
        
        self.assertEqual(b_residuals.size()[0], TIME_STAMP_COUNT)
        self.assertEqual(torch.sum(b_residuals).item(), 0.0)

    def test_calc_advantages(self):
        TIME_STAMP_COUNT = 3
        time_stamps = []
        """
        states =  0 1 0 
        rewards = 0 0 1 
        r       = .5 -1 1
        pow       1 .25 .0625
        """
        for i in range(TIME_STAMP_COUNT):
            reward = 0 if i < TIME_STAMP_COUNT - 1 else 1
            done = i == TIME_STAMP_COUNT - 1 
            time_stamps.append(TimeStamp(i % 2, 1, reward, i % 2, done))

        advantageEstimator = AdvantageEstimator(.5, .5)
        advantage = advantageEstimator.calc_advantages(time_stamps)

        self.assertEqual(advantage[0], 0.3125)
        self.assertEqual(advantage[1], -0.75)
        self.assertEqual(advantage[2], 1)


        

if __name__ == '__main__':
    unittest.main()


