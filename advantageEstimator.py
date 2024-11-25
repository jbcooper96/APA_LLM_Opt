import torch

class AdvantageEstimator:
    def __init__(self, lam=.9, gamma=.9):
        self.lam = lam
        self.gamma = gamma

    def calc_advantages(self, timestamps):
        bellman_residuals = self.calc_bellman_residuals(timestamps)
        gamma_lam_pow = torch.arange(0, len(timestamps))
        gamma_lam = torch.ones(len(timestamps)) * self.lam * self.gamma
        gamma_lam = torch.pow(gamma_lam, gamma_lam_pow)

        advantages = torch.ones(len(timestamps))
        for i in range(len(timestamps)):
            num_of_elems = len(timestamps) - i
            advantages_to_sum = torch.mul(gamma_lam[:num_of_elems], bellman_residuals[i:])
            advantages[i] = torch.sum(advantages_to_sum)

        return advantages


    
    def calc_bellman_residuals(self, timestamps):
        res = torch.ones(len(timestamps))
        for i in range(len(timestamps)):
            next_state_value = timestamps[i + 1].value_est if i + 1 < len(timestamps) else 0
            res[i] = timestamps[i].reward + self.lam * next_state_value - timestamps[i].value_est

        return res