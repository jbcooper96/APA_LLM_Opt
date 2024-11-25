class TimeStamp:
    def __init__(self, state=None, action=None, reward=None, value_est=None, done=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.value_est = value_est
        self.done = done