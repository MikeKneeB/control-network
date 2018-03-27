import numpy as np

class MissileEnv:

    #reset returning an observation
    #render
    #step returning an observation, reward, d, i?

    def __init__(self):
        self.max_bounds = np.array([1, 1, 1])
        self.target_pos = np.array([0, 0, 0])
        self.target_vel = np.array([0, 0, 0])
        self.missile_pos = np.array([0, 0, 0])
        self.missile_vel = np.array([0, 0, 0])

    def reset():
