import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MissileEnv:

    #reset returning an observation
    #render
    #step returning an observation, reward, done, i?
    #obs dim: 13
    #act dim: 3?

    def __init__(self):
        self.target_pos = np.array([np.float64(10.), np.float64(0.), np.float64(0.)])
        self.missile_pos = np.array([np.float64(0.), np.float64(0.), np.float64(0.)])

    def make_obs(self):
        return self.target_pos - self.missile_pos

    def render(self):
        pass

    def reset(self):
        self.target_pos = np.array([get_ran10(), get_ran10(), get_ran10()])
        self.missile_pos = np.array([get_ran10(), get_ran10(), get_ran10()])
        return self.make_obs()

    def step(self, action):
        #self.target_update()
        self.missile_pos += action
        # if np.linalg.norm(self.missile_vel) > 1:
        #     self.missile_vel = normalise(self.missile_vel)
        # self.time += self.d_time
        return self.make_obs(), self.calculate_reward(), self.hit(), None

    def target_update(self):
        accel_mag = 1
        accel_dir = normalise(np.cross(self.target_vel, np.array([np.float64(0.), np.float64(0.), np.float64(1.)])))
        accel = accel_mag * accel_dir
        self.target_pos += self.target_vel * self.d_time + 0.5 * accel * self.d_time * self.d_time
        self.target_vel += accel * self.d_time

    def calculate_reward(self):
        # linear dist?
        diff = self.target_pos - self.missile_pos
        return [-np.linalg.norm(diff)]

    def hit(self):
        pass

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_ran10():
    return (0.5 - np.random.ranf()) * 20

if __name__ == '__main__':
    env = MissileEnv()
    x_t = []
    x_m = []
    y_t = []
    y_m = []
    z_t = []
    z_m = []
    def ran_acc():
        return (0.5 - np.random.ranf())
    for i in range(500):
        o, r, d, i = env.step(np.array([ran_acc(), ran_acc(), ran_acc()]))
        x_t.append(env.target_pos[0])
        y_t.append(env.target_pos[1])
        z_t.append(env.target_pos[2])
        x_m.append(env.missile_pos[0])
        y_m.append(env.missile_pos[1])
        z_m.append(env.missile_pos[2])
        print(r)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_t, y_t, z_t, c = 'r', marker = 'o')
    ax.scatter(x_m, y_m, z_m, c = 'b', marker = 'x')
    plt.show()
