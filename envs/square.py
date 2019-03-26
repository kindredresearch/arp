import numpy as np
import matplotlib.pyplot as plt
import gym
from arp.arp import ARProcess
import time

class SquareEnvironment(gym.core.Env):
    def __init__(self,
                 size=10,
                 target_size = 0.5,
                 dt=0.1,
                 n_steps=10000,
                 visualize=False,
                 velocity_lim = 1.0):
        plt.ion()
        self.size = size
        self.n_steps = n_steps
        self.target_size = target_size
        self.velocity = np.zeros(2)
        self.pos = np.zeros(2)
        self.dt = dt
        self.visualize = visualize
        self.time = 0
        self.velocity_lim = velocity_lim
        self.trajectory = []
        self.env = self
        self.current_steps = 0
        from gym.spaces import Box as GymBox
        Box = GymBox
        self._observation_space = Box(
        low= -np.ones(6),
        high=np.ones(6)
        )
        self._action_space = Box(low=-np.ones(2), high=np.ones(2))
        self.fig = None
        if self.visualize:
            plt.ion()
            self.fig = plt.figure(figsize=(6,6))
            self.ax = self.fig.add_subplot(111)
            self.hl_target, = self.ax.plot([], [], markersize=25, marker="o", color='r')
            self.hl_agent, = self.ax.plot([], [], markersize=10, marker="o", color='b')
            self.hl, = self.ax.plot([], [])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("Agent Trajectory")

    def step(self, action):
        self.current_steps += 1
        self.velocity = np.clip(action, - self.velocity_lim, self.velocity_lim).flatten()
        self.pos += self.velocity * self.dt
        clipped_pos = np.clip(self.pos, -self.size/2, self.size/2)
        self.velocity[clipped_pos!=self.pos] = 0
        self.pos = clipped_pos
        reward = np.linalg.norm(self.pos - self.target_pos) < self.target_size
        done = reward > 0 or self.current_steps >= self.n_steps
        reward -= self.dt
        self.time += self.dt
        if self.visualize:
            self.trajectory.append(self.pos)
            self.hl_target.set_xdata(self.target_pos[0])
            self.hl_target.set_ydata(self.target_pos[1])
            self.hl_agent.set_xdata(self.pos[0])
            self.hl_agent.set_ydata(self.pos[1])
            self.hl.set_xdata(np.array(self.trajectory)[:, 0])
            self.hl.set_ydata(np.array(self.trajectory)[:, 1])
            self.ax.set_ylim([-self.size/2, self.size/2])
            self.ax.set_xlim([-self.size/2, self.size/2])
            time.sleep(0.02)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        new_ob = np.hstack([self.pos * 2/self.size, self.velocity, (self.target_pos - self.pos) * 2/self.size])
        return new_ob, reward, done, {}

    def reset(self):
        self.velocity = np.zeros(2)
        self.pos = np.zeros(2)
        self.current_steps = 0
        self.time = 0
        self.trajectory = []
        self.target_pos = 2 * np.random.random(size = (2,)) - 1
        self.target_pos /= np.linalg.norm(self.target_pos)
        self.target_pos *= self.size/4
        new_ob = np.hstack([self.pos, self.velocity, self.target_pos])
        return new_ob

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

if __name__ == "__main__":
    plt.ion()
    env = SquareEnvironment(visualize=True)
    ob = env.reset()
    p = 3
    alpha = 0.8
    ar = ARProcess(p=p, alpha=alpha, size=env.action_space.shape[-1])
    ar.reset()
    steps = 0
    while steps < 10000:
        x, _ = ar.step()
        ob, reward, done, _ = env.step(x)
        if done:
            env.reset()
            ar.reset()





