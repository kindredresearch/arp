import numpy as np
import matplotlib.pyplot as plt
import time
import gym


class SquareEnvironment(gym.core.Env):
    def __init__(self,
                 size=10,
                 target_size = 0.5,
                 control="velocity",
                 dt=0.1,
                 n_steps=1000,
                 visualize=False,
                 velocity_lim = 1.0):
        plt.ion()
        self.size = size
        self.n_steps = n_steps
        self.target_size = target_size
        self.control = control
        self.velocity = np.zeros(2)
        self.pos = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.dt = dt
        self.visualize = visualize
        self.time = 0
        self.velocity_lim = velocity_lim
        self.trajectory = []
        self.env = self
        self.current_steps = 0
        self.np_random = np.random
        from gym.spaces import Box as GymBox  # use this for baselines algos
        Box = GymBox
        self._observation_space = Box(
        low= -np.ones(6),
        high=np.ones(6)
        )
        self._action_space = Box(low=-np.ones(2), high=np.ones(2))
        self.fig = None
        if self.visualize:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.hl_target, = self.ax.plot([], [], markersize=10, marker="o", color='r')
            self.hl, = self.ax.plot([], [])

    def step(self, action):
        self.current_steps += 1
        if self.control == "velocity":
            self.velocity = np.clip(action, -1, 1).flatten()
        elif self.control == "acceleration":
            self.acceleration = np.clip(action, -1, 1).flatten()
        self.velocity += self.acceleration * self.dt
        self.velocity = np.clip(self.velocity, - self.velocity_lim, self.velocity_lim)
        self.pos += self.velocity * self.dt + self.acceleration * self.dt**2/2
        clipped_pos = np.clip(self.pos, -self.size/2, self.size/2)
        self.velocity[clipped_pos!=self.pos] = 0
        self.acceleration[clipped_pos!=self.pos] = 0
        self.pos = clipped_pos
        reward = np.linalg.norm(self.pos - self.target_pos) < self.target_size
        done = reward > 0 or self.current_steps >= self.n_steps
        reward -= 0.1 * self.dt
        self.time += self.dt
        if self.visualize:
            self.trajectory.append(self.pos)
            self.hl_target.set_xdata(self.target_pos[0])
            self.hl_target.set_ydata(self.target_pos[1])
            self.hl.set_xdata(np.array(self.trajectory)[:, 0])
            self.hl.set_ydata(np.array(self.trajectory)[:, 1])
            self.ax.set_ylim([-self.size/2, self.size/2])
            self.ax.set_xlim([-self.size/2, self.size/2])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        new_ob = np.hstack([self.pos * 2/self.size, self.velocity, (self.target_pos - self.pos) * 2/self.size])
        return new_ob, reward, done, {}

    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.hl_target, = self.ax.plot(self.target_pos[0], self.target_pos[1], markersize=10, marker="o", color='r')
            self.hl, = self.ax.plot([], [])

        self.hl_target.set_xdata(self.target_pos[0])
        self.hl_target.set_ydata(self.target_pos[1])
        self.hl.set_xdata(np.array(self.trajectory)[:, 0])
        self.hl.set_ydata(np.array(self.trajectory)[:, 1])
        self.ax.set_ylim([-self.size / 2, self.size / 2])
        self.ax.set_xlim([-self.size / 2, self.size / 2])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        self.velocity = np.zeros(2)
        self.pos = np.zeros(2)
        self.acceleration = np.zeros(2)
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
    env = SquareEnvironment(visualize=False, control='velocity', dt=1, target_type='random')
    ob = env.reset()
    ep_returns = []
    ep_return = 0
    ac = np.zeros(2,)
    value = 0.0
    alpha = value
    beta = value
    gamma = value
    sigma = np.sqrt((1 - alpha * beta) * (1 - gamma * beta) * (1 - alpha * gamma) / (
            1 + alpha * beta + alpha * gamma + beta * gamma - alpha * beta * gamma * (
            alpha * beta * gamma + alpha + beta + gamma)))
    a = np.random.normal(size=ac.shape) * sigma
    v = np.random.normal(size=ac.shape) * sigma * np.sqrt((1 + alpha * beta) / (1 - alpha * beta))
    noise = np.random.normal(size=ac.shape)
    x_tick = 1

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    hl11, = ax2.plot([], [])
    steps = 0
    while steps < 100000:
        steps += 1
        rnd = np.random.normal(size=ac.shape)
        noise = gamma * noise + np.sqrt(1 - gamma ** 2) * v
        v = beta * v + np.sqrt(1 - beta ** 2) * a
        a = alpha * a + np.sqrt(1 - alpha ** 2) * sigma * rnd
        action = noise
        #action = rnd
        ob, reward, done = env.step(action)
        ind_to_reset = np.abs(np.abs(ob[:2]) - env.size/2) < 1e-3
        if np.any(ind_to_reset):
            rnd = np.random.normal(size=ac.shape)
            a[ind_to_reset] = (np.abs(np.random.normal(size=ac.shape))*( - np.sign(ob[:2])) * sigma)[ind_to_reset]
            v[ind_to_reset] = (np.abs(np.random.normal(size=ac.shape)) * ( - np.sign(ob[:2])) * sigma * np.sqrt((1 + alpha * beta) / (1 - alpha * beta)))[ind_to_reset]
            noise[ind_to_reset] = (np.abs(np.random.normal(size=ac.shape))*(-np.sign(ob[:2])))[ind_to_reset]
        ep_return += reward
        if done:
            ep_returns.append(ep_return)
            print(ep_return)
            ep_return = 0
            ob = env.reset()

            alpha = value
            beta = value
            gamma = value
            sigma = np.sqrt((1 - alpha * beta) * (1 - gamma * beta) * (1 - alpha * gamma) / (
                    1 + alpha * beta + alpha * gamma + beta * gamma - alpha * beta * gamma * (
                    alpha * beta * gamma + alpha + beta + gamma)))
            a = np.random.normal(size=ac.shape) * sigma
            v = np.random.normal(size=ac.shape) * sigma * np.sqrt((1 + alpha * beta) / (1 - alpha * beta))
            noise = np.random.normal(size=ac.shape)

            hl11.set_xdata(np.arange(1, len(ep_returns) + 1))
            ax2.set_xlim([x_tick, len(ep_returns) * x_tick])
            hl11.set_ydata(ep_returns)
            ax2.set_ylim([np.min(ep_returns), np.max(ep_returns) + 50])
            time.sleep(0.01)
            fig.canvas.draw()
            fig.canvas.flush_events()
    print("mean return", np.mean(ep_returns), len(ep_returns))




