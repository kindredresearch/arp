#!/usr/bin/env python3
from arp.arp import ARProcess
from baselines.common import tf_util as U
from common.utils import square_arg_parser
import time
import matplotlib.pyplot as plt
import numpy as np
from rl_experiments.normalized_env import NormalizedEnv
from examples.square import SquareEnvironment

def train(dt, num_timesteps, seed, p, alpha):
    from common import ar_mlp_policy
    from ar_ppo import ar_pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    np.random.seed(seed)
    env = SquareEnvironment(visualize=False, dt=dt, n_steps=int(1000/dt))
    env = NormalizedEnv(env)
    ar = ARProcess(p, alpha, size=env.action_space.shape[-1])
    def policy_fn(name, ob_space, ac_space):
        return ar_mlp_policy.ARMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
           hid_size=64, num_hid_layers=2, phi=ar.phi, sigma_z=ar.sigma_z)
    # Plot learning curve
    plt.ion()
    time.sleep(5.0)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    hl1, = ax1.plot([], [], markersize=10, color='r')
    def mujoco_callback(locals, globals):
        if not 'seg' in locals:
            return
        episodic_returns = locals['episodic_returns']
        episodic_lengths = locals['episodic_lengths']
        ep_rets = locals['seg']['ep_rets']
        ep_lens = locals['seg']['ep_lens']
        if len(ep_rets):
            episodic_returns += ep_rets
            episodic_lengths += ep_lens
            window_size_steps = 5000
            x_tick = 1000
            if episodic_lengths:
                ep_lens = np.array(episodic_lengths)
                returns = np.array(episodic_returns)
            cum_episode_lengths = np.cumsum(ep_lens)
            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
                rets = []
                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))
                if len(rets) > 0:
                    hl1.set_xdata(np.arange(1, len(rets) + 1) * x_tick)
                    ax1.set_xlim([x_tick, len(rets) * x_tick])
                    hl1.set_ydata(rets)
                    ax1.set_ylim([np.min(rets), np.max(rets) + 50])
            time.sleep(0.01)
            fig.canvas.draw()
            fig.canvas.flush_events()

    ar_pposgd_simple.learn(env, policy_fn,
                           callback=mujoco_callback,
                           max_timesteps=num_timesteps,
                           timesteps_per_actorbatch=8192,
                           clip_param=0.2, entcoeff=0.0,
                           optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=256,
                           gamma=0.995, lam=0.995, schedule='linear', ar=ar
                           )
    env.close()

def main():
    args = square_arg_parser().parse_args()
    train(num_timesteps=args.num_timesteps, seed=args.seed, dt=args.dt, p=args.p, alpha=args.alpha)

if __name__ == '__main__':
    main()
