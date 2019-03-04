#!/usr/bin/env python3
from arp.arp import ARProcess
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import time
import matplotlib.pyplot as plt
import numpy as np
import _pickle as pkl
from os.path import join, realpath
import tensorflow as tf
import random
from rl_experiments.normalized_env import NormalizedEnv

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    from ar_ppo import ar_mlp_policy, ar_pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    env = make_mujoco_env(env_id, seed)
    env = NormalizedEnv(env)
    p = 3
    alpha = 0.5
    ar = ARProcess(p, alpha, size=env.action_space.shape[-1])
    def policy_fn(name, ob_space, ac_space):
        return ar_mlp_policy.ARMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
           hid_size=64, num_hid_layers=2, phi=ar.phi, sigma_z=ar.sigma_z)


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
                           timesteps_per_actorbatch=2048,
                           clip_param=0.2, entcoeff=0.0,
                           optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                           gamma=0.995, lam=0.995, schedule='linear', ar=ar
                           )
    env.close()

def main():
    seed = 1
    home_dir = realpath(realpath(__file__) + "/../../..")
    random_state = pkl.load(open(join(home_dir, "rl_experiments/rl_experiments/random_states.pkl"), "rb"))[seed - 1].get_state()
    np.random.set_state(random_state)
    seed = np.random.randint(1, 2**31 - 1)
    tf.set_random_seed(seed)
    random.seed(seed)
    rand_num = np.random.random()
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
