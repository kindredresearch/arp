#!/usr/bin/env python3
from arp.arp import ARProcess
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

def train(env_id, num_timesteps, seed, ar):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    from ar_ppo import ar_mlp_policy, ar_pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return ar_mlp_policy.ARMlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, phi=ar.phi, sigma_z=ar.sigma_z)
    env = make_mujoco_env(env_id, seed)
    ar_pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', ar=ar
        )
    env.close()

def main():
    p = 3
    alpha = 0.8
    ar = ARProcess(p, alpha)
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, ar=ar)

if __name__ == '__main__':
    main()
