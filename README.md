# Autoregressive policies for continuous control reinforcement learning

This repository provides the implementation of autoregressive policies (ARPs) for continuous control deep reinforcement learning together with learning examples based on Open AI Baselines PPO and TRPO algorithms. The examples are provided for OpenAI Gym Mujoco environments and for Square sparse reward environment, discussed in the [paper](https://arxiv.org/abs/1903.11524).   



Tensorflow >= 1.12, [OpenAI Baselines](https://github.com/openai/baselines) and [OpenAI Gym](https://github.com/openai/gym) are required to run learning examples.
NumPy only is required to build and plot stationary AR processes.

# Examples

1. To generate and plot noise trajectories based on AR processes at different orders and smoothing parameter values

`python ./examples/make_noise.py`

2. To run ARP with OpenAI Baselines PPO on a Square environment

`python ./examples/run_square_ppo.py --dt 0.1 --p 3 --alpha 0.8 --num-timesteps=500000`

3. To run ARP with OpenAI Baselines PPO on a Mujoco environment

`python ./examples/run_mujoco_ppo.py --env Reacher-v2 --p 3 --alpha 0.5 --num-timesteps=1000000`

4. To run ARP with OpenAI Baselines TRPO on a Mujoco environment

`python ./examples/run_mujoco_trpo.py --env Reacher-v2 --p 3 --alpha 0.5 --num-timesteps=1000000`

# Reference

*Autoregressive Policies for Continuous Control Deep Reinforcement Learning.*<br/>
Dmytro Korenkevych, A. Rupam Mahmood, Gautham Vasan, James Bergstra. arXiv preprint, 2019.<br/>
[paper](https://arxiv.org/abs/1903.11524) | [video](https://youtu.be/NCpyXBNqNmw)
