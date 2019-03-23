import tensorflow as tf
import argparse

def mujoco_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--p', help='AR process order p', type=int, default=3)
    parser.add_argument('--alpha', help='AR process smoothing coefficient alpha', type=float, default=0.5)
    return parser

def square_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', help='Time step duration', type=float, default=0.1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e5))
    parser.add_argument('--p', help='AR process order p', type=int, default=3)
    parser.add_argument('--alpha', help='AR process smoothing coefficient alpha', type=float, default=0.8)
    return parser

def dense(x, size, name, weight_init=None, bias=True):
   w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
   ret = tf.matmul(x, w)
   if bias:
       b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
       return ret + b
   else:
       return ret