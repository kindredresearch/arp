import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from baselines.a2c.utils import fc
from tensorflow.python.ops import math_ops
from baselines.common.distributions import Pd, PdType, DiagGaussianPdType, DiagGaussianPd

class ARPd(Pd):
    """
    A particular probability distribution
    """
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError


class ARPdType(PdType):
    """
    Parametrized family of probability distributions
    """
    def pdfromflat(self, flat, *args):
         return self.pdclass()(flat, *args)


class ARDiagGaussianPd(DiagGaussianPd):
    def __init__(self, flat, phi=None, sigma_z=1.0):
        self.flat = flat
        self.sigma_z = sigma_z
        if not phi is None:
            p = len(phi)
            self.phi = np.array(phi).reshape((len(phi), 1))
        else:
            p = 0
        params = tf.split(axis=len(flat.shape)-1, num_or_size_splits=p + 2, value=flat)
        ac_dim = flat.shape.as_list()[-1]//(p + 2)
        self.mean = tf.reshape(flat[..., :-ac_dim], [-1] + flat.shape.as_list()[1:-1] + [p + 1, ac_dim])
        self.logstd = flat[..., -ac_dim:] + np.log(self.sigma_z)
        #self.mean3, self.mean2, self.mean1, self.mean, self.logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=5, value=flat)
        self.std = tf.exp(self.logstd)
    def neglogp(self, acs, init_mask):
        h = tf.reduce_sum(self.phi[::-1] * (acs[..., :-1, :] - self.mean[..., :-1, :]) * init_mask, axis=-2)
        #self.coeffs[0] * (acs[:, 0, :] - self.mean3) + self.coeffs[1] * (acs[:, 1, :] - self.mean2) + self.coeffs[2] * (acs[:, 2, :] - self.mean1)
        return 0.5 * tf.reduce_sum(tf.square((acs[:, -1, :] - self.mean[..., -1, :] - h) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(acs)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self, acs, init_mask):
        h = tf.reduce_sum(self.phi[::-1] * (acs - self.mean[..., :-1, :]) * init_mask, axis=-2)
        return self.mean[..., -1, :] + h + self.std * tf.random_normal(tf.shape(self.mean[..., -1, :]))
    def logp(self, x, init_mask):
        return - self.neglogp(x, init_mask)

class ARDiagGaussianPdType(DiagGaussianPdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return ARDiagGaussianPd

def make_ar_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return ARDiagGaussianPdType(ac_space.shape[0])
    else:
        raise NotImplementedError
