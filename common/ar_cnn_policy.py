from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
from common.utils import dense
import tensorflow as tf
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras import regularizers as rgl
import gym
from common.distributions import make_ar_pdtype
import numpy as np

class ARMlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, gaussian_fixed_var=True, sigma_z=1.0, phi=None, normalize=True, img_dim=(256,256)):
        assert isinstance(ob_space, gym.spaces.Box)
        self.pdtype = pdtype = make_ar_pdtype(ac_space)
        sequence_length = None
        self.sigma_z = sigma_z
        if not phi is None:
            p = len(phi)
        else:
            p = 0

        l2_reg = 0.0
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length, p + 1] + list(ob_space.shape))
        if normalize:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space.shape[-1])

        if normalize:
            ob = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)        

        ob_image = ob[:, :, :(4 * np.product(img_dim))]
        ob_image = tf.reshape(ob_image, [tf.shape(ob_image)[0], 4] + list(img_dim) + [1])
        ob_scalar = ob[:, :, (4 * np.product(img_dim)):]
        ob_scalar = tf.reshape(ob_scalar, [tf.shape(ob_scalar)[0], tf.shape(ob_scalar)[-1]])


        acs = U.get_placeholder(name="ac", dtype=tf.float32, shape=[sequence_length, p] + list(ac_space.shape))
        past_x = U.get_placeholder(name="past_x", dtype=tf.float32, shape=[sequence_length, p] + list(ac_space.shape))
        update_mask = U.get_placeholder(name="update_mask", dtype=tf.float32, shape=[sequence_length, p, 1])

        x_image = ob_image
        # x_image = tf.layers.conv2d(x_image, 32, [8, 8], [4, 4],
        #                            name="l1",
        #                            activation=tf.nn.relu,
        #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
        x_image = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4),
                                         activation="relu",
                                         kernel_regularizer=rgl.l2(l2_reg)))(x_image)
        first_conv_out = x_image
        # x_image = tf.layers.conv2d(x_image, 64, [4, 4], [2, 2],
        #                            name="l2",
        #                            activation=tf.nn.relu,
        #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
        x_image = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2),
                                         activation="relu",
                                         kernel_regularizer=rgl.l2(l2_reg)))(x_image)
        second_conv_out = x_image
        # x_image = tf.layers.conv2d(x_image, 64, [4, 4], [2, 2],
        #                            name="l3",
        #                            activation=tf.nn.relu,
        #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
        x_image = TimeDistributed(Conv2D(128, (5, 5), strides=(1, 1),
                                         activation="relu",
                                         kernel_regularizer=rgl.l2(l2_reg)))(x_image)
        third_conv_out = x_image
        x_image = TimeDistributed(Conv2D(128, (5, 5), strides=(1, 1),
                                         activation="relu",
                                         kernel_regularizer=rgl.l2(l2_reg)))(x_image)
        x_image = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1),
                                         activation="relu",
                                         kernel_regularizer=rgl.l2(l2_reg)))(x_image)
        fourth_conv_out = x_image
        # x_image = tf.layers.conv2d(x_image, 32, [4, 4], [2, 2],
        #                            name="l3",
        #                            activation=tf.nn.relu,
        #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
        # x_image = tf.nn.relu(U.conv2d(x_image, 32, "l3", [3, 3], [2, 2], pad="VALID"))

        x_image = TimeDistributed(Lambda(tf.contrib.layers.spatial_softmax))(x_image)

        x_image = U.flattenallbut0(x_image)
        x_scalar = ob_scalar
        # x_scalar = tf.nn.relu(tf.layers.dense(x_scalar, 256, name='lin', kernel_initializer=U.normc_initializer(1.0),
        #                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)))
        x_image = tf.nn.relu(tf.layers.dense(x_image, 256, name='lin1', kernel_initializer=U.normc_initializer(1.0),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)))
        x = tf.concat([x_image, x_scalar], 1)
        # x = x_image
        x = tf.nn.relu(tf.layers.dense(x, 128, name='lin2', kernel_initializer=U.normc_initializer(1.0),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)))
        x = tf.nn.relu(tf.layers.dense(x, 128, name='lin3', kernel_initializer=U.normc_initializer(1.0),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)))
        # x = tf.nn.relu(tf.layers.dense(x, 256, name='lin3', kernel_initializer=U.normc_initializer(1.0),
        #                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)))        
        last_out = x
        last_out = tf.reshape(last_out, [-1, p+1] + tf.shape(x)[1:])
        last_out = last_out[:, -1, :]        
        self.vpred = dense(last_out, 1, name='vf', weight_init=U.normc_initializer(1.0))[:,0]        
        last_out = x        
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, name='final', weight_init=U.normc_initializer(0.01))
            mean = tf.reshape(mean, [-1, mean.shape[-1] * (p + 1)])
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean[:, :ac_space.shape[-1]] * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], name='final', weight_init=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam, phi, sigma_z)

        self.state_in = []
        self.state_out = []
        ac, past_x_next = self.pd.sample(acs, past_x, update_mask)
        self._act = U.function([ob, acs, past_x, update_mask], [ac, self.vpred, mean, logstd, past_x_next])

    def act(self, ob, acs, past_x, update_mask):
        ac1, vpred1, mean1, logstd, past_x_next =  self._act(ob[None], acs[None], past_x[None], update_mask[None])
        return ac1[0], vpred1[0], mean1[0], logstd[0], past_x_next[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

