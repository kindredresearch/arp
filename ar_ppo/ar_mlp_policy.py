from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from ar_ppo.distributions import make_ar_pdtype

class ARMlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, sigma_z=1.0, phi=None, normalize=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.pdtype = pdtype = make_ar_pdtype(ac_space)
        sequence_length = None
        self.sigma_z = sigma_z
        if not phi is None:
            p = len(phi)
        else:
            p = 0
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length, p + 1] + list(ob_space.shape))
        acs = U.get_placeholder(name="ac", dtype=tf.float32, shape=[sequence_length, p] + list(ac_space.shape))
        past_x = U.get_placeholder(name="past_x", dtype=tf.float32, shape=[sequence_length, p] + list(ac_space.shape))
        update_mask = U.get_placeholder(name="update_mask", dtype=tf.float32, shape=[sequence_length, p, 1])

        if normalize:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space.shape[-1])

        with tf.variable_scope('vf'):
            if normalize:
                obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            else:
                obz = ob
            last_out = obz[:, -1, :]
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, name="fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred = U.dense(last_out, 1, name='final', weight_init=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            obz = tf.reshape(obz, [-1, obz.shape[-1]])
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, name='fc%i'%(i+1), weight_init=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = U.dense(last_out, pdtype.param_shape()[0]//2, name='final', weight_init=U.normc_initializer(0.01))
                mean = tf.reshape(mean, [-1, mean.shape[-1] * (p + 1)])
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean[:, :ac_space.shape[-1]] * 0.0 + logstd], axis=1)
            else:
                pdparam = U.dense(last_out, pdtype.param_shape()[0], name='final', weight_init=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(pdparam, phi, sigma_z)

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = U.switch(stochastic, self.pd.sample(acs, init_mask), self.pd.mode())
        ac, past_x_next = self.pd.sample(acs, past_x, update_mask)
        self._act = U.function([stochastic, ob, acs, past_x, update_mask], [ac, self.vpred, mean, logstd, past_x_next])

    def act(self, stochastic, ob, acs, past_x, update_mask):
        ac1, vpred1, mean1, logstd, past_x_next =  self._act(stochastic, ob[None], acs[None], past_x[None], update_mask[None])
        return ac1[0], vpred1[0], mean1[0], logstd[0], past_x_next[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

