import tensorflow as tf
import numpy as np

from functools import partial

import attend
from attend import tf_util


class Sampler:
    pass


class StandardSampler(Sampler):
    def __init__(self, *args):
        pass

    def prepare(self, *args):
        pass

    def sample(self, truth, output, is_training):
        if is_training:
            return truth
        else:
            return output


class ScheduledSampler(Sampler):
    def __init__(self, sampling_min):
        self._sampling_min = sampling_min

    def prepare(self, decay_steps, is_training):
        # print('Preparing. Training?', is_training)
        if not is_training:
            return

        step = tf.train.get_global_step()

        with tf.name_scope('epsilon'):
            epsilon = self.calculate_epsilon(step, decay_steps)
            epsilon = tf.cast(epsilon, tf.float32)
        tf.summary.scalar('sampling_epsilon', epsilon, family='train')
        self.epsilon = epsilon
        tf_util.add_to_collection(attend.GraphKeys.SAMPLING_EPSILON, epsilon)

    def calculate_epsilon(self, step, decay_steps):
        epsilon = self._epsilon_scheme(step, decay_steps)
        return epsilon

    def sample(self, truth, output, is_training):
        if not is_training:
            return output

        B = tf.gather(tf.shape(truth), 0, name='batch_size')
        randoms = tf.random_uniform([B])
        epsilon = self.epsilon

        def _sample(i):
            return tf.cond(epsilon > randoms[i],
                       true_fn=lambda: truth[i],
                       false_fn=lambda: output[i],
                           name='truth_if_eps_gt_random')

        selected = tf.map_fn(_sample, tf.range(B), dtype=tf.float32,
                             name='sample_over_batch')

        return selected


class FixedEpsilonSampler(ScheduledSampler):
    """
    Used for imitating scheduled sampler at a specific point,
    useful for debugging training/inference
    """

    def __init__(self, *args, epsilon):
        super().__init__(*args)
        self.epsilon = epsilon

    def _epsilon_scheme(self, *args):
        return self.epsilon

    # Allow the fixed epsilon scheduler to sample even outside training
    # TODO find more elegant solution
    def prepare(self, decay_steps, is_training):
        return super().prepare(decay_steps, True)

    def sample(self, truth, output, is_training):
        return super().sample(truth, output, True)


class LinearScheduledSampler(ScheduledSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _epsilon_scheme(self, time_step, decay_steps):
        return self._linear_tf(self._sampling_min, decay_steps, time_step)

    def _linear_tf(self, p_min, decay_steps, x):
        with tf.name_scope('linear_eps'):
            b = 1. # Offset 1, starting point
            a = (p_min - b) / decay_steps
            o = a * tf.cast(x, tf.float32) + b
            return tf.maximum(p_min, o)


class InverseSigmoidScheduledSampler(ScheduledSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _epsilon_scheme(self, time_step, decay_steps):
        k = self._solve_for_k(self._sampling_min, decay_steps)
        return self._inverse_sigmoid_tf(k, self._sampling_min, time_step)

    def _inverse_sigmoid_np(self, k, p_min, x):
        return p_min + k / (k + np.exp(x/k)) / (1. / (1-p_min))

    def _inverse_sigmoid_tf(self, k, p_min, x):
        with tf.name_scope('inverse_sigmoid_eps'):
            return p_min + k / (k + tf.exp(x/k)) / (1. / (1-p_min))

    def _solve_for_k(self, p_min, decay_steps):
        # Solve for parameter k such that the inverse sigmoid gently curves
        # from 1 to p_min with the half-way point exactly halfway the decay
        from scipy.optimize import fsolve
        M = p_min + (1 - p_min) / 2
        X = decay_steps / 2
        x0 = X / np.log(X)
        inverse_sigmoid_instance = partial(self._inverse_sigmoid_np, p_min=p_min, x=X)
        k = fsolve(lambda k: inverse_sigmoid_instance(k) - M, x0)[0]
        return k


samplers = dict(
    none=StandardSampler,
    standard=StandardSampler,
    linear=LinearScheduledSampler,
    inverse_sigmoid=InverseSigmoidScheduledSampler,
    fixed=FixedEpsilonSampler,
    fixed_epsilon=FixedEpsilonSampler
)
samplers[None] = samplers['none']


def get_sampler(s):
    if s not in samplers:
        raise ValueError('No sampler found for `{}`'.format(s))
    return samplers[s]
