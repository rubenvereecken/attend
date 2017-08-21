import tensorflow as tf
import numpy as np

from functools import partial

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

    def prepare(self, step, decay_steps, is_training):
        if not is_training:
            return

        epsilon = self.calculate_epsilon(step, decay_steps)
        epsilon = tf.cast(epsilon, tf.float32)
        tf.summary.scalar('sampling_epsilon', epsilon, family='train')
        self.epsilon = epsilon

    def calculate_epsilon(self, step, decay_steps):
        epsilon = self._epsilon_scheme(step, decay_steps)
        return epsilon

    def sample(self, truth, output, is_training):
        if not is_training:
            return output

        B = tf.shape(truth)[0]
        randoms = tf.random_uniform([B])
        epsilon = self.epsilon

        def _sample(i):
            return tf.cond(epsilon > randoms[i],
                       true_fn=lambda: truth[i],
                       false_fn=lambda: output[i])

        selected = tf.map_fn(_sample, tf.range(B), dtype=tf.float32)

        return selected


class LinearScheduledSampler(ScheduledSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _epsilon_scheme(self, time_step, decay_steps):
        return self._linear_tf(self._sampling_min, decay_steps, time_step)

    def _linear_tf(self, p_min, decay_steps, x):
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
    inverse_sigmoid=InverseSigmoidScheduledSampler
)
samplers[None] = samplers['none']


def get_sampler(s):
    if s not in samplers:
        raise ValueError('No sampler found for `{}`'.format(s))
    return samplers[s]
