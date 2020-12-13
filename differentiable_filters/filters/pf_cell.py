# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:56:47 2020

@author: akloss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from differentiable_filtering import filter_cell_base as base


class PFCell(base.FilterCellBase):
    def __init__(self, param, context):
        """
        RNN cell implementing a Differentiable Particle Filter

        Parameters
        ----------
        param : dict
            parmaters
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)
        """
        base.FilterCellBase.__init__(self, param, context)

        # number of particles to sample
        self.num_particles = param['num_samples']
        # use a learned or an analytical gaussian likelihood for observations
        self.learned_likelihood = param['learned_likelihood']

        # resample every x steps
        self.resample_rate = param['resample_rate']

        # proportion of uniform resampling
        self.alpha_init = float(param['alpha'])
        self.alpha_min = 0.
        self.alpha = float(param['alpha'])
        # schedule for decreasing the uniform resampling rate
        self.alpha_schedule = param['alpha_schedule']
        if self.alpha_schedule == 1.:
            self.alpha_min = self.alpha_init

        # for sampling noise in the process model (= drawing new particles
        # from the distribution of the next state)
        self.sam = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros((self.dim_x)), scale_diag=np.ones((self.dim_x)))

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # particles, weights and step number
        return [[self.num_particles * self.dim_x], [self.num_particles], [1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # articles, weights, mean state, observations, R, Q
        return ([self.num_particles * self.dim_x], [self.num_particles],
                [self.dim_x], [self.dim_x * self.dim_x],
                [self.dim_z], [self.dim_z * self.dim_z],
                [self.dim_x * self.dim_x])

    def sample_from_start(self, initial_state, initial_covariance):
        """
        Create a set of particles from the initial state

        Parameters
        ----------
        initial_state : tensor [batch_size, dim_x]
            the initial state around which we sample particles
        initial_covariance : tensor [batch_size, dim_x, dim_x]
            the covariance for sampling the initial particle set

        Returns
        -------
        particles : tensor [batch_size, num_particles, dim_x]
            the initial particle set
        weights : tensor [batch_size, num_particles]
            the weights of the initial particle set (uniform)

        """
        # we sample the particles according to the initial covariance around
        # the given (potentially noisy) initial state
        sam = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(self.dim_x, dtype=np.float32),
            scale_diag=initial_covariance)

        # add sampled noise to the initial mean state to
        # generate different particles
        noise = sam.sample(sample_shape=[self.batch_size * self.num_particles])
        noise = tf.reshape(noise,
                           [self.batch_size, self.num_particles, self.dim_x])

        particles = tf.tile(initial_state[:, None, :],
                            [1, self.num_particles, 1])
        particles = particles + noise

        # make sure the states are valid
        particles = self.context.correct_state(particles, diff=False)

        # uniform weights for the particles
        weights = tf.ones([self.batch_size, self.num_particles])
        weights /= float(self.num_particles)
        weights = tf.math.log(weights)

        return particles, weights

    def call(self, inputs, states, training):
        """
        The function that contains the logic for one RNN step calculation.

        Parameters
        ----------
        inputs : list of tensors
            the input tensors, which is a slice from the overall RNN input
            by the time dimension (usually the second dimension).
        states : list of tensors
            the state tensor from previous step as specified by state_size. In
            the case of timestep 0, it will be the
            initial state user specified, or zero filled tensor otherwise.
        training : bool
            if the cell is run in training or test mode

        Returns
        -------
        output : list of tensors
            output tensors as defined in output_size
        new_state : list of tensors
            the new predicted state as defined in state_size
        """
        # turn off the '/rnn' name scope to improve summary logging
        with tf.name_scope(""):
            # get the inputs
            raw_observations, actions = inputs
            particles_old, weights_old, step = states

            particles_old = tf.reshape(particles_old,
                                       [self.batch_size, self.num_particles,
                                        self.dim_x])
            weights_old = tf.reshape(weights_old,
                                     [self.batch_size, self.num_particles])

            if self.debug:
                particles_old = tf.Print(particles_old, [tf.squeeze(step)],
                                         message='--------------- \n step \n ')
                particles_old = tf.Print(particles_old, [particles_old],
                                         summarize=1000,
                                         message='particles old \n')
                particles_old = tf.Print(particles_old, [weights_old],
                                         summarize=1000,
                                         message='weights old \n')

            ###################################################################
            # resample if desired
            particles_re, weights_re = self._resample(particles_old,
                                                      weights_old, training)

            # we don't necessarily resample at every step (and never at the
            # first)
            weights_re = \
                tf.cond(tf.equal(tf.math.mod(step[0, 0] + 1,
                                             self.resample_rate), 0),
                        lambda: weights_re, lambda: weights_old)
            particles_re = \
                tf.cond(tf.equal(tf.math.mod(step[0, 0] + 1,
                                             self.resample_rate), 0),
                        lambda: particles_re, lambda: particles_old)

            if self.debug:
                particles_re = tf.Print(particles_re, [particles_re],
                                        summarize=1000,
                                        message='particles resampled\n')
                particles_re = tf.Print(particles_re, [weights_re],
                                        summarize=1000,
                                        message='weights resampled\n')
                particles_re = tf.Print(particles_re, [actions],
                                        summarize=1000,
                                        message='actions\n')

            ###################################################################
            # preproess the raw observations
            z, encoding = self.context.sensor_model(raw_observations, training)

            if self.learn_r:
                R = self.context.observation_noise(
                    encoding, hetero=self.hetero_r,
                    diag=self.diagonal_covar, training=training)
            else:
                R = tf.tile(self.context.R[None, :, :],
                            [self.batch_size, 1, 1])

            ###################################################################
            # predict the next state for each particle
            particles_pred, Q, = \
                self._prediction_step(particles_re, weights_re, actions,
                                      training=training)

            if self.debug:
                particles_pred = tf.Print(particles_pred, [particles_pred],
                                          summarize=1000,
                                          message='particles pred\n')
                particles_pred = tf.Print(particles_pred, [Q],
                                          summarize=1000, message='Q\n')
                particles_pred = tf.Print(particles_pred, [R],
                                          summarize=1000, message='R\n')

            ###################################################################
            # update the particle weights according to the observations
            weights_updated = \
                self._observation_update(particles_pred, weights_re, z, R,
                                         encoding, training)
            # this lets us emulate not getting observations at every step
            weights_new = \
                tf.cond(tf.equal(tf.math.mod(step[0, 0],
                                             self.update_rate), 0),
                        lambda: weights_updated, lambda: weights_re)

            if self.debug:
                weights_new = tf.Print(weights_new, [weights_new],
                                       summarize=1000, message='weights up\n')

            ###################################################################
            # estimate the mean state
            mean_state, mean_covar = self._get_gaussian(particles_pred,
                                                        weights_new)

            if self.debug:
                mean_covar = tf.Print(mean_covar, [mean_state],
                                      summarize=1000, message='mean state\n')
                mean_covar = tf.Print(mean_covar, [mean_covar],
                                      summarize=1000, message='mean covar\n')

            # the recurrent state contains the updated particles and weights
            new_state = (tf.reshape(particles_pred, [self.batch_size, -1]),
                         tf.reshape(weights_new, [self.batch_size, -1]),
                         step + 1)

            # we can output additional output, but it also makes sense to add
            # the predicted state here again to have access to the full
            # sequence of states after running the network
            output = (tf.reshape(particles_pred, [self.batch_size, -1]),
                      tf.reshape(weights_new, [self.batch_size, -1]),
                      mean_state,
                      tf.reshape(mean_covar, [self.batch_size, self.dim_x**2]),
                      tf.reshape(z, [self.batch_size, -1]),
                      tf.reshape(R, [self.batch_size, -1]),
                      tf.reshape(Q, [self.batch_size, -1]))

            return output, new_state

    def _prediction_step(self, particles_old, weights_old, actions, training):
        """
        Prediction step of the PF

        Parameters
        ----------
        particles_old : tensor [batch_size, num_particles, dim_x]
            the particles representing the current belief about the state
        weights_old : tensor [batch_size, num_particles]
            their weights
        actions : tensor [batch_size, dim_u]
            the current actions
        training : bool
            training or testing?

        Returns
        -------
        particles_pred : tensor [batch_size, num_particles, dim_x]
            the predicted state for each particle
        Q : tensor [batch_size, dim_x, dim_x]
            the predicted process noise for this time step
            (averaged over particles)
        """
        # move the particle-dimension into the batch dimension
        particles_old = tf.reshape(particles_old, [-1, self.dim_x])
        # tile the actions to match the shape of the sigma points
        actions = tf.tile(actions[:, None, :], [1, self.num_particles, 1])
        actions = tf.reshape(actions, [-1, 2])

        particles_pred, _ = \
            self.context.process_model(particles_old, actions,
                                       self.learn_process, training)

        # get the process noise
        # must be defined via a FilterContext
        if self.learn_q:
            Q = self.context.process_noise(
                particles_old, actions, learned=self.learn_process,
                hetero=self.hetero_q, diag=self.diagonal_covar,
                training=training)
        else:
            Q = tf.tile(self.context.Q[None, :, :], [self.batch_size, 1, 1])
        # Q = self._make_valid(Q)
        if self.learn_q and self.hetero_q:
            Q = tf.reshape(Q, [self.batch_size, self.num_particles, self.dim_x,
                               self.dim_x])

        # restore the particle dimension
        particles_pred = tf.reshape(particles_pred,
                                    [self.batch_size, self.num_particles,
                                     self.dim_x])

        # sample noise according to the estimated process noise
        noise = \
            self.sam.sample(sample_shape=[self.batch_size*self.num_particles])
        noise = tf.reshape(noise, [self.batch_size, self.num_particles,
                                   self.dim_x])
        # whiten the noise to reduce undesired correlations in the sampled
        # sigma points
        noise = self.zca_whiten(noise)
        noise = tf.reshape(noise, [self.batch_size, self.num_particles,
                                   self.dim_x, 1])

        scale = tf.linalg.sqrtm(tf.cast(Q, tf.float64))
        if not (self.learn_q and self.hetero_q):
            # when using heteroscedastic noise, we get one Q per particle, but
            # with constant noise, we only get one Q for all particles in the
            # batch
            scale = tf.tile(scale[:, None, :, :],
                            [1, self.num_particles, 1, 1])
        # replace nans and infs in scale with zeros (off-diagonal) or ones
        # (diagonal)
        scale = tf.where(tf.math.is_finite(scale), scale,
                         tf.eye(self.dim_x,
                                batch_shape=[self.batch_size,
                                             self.num_particles],
                                dtype=tf.float64, name=None))
        noise = tf.linalg.matmul(scale, noise)

        # add the noise to the particles
        particles_pred += tf.squeeze(tf.cast(noise, tf.float32), axis=-1)
        particles_pred = self.context.correct_state(particles_pred, diff=False)

        # when we use heteroscedastic noise, we get one Q per particle, so
        # we output the weighted mean
        if self.learn_q and self.hetero_q:
            # weights are in log scale, to turn them into a distribution, we
            # exponentiate and normalize them == apply the softmax transform
            weights_dist = tf.nn.softmax(weights_old, axis=-1)
            Q = tf.reduce_sum(tf.multiply(Q, weights_dist[:, :, None, None]),
                              axis=1)

        return particles_pred, Q

    def _observation_update(self, particles_pred, weights_old, z, R, encoding,
                            training):
        """
        Update step of the PF

        Parameters
        ----------
        particles_pred : tensor [batch_size, num_particles, dim_x]
            the particles representing the current belief about the state
        weights_old : tensor [batch_size, num_particles]
            their weights
        z : tensor [batch_size, dim_z]
            the observations for this time step
        R : tensor [batch_size, dim_z, dim_z]
            the observation noise for this time step
        encoding : [batch_size, ?]
            encoding of the raw observations (used if learned_likelihood=True)
        training : bool
            training or testing?

        Returns
        -------
        weights_new : tensor [batch_size, num_particles]
            the update weights

        """
        """
        Updates the weights of the particles according to the current
        observation
        """
        # get the predicted observation for each particle
        zs_pred, _ = \
            self.context.observation_model(tf.reshape(particles_pred,
                                                      [-1, self.dim_x]),
                                           training=training)
        # restore the particle dimension
        zs_pred = tf.reshape(zs_pred, [self.batch_size, -1, self.dim_z])

        if self.debug:
            zs_pred = tf.Print(zs_pred, [zs_pred], summarize=1000,
                               message='z pred\n')
            zs_pred = tf.Print(zs_pred, [z], summarize=1000, message='z\n')

        if self.learned_likelihood:
            # train a neural network to output a likelihood per particle
            like = self.context.likelihood(zs_pred, encoding,
                                           training=training)
            like = tf.reshape(like, weights_old.get_shape())

            like = tf.where(tf.greater_equal(like, -20), like,
                            tf.ones_like(like)*-20)
        else:
            # for each particle, compute p(z|x), which we assume to be gaussian
            # tile the observations and observation noise
            z = tf.tile(z[:, None, :], [1, self.num_particles, 1])
            # calculate the error and correct for periodicity
            diff = self.context.correct_observation_diff(z - zs_pred)
            diff = tf.expand_dims(diff, axis=-1)

            diff = tf.where(tf.math.is_finite(diff), diff,
                            tf.ones_like(diff)*1e5/self.context.scale)

            R = tf.tile(R[:, None, :, :], [1, self.num_particles, 1, 1])
            inv = tf.linalg.inv(R)

            diff = tf.cast(diff, tf.float64)
            inv = tf.cast(inv, tf.float64)
            exponent = tf.matmul(tf.linalg.matrix_transpose(diff),
                                 tf.matmul(inv, diff))
            exponent = exponent * -0.5
            exponent = tf.cast(exponent, tf.float32)
            exponent = tf.reshape(exponent,
                                  [self.batch_size, self.num_particles])

            # normalizer = tf.sqrt((2*np.pi)**self.dim_z * tf.linalg.det(R))
            normalizer = tf.linalg.det(R)
            like = exponent - tf.math.log(normalizer)

            # shift the likelihoods such that the maximum likelihood is at
            # least 1/(num particles)
            # this shift is automatically undone when we take the softmax of
            # the weights
            max_like = tf.reduce_max(like, axis=-1, keepdims=True)
            bias = tf.maximum(tf.log(1/float(self.num_particles)) - max_like,
                              0) * tf.ones_like(max_like)
            bias = tf.tile(bias, [1, self.num_particles])
            like += bias

            like = tf.where(tf.greater_equal(like, -20), like,
                            tf.ones_like(like)*-20)

        # this multiplication with the old weights is often skipped, because
        # weights_old are uniform after resampling.
        # Since we use soft resampling, this is not true for alpha != 0
        weights_new = like + weights_old
        weights_new = tf.where(tf.math.is_inf(weights_new),
                               -tf.ones_like(weights_new) * 30, weights_new)
        weights_new = tf.where(tf.math.is_nan(weights_new),
                               -tf.ones_like(weights_new) * 30, weights_new)

        return weights_new

    def _resample(self, particles, weights, training):
        """
        Resample the particles to discard particles with low weights

        Parameters
        ----------
        particles : tensor [batch_size, num_particles, dim_x]
            old particle set
        weights : tensor [batch_size, num_particles]
            their weights
        training : bool
            training or testing?

        Returns
        -------
        new_particles: tensor [batch_size, num_particles, dim_x]
            resampled particle set
        new_weights : tensor [batch_size, num_particles]
            their weights
        """
        # weights are in log scale, to turn them into a distribution, we
        # exponentiate and normalize them == apply the softmax transform
        weights = tf.nn.softmax(weights, axis=-1)

        # soft resampling - this maintains a gradient between old and new
        # weights
        rate = tf.cond(training, lambda: self.alpha,
                       lambda: tf.cast(self.alpha_min, tf.float32))
        resample_prob = (1 - rate) * weights + rate/float(self.num_particles)
        new_weights = weights / resample_prob

        # systematic resampling: the samples are evenly distributed over the
        # original particles
        base_inds = \
            tf.linspace(0.0, (self.num_particles-1.)/float(self.num_particles),
                        self.num_particles)
        random_offsets = tf.random.uniform([self.batch_size], 0.0,
                                           1.0 / float(self.num_particles))
        # shape: batch_size x num_resampled
        inds = random_offsets[:, None] + base_inds[None, :]
        cum_probs = tf.cumsum(resample_prob, axis=1)

        # shape: batch_size x num_resampled x num_particles
        inds_matching = inds[:, :, None] < cum_probs[:, None, :]
        samples = tf.cast(tf.argmax(tf.cast(inds_matching, 'int32'),
                                    axis=2), 'int32')

        # compute 1D indices into the 2D array
        idx = samples + self.num_particles * tf.tile(
            tf.reshape(tf.range(self.batch_size), [self.batch_size, 1]),
            [1, self.num_particles])

        # index using the 1D indices and reshape again
        new_particles = \
            tf.gather(tf.reshape(particles,
                                 [self.batch_size*self.num_particles,
                                  self.dim_x]), idx)
        new_particles = \
            tf.reshape(new_particles,
                       [self.batch_size, self.num_particles, self.dim_x])

        new_weights = tf.gather(tf.reshape(new_weights,
                                           [self.batch_size*self.num_particles,
                                            1]), idx)

        new_weights = tf.reshape(new_weights,
                                 [self.batch_size, self.num_particles])
        # renormalize
        new_weights /= tf.reduce_sum(new_weights, axis=-1, keepdims=True)

        # return into log scale
        new_weights = tf.math.log(new_weights)

        return new_particles, new_weights

    def _get_gaussian(self, particles, weights):
        """
        Aprroximate the particle distribution with a single Gaussian

        Parameters
        ----------
        particles : tensor [batch_size, num_particles, dim_x]
            particle set
        weights : tensor [batch_size, num_particles]
            their weights

        Returns
        -------
        mean : tensor [batch_size, dim_x]
            (weighted) mean of the particle set
        covar : tensor [batch_size, dim_x, dim_x]
            covariance of the particle set
        """
        # weights are in log scale, to turn them into a distribution, we
        # exponentiate and normalize them == apply the softmax transform
        weights = tf.nn.softmax(weights, axis=-1)
        weights = tf.cast(weights, tf.float64)
        particles = tf.cast(particles, tf.float64)

        weights = tf.reshape(weights, [self.batch_size, self.num_particles, 1])
        # count the number of extremely small weights
        small = tf.where(tf.less(weights, 1e-9),
                         tf.ones_like(weights), tf.zeros_like(weights))
        small = tf.reduce_sum(small, axis=1)
        small = tf.reshape(small, [self.batch_size])

        # make sure they are normalized
        div = tf.reduce_sum(weights, axis=1, keepdims=True)
        weights = weights / div

        mean_state = \
            self.context.weighted_state_mean_with_angles(particles, weights)

        mean_diff = particles - \
            tf.tile(mean_state[:, None, :], [1, self.num_particles, 1])
        mean_diff = self.context.correct_state(mean_diff)

        # remove nans and infs and replace them with high values
        mean_diff = tf.where(tf.math.is_finite(mean_diff), mean_diff,
                             tf.ones_like(mean_diff)*1e3/self.context.scale)

        # batch_size, dim_x, dim_x
        cov = tf.matmul(mean_diff[:, :, :, None], mean_diff[:, :, None, :])
        mean_covar = tf.reduce_sum(cov * weights[:, :, :, None], axis=1)
        mean_covar /= div

        mean_state = tf.cast(mean_state, tf.float32)
        mean_covar = tf.cast(mean_covar, tf.float32)
        mean_covar = self._make_valid(mean_covar)

        if self.param['problem'] == 'pushing':
            # for the circular object, the orientation is always zero, leading
            # to 0 variance between particles and thus a degenerate covariance
            # matrix
            ob = tf.squeeze(self.context.ob)
            ob = tf.strings.regex_replace(ob, "\000", "")
            ob = tf.strings.regex_replace(ob, "\00", "")

            mod_covar = \
                tf.concat([mean_covar[:, :2],
                           tf.tile(np.array([[[0, 0, 1e-5, 0, 0, 0, 0, 0,
                                               0, 0]]], dtype=np.float32),
                                   [self.batch_size, 1, 1]),
                           mean_covar[:, 3:]], axis=1)
            mean_covar = tf.where(tf.equal(ob, 'ellip1'),
                                  mod_covar, mean_covar)

        return mean_state, mean_covar
