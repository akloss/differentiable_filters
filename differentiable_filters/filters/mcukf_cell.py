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
from differentiable_filters.filters import filter_cell_base as base


class MCUKFCell(base.FilterCellBase):
    def __init__(self, param, context):
        """
        RNN cell implementing a Differentiable Monte Carlo Unscented Kalman
        Filter

        Parameters
        ----------
        param : dict
            parmaters
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)
        """
        base.FilterCellBase.__init__(self, param, context)

        # number of pseudo sigma points to sample
        self.num_sigma = param['num_samples']

        # for sampling pseudo sigma points
        # we generate the sigma points by sampling offsets to last mean state
        # according to the last estimated covariance
        # this is doen via the reparametrization trick
        self.sam = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros((self.dim_x)), scale_diag=np.ones((self.dim_x)))

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # estimated state, its covariance, and the step number
        return [[self.dim_x], [self.dim_x * self.dim_x], [1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # estimated state and covariance, observations, R, Q
        return ([self.dim_x], [self.dim_x * self.dim_x],
                [self.dim_z], [self.dim_z * self.dim_z],
                [self.dim_x * self.dim_x])

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
            state_old, covar_old, step = states

            state_old = tf.reshape(state_old, [self.batch_size, self.dim_x])
            covar_old = tf.reshape(covar_old,
                                   [self.batch_size, self.dim_x, self.dim_x])

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
            # calculate the sigma points (state samples)
            sigma_points = \
                self._get_sigma_points(state_old, covar_old)

            if self.debug:
                sigma_points = tf.Print(sigma_points, [tf.squeeze(step)],
                                        message='---------------- \n step \n ')
                sigma_points = tf.Print(sigma_points, [state_old],
                                        summarize=1000, message='state old \n')
                sigma_points = tf.Print(sigma_points, [covar_old],
                                        summarize=1000, message='covar old \n')
                sigma_points = tf.Print(sigma_points, [sigma_points[0:3]],
                                        summarize=10000,
                                        message='sigma points\n')
                sigma_points = tf.Print(sigma_points, [actions[0:3]],
                                        summarize=10000, message='actions\n')

            # predict the next state for each sigma point
            sigma_points_pred, Q, = \
                self._prediction_step(sigma_points, actions,
                                      training=training)

            # calcultate the new mean and covariance from the sigma points
            weights = np.ones([self.batch_size,
                               self.num_sigma, 1])/float(self.num_sigma)
            state_pred = \
                self.context.weighted_state_mean_with_angles(sigma_points_pred,
                                                             weights)

            diff = sigma_points_pred - state_pred[:, None, :]
            diff = self.context.correct_state(diff)
            diff_square = tf.matmul(diff[:, :, :, None], diff[:, :, None, :])
            covar_pred = tf.reduce_mean(diff_square, axis=1)
            covar_pred += Q

            if self.debug:
                sigma_points_pred = tf.Print(sigma_points_pred,
                                             [sigma_points_pred],
                                             summarize=100000,
                                             message='sigma points pred\n')
                sigma_points_pred = tf.Print(sigma_points_pred, [state_pred],
                                             summarize=10000,
                                             message='state pred\n')
                sigma_points_pred = tf.Print(sigma_points_pred, [covar_pred],
                                             summarize=10000,
                                             message='covar pred\n')
                sigma_points_pred = tf.Print(sigma_points_pred, [Q],
                                             summarize=100000, message='Q\n')
                sigma_points_pred = tf.Print(sigma_points_pred, [R],
                                             summarize=100000, message='R\n')

            # get the expected observations for each sigma point
            z_points_pred, H = \
                self.context.observation_model(tf.reshape(sigma_points_pred,
                                                          [-1, self.dim_x]),
                                               training=training)
            # restore the sigma dimension
            z_points_pred = tf.reshape(z_points_pred,
                                       [self.batch_size, -1, self.dim_z])
            # and compute their mean
            z_pred = \
                self.context.weighted_observation_mean_with_angles(z_points_pred,
                                                                   weights)

            if self.debug:
                z_pred = tf.Print(z_pred, [z_points_pred], summarize=100000,
                                  message='zs pred\n')
                z_pred = tf.Print(z_pred, [z_pred], summarize=10000,
                                  message='mean z pred\n')

            ###################################################################
            # update the predictions with the observations
            state_up, covar_up = \
                self._update(state_pred, covar_pred, sigma_points_pred,
                             z_pred, z_points_pred, H, z, R)

            # this lets us emulate not getting observations at every step
            state = tf.cond(tf.equal(tf.math.mod(step[0, 0],
                                                 self.update_rate), 0),
                            lambda: state_up, lambda: state_pred)
            covar = tf.cond(tf.equal(tf.math.mod(step[0, 0],
                                                 self.update_rate), 0),
                            lambda: covar_up, lambda: covar_pred)

            # now flatten the tensors again for output
            state = tf.reshape(state, [self.batch_size, -1])
            covar = tf.reshape(covar, [self.batch_size, -1])

            # the recurrent state contains the updated state estimate
            new_state = (state, covar, step + 1)

            # we can output additional output, but it also makes sense to add
            # the predicted state here again to have access to the full
            # sequence of states after running the network
            output = (tf.reshape(state, [self.batch_size, -1]),
                      tf.reshape(covar, [self.batch_size, -1]),
                      tf.reshape(z, [self.batch_size, -1]),
                      tf.reshape(R, [self.batch_size, -1]),
                      tf.reshape(Q, [self.batch_size, -1]))

            return output, new_state

    def _get_sigma_points(self, state_old, covar_old):
        """
        Sample pseudo sigma points around the current state according to the
        estimated uncertainty

        Parameters
        ----------
        state_old : tensor [batch_size, dim_x]
            the mean of the old state estimate
        covar_old : tensor [batch_size, dim_x, dim_x]
            the covariance of the old state estimate

        Returns
        -------
        sigma_points : tensor [batch_size, num_sigma, dim_x]
            the sigma points

        """
        # reparametrization trick: sample from the standard normal dist.
        # and scale with the current standard deviation
        noise = self.sam.sample(sample_shape=[self.batch_size*self.num_sigma])
        noise = tf.reshape(noise, [self.batch_size, self.num_sigma,
                                   self.dim_x])

        # whiten the noise to reduce undesired correlations in the sampled
        # sigma points
        noise = self.zca_whiten(noise)

        noise = tf.reshape(noise, [self.batch_size, self.num_sigma,
                                   self.dim_x, 1])

        covar_old = self._make_valid(covar_old)
        scale = tf.linalg.sqrtm(tf.cast(covar_old, tf.float64))

        # replace nans and infs in scale with zeros (off-diagonal) or ones
        # (diagonal)
        scale = tf.where(tf.math.is_finite(scale), scale,
                         tf.eye(self.dim_x, batch_shape=[self.batch_size],
                                dtype=tf.float64, name=None))

        scale = tf.tile(scale[:, None, :, :], [1, self.num_sigma, 1, 1])
        noise = tf.linalg.matmul(scale, noise)
        noise = tf.squeeze(noise, axis=-1)

        sigma_points = tf.tile(state_old[:, None, :], [1, self.num_sigma, 1])
        sigma_points = sigma_points + tf.cast(noise, tf.float32)

        sigma_points = self.context.correct_state(sigma_points, diff=False)

        return sigma_points

    def _prediction_step(self, sigma_points, actions, training):
        """
        Prediction step of the MCUKF

        Parameters
        ----------
        sigma_points : tensor [batch_size, num_sigma, dim_x]
            the old sigma points
        actions : tensor [batch_size, dim_u]
            the current actions
        training : bool
            training or testing?

        Returns
        -------
        sigma_points_pred : tensor [batch_size, num_sigma, dim_x]
            the predicted sigma points
        Q : tensor [batch_size, dim_x, dim_x]
            the predicted process noise for this time step (averaged over
            sigma points)
        """
        # move the sigma-dimension into the batch dimension
        sigma_points = tf.reshape(sigma_points, [-1, self.dim_x])
        # tile the actions to match the shape of the sigma points
        actions = tf.tile(actions[:, None, :], [1, self.num_sigma, 1])
        actions = tf.reshape(actions, [-1, 2])

        sigma_points_pred, _ = \
            self.context.process_model(sigma_points, actions,
                                       self.learn_process, training)

        sigma_points_pred = self.context.correct_state(sigma_points_pred,
                                                       diff=False)

        # get the process noise
        if self.learn_q:
            Q = self.context.process_noise(
                sigma_points, actions, learned=self.learn_process,
                hetero=self.hetero_q, diag=self.diagonal_covar,
                training=training)
        else:
            Q = tf.tile(self.context.Q[None, :, :], [self.batch_size, 1, 1])

        # restore the sigma dimension
        sigma_points_pred = tf.reshape(sigma_points_pred,
                                       [self.batch_size, self.num_sigma,
                                        self.dim_x])
        # when we use heteroscedastic noise, we get one Q per sigma point, so
        # we use the weighted mean
        if self.learn_q and self.hetero_q:
            Q = tf.reshape(Q, [self.batch_size, self.num_sigma, self.dim_x,
                               self.dim_x])
            Q = tf.reduce_mean(Q, axis=1)

        return sigma_points_pred, Q

    def _update(self, state_pred, covar_pred, sigma_points_pred, z_pred,
                z_points_pred, H, z, R):
        """
        Update step of the MCUKF

        Parameters
        ----------
        state_pred : tensor [batch_size, dim_x]
            the predicted mean state from the process model
        covar_pred : tensor [batch_size, dim_x, dim_x]
            it's predicted covariance
        sigma_points_pred : [batch_size, num_sigma, dim_x]
            the predicted sigma points from the process model
        z_pred : tensor [batch_size, dim_z]
            the mean predicted observations for the predicted sigma points
        z_points_pred : tensor [batch_size, num_sigma, dim_z]
            the predicted observations for the predicted sigma points
        H : tensor [batch_size, dim_z, dim_x]
            Jacobian of the the observation model wrt the predicted state
        z : tensor [batch_size, dim_z]
            the observations for this time step
        R : tensor [batch_size, dim_z, dim_z]
            the observation noise for this time st

        Returns
        -------
        state_up: tensor [batch_size, dim_x]
            the updated state
        covar_up: tensor  [batch_size, dim_x, dim_x]
            the updated state covariance
        """
        # Compute the Kalman gains and the innovation
        K = self._KF_gain(state_pred, covar_pred, sigma_points_pred,
                          z_pred, z_points_pred, R)
        innovation = self.context.correct_observation_diff(z - z_pred)
        innovation = tf.expand_dims(innovation, -1)
        update = tf.squeeze(tf.matmul(K, innovation), -1)

        if self.debug:
            update = tf.Print(update, [z], summarize=10000, message='z \n')
            update = tf.Print(update, [K], summarize=10000, message='K \n')
            update = tf.Print(update, [update], summarize=10000,
                              message='update \n')

        # Compute the new state
        state_up = state_pred + update
        # correct the updated state, e.g.\ warping angles
        state_up = self.context.correct_state(state_up, diff=False)

        # Joseph form update: this is an alternative form of writing the update
        # of the covariance matrix that keeps the covariance matrix
        # positive semidefinite (i.e. it remains a correct covariance matrix)
        mult = tf.eye(self.dim_x) - tf.matmul(K, H)
        covar_up = \
            tf.matmul(mult, tf.matmul(covar_pred,
                                      tf.linalg.matrix_transpose(mult)))
        covar_up += tf.matmul(K, tf.matmul(R, tf.linalg.matrix_transpose(K)))

        covar_up = self._make_valid(covar_up)
        return state_up, covar_up

    def _KF_gain(self, state_pred, covar_pred, sigma_points_pred,
                 z_pred, z_points_pred, R):
        """
        Computing the Kalman Gain

        Parameters
        ----------
        state_pred : tensor [batch_size, dim_x]
            the predicted mean state from the process model
        covar_pred : tensor [batch_size, dim_x, dim_x]
            the predicted covariance of the current state estimate
        sigma_points_pred : [batch_size, num_sigma, dim_x]
            the predicted sigma points from the process model
        z_pred : tensor [batch_size, dim_z]
            the mean predicted observations for the predicted sigma points
        z_points_pred : tensor [batch_size, num_sigma, dim_z]
            the predicted observations for the predicted sigma points
        R : tensor [batch_size, dim_z, dim_z]
            the observation noise for this time step

        Returns
        -------
        K : tensor [batch_size, dim_x, dim_x]
            the Kalman Gain

        """
        diff_z = z_points_pred - z_pred[:, None, :]
        diff_z = tf.reshape(diff_z, [-1, self.dim_z])
        diff_z = self.context.correct_observation_diff(diff_z)
        diff_z = tf.cast(diff_z, tf.float64)
        diff_z = tf.reshape(diff_z,
                            [self.batch_size, self.num_sigma, self.dim_z])
        diff_square = tf.matmul(diff_z[:, :, :, None], diff_z[:, :, None, :])
        S = tf.cast(tf.reduce_mean(diff_square, axis=1), tf.float32)
        S += R
        S = self._make_valid(S)

        diff_state = sigma_points_pred - state_pred[:, None, :]
        diff_state = tf.reshape(diff_state, [-1, self.dim_x])
        diff_state = self.context.correct_state(diff_state)
        diff_state = tf.reshape(diff_state,
                                [self.batch_size, self.num_sigma, self.dim_x])
        z_state = tf.matmul(tf.cast(diff_state[:, :, :, None], tf.float64),
                            diff_z[:, :, None, :])
        z_state_covar = tf.reduce_mean(z_state, axis=1)
        z_state_covar = tf.cast(z_state_covar, tf.float32)

        K = tf.matmul(z_state_covar, tf.linalg.inv(S))
        return K
