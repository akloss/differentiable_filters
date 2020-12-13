# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:56:47 2020

@author: akloss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals

import tensorflow as tf
from differentiable_filters.filters import filter_cell_base as base


class UKFCell(base.FilterCellBase):
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

        # parameter for setting the sigma points
        if 'scaled' in param.keys() and \
                param['scaled'] is not None:
            self.scaled = param['scaled']
        else:
            self.scaled = False

        if self.scaled:
            self.kappa = 0
            self.beta = 2
            if 'scaled_alpha' in param.keys() and \
                    param['scaled_alpha'] is not None:
                self.alpha = param['scaled_alpha']
            else:
                self.alpha = 0.5
        else:
            self.beta = 0
            self.alpha = 1
            self.kappa = param['kappa']

        self.num_sigma = 2 * self.dim_x + 1

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
            sigma_points, weights_m, weights_c = \
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

            # predict the next state
            sigma_points_pred, Q = \
                self._prediction_step(sigma_points, weights_m, actions,
                                      training=training)

            # calcultate the new mean and covariance from the sigma points
            # mult = tf.multiply(sigma_points_pred, weights_m)
            # state_pred = tf.reduce_sum(mult, axis=1)
            state_pred = \
                self.context.weighted_state_mean_with_angles(sigma_points_pred,
                                                             weights_m)

            if self.kappa > 0:
                diff = sigma_points_pred - state_pred[:, None, :]
                diff = self.context.correct_state(diff)
                diff_square = tf.matmul(diff[:, :, :, None],
                                        diff[:, :, None, :])
                covar_pred = \
                    tf.reduce_sum(tf.multiply(diff_square,
                                              weights_c[:, :, :, None]),
                                  axis=1)
            else:
                # for negative kappa, the above would not be guaranteed to give
                # a positive semidefinite covariance matrix
                # this alternative formulation was proposed in appedix III of
                # Julier, Simon J. et al. “A new method for the nonlinear
                # transformation of means and covariances in filters and
                # estimators.” IEEE Trans. Automat. Contr. 45 (2000): 477-482.
                diff = sigma_points_pred[:, 1:] - sigma_points_pred[:, 0:1]
                diff = self.context.correct_state(diff)
                diff_square = tf.matmul(diff[:, :, :, None],
                                        diff[:, :, None, :])
                covar_pred = \
                    tf.reduce_sum(tf.multiply(diff_square,
                                              weights_c[:, 1:, :, None]),
                                  axis=1)
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
                                                                   weights_m)
            if self.debug:
                z_pred = tf.Print(z_pred, [z_points_pred], summarize=100000,
                                  message='zs pred\n')
                z_pred = tf.Print(z_pred, [z_pred], summarize=10000,
                                  message='mean z pred\n')

            ###################################################################
            # update the predictions with the observations
            state_up, covar_up = \
                self._update(state_pred, covar_pred, sigma_points_pred,
                             weights_c, z_pred, z_points_pred, H, z, R)

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
        Sigma points as proposed by S.J. Julier, The Scaled Unscented
        Transformation and used e.g. in E. A. Wan and R. van der Merwe,
        The Unscented Kalman Filter for Nonlinear Estimation.

        Setting alpha = 1 and beta = 0 recovers the paramtrization proposed in
        S.J. Julier and J.K. Uhlmann, A New Extension of the Kalman Filter to
        Non linear Systems

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
        weights_m : tensor [batch_size, num_sigma]
            weights for computing the mean of the sigma points
        weights_c : tensor [batch_size, num_sigma]
            weights for computing the covariance of the sigma points
        """
        covar_old = self._make_valid(covar_old)
        lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        covar = (lambda_ + self.dim_x) * covar_old
        U = tf.matrix_square_root(tf.cast(covar, tf.float64))
        U = tf.cast(U, tf.float32)

        # replace nans and infs in U with zeros (off-diagonal) or ones
        # (diagonal)
        U = tf.where(tf.math.is_finite(U), U,
                     tf.eye(self.dim_x, batch_shape=[self.batch_size],
                            dtype=tf.float32, name=None))

        states = tf.tile(state_old[:, None, :], [1, self.dim_x, 1])
        sigmas_pos = states + U
        sigmas_neg = states - U

        sigma_points = tf.concat([state_old[:, None, :], sigmas_pos,
                                  sigmas_neg], axis=1)

        sigma_points = self.context.correct_state(sigma_points, diff=False)

        weights = 0.5 / (self.dim_x + lambda_)
        weights_m = weights * tf.ones([self.batch_size, self.dim_x, 1],
                                      dtype=tf.float32)
        weights_c = weights * tf.ones([self.batch_size, 2*self.dim_x, 1])

        wm_s = lambda_/(self.dim_x + lambda_)
        wm_s = wm_s * tf.ones([self.batch_size, 1, 1])
        weights_m = tf.concat([wm_s, weights_m, weights_m], axis=1)

        wc_s = lambda_/(self.dim_x + lambda_) + \
            (1 - self.alpha**2 + self.beta)
        wc_s = wc_s * tf.ones([self.batch_size, 1, 1])
        weights_c = tf.concat([wc_s, weights_c], axis=1)

        return sigma_points, weights_m, weights_c

    def _prediction_step(self, sigma_points, weights_m, actions, training):
        """
        Prediction step of the UKF

        Parameters
        ----------
        sigma_points : tensor [batch_size, num_sigma, dim_x]
            the old sigma points
        weights_m : tensor [batch_size, num_sigma]
            weights for computing the mean of the sigma points
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
        if self.hetero_q and self.learn_q:
            Q = tf.reshape(Q, [self.batch_size, self.num_sigma, self.dim_x,
                               self.dim_x])
            Q = tf.reduce_sum(tf.multiply(Q, weights_m[:, :, :, None]), axis=1)

        return sigma_points_pred, Q

    def _update(self, state_pred, covar_pred, sigma_points_pred, weights_c,
                z_pred, z_points_pred, H, z, R):
        """
        Update step of the UKF

        Parameters
        ----------
        state_pred : tensor [batch_size, dim_x]
            the predicted mean state from the process model
        covar_pred : tensor [batch_size, dim_x, dim_x]
            it's predicted covariance
        sigma_points_pred : [batch_size, num_sigma, dim_x]
            the predicted sigma points from the process model
        weights_c : tensor [batch_size, num_sigma]
            weights for computing the covariance of the sigma points
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
                          weights_c, z_pred, z_points_pred, R)
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
        # correct the updated state, e.g. warping angles
        state_up = self.context.correct_state(state_up, diff=False)

        # Joseph form update: this is an alternative form of writing the update
        # of the covariance matrix that keeps the covariance matrix
        # positive semidefinite (i.e. it remains a correct covariance matrix)
        mult = tf.eye(self.dim_x) - tf.matmul(K, H)
        covar_up = \
            tf.matmul(mult, tf.matmul(covar_pred,
                                      tf.linalg.matrix_transpose(mult)))
        covar_up += tf.matmul(K, tf.matmul(R,
                                           tf.linalg.matrix_transpose(K)))

        covar_up = self._make_valid(covar_up)
        return state_up, covar_up

    def _KF_gain(self, state_pred, covar_pred, sigma_points_pred, weights_c,
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
        weights_c : tensor [batch_size, num_sigma]
            weights for computing the covariance of the sigma points
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
        diff_z = tf.cast(diff_z, tf.float64)
        diff_z = tf.reshape(diff_z, [-1, self.dim_z])
        diff_z = self.context.correct_observation_diff(diff_z)
        diff_z = tf.reshape(diff_z,
                            [self.batch_size, self.num_sigma, self.dim_z])
        diff_square = tf.matmul(diff_z[:, :, :, None], diff_z[:, :, None, :])
        S = tf.multiply(diff_square, tf.cast(weights_c[:, :, :, None],
                                             tf.float64))
        S = tf.cast(tf.reduce_sum(S, axis=1), tf.float32) + R
        S = self._make_valid(S)

        diff_state = sigma_points_pred - state_pred[:, None, :]
        diff_state = tf.reshape(diff_state, [-1, self.dim_x])
        diff_state = self.context.correct_state(diff_state)
        diff_state = tf.reshape(diff_state,
                                [self.batch_size, self.num_sigma, self.dim_x])
        z_state = tf.matmul(tf.cast(diff_state[:, :, :, None], tf.float64),
                            diff_z[:, :, None, :])
        z_state_covar = \
            tf.reduce_sum(tf.multiply(z_state,
                                      tf.cast(weights_c[:, :, :, None],
                                              tf.float64)), axis=1)
        z_state_covar = tf.cast(z_state_covar, tf.float32)

        K = tf.matmul(z_state_covar, tf.linalg.inv(S))
        return K
