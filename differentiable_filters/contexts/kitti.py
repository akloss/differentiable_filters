# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:04:00 2020

@author: akloss
"""

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import csv
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


from differentiable_filters.contexts import base_context as base
from differentiable_filters.contexts.base_layer import BaseLayer
from differentiable_filters.contexts import recordio as tfr


class Context(base.BaseContext):
    def __init__(self, param, mode):
        base.BaseContext.__init__(self, param, mode)
        self.debug = param['debug']
        if 'normalize' in param.keys():
            self.normalize = param['normalize']
        else:
            self.normalize = 'layer'

        # the state size
        self.dim_x = 5
        self.dim_u = 0
        self.dim_z = 2

        self.x_names = ['x', 'y', 'theta', 'v', 'theta_dot']
        self.z_names = ['v', 'theta_dot']

        self.dt = 0.103

        self.scale = param['scale']
        self.sl = param['sequence_length']

        # define initial values for the process noise q and observation noise r
        # diagonals
        # Important: All values are standard-deviations, so they are
        # squared for forming the covariance matrices
        if param['q_diag'] is not None:
            cov_string = param['q_diag']
            cov = list(map(lambda x: float(x), cov_string.split(' ')))
            self.q_diag = np.array(cov).astype(np.float32)
        else:
            self.q_diag = np.ones((self.dim_x)).astype(np.float32)
        self.q_diag = self.q_diag.astype(np.float32) / self.scale

        if param['r_diag'] is not None:
            cov_string = param['r_diag']
            cov = list(map(lambda x: float(x), cov_string.split(' ')))
            self.r_diag = np.array(cov).astype(np.float32)
        else:
            self.r_diag = np.ones((self.dim_z)).astype(np.float32)
        self.r_diag = self.r_diag.astype(np.float32) / self.scale

        # if the noise matrices are not learned, we construct the fixed
        # covariance matrices here
        q = np.diag(np.square(self.q_diag))
        self.Q = tf.convert_to_tensor(q, dtype=tf.float32)
        r = np.diag(np.square(self.r_diag))
        self.R = tf.convert_to_tensor(r, dtype=tf.float32)

        # fixed noise for initial covariance with sqrt(diag) = [5. 5. 5. 5.]
        self.noise_list = \
            [np.array([0., 0., 0., 0., 0]).astype(np.float32),
             np.array([0, 0, 0, -5.09878697, 2.4157549]).astype(np.float32),
             np.array([0, 0, 0, 1.13696468, 7.36979061]).astype(np.float32)]

        if mode == 'filter':
            train_sensor_model = param['train_sensor_model']
            train_process_model = param['train_process_model']
            train_q = param['train_q']
            train_r = param['train_r']
            if param['filter'] == 'lstm':
                train_process_model = False
                train_q = False
                train_r = False
            # tensorflow does not allow summaries inside rnn-loops
            summary = False
        else:
            train_sensor_model = True
            train_process_model = True
            train_q = True
            train_r = True
            summary = True

        # all layers used in the context need to be instantiated here, but we
        # cannot instantiate layers that will not be used
        if mode == 'filter' or mode == 'pretrain_obs':
            self.sensor_model_layer = \
                SensorModel(self.batch_size, self.normalize, summary,
                            train_sensor_model)
            self.observation_model_layer = ObservationModel(self.dim_z,
                                                            self.batch_size)
            # group the layers for easier checkpoint restoring
            self.observation_models = {'sensor': self.sensor_model_layer,
                                       'obs': self.observation_model_layer}
            self.update_ops += self.sensor_model_layer.updateable
        else:
            self.observation_models = {}

        lstm_no_noise = param['filter'] == 'lstm' and \
            not param['lstm_structure'] == 'full'
        self.observation_noise_models = {}
        if param['learn_r'] and param['hetero_r'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                not lstm_no_noise or mode == 'pretrain_obs':
            self.observation_noise_hetero_diag = \
                ObservationNoise(self.batch_size, self.dim_z, self.r_diag,
                                 self.scale, hetero=True, diag=True,
                                 trainable=train_r, summary=summary)
            self.observation_noise_models['het_diag'] = \
                self.observation_noise_hetero_diag
        if param['learn_r'] and param['hetero_r'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                not lstm_no_noise or mode == 'pretrain_obs':
            self.observation_noise_hetero_full = \
                ObservationNoise(self.batch_size, self.dim_z, self.r_diag,
                                 self.scale, hetero=True, diag=False,
                                 trainable=train_r, summary=summary)
            self.observation_noise_models['het_full'] = \
                self.observation_noise_hetero_full
        if param['learn_r'] and not param['hetero_r'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                not lstm_no_noise or mode == 'pretrain_obs':
            self.observation_noise_const_diag = \
                ObservationNoise(self.batch_size, self.dim_z, self.r_diag,
                                 self.scale, hetero=False, diag=True,
                                 trainable=train_r, summary=summary)
            self.observation_noise_models['const_diag'] = \
                self.observation_noise_const_diag
        if param['learn_r'] and not param['hetero_r'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                not lstm_no_noise or mode == 'pretrain_obs':
            self.observation_noise_const_full = \
                ObservationNoise(self.batch_size, self.dim_z, self.r_diag,
                                 self.scale, hetero=False, diag=False,
                                 trainable=train_r, summary=summary)
            self.observation_noise_models['const_full'] = \
                self.observation_noise_const_full
        if param['learned_likelihood'] and mode == 'filter' and \
                not lstm_no_noise or mode == 'pretrain_obs':
            self.likelihood_layer = Likelihood(self.dim_z,
                                               trainable=train_r,
                                               summary=summary)
            self.observation_noise_models['like'] = self.likelihood_layer

        self.process_models = {}
        lstm_unstructured = param['filter'] == 'lstm' and \
            (param['lstm_structure'] == 'none' or
             param['lstm_structure'] == 'lstm' or
             param['lstm_structure'] == 'lstm1' or
             param['lstm_structure'] == 'lstm1_1' or
             param['lstm_structure'] == 'lstm1_2' or
             param['lstm_structure'] == 'lstm2_1' or
             param['lstm_structure'] == 'lstm2_2')
        if mode == 'filter' and not lstm_unstructured and \
                param['learn_process'] or mode == 'pretrain_process':
            self.process_model_learned_layer = \
                ProcessModel(self.batch_size, self.dim_x, self.dim_u, self.dt,
                             self.scale, learned=True,
                             jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['learned'] = self.process_model_learned_layer
        if mode == 'filter' and not lstm_unstructured and \
                not param['learn_process'] or mode == 'pretrain_process':
            self.process_model_analytical_layer = \
                ProcessModel(self.batch_size, self.dim_x, self.dim_u, self.dt,
                             self.scale, learned=False,
                             jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['ana'] = self.process_model_analytical_layer

        self.process_noise_models = {}
        if param['learn_q'] and param['hetero_q'] and not lstm_no_noise and \
                param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] or mode == 'pretrain_process':
            self.process_noise_hetero_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_lrn'] = \
                self.process_noise_hetero_diag_lrn
        if param['learn_q'] and param['hetero_q'] and not lstm_no_noise and \
                not param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] or mode == 'pretrain_process':
            self.process_noise_hetero_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_full_lrn'] = \
                self.process_noise_hetero_full_lrn
        if param['learn_q'] and not param['hetero_q'] and \
                not lstm_no_noise and param['diagonal_covar'] and \
                mode == 'filter' and param['learn_process'] or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_lrn'] = \
                self.process_noise_const_diag_lrn
        if param['learn_q'] and not param['hetero_q'] and \
                not lstm_no_noise and not param['diagonal_covar'] and \
                mode == 'filter' and param['learn_process'] or \
                mode == 'pretrain_process':
            self.process_noise_const_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=True, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_lrn'] = \
                self.process_noise_const_full_lrn
        if param['learn_q'] and param['hetero_q'] and not lstm_no_noise and \
                param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] or mode == 'pretrain_process':
            self.process_noise_hetero_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=False,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_ana'] = \
                self.process_noise_hetero_diag_ana
        if param['learn_q'] and param['hetero_q'] and not lstm_no_noise and \
                not param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] or mode == 'pretrain_process':
            self.process_noise_hetero_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['het_full_ana'] = \
                self.process_noise_hetero_full_ana
        if param['learn_q'] and not param['hetero_q'] and \
                not lstm_no_noise and param['diagonal_covar'] and \
                mode == 'filter' and not param['learn_process'] or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_ana'] = \
                self.process_noise_const_diag_ana
        if param['learn_q'] and not param['hetero_q'] and \
                not lstm_no_noise and not param['diagonal_covar'] and \
                mode == 'filter' and not param['learn_process'] or \
                mode == 'pretrain_process':
            self.process_noise_const_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_ana'] = \
                self.process_noise_const_full_ana

    def initial_from_observed(self, base_state, init_z, base_covar, init_R):
        state = tf.concat([base_state[:, :3], init_z], axis=-1)
        covar = \
            tf.concat([base_covar[:, :3, :],
                       tf.concat([base_covar[:, 3:, :3], init_R], axis=-1)],
                      axis=1)
        return state, covar

    ###########################################################################
    # loss functions
    ###########################################################################
    def get_filter_loss(self, prediction, label, step, training):
        """
        Compute the loss for the filtering application - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step
            training: boolean tensor, indicates if we compute a loss for
            training or testing

        Returns:
            loss: the total loss for training the filtering application
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        particles, weights, states, covars, init_s, init_c, z, r, q = \
            prediction
        states = tf.reshape(states, [self.batch_size, -1, self.dim_x])
        covars = tf.reshape(covars, [self.batch_size, -1, self.dim_x,
                                     self.dim_x])
        seq_label, mv_tr, mv_rot = label

        diff = seq_label - states
        diff = self.correct_state(diff)

        # get the likelihood
        if self.param['filter'] == 'pf' and self.param['mixture_likelihood']:
            num = particles.get_shape()[2].value
            seq_label_tiled = tf.tile(seq_label[:, :, None, :], [1, 1, num, 1])
            particle_diff = self.correct_state(seq_label_tiled - particles)
            likelihood = self._mixture_likelihood(particle_diff, weights)
        else:
            likelihood = self._likelihood(diff, covars, reduce_mean=False)

        # compensate for scaling
        offset = tf.ones_like(likelihood)*tf.math.log(self.scale)*2*self.dim_x
        likelihood += 0.5 * offset

        # compute the errors of the predicted states
        total_mse, total_dist = self._mse(diff, reduce_mean=False)
        total_mse *= self.scale**2
        total_dist *= self.scale

        # compute component-wise distances
        dists = []
        for i in range(self.dim_x):
            _, dist = self._mse(diff[:, :, i:i+1], reduce_mean=False)
            dists += [dist*self.scale]

        # compute the output metric
        m_per_tr, deg_per_tr = \
            self._output_loss(states, seq_label, mv_tr)

        # compute the endpoint error
        _, endpoint_error_tr = self._mse(diff[:, -1, 0:2], reduce_mean=False)
        endpoint_error_tr *= self.scale
        _, endpoint_error_rot = self._mse(diff[:, -1, 2:3], reduce_mean=False)
        endpoint_error_rot *= self.scale

        # compute the error in the predicted observations (only for monitoring)
        diff_obs = self.correct_observation_diff(seq_label[:, :, 3:] - z)
        mse_v_obs, dist_v_obs = self._mse(diff_obs[:, :, :1],
                                          reduce_mean=False)
        mse_dt_obs, dist_dt_obs = self._mse(diff_obs[:, :, 1:],
                                            reduce_mean=False)

        dist_v_obs = dist_v_obs * self.scale
        dist_dt_obs = dist_dt_obs * self.scale

        _, dist_obs = self._mse(diff_obs, reduce_mean=False)
        dist_obs *= self.scale

        # get the weight decay
        wd = []
        for la in self.observation_models.values():
            wd += la.losses
        for la in self.observation_noise_models.values():
            wd += la.losses
        for la in self.process_models.values():
            wd += la.losses
        for la in self.process_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        # add a bias to all losses that use the likelihood, to set off
        # possible negative values of the likelihood
        total_tracking = tf.reduce_mean(total_mse)
        if self.loss == 'like':
            total_loss = tf.reduce_mean(likelihood)
        elif self.loss == 'error':
            total_loss = total_tracking
        elif self.loss == 'mixed':
            total_loss = (total_tracking +
                          tf.reduce_mean(likelihood)) / 2.
        elif self.loss == 'mixed_error':
            total_loss = total_tracking * 0.75 + \
                tf.reduce_mean(likelihood) * 0.25
        elif self.loss == 'mixed_like':
            total_loss = total_tracking * 0.25 + \
                tf.reduce_mean(likelihood) * 0.75
        elif self.loss == 'mixed_curr':
            total_loss = tf.cond(tf.less(step, self.epoch_size * 5),
                                 lambda: total_tracking +
                                 tf.reduce_mean(likelihood)/10000.,
                                 lambda: (total_tracking +
                                          tf.reduce_mean(likelihood)) / 2.)

        if self.loss == 'mixed_curr':
            total_loss_val = tf.reduce_mean(likelihood)
        else:
            total_loss_val = total_loss

        if self.loss != 'error':
            total_loss_val += 1000

        total = tf.cond(training,
                        lambda: 100 * total_loss + wd, lambda: total_loss_val)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood', tf.reduce_mean(likelihood))
        tf.summary.scalar('loss/tracking', total_tracking)
        tf.summary.scalar('out/m_per_tr', m_per_tr)
        tf.summary.scalar('out/deg_per_tr', deg_per_tr)
        for i, name in enumerate(self.x_names):
            tf.summary.scalar('tracking_loss/' + name,
                              tf.reduce_mean(dists[i]))
        tf.summary.scalar('observation_loss/dist_v',
                          tf.reduce_mean(dist_v_obs))
        tf.summary.scalar('observation_loss/dist_dt',
                          tf.reduce_mean(dist_dt_obs))
        return total, [likelihood, total_dist, dist_obs, total_mse,
                       endpoint_error_tr, endpoint_error_rot, m_per_tr,
                       deg_per_tr, dist_v_obs, dist_dt_obs, wd] + dists, \
            ['likelihood', 'dist', 'dist_obs', 'mse', 'end_tr', 'end_rot',
             'm_tr', 'deg_tr', 'v_obs', 'dt_obs', 'wd'] + self.x_names

    def _output_loss(self, pred, label, mv_tr):
        endpoint_error = self._compute_sq_distance(pred[:, -1, 0:2],
                                                   label[:, -1, 0:2])

        endpoint_error_rot = self._compute_sq_distance(pred[:, -1, 2:3],
                                                       label[:, -1, 2:3], True)

        m_per_tr = tf.reduce_mean(endpoint_error**0.5/mv_tr)
        deg_per_tr = tf.reduce_mean(endpoint_error_rot**0.5/mv_tr)

        return m_per_tr, deg_per_tr

    def _compute_sq_distance(self, pred, label, rotation=False):
        diff = pred - label
        if rotation:
            diff = self._correct_theta(diff, self.scale)
        diff = tf.square(diff)
        diff = tf.reduce_sum(diff, axis=-1)

        return diff

    def get_observation_loss(self, prediction, label, step, training):
        """
        Compute the loss for the observation functions - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step
            training: are we doing training or validation

        Returns:
            loss: the total loss for training the observation preprocessing
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        z, R_const_diag, R_const_tri, R_het_diag, R_het_tri, like_good, \
            like_bad = prediction

        label = label[0]
        diff = label - z

        likelihood_const_diag = self._likelihood(diff, R_const_diag,
                                                 reduce_mean=False)
        likelihood_const_tri = self._likelihood(diff, R_const_tri,
                                                reduce_mean=False)
        likelihood_het_diag = self._likelihood(diff, R_het_diag,
                                               reduce_mean=False)
        likelihood_het_tri = self._likelihood(diff, R_het_tri,
                                              reduce_mean=False)

        likelihood = (likelihood_const_diag + likelihood_const_tri +
                      likelihood_het_diag + likelihood_het_tri) / 4.

        # compute the errors of the predicted observations
        mse_v, dist_v = self._mse(diff[:, 0:1], reduce_mean=False)
        # undo the overall scaling for dist and mse
        dist_v = dist_v * self.scale
        mse_v *= self.scale**2
        mse_dt, dist_dt = self._mse(diff[:, 1:], reduce_mean=False)
        dist_dt = dist_dt * self.scale
        mse_dt *= self.scale**2

        # compute the loss for the learned likelihood model of the pf
        good_loss = tf.reduce_mean(-tf.math.log(tf.maximum(like_good, 1e-6)))
        bad_loss = \
            tf.reduce_mean(-tf.math.log(tf.maximum(1.0 - like_bad, 1e-6)))
        like_loss = good_loss + bad_loss

        wd = []
        for la in self.observation_models.values():
            wd += la.losses
        for la in self.observation_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        obs = tf.reduce_mean(mse_v) + tf.reduce_mean(mse_dt)
        total_loss = \
            tf.cond(tf.less(step, self.epoch_size*10),
                    lambda: 10 * obs,
                    lambda: tf.reduce_mean(likelihood) + 10 * obs)
        total_loss = \
            tf.cond(tf.less(step, self.epoch_size*15),
                    lambda: total_loss,
                    lambda: tf.reduce_mean(likelihood) + 10 * like_loss + wd +
                    10 * obs)

        total_train = total_loss
        total_val = tf.reduce_mean(likelihood) + like_loss + 10 * obs + 100

        total = tf.cond(training,
                        lambda: total_train, lambda: total_val)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood_const_diag',
                          tf.reduce_mean(likelihood_const_diag))
        tf.summary.scalar('loss/likelihood_const_tri',
                          tf.reduce_mean(likelihood_const_tri))
        tf.summary.scalar('loss/likelihood_het_diag',
                          tf.reduce_mean(likelihood_het_diag))
        tf.summary.scalar('loss/likelihood_het_tri',
                          tf.reduce_mean(likelihood_het_tri))
        tf.summary.scalar('loss/dist_v', tf.reduce_mean(dist_v))
        tf.summary.scalar('loss/dist_dt', tf.reduce_mean(dist_dt))
        tf.summary.scalar('loss/like_good', good_loss)
        tf.summary.scalar('loss/like_bad', bad_loss)
        tf.summary.scalar('loss/like_loss', like_loss)
        return total, [likelihood_const_diag, likelihood_const_tri,
                       likelihood_het_diag, likelihood_het_tri, dist_v,
                       dist_dt, like_loss, wd], \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri', 'v', 'dt', 'like',
             'wd']

    def get_process_loss(self, prediction, labels, step, training):
        """
        Compute the loss for the process functions - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step
            training: boolean tensor, indicates if we compute a loss for
            training or testing

        Returns:
            loss: the total loss for training the process model
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        state, Q_const_diag, Q_const_tri, Q_het_diag, Q_het_tri, \
            state_ana, Q_const_diag_ana, Q_const_tri_ana, Q_het_diag_ana, \
            Q_het_tri_ana = prediction

        label, start = labels
        diff = label - state
        diff = self.correct_state(diff)

        likelihood_const_diag = self._likelihood(diff, Q_const_diag,
                                                 reduce_mean=False)
        likelihood_const_tri = self._likelihood(diff, Q_const_tri,
                                                reduce_mean=False)
        likelihood_het_diag = self._likelihood(diff, Q_het_diag,
                                               reduce_mean=False)
        likelihood_het_tri = self._likelihood(diff, Q_het_tri,
                                              reduce_mean=False)

        likelihood = (likelihood_const_diag + likelihood_const_tri +
                      likelihood_het_diag + likelihood_het_tri) / 4.

        diff_ana = label - state_ana
        diff_ana = self.correct_state(diff_ana)

        likelihood_const_diag_ana = self._likelihood(diff_ana,
                                                     Q_const_diag_ana,
                                                     reduce_mean=False)
        likelihood_const_tri_ana = self._likelihood(diff_ana, Q_const_tri_ana,
                                                    reduce_mean=False)
        likelihood_het_diag_ana = self._likelihood(diff_ana, Q_het_diag_ana,
                                                   reduce_mean=False)
        likelihood_het_tri_ana = self._likelihood(diff_ana, Q_het_tri_ana,
                                                  reduce_mean=False)

        likelihood_ana = \
            (likelihood_const_diag_ana + likelihood_const_tri_ana +
             likelihood_het_diag_ana + likelihood_het_tri_ana) / 4.

        # compute the errors of the predicted states
        mses = []
        dists = []
        dists_ana = []
        for i in range(self.dim_x):
            mse, dist = self._mse(diff[:, i:i+1], reduce_mean=False)
            _, dist_ana = self._mse(diff_ana[:, i:i+1], reduce_mean=False)
            # undo the overall scaling for dist and mse
            mses += [mse*self.scale**2]
            dists += [dist*self.scale]
            dists_ana += [dist_ana*self.scale]
        mse = tf.add_n(mses)
        dist = tf.add_n(dists)
        dist_ana = tf.add_n(dists_ana)

        wd = []
        for la in self.process_models.values():
            wd += la.losses
        for la in self.process_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        total_loss = \
            tf.cond(tf.less(step, 10000),
                    lambda: 10 * tf.reduce_mean(mse),
                    lambda: tf.reduce_mean(likelihood) +
                    tf.reduce_mean(likelihood_ana) + 10 * tf.reduce_mean(mse))

        total = \
            tf.cond(training,
                    lambda: total_loss + wd,
                    lambda: tf.reduce_mean(likelihood) + 100 +
                    tf.reduce_mean(likelihood_ana) + 10 * tf.reduce_mean(mse))

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood_const_diag',
                          tf.reduce_mean(likelihood_const_diag))
        tf.summary.scalar('loss/likelihood_const_tri',
                          tf.reduce_mean(likelihood_const_tri))
        tf.summary.scalar('loss/likelihood_het_diag',
                          tf.reduce_mean(likelihood_het_diag))
        tf.summary.scalar('loss/likelihood_het_tri',
                          tf.reduce_mean(likelihood_het_tri))
        tf.summary.scalar('loss/likelihood_const_diag_ana',
                          tf.reduce_mean(likelihood_const_diag_ana))
        tf.summary.scalar('loss/likelihood_const_tri_ana',
                          tf.reduce_mean(likelihood_const_tri_ana))
        tf.summary.scalar('loss/likelihood_het_diag_ana',
                          tf.reduce_mean(likelihood_het_diag_ana))
        tf.summary.scalar('loss/likelihood_het_tri_ana',
                          tf.reduce_mean(likelihood_het_tri_ana))
        for i, name in enumerate(self.x_names):
            tf.summary.scalar('tracking_loss/' + name,
                              tf.reduce_mean(dists[i]))
            tf.summary.scalar('tracking_loss/' + name + '_ana',
                              tf.reduce_mean(dists_ana[i]))

        for i in range(min(self.batch_size, 3)):
            tf.summary.scalar('label/x_' + str(i), label[i, 0])
            tf.summary.scalar('label/y_' + str(i), label[i, 1])
            tf.summary.scalar('label/theta_' + str(i), label[i, 2])
            tf.summary.scalar('label/v_' + str(i), label[i, 3])
            tf.summary.scalar('label/theta_dot_' + str(i), label[i, 4])
            tf.summary.scalar('start/x_' + str(i), start[i, 0])
            tf.summary.scalar('start/y_' + str(i), start[i, 1])
            tf.summary.scalar('start/theta_' + str(i), start[i, 2])
            tf.summary.scalar('start/v_' + str(i), start[i, 3])
            tf.summary.scalar('start/theta_dot_' + str(i), start[i, 4])
            tf.summary.scalar('pred/x_' + str(i), state[i, 0])
            tf.summary.scalar('pred/y_' + str(i), state[i, 1])
            tf.summary.scalar('pred/theta_' + str(i), state[i, 2])
            tf.summary.scalar('pred/v_' + str(i), state[i, 3])
            tf.summary.scalar('pred/theta_dot_' + str(i), state[i, 4])

        return total, \
            [likelihood_const_diag, likelihood_const_tri, likelihood_het_diag,
             likelihood_het_tri, likelihood_const_diag_ana,
             likelihood_const_tri_ana, likelihood_het_diag_ana,
             likelihood_het_tri_ana, wd] + dists + dists_ana, \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri',
             'likelihood_const_diag_ana', 'likelihood_const_tri_ana',
             'likelihood_het_diag_ana', 'likelihood_het_tri_ana',
             'wd'] + self.x_names + list(map(lambda x: x + '_ana',
                                             self.x_names))

    ###########################################################################
    # keeping the state correct
    ###########################################################################
    def correct_state(self, state, diff=True):
        """
        Correct the state to make sure theta is in the right interval
        Args:
            state: The current state
        Returns:
            state: The corrected state
        """
        shape = state.get_shape().as_list()
        if len(shape) > 2:
            state = tf.reshape(state, [-1, self.dim_x])
        # correct theta and dt
        state = \
            tf.concat([state[:, :2],
                       self._correct_theta(state[:, 2:3], self.scale),
                       state[:, 3:4],
                       self._correct_theta(state[:, 4:], self.scale)],
                      axis=-1)
        if len(shape) > 2:
            state = tf.reshape(state, shape[:-1] + [self.dim_x])
        return state

    def correct_observation_diff(self, diff):
        """
        Correct a difference in observations to account for angle intervals
        Args:
            state: The difference
        Returns:
            state: The corrected difference
        """
        shape = diff.get_shape().as_list()
        if len(shape) > 2:
            diff = tf.reshape(diff, [-1, self.dim_z])
        diff = tf.concat([diff[:, :1],
                          self._correct_theta(diff[:, 1:2], self.scale)],
                         axis=-1)
        if len(shape) > 2:
            diff = tf.reshape(diff, shape[:-1] + [self.dim_z])
        return diff

    def _correct_theta(self, theta, sc=1):
        # undo scaling
        theta = theta * sc
        theta = theta - tf.ones_like(theta) * 180.
        theta = tf.math.mod(theta, tf.ones_like(theta) * 360.)
        theta = theta - tf.ones_like(theta) * 180.
        # redo scaling
        theta = theta / sc
        return theta

    def weighted_state_mean_with_angles(self, points, weights):
        ps = tf.concat([points[:, :, :2],
                        tf.sin(points[:, :, 2:3]*self.scale*np.pi/180.),
                        tf.cos(points[:, :, 2:3]*self.scale*np.pi/180.),
                        points[:, :, 3:4],
                        tf.sin(points[:, :, 4:5]*self.scale*np.pi/180.),
                        tf.cos(points[:, :, 4:5]*self.scale*np.pi/180.)],
                       axis=-1)
        mult = tf.multiply(ps, weights)
        mean = tf.reduce_sum(mult, axis=1)

        ang1 = tf.math.atan2(mean[:, 2:3], mean[:, 3:4])*180/np.pi
        ang2 = tf.math.atan2(mean[:, 5:6], mean[:, 6:7])*180/np.pi

        out = tf.concat([mean[:, :2], ang1/self.scale, mean[:, 4:5],
                         ang2/self.scale], axis=-1)
        return out

    def weighted_observation_mean_with_angles(self, points, weights, axis=1):
        ps = tf.concat([points[:, :, :1],
                        tf.sin(points[:, :, 1:2]*self.scale*np.pi/180.),
                        tf.cos(points[:, :, 1:2]*self.scale*np.pi/180.)],
                       axis=-1)
        mult = tf.multiply(ps, weights)
        mean = tf.reduce_sum(mult, axis=axis)

        ang = tf.math.atan2(mean[:, 1:2], mean[:, 2:3])*180/np.pi

        out = tf.concat([mean[:, :1], ang/self.scale], axis=-1)
        return out

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        keys = ['state', 'image', 'image_diff']

        record_meta = tfr.RecordMeta.load(path, name + '_' + data_mode + '_')

        def _parse_function_obs_train(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)
            state = features['state']
            thet = self._correct_theta(state[:, 4:]*180/np.pi, 1)
            label = tf.concat([state[:, 3:4]/self.scale, thet/self.scale],
                              axis=-1)
            im = features['image']
            diff = features['image_diff']

            # we use several steps of the sequence
            start_inds = np.random.randint(0, im.get_shape()[0].value-1, 30)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            ims = []
            diffs = []
            labels = []
            good_zs = []
            bad_zs = []
            for si in start_inds:
                ims += [im[si]]
                diffs += [diff[si]]
                labels += [label[si]]
                good_noise = np.random.normal(loc=0, scale=1e-2, size=(6, 2))
                good_noise[0, :] = 0
                good_noise[:, 0:1] /= self.scale
                good_noise[:, 1:2] /= self.scale
                good_zs += [tf.tile(label[si:si+1], [6, 1]) + good_noise]
                bad_noise = np.random.normal(loc=4, scale=2, size=(6, 2))
                bad_noise[3:] = np.random.normal(loc=-4, scale=2, size=(3, 2))
                bad_noise[:, 0:1] /= self.scale
                bad_noise[:, 1:2] /= self.scale
                bad_zs += [tf.tile(label[si:si+1], [6, 1]) + bad_noise]

            values = [(tf.stack(ims), tf.stack(diffs)), tf.stack(labels),
                      tf.stack(good_zs), tf.stack(bad_zs)]
            labels = [tf.stack(labels)]
            return tuple(values), tuple(labels)

        def _parse_function_obs_val(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)

            state = features['state']
            thet = self._correct_theta(state[:, 4:]*180/np.pi, 1)
            scv = 1./self.scale
            sctd = 1./self.scale
            label = tf.concat([state[:, 3:4]*scv, thet*sctd], axis=-1)
            im = features['image']
            diff = features['image_diff']

            ims = []
            diffs = []
            labels = []
            good_zs = []
            bad_zs = []
            # use every third data point
            start_inds = np.arange(0, im.get_shape()[0].value-1, 3)
            for si in start_inds:
                ims += [im[si]]
                diffs += [diff[si]]
                labels += [label[si]]
                good_noise = np.random.normal(loc=0, scale=1e-2, size=(6, 2))
                good_noise[0, :] = 0
                good_noise[:, 0:1] *= scv
                good_noise[:, 1:2] *= sctd
                good_zs += [tf.tile(label[si:si+1], [6, 1]) + good_noise]
                bad_noise = np.random.normal(loc=4, scale=2, size=(6, 2))
                bad_noise[3:] = np.random.normal(loc=-4, scale=2, size=(3, 2))
                bad_noise[:, 0:1] *= scv
                bad_noise[:, 1:2] *= sctd
                bad_zs += [tf.tile(label[si:si+1], [6, 1]) + bad_noise]

            values = [(tf.stack(ims), tf.stack(diffs)), tf.stack(labels),
                      tf.stack(good_zs), tf.stack(bad_zs)]
            labels = [tf.stack(labels)]

            return tuple(values), tuple(labels)

        def _parse_function_process_train(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)
            state = features['state']
            state = \
                tf.concat([state[:, :2],
                           self._correct_theta(state[:, 2:3]*180./np.pi, 1),
                           state[:, 3:4],
                           self._correct_theta(state[:, 4:]*180/np.pi, 1)],
                          axis=-1)
            # apply scaling
            state = state/self.scale

            # calculate the movements
            mv = state[1:] - state[:-1]
            actions = mv[:, 3:]

            # we use several steps of the sequence
            start_inds = np.random.randint(1, state.get_shape()[0].value-1, 30)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            starts = []
            labels = []
            acs = []
            for si in start_inds:
                starts += [state[si-1]]
                labels += [state[si]]
                acs += [actions[si]]

            values = [tf.stack(starts), tf.stack(acs)]
            labels = [tf.stack(labels), tf.stack(starts)]

            return tuple(values), tuple(labels)

        def _parse_function_process_val(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)

            state = features['state']
            state = \
                tf.concat([state[:, :2],
                           self._correct_theta(state[:, 2:3]*180./np.pi, 1),
                           state[:, 3:4],
                           self._correct_theta(state[:, 4:]*180/np.pi, 1)],
                          axis=-1)
            # apply scaling
            state = state/self.scale
            # calculate the movements
            mv = state[1:] - state[:-1]
            actions = mv[:, 3:]

            # use every fith data point
            start_inds = np.arange(1, state.get_shape()[0].value-1, 5)
            starts = []
            labels = []
            acs = []
            for si in start_inds:
                starts += [state[si-1]]
                labels += [state[si]]
                acs += [actions[si]]

            values = [tf.stack(starts), tf.stack(acs)]
            labels = [tf.stack(labels), tf.stack(starts)]

            return tuple(values), tuple(labels)

        def _parse_function_train(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)
            state = features['state']
            state = \
                tf.concat([state[:, :2],
                           self._correct_theta(state[:, 2:3]*180./np.pi, 1),
                           state[:, 3:4],
                           self._correct_theta(state[:, 4:]*180/np.pi, 1)],
                          axis=-1)
            # apply scaling
            state = state/self.scale

            # calculate the movements
            mv = state[1:] - state[:-1]
            actions = mv[:, 3:]
            mv_tr = mv[:, :2]
            mv_rot = self._correct_theta(mv[:, 2:3], self.scale)

            im = features['image']
            diff = features['image_diff']

            if self.sl == 100:
                start_inds = [0]
            else:
                # we use several sub-sequences of the sequence, such that
                # the overall amount of data stays the same between sequence
                # lengths (maximum length is 100)
                num = 100 // self.sl
                start_inds = \
                    np.random.randint(0, im.get_shape()[0].value-self.sl-1,
                                      num)
                self.train_multiplier = len(start_inds)

            # prepare the lists of output tensors
            ims = []
            diffs = []
            starts = []
            start_ims = []
            start_diffs = []
            states = []
            mv_trs = []
            mv_rots = []
            acs = []
            for si in start_inds:
                end = si+self.sl+1
                ims += [im[si+1:end]]
                start_ims += [im[si]]
                start_diffs += [im[si]]
                diffs += [diff[si+1:end]]
                states += [state[si+1:end]]
                starts += [state[si]]
                mv_trs += [tf.reduce_sum(tf.norm(mv_tr[si:end], axis=-1))]
                mv_rots += [tf.reduce_sum(tf.abs(mv_rot[si:end]))]
                acs += [actions[si+1:end]]

            # observations, actions, initial observations, initial state,
            # info
            values = [(tf.stack(ims), tf.stack(diffs)), tf.stack(acs),
                      (tf.stack(start_ims), tf.stack(start_diffs)),
                      tf.stack(starts), tf.zeros([len(ims)])]
            labels = [tf.stack(states), tf.stack(mv_trs), tf.stack(mv_rots)]

            return tuple(values), tuple(labels)

        def _parse_function_val(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)
            state = features['state']
            state = \
                tf.concat([state[:, :2],
                           self._correct_theta(state[:, 2:3]*180./np.pi, 1),
                           state[:, 3:4],
                           self._correct_theta(state[:, 4:]*180/np.pi, 1)],
                          axis=-1)
            # apply scaling
            state = state/self.scale

            # calculate the movements
            mv = state[1:] - state[:-1]
            actions = mv[:, 3:]
            mv_tr = mv[:, :2]
            mv_rot = self._correct_theta(mv[:, 2:3], self.scale)

            im = features['image']
            diff = features['image_diff']

            if self.sl == 100:
                start_inds = [0]
            else:
                # we use several sub-sequences of the testsequence, such that
                # the overall amount of data stays teh same between sequence
                # lengths (maximum length is 100)
                num = 100 // self.sl
                # we use several sub-sequences of the testsequence
                start_inds = \
                    np.arange(0, im.get_shape()[0].value-self.sl-1,
                              (self.sl+1)//2)
                start_inds = start_inds[:num]

            # prepare the lists of output tensors
            ims = []
            diffs = []
            starts = []
            start_ims = []
            start_diffs = []
            states = []
            mv_trs = []
            mv_rots = []
            acs = []
            for si in start_inds:
                end = si+self.sl+1
                ims += [im[si+1:end]]
                start_ims += [im[si]]
                start_diffs += [im[si]]
                diffs += [diff[si+1:end]]
                states += [state[si+1:end]]
                starts += [state[si]]
                mv_trs += [tf.reduce_sum(tf.norm(mv_tr[si:end], axis=-1))]
                mv_rots += [tf.reduce_sum(tf.abs(mv_rot[si:end]))]
                acs += [actions[si+1:end]]

            # observations, actions, initial observations, initial state,
            # info
            values = [(tf.stack(ims), tf.stack(diffs)), tf.stack(acs),
                      (tf.stack(start_ims), tf.stack(start_diffs)),
                      tf.stack(starts), tf.zeros([len(ims)])]
            labels = [tf.stack(states), tf.stack(mv_trs), tf.stack(mv_rots)]

            return tuple(values), tuple(labels)

        def _parse_function_test(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)

            state = features['state']
            state = \
                tf.concat([state[:, :2],
                           self._correct_theta(state[:, 2:3]*180./np.pi, 1),
                           state[:, 3:4],
                           self._correct_theta(state[:, 4:]*180/np.pi, 1)],
                          axis=-1)
            # apply scaling
            state = state/self.scale

            # calculate the movements
            mv = state[1:] - state[:-1]
            actions = mv[:, 3:]
            mv_tr = mv[:, :2]
            mv_rot = self._correct_theta(mv[:, 2:3], self.scale)

            im = features['image']
            diff = features['image_diff']

            # we use several sub-sequences of the testsequence
            start_inds = \
                np.arange(0, im.get_shape()[0].value-self.sl-1, self.sl//3)

            self.test_multiplier = len(start_inds)

            # prepare the lists of output tensors
            ims = []
            diffs = []
            starts = []
            start_ims = []
            start_diffs = []
            states = []
            mv_trs = []
            mv_rots = []
            acs = []
            for si in start_inds:
                end = si+self.sl+1
                ims += [im[si+1:end]]
                start_ims += [im[si]]
                start_diffs += [im[si]]
                diffs += [diff[si+1:end]]
                states += [state[si+1:end]]
                starts += [state[si]]
                mv_trs += [tf.reduce_sum(tf.norm(mv_tr[si:end], axis=-1))]
                mv_rots += [tf.reduce_sum(tf.abs(mv_rot[si:end]))]
            # prepare the lists of output tensors
            ims = []
            diffs = []
            starts = []
            start_ims = []
            start_diffs = []
            states = []
            mv_trs = []
            mv_rots = []
            acs = []
            for si in start_inds:
                end = si+self.sl+1
                ims += [im[si+1:end]]
                start_ims += [im[si]]
                start_diffs += [im[si]]
                diffs += [diff[si+1:end]]
                states += [state[si+1:end]]
                starts += [state[si]]
                mv_trs += [tf.reduce_sum(tf.norm(mv_tr[si:end], axis=-1))]
                mv_rots += [tf.reduce_sum(tf.abs(mv_rot[si:end]))]
                acs += [actions[si+1:end]]

            # observations, actions, initial observations, initial state,
            # info
            values = [(tf.stack(ims), tf.stack(diffs)), tf.stack(acs),
                      (tf.stack(start_ims), tf.stack(start_diffs)),
                      tf.stack(starts), tf.zeros([len(ims)])]
            labels = [tf.stack(states), tf.stack(mv_trs), tf.stack(mv_rots)]

            return tuple(values), tuple(labels)

        if train_mode == 'filter':
            if data_mode == 'train':
                dataset = dataset.map(_parse_function_train,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'val':
                dataset = dataset.map(_parse_function_val,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'test':
                dataset = dataset.map(_parse_function_test,
                                      num_parallel_calls=num_threads)
            dataset = \
                dataset.flat_map(lambda x, y:
                                 tf.data.Dataset.from_tensor_slices((x, y)))
        elif train_mode == 'pretrain_obs':
            if data_mode == 'train':
                dataset = dataset.map(_parse_function_obs_train,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'val' or data_mode == 'test':
                dataset = dataset.map(_parse_function_obs_val,
                                      num_parallel_calls=num_threads)
            dataset = \
                dataset.flat_map(lambda x, y:
                                 tf.data.Dataset.from_tensor_slices((x, y)))
        elif train_mode == 'pretrain_process':
            if data_mode == 'train':
                dataset = dataset.map(_parse_function_process_train,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'val' or data_mode == 'test':
                dataset = dataset.map(_parse_function_process_val,
                                      num_parallel_calls=num_threads)
            dataset = \
                dataset.flat_map(lambda x, y:
                                 tf.data.Dataset.from_tensor_slices((x, y)))
        else:
            self.log.error('unknown training mode: ' + train_mode)

        return dataset

    ######################################
    # Evaluation
    ######################################
    def save_log(self, log_dict, out_dir, step, num, mode):
        if mode == 'filter':
            keys = ['noise_num', 'likelihood', 'likelihood_std', 'dist',
                    'dist_std',
                    'dist_obs', 'dist_obs_std', 'm_tr', 'm_tr_std', 'deg_tr',
                    'deg_tr_std',
                    'x', 'x_std', 'y', 'y_std', 'theta', 'theta_std',
                    'v', 'v_std', 'theta_dot', 'theta_dot_std',
                    'end_tr', 'end_tr_std', 'end_rot', 'end_rot_std']

            log_file = open(os.path.join(out_dir, str(step) + '_res.csv'), 'a')
            log = csv.DictWriter(log_file, fieldnames=keys)
            if num == 0:
                log.writeheader()

            non_num = [str, bool, np.str, np.bool]

            row = {}
            for k, v in log_dict.items():
                if k in keys and type(v[0]) not in non_num:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)
            row['noise_num'] = num
            log.writerow(row)
            log_file.close()
        else:
            row = {}
            for k, v in log_dict.items():
                if type(v[0]) not in [str, bool, np.str, np.bool]:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)

            log_file = open(os.path.join(out_dir, str(step) + '_res.csv'),
                            'w')
            log = csv.DictWriter(log_file, sorted(row.keys()))
            log.writeheader()
            log.writerow(row)
            log_file.close()
        return

    def _eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def plot_tracking(self, seq_pred, cov_pred, z, seq, q_pred, r_pred,
                      out_dir, num, diffs, likes):
        pos_pred = np.squeeze(seq_pred[:, :2])
        or_pred = np.squeeze(seq_pred[:, 2:3])
        v_pred = np.squeeze(seq_pred[:, 3:4])
        td_pred = np.squeeze(seq_pred[:, 4:5])

        diffs = np.squeeze(diffs)
        likes = np.squeeze(likes)

        if z is not None:
            v_obs = np.squeeze(z[:, 0])
            td_obs = np.squeeze(z[:, 1])

        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)
            pos_cov_x = np.sqrt(np.squeeze(cov_pred[:, 0, 0]))
            pos_cov_y = np.sqrt(np.squeeze(cov_pred[:, 1, 1]))
            or_cov = np.sqrt(np.squeeze(cov_pred[:, 2, 2]))
            v_cov = np.sqrt(np.squeeze(cov_pred[:, 3, 3]))
            td_cov = np.sqrt(np.squeeze(cov_pred[:, 4, 4]))
            q_pred = q_pred.reshape(self.sl, self.dim_x, self.dim_x)
            r_pred = r_pred.reshape(self.sl, self.dim_z, self.dim_z)
            qx = np.sqrt(np.squeeze(q_pred[:, 0, 0]))
            qy = np.sqrt(np.squeeze(q_pred[:, 1, 1]))
            qt = np.sqrt(np.squeeze(q_pred[:, 2, 2]))
            qv = np.sqrt(np.squeeze(q_pred[:, 3, 3]))
            qdt = np.sqrt(np.squeeze(q_pred[:, 4, 4]))
            rv = np.sqrt(np.squeeze(r_pred[:, 0, 0]))
            rdt = np.sqrt(np.squeeze(r_pred[:, 1, 1]))

        fig, ax = plt.subplots(6, figsize=[12, 25])
        ts = np.arange(pos_pred.shape[0])
        ax[0].plot(ts, pos_pred[:, 0], '-r', label='x predicted')
        ax[0].plot(ts, seq[:, 0], '--g', label='x true')
        ax[0].plot(ts, pos_pred[:, 1], '-m', label='y predicted')
        ax[0].plot(ts, seq[:, 1], '--c', label='y true')
        ax[0].set_title('position')
        ax[0].legend()
        ax[1].plot(ts, or_pred, '-r', label='predicted')
        ax[1].plot(ts, seq[:, 2], '--g', label='true')
        ax[1].set_title('heading')
        ax[1].legend()
        ax[2].plot(ts, v_pred, '-r', label='predicted')
        ax[2].plot(ts, seq[:, 3], '--g', label='true')
        ax[2].plot(ts, v_obs, 'kx', label='observed')
        ax[2].set_title('v')
        ax[2].legend()
        ax[3].plot(ts, td_pred, '-r', label='predicted')
        ax[3].plot(ts, seq[:, 4], '--g', label='true')
        ax[3].plot(ts, td_obs, 'kx', label='observed')
        ax[3].set_title('angular velocity')
        ax[3].legend()
        ax[4].plot(ts, qx, '-b', label='x predicted')
        ax[4].plot(ts, qy, '-c', label='y predicted')
        ax[4].plot(ts, qt, '-y', label='theta predicted')
        ax[4].plot(ts, qv, '-r', label='v predicted')
        ax[4].plot(ts, qdt, '-m', label='dt predicted')
        ax[4].set_title('q')
        ax[4].legend()
        ax[5].plot(ts, rv, '-r', label='v predicted')
        ax[5].plot(ts, rdt, '-m', label='dt predicted')
        ax[5].set_title('r')
        ax[5].legend()

        if cov_pred is not None:
            ax[0].fill_between(ts, pos_pred[:, 0] - 2 * pos_cov_x,
                               pos_pred[:, 0] + 2 * pos_cov_x,
                               color="lightblue")
            ax[0].fill_between(ts, pos_pred[:, 1] - 2 * pos_cov_y,
                               pos_pred[:, 1] + 2 * pos_cov_y,
                               color="lightblue")
            ax[1].fill_between(ts, (or_pred - or_cov), (or_pred + or_cov),
                               color="lightblue")
            ax[2].fill_between(ts, v_pred - v_cov,
                               v_pred + v_cov, color="lightblue")
            ax[3].fill_between(ts, td_pred - td_cov,
                               td_pred + td_cov, color="lightblue")

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85,
                            wspace=0.1, hspace=0.3)

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking"),
                    bbox_inches="tight")

        log_file = open(os.path.join(out_dir, str(num) + '_seq.csv'), 'w')

        keys = ['t', 'rmse', 'like', 'x', 'y', 'om', 'v', 'rot', 'x_p', 'y_p',
                'om_p', 'v_p', 'rot_p']
        if cov_pred is not None and z is not None:
            keys += ['x_c', 'y_c', 'om_c', 'v_c', 'rot_c', 'v_ob', 'rot_ob',
                     'q_x', 'q_y', 'q_t', 'q_v', 'q_dt', 'r_v', 'r_td']
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                row = {'t': t, 'rmse': diffs[t], 'like': likes[t],
                       'x': seq[t, 0], 'y': seq[t, 1], 'om': seq[t, 2],
                       'v': seq[t, 3], 'rot': seq[t, 4],
                       'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                       'om_p': seq_pred[t, 2], 'v_p': seq_pred[t, 3],
                       'rot_p': seq_pred[t, 4],
                       'x_c': pos_cov_x[t], 'y_c': pos_cov_x[t],
                       'om_c': or_cov[t], 'v_c': v_cov[t], 'rot_c': td_cov[t],
                       'v_ob': v_obs[t], 'rot_ob': td_obs[t],
                       'q_x': qx[t], 'q_y': qy[t], 'q_t': qt[t], 'q_v': qv[t],
                       'q_dt': qdt[t], 'r_v': rv[t], 'r_td': rdt[t]}
                log.writerow(row)
        else:
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                row = {'t': t, 'rmse': diffs[t], 'like': likes[t],
                       'x': seq[t, 0], 'y': seq[t, 1], 'om': seq[t, 2],
                       'v': seq[t, 3], 'rot': seq[t, 4],
                       'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                       'om_p': seq_pred[t, 2], 'v_p': seq_pred[t, 3],
                       'rot_p': seq_pred[t, 4]}
                log.writerow(row)
        log_file.close()

    def plot_trajectory(self, particles, weights, seq, cov_pred, seq_pred,
                        out_dir, num):
        print('2d plot')
        if particles is not None:
            particles = particles.reshape(self.sl, -1, self.dim_x)
            weights = weights.reshape(self.sl, -1)
        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)

        pos_pred = np.squeeze(seq_pred[:, :2])
        minx = min(np.min(seq[:, 0]), np.min(pos_pred[:, 0]))
        miny = min(np.min(seq[:, 1]), np.min(pos_pred[:, 1]))
        maxx = max(np.max(seq[:, 0]), np.max(pos_pred[:, 0]))
        maxy = max(np.max(seq[:, 1]), np.max(pos_pred[:, 1]))

        x_len = maxx - minx + 20
        y_len = maxy - miny + 20

        if not np.isfinite(x_len) or not np.isfinite(y_len) or \
                x_len > 200 or y_len > 200:
            return

        fig, ax = plt.subplots(figsize=[x_len/5., y_len/5.])
        ax.set_xlim(minx-10, maxx+10)
        ax.set_ylim(miny-10, maxy+10)
        ax.set_aspect('equal')
        for i in range(self.sl - 1):
            if cov_pred is not None:
                # plot the confidence ellipse
                vals, vecs = self._eigsorted(cov_pred[i, :2, :2])
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 4 * np.sqrt(vals)
                ellip = Ellipse(xy=pos_pred[i], width=width, height=height,
                                angle=theta, alpha=0.1)
                ax.add_artist(ellip)

            if particles is not None:
                # sort the particles by weight
                p = weights[i].argsort()
                par = particles[i][p]
                wei = weights[i][p]
                # plot the 20 best weighted particles with colour depending on
                # weight
                if i == 0:
                    ax.scatter(par[:20, 0], par[:20, 1],
                               c=wei[:20], cmap='jet', marker='x',
                               alpha=0.5, label='particles')
                else:
                    ax.scatter(par[:20, 0], par[:20, 1],
                               c=wei[:20], cmap='jet', marker='x',
                               alpha=0.5)
            # plot a marker for the starting point of the sequence
            if i == 0:
                ax.plot(seq[i, 0], seq[i, 1], 'cx', markersize=15.,
                        label='start')
                # plot the mean trajectory
                ax.plot([pos_pred[i, 0], pos_pred[i+1, 0]],
                        [pos_pred[i, 1], pos_pred[i+1, 1]], '-r',
                        label='predicted')

                # plot the real trajectory
                ax.plot([seq[i, 0], seq[i+1, 0]], [seq[i, 1], seq[i+1, 1]],
                        '-g', label='true')
            else:
                # plot the mean trajectory
                ax.plot([pos_pred[i, 0], pos_pred[i+1, 0]],
                        [pos_pred[i, 1], pos_pred[i+1, 1]], '-r')

                # plot the real trajectory
                ax.plot([seq[i, 0], seq[i+1, 0]],
                        [seq[i, 1], seq[i+1, 1]], '-g')
            # plot the mean trajectory
            ax.plot(pos_pred[i, 0], pos_pred[i, 1], 'ro')
            ax.plot(seq[i, 0], seq[i, 1], 'go')

        ax.legend()

        # plot the last step
        if cov_pred is not None:
            vals, vecs = self._eigsorted(cov_pred[-1, :2, :2])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 2 * np.sqrt(vals)
            ellip = Ellipse(xy=pos_pred[-1], width=width, height=height,
                            angle=theta, alpha=0.1)
            ax.add_artist(ellip)

        # plot the mean trajectory
        ax.plot(pos_pred[-1, 0], pos_pred[-1, 1], 'ro')

        # plot the real trajectory
        ax.plot(seq[-1, 0], seq[-1, 1], 'go')

        if particles is not None:
            p = weights[-1].argsort()
            par = particles[-1][p]
            wei = weights[-1][p]
            ps = weights[0].argsort()
            par_s = particles[0][ps]
            wei_s = weights[0][ps]
            # save final particles and weights
            name = os.path.join(out_dir, str(num))
            np.save(name + '_end_particles', par)
            np.save(name + '_end_weights', wei)
            np.save(name + '_start_particles', par_s)
            np.save(name + '_start_weights', wei_s)

            # plot the particles with colour depending on weight
            ax.scatter(par[:20, 0], par[:20, 1], c=wei[:20], cmap='jet',
                       marker='x', alpha=0.5)

            # make an extra figure of the last state + particles
            dx = pos_pred[-1, 0]
            dy = pos_pred[-1, 1]
            minx = min(seq[-1, 0] - dx, np.min(par[:, 0]) - dx)
            miny = min(seq[-1, 1] - dy, np.min(par[:, 1]) - dy)
            maxx = max(seq[-1, 0] - dx, np.max(par[:, 0]) - dx)
            maxy = max(seq[-1, 1] - dy, np.max(par[:, 1]) - dy)
            x_len = maxx - minx + 20
            y_len = maxy - miny + 20
            if not np.isfinite(x_len) or not np.isfinite(y_len) or \
                    x_len > 200 or y_len > 200:
                return
            fig2, ax2 = plt.subplots(figsize=[x_len/5., y_len/5.])
            ax2.set_xlim(minx-10, maxx+10)
            ax2.set_ylim(miny-10, maxy+10)
            ax2.set_aspect('equal')

            ax2.plot(0, 0, 'ro')
            ax2.plot(seq[-1, 0] - dx, seq[-1, 1] - dy, 'go')
            ax2.scatter(par[:, 0] - dx, par[:, 1] - dy, c=wei, cmap='jet',
                        marker='x', alpha=1)
            # plot the support ellipse of each particle
            w = 2 * 2 * self.param['mixture_std']
            for i in range(par.shape[0]):
                ellip = Ellipse(xy=par[i, :2] - pos_pred[-1], width=w,
                                height=w, angle=0, alpha=0.1)
                ax2.add_artist(ellip)
        else:
            # first and save final covarance and state
            name = os.path.join(out_dir, str(num))
            np.save(name + '_end_state', pos_pred[-1])
            np.save(name + '_end_covar', cov_pred[-1])
            np.save(name + '_start_state', pos_pred[-1])
            np.save(name + '_start_covar', cov_pred[-1])

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking_2d"),
                    bbox_inches="tight")
        if particles is not None:
            fig2.savefig(os.path.join(out_dir, str(num) + "_final_2d"),
                         bbox_inches="tight")


class SensorModel(BaseLayer):
    def __init__(self, batch_size, normalize, summary, trainable):
        super(SensorModel, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.normalize = normalize

        self.c1 = self._conv_layer('conv1', 7, 16, trainable=trainable)
        self.c2 = self._conv_layer('conv2', 5, 16, stride=[1, 2],
                                   trainable=trainable)
        self.c3 = self._conv_layer('conv3', 5, 16, stride=[1, 2],
                                   trainable=trainable)
        self.c4 = self._conv_layer('conv4', 5, 16, stride=[2, 2],
                                   trainable=trainable)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc1 = self._fc_layer('fc1', 128, trainable=trainable)
        self.fc2 = self._fc_layer('fc2', 128, trainable=trainable)
        self.v = self._fc_layer('fc_v', 1, trainable=trainable,
                                activation=None)
        self.td = self._fc_layer('fc_td', 1, trainable=trainable,
                                 activation=None)

        if self.normalize == 'layer':
            self.n1 =\
                tf.keras.layers.LayerNormalization(name='norm1',
                                                   trainable=trainable)
            self.n2 =\
                tf.keras.layers.LayerNormalization(name='norm2',
                                                   trainable=trainable)
            self.n3 =\
                tf.keras.layers.LayerNormalization(name='norm3',
                                                   trainable=trainable)
            self.n4 = \
                tf.keras.layers.LayerNormalization(name='norm4',
                                                   trainable=trainable)
        elif self.normalize == 'batch':
            self.n1 =\
                tf.keras.layers.BatchNormalization(name='norm1',
                                                   trainable=trainable)
            self.n2 =\
                tf.keras.layers.BatchNormalization(name='norm2',
                                                   trainable=trainable)
            self.n3 =\
                tf.keras.layers.BatchNormalization(name='norm3',
                                                   trainable=trainable)
            self.n4 = \
                tf.keras.layers.BatchNormalization(name='norm4',
                                                   trainable=trainable)
            self.updateable = [self.n1, self.n2, self.n3, self.n4]

    def call(self, inputs, training):
        # unpack the inputs
        images, images_diff = inputs
        if self.summary:
            tf.summary.image('im', images[0:1]/255.)
            tf.summary.image('im_diff', images_diff[0:1])

        # subtract the mean image
        in_im = tf.concat([images[:, :, :, 0:1] - 89.,
                           images[:, :, :, 1:2] - 94.,
                           images[:, :, :, 2:3] - 93.,
                           images_diff], axis=-1)

        conv1 = self.c1(in_im)
        if self.normalize == 'layer':
            conv1 = self.n1(conv1)
        elif self.normalize == 'batch':
            conv1 = self.n1(conv1, training)
        if self.summary:
            tf.summary.image('conv1_im',
                             tf.slice(conv1, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv1_out', conv1)

        # conv 2
        conv2 = self.c2(conv1)
        if self.normalize == 'layer':
            conv2 = self.n2(conv2)
        elif self.normalize == 'batch':
            conv2 = self.n2(conv2, training)
        if self.summary:
            tf.summary.image('conv2_im',
                             tf.slice(conv2, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv2_out', conv2)

        # conv 3
        conv3 = self.c3(conv2)
        if self.normalize == 'layer':
            conv3 = self.n3(conv3)
        elif self.normalize == 'batch':
            conv3 = self.n3(conv3, training)
        if self.summary:
            tf.summary.image('conv3_im',
                             tf.slice(conv3, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv3_out', conv3)

        # conv 4
        conv4 = self.c4(conv3)
        if self.normalize == 'layer':
            conv4 = self.n4(conv4)
        elif self.normalize == 'batch':
            conv4 = self.n4(conv4, training)
        if self.summary:
            tf.summary.image('conv4_im',
                             tf.slice(conv4, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv4_out', conv4)

        input_data = tf.reshape(conv4, [self.batch_size, -1])
        input_data = self.dropout(input_data, training)
        fc1 = self.fc1(input_data)
        fc2 = self.fc2(fc1)
        v = self.v(fc2)
        td = self.td(fc2)

        if self.summary:
            tf.summary.histogram('fc1_out', fc1)
            tf.summary.histogram('fc2_out', fc2)
            tf.summary.histogram('fc_v_out', v)
            tf.summary.histogram('fc_td_out', td)
        z = tf.concat([v, td], axis=1)

        return z, fc2


class ObservationNoise(BaseLayer):
    def __init__(self, batch_size, dim_z, r_diag, scale, hetero, diag,
                 trainable, summary):
        super(ObservationNoise, self).__init__()

        self.hetero = hetero
        self.diag = diag
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.r_diag = r_diag
        self.scale = scale
        self.summary = summary
        self.trainable = trainable

    def build(self, input_shape):
        init_const = np.ones(self.dim_z) * 1e-3/self.scale**2
        init = np.sqrt(np.square(self.r_diag) - init_const)
        # the constant bias keeps the predicted covariance away from zero
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_z],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))
        num = self.dim_z * (self.dim_z + 1) / 2
        wd = 1e-3*self.scale**2

        if self.hetero and self.diag:
            # for heteroscedastic noise with diagonal covariance matrix
            self.het_diag_fc = self._fc_layer('het_diag', self.dim_z, mean=0,
                                              std=1e-3, activation=None,
                                              trainable=self.trainable)
            self.het_diag_init_bias = \
                self.add_weight(name='het_diag_init_bias',
                                shape=[self.dim_z],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif not self.hetero and self.diag:
            # for constant noise with diagonal covariance matrix
            self.const_diag = \
                self.add_weight(name='const_diag',
                                shape=[self.dim_z],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif self.hetero and not self.diag:
            # for heteroscedastic noise with full covariance matrix
            self.het_full_fc = self._fc_layer('het_tri', num, mean=0, std=1e-3,
                                              activation=None,
                                              trainable=self.trainable)
            self.het_full_init_bias = \
                self.add_weight(name='het_full_init_bias',
                                shape=[self.dim_z],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        else:
            # for constant noise with full covariance matrix
            self.const_full = \
                self.add_weight(name='const_tri', shape=[num],
                                initializer=tf.constant_initializer(0.),
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                trainable=self.trainable)
            self.const_full_init_bias = \
                self.add_weight(name='const_full_init_bias',
                                shape=[self.dim_z],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        if self.hetero and self.diag:
            diag = self.het_diag_fc(inputs)
            if self.summary:
                tf.summary.histogram('het_diag_out', diag)
            diag = tf.square(diag + self.het_diag_init_bias)
            diag += self.bias_fixed
            R = tf.linalg.diag(diag)
        elif not self.hetero and self.diag:
            diag = self.const_diag
            diag = tf.square(diag) + self.bias_fixed
            R = tf.linalg.tensor_diag(diag)
            R = tf.tile(R[None, :, :], [self.batch_size, 1, 1])
        elif self.hetero and not self.diag:
            tri = self.het_full_fc(inputs)
            if self.summary:
                tf.summary.histogram('het_tri_out', tri)

            R = tf.contrib.distributions.fill_triangular(tri)
            R += tf.linalg.diag(self.het_full_init_bias)
            R = tf.matmul(R, tf.linalg.matrix_transpose(R))
            R = R + tf.linalg.diag(self.bias_fixed)
        else:
            tri = self.const_full
            R = tf.contrib.distributions.fill_triangular(tri)
            R += tf.linalg.diag(self.const_full_init_bias)
            R = tf.matmul(R, tf.linalg.matrix_transpose(R))
            R = R + tf.linalg.diag(self.bias_fixed)
            R = tf.tile(R[None, :, :], [self.batch_size, 1, 1])

        return R


class Likelihood(BaseLayer):
    def __init__(self, dim_z, trainable, summary):
        super(Likelihood, self).__init__()
        self.summary = summary
        self.dim_z = dim_z

        self.fc1 = self._fc_layer('fc1', 128, trainable=trainable)
        self.fc2 = self._fc_layer('fc2', 128, trainable=trainable)
        self.fc3 = self._fc_layer('fc4', 1, trainable=trainable,
                                  activation=None)

    def call(self, inputs, training):
        # unpack the inputs
        particles, encoding = inputs
        bs = particles.get_shape()[0].value
        num_pred = particles.get_shape()[1].value

        # tile the encoding
        encoding = tf.tile(encoding[:, None, :], [1, num_pred, 1])
        input_data = tf.concat([encoding, particles], axis=-1)
        input_data = tf.reshape(input_data, [bs * num_pred, -1])

        fc1 = self.fc1(input_data)
        if self.summary:
            tf.summary.histogram('fc1_out', fc1)
        fc2 = self.fc2(fc1)
        if self.summary:
            tf.summary.histogram('fc2_out', fc2)
        like = self.fc3(fc2)

        if self.summary:
            tf.summary.histogram('like', like)

        return like


class ObservationModel(BaseLayer):
    def __init__(self, dim_z, batch_size):
        super(ObservationModel, self).__init__()
        self.dim_z = dim_z
        self.batch_size = batch_size

    def call(self, inputs, training):
        bs = inputs.get_shape()[0].value // self.batch_size
        H = tf.concat(
            [tf.tile(np.array([[[0, 0, 0, 1, 0]]], dtype=np.float32),
                     [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 1]]], dtype=np.float32),
                     [self.batch_size, 1, 1])], axis=1)

        z_pred = tf.matmul(tf.tile(H, [bs, 1, 1]),
                           tf.expand_dims(inputs, -1))
        z_pred = tf.reshape(z_pred, [bs*self.batch_size, self.dim_z])
        return z_pred, H


class ProcessModel(BaseLayer):
    def __init__(self, batch_size, dim_x, dim_u, dt, scale, learned, jacobian,
                 summary, trainable):
        super(ProcessModel, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dt = dt
        self.scale = scale
        self.learned = learned
        self.jacobian = jacobian

        if learned:
            self.fc1 = self._fc_layer('fc1', 32, trainable=trainable)
            self.fc2 = self._fc_layer('fc2', 64, trainable=trainable)
            self.fc3 = self._fc_layer('fc3', 64, trainable=trainable)
            self.update = self._fc_layer('fc4', self.dim_x, activation=None,
                                         trainable=trainable)

    def call(self, inputs, training):
        # unpack the inputs
        last_state, actions = inputs
        if self.learned:
            # give the learned network access to sin and cosine
            theta = tf.slice(last_state, [0, 2], [-1, 1])
            ct = tf.cos(theta*np.pi/180.)
            st = tf.sin(theta*np.pi/180.)
            # remove the position and raw orientation from the input data
            in_data = tf.concat([last_state[:, 3:], ct, st], axis=-1)
            # in_data = last_state

            fc1 = self.fc1(in_data)
            if self.summary:
                tf.summary.histogram('fc1_out', fc1)
            fc2 = self.fc2(fc1)
            if self.summary:
                tf.summary.histogram('fc2_out', fc2)
            fc3 = self.fc3(fc2)
            if self.summary:
                tf.summary.histogram('fc3_out', fc3)
            update = self.update(fc3)
            if self.summary:
                tf.summary.histogram('update_out', update)
            new_state = last_state + update
            if self.jacobian:
                F = self._compute_jacobian(new_state, last_state)
            else:
                F = None
        else:
            # split the state into parts and undo scaling
            x = tf.slice(last_state, [0, 0], [-1, 1]) * self.scale
            y = tf.slice(last_state, [0, 1], [-1, 1]) * self.scale
            theta = tf.slice(last_state, [0, 2], [-1, 1]) * self.scale
            v = tf.slice(last_state, [0, 3], [-1, 1]) * self.scale
            theta_dot = tf.slice(last_state, [0, 4], [-1, 1]) * self.scale

            # construct new state and rescale
            x_pred = (x + tf.cos(theta*np.pi/180.) * v * self.dt) / self.scale
            y_pred = (y + tf.sin(theta*np.pi/180.) * v * self.dt) / self.scale
            theta_pred = (theta + theta_dot * self.dt) / self.scale
            v_pred = v / self.scale
            theta_dot_pred = theta_dot / self.scale

            new_state = tf.concat([x_pred, y_pred, theta_pred, v_pred,
                                   theta_dot_pred], axis=1)
            if self.jacobian:
                one = tf.ones_like(v)
                zero = tf.zeros_like(v)
                f = np.pi/180.
                F = tf.concat(
                    [tf.concat([one, zero, -tf.sin(theta*f)*v*self.dt*f,
                                tf.cos(theta*f)*self.dt, zero],
                               axis=1)[:, None, :],
                     tf.concat([zero, one, tf.cos(theta*f)*v*self.dt*f,
                                tf.sin(theta*f)*self.dt, zero],
                               axis=1)[:, None, :],
                     tf.concat([zero, zero, one, zero, self.dt*one],
                               axis=1)[:, None, :],
                     tf.concat([zero, zero, zero, one, zero],
                               axis=1)[:, None, :],
                     tf.concat([zero, zero, zero, zero, one],
                               axis=1)[:, None, :]],
                    axis=1)
            else:
                F = None
        if self.jacobian:
            F = tf.stop_gradient(F)

        return new_state, F


class ProcessNoise(BaseLayer):
    def __init__(self, batch_size, dim_x, q_diag, scale, hetero, diag, learned,
                 trainable, summary):
        super(ProcessNoise, self).__init__()
        self.hetero = hetero
        self.diag = diag
        self.learned = learned
        self.trainable = trainable

        self.dim_x = dim_x
        self.q_diag = q_diag
        self.scale = scale
        self.batch_size = batch_size
        self.summary = summary

    def build(self, input_shape):
        init_const = np.ones(self.dim_x) * 1e-3/(self.scale**2)
        init = np.sqrt(np.square(self.q_diag) - init_const)
        # the constant bias keeps the predicted covariance away from zero
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_x],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))
        num = self.dim_x * (self.dim_x + 1) / 2
        wd = 1e-3*self.scale**2

        if self.hetero and self.diag and self.learned:
            # for heteroscedastic noise with diagonal covariance matrix
            self.het_diag_lrn_fc1 = self._fc_layer('het_diag_lrn_fc1', 32,
                                                   trainable=self.trainable)
            self.het_diag_lrn_fc2 = self._fc_layer('het_diag_lrn_fc2', 32,
                                                   trainable=self.trainable)
            self.het_diag_lrn_fc3 = \
                self._fc_layer('het_diag_lrn_fc3', self.dim_x, mean=0,
                               std=1e-3, activation=None,
                               trainable=self.trainable)
            self.het_diag_lrn_init_bias = \
                self.add_weight(name='het_diag_lrn_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif not self.hetero and self.diag and self.learned:
            # for constant noise with diagonal covariance matrix
            self.const_diag_lrn = \
                self.add_weight(name='const_diag_lrn', shape=[self.dim_x],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif self.hetero and not self.diag and self.learned:
            # for heteroscedastic noise with full covariance matrix
            self.het_full_lrn_fc1 = self._fc_layer('het_full_lrn_fc1', 32,
                                                   trainable=self.trainable)
            self.het_full_lrn_fc2 = self._fc_layer('het_full_lrn_fc2', 32,
                                                   trainable=self.trainable)
            self.het_full_lrn_fc3 = \
                self._fc_layer('het_full_lrn_fc3', num, mean=0, std=1e-3,
                               activation=None, trainable=self.trainable)
            self.het_full_lrn_init_bias = \
                self.add_weight(name='het_full_lrn_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif not self.hetero and not self.diag and self.learned:
            # for constant noise with full covariance matrix
            self.const_full_lrn = \
                self.add_weight(name='const_tri_lrn', shape=[num],
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(0.),
                                trainable=self.trainable)
            self.const_full_lrn_init_bias = \
                self.add_weight(name='const_full_lrn_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif self.hetero and self.diag and not self.learned:
            # for heteroscedastic noise with diagonal covariance matrix
            self.het_diag_ana_fc1 = self._fc_layer('het_diag_ana_fc1', 32,
                                                   trainable=self.trainable)
            self.het_diag_ana_fc2 = self._fc_layer('het_diag_ana_fc2', 32,
                                                   trainable=self.trainable)
            self.het_diag_ana_fc3 = \
                self._fc_layer('het_diag_ana_fc3', self.dim_x, mean=0,
                               std=1e-3, activation=None,
                               trainable=self.trainable)
            self.het_diag_ana_init_bias = \
                self.add_weight(name='het_diag_ana_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif not self.hetero and self.diag and not self.learned:
            # for constant noise with diagonal covariance matrix
            self.const_diag_ana = \
                self.add_weight(name='const_diag_ana', shape=[self.dim_x],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif self.hetero and not self.diag and not self.learned:
            # for heteroscedastic noise with full covariance matrix
            self.het_full_ana_fc1 = self._fc_layer('het_full_ana_fc1', 32,
                                                   trainable=self.trainable)
            self.het_full_ana_fc2 = self._fc_layer('het_full_ana_fc2', 32,
                                                   trainable=self.trainable)
            self.het_full_ana_fc3 = \
                self._fc_layer('het_full_ana_fc3', num, mean=0, std=1e-3,
                               activation=None, trainable=self.trainable)
            self.het_full_ana_init_bias = \
                self.add_weight(name='het_full_ana_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))
        elif not self.hetero and not self.diag and not self.learned:
            # for constant noise with full covariance matrix
            self.const_full_ana = \
                self.add_weight(name='const_tri_ana', shape=[num],
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(0.),
                                trainable=self.trainable)
            self.const_full_ana_init_bias = \
                self.add_weight(name='const_full_ana_init_bias',
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                shape=[self.dim_x], trainable=self.trainable,
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        old_state, actions = inputs

        # remove the absolute position from the old state
        old_state = old_state[:, 3:]
        if self.learned:
            if self.hetero and self.diag:
                fc1 = self.het_diag_lrn_fc1(old_state)
                fc2 = self.het_diag_lrn_fc2(fc1)
                diag = self.het_diag_lrn_fc3(fc2)
                if self.summary:
                    tf.summary.histogram('het_diag_lrn_fc1_out', fc1)
                    tf.summary.histogram('het_diag_lrn_fc2_out', fc2)
                    tf.summary.histogram('het_diag_lrn_fc3_out', diag)
                diag = tf.square(diag + self.het_diag_lrn_init_bias)
                diag += self.bias_fixed
                Q = tf.linalg.diag(diag)
            elif not self.hetero and self.diag:
                diag = self.const_diag_lrn
                diag = tf.square(diag) + self.bias_fixed
                Q = tf.linalg.tensor_diag(diag)
                Q = tf.tile(Q[None, :, :], [self.batch_size, 1, 1])
            elif self.hetero and not self.diag:
                fc1 = self.het_full_lrn_fc1(old_state)
                fc2 = self.het_full_lrn_fc2(fc1)
                tri = self.het_full_lrn_fc3(fc2)
                if self.summary:
                    tf.summary.histogram('het_full_lrn_fc1_out', fc1)
                    tf.summary.histogram('het_full_lrn_fc2_out', fc2)
                    tf.summary.histogram('het_full_lrn_out', tri)

                Q = tf.contrib.distributions.fill_triangular(tri)
                Q += tf.linalg.diag(self.het_full_lrn_init_bias)
                Q = tf.matmul(Q, tf.linalg.matrix_transpose(Q))
                Q = Q + tf.linalg.diag(self.bias_fixed)
            else:
                tri = self.const_full_lrn
                Q = tf.contrib.distributions.fill_triangular(tri)
                Q += tf.linalg.diag(self.const_full_lrn_init_bias)
                Q = tf.matmul(Q, tf.linalg.matrix_transpose(Q))
                Q = Q + tf.linalg.diag(self.bias_fixed)
                Q = tf.tile(Q[None, :, :], [self.batch_size, 1, 1])
        else:
            if self.hetero and self.diag:
                fc1 = self.het_diag_ana_fc1(old_state)
                fc2 = self.het_diag_ana_fc2(fc1)
                diag = self.het_diag_ana_fc3(fc2)
                if self.summary:
                    tf.summary.histogram('het_diag_ana_fc1_out', fc1)
                    tf.summary.histogram('het_diag_ana_fc2_out', fc2)
                    tf.summary.histogram('het_diag_ana_fc3_out', diag)
                diag = tf.square(diag + self.het_diag_ana_init_bias)
                diag += self.bias_fixed
                Q = tf.linalg.diag(diag)
            elif not self.hetero and self.diag:
                diag = self.const_diag_ana
                diag = tf.square(diag) + self.bias_fixed
                Q = tf.linalg.tensor_diag(diag)
                Q = tf.tile(Q[None, :, :], [self.batch_size, 1, 1])
            elif self.hetero and not self.diag:
                fc1 = self.het_full_ana_fc1(old_state)
                fc2 = self.het_full_ana_fc2(fc1)
                tri = self.het_full_ana_fc3(fc2)
                if self.summary:
                    tf.summary.histogram('het_full_ana_fc1_out', fc1)
                    tf.summary.histogram('het_full_ana_fc2_out', fc2)
                    tf.summary.histogram('het_full_ana_out', tri)

                Q = tf.contrib.distributions.fill_triangular(tri)
                Q += tf.linalg.diag(self.het_full_ana_init_bias)
                Q = tf.matmul(Q, tf.linalg.matrix_transpose(Q))
                Q = Q + tf.linalg.diag(self.bias_fixed)
            else:
                tri = self.const_full_ana
                Q = tf.contrib.distributions.fill_triangular(tri)
                Q += tf.linalg.diag(self.const_full_ana_init_bias)
                Q = tf.matmul(Q, tf.linalg.matrix_transpose(Q))
                Q = Q + tf.linalg.diag(self.bias_fixed)
                Q = tf.tile(Q[None, :, :], [self.batch_size, 1, 1])

        return Q
