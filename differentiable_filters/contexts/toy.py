# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:04:00 2020

@author: akloss
"""

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf
import tensorflow_probability as tfp
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

        # the state size
        self.dim_x = 4
        self.dim_u = 0
        self.dim_z = 2

        self.x_names = ['x', 'y', 'vx', 'vy']
        self.z_names = ['x', 'y']

        self.spring_force = 0.05
        self.drag_force = 0.0075

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

        # fixed initial noise values for testing
        if param['initial_covar'] is not None:
            cov_string = param['initial_covar']
            self.initial_covariance = list(map(lambda x: float(x),
                                               cov_string.split(' ')))
        else:
            self.initial_covariance = [1., 1., 1., 1.]

        if self.initial_covariance == [10.0, 10.0, 10.0, 10.0]:
            self.noise_list = \
                [np.array([0., 0., 0., 0.]).astype(np.float32)*2,
                 np.array([3.0377, 18.574, 7.65, -26.14]).astype(np.float32),
                 np.array([-17.30, 9.05, 3.23, 9.94]).astype(np.float32),
                 np.array([3.1, 1.197, -17.166, -1.94]).astype(np.float32),
                 np.array([-12.76, 26.25, -1.98, -1.87]).astype(np.float32)]
        elif self.initial_covariance == [5.0, 5.0, 5.0, 5.0]:
            self.noise_list = \
                [np.array([0., 0., 0., 0.]).astype(np.float32),
                 np.array([-5.587, 3.782, -5.895, -3.441]).astype(np.float32),
                 np.array([-10.428, 1.034, 9.516, -1.171]).astype(np.float32),
                 np.array([3.053, 4.262, -6.058, 4.927]).astype(np.float32),
                 np.array([2.631, 12.652, 7.648, 5.688]).astype(np.float32)]
        else:
            # initial covariance [1. 1. 1. 1.]
            self.noise_list = \
                [np.array([0., 0., 0., 0.]).astype(np.float32),
                 np.array([-5.587, 3.782, -5.895, -3.441]).astype(np.float32),
                 np.array([-10.428, 1.034, 9.516, -1.171]).astype(np.float32),
                 np.array([3.053, 4.262, -6.058, 4.927]).astype(np.float32),
                 np.array([2.631, 12.652, 7.648, 5.688]).astype(np.float32)]
            self.noise_list = list(map(lambda x: x/5., self.noise_list))

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
            self.sensor_model_layer = SensorModel(self.batch_size,
                                                  summary, train_sensor_model)
            self.observation_model_layer = ObservationModel(self.dim_z,
                                                            self.batch_size)
            # group the layers for easier checkpoint restoring
            self.observation_models = {'sensor': self.sensor_model_layer,
                                       'obs': self.observation_model_layer}
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
        if param['learned_likelihood'] and mode == 'filter' or \
                mode == 'pretrain_obs':
            self.likelihood_layer = Likelihood(self.dim_z, trainable=train_r,
                                               summary=summary)
            self.observation_noise_models['like'] = self.likelihood_layer

        self.process_models = {}
        lstm_unstructured = param['filter'] == 'lstm' and \
            (param['lstm_structure'] == 'none' or
             param['lstm_structure'] == 'lstm' or
             param['lstm_structure'] == 'lstm1')
        if mode == 'filter' and not lstm_unstructured and \
                param['learn_process'] or mode == 'pretrain_process':
            self.process_model_learned_layer = \
                ProcessModel(self.batch_size, self.dim_x, self.dim_u,
                             self.spring_force, self.drag_force,
                             scale=self.scale,
                             learned=True, jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['learned'] = self.process_model_learned_layer
        if mode == 'filter' and not lstm_unstructured and \
                not param['learn_process'] or mode == 'pretrain_process':
            self.process_model_analytical_layer = \
                ProcessModel(self.batch_size, self.dim_x, self.dim_u,
                             self.spring_force, self.drag_force,
                             scale=self.scale,
                             learned=False, jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['ana'] = self.process_model_analytical_layer

        self.process_noise_models = {}
        if param['learn_q'] and param['hetero_q'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_hetero_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_lrn'] = \
                self.process_noise_hetero_diag_lrn
        if param['learn_q'] and param['hetero_q'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_hetero_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_full_lrn'] = \
                self.process_noise_hetero_full_lrn
        if param['learn_q'] and not param['hetero_q'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_lrn'] = \
                self.process_noise_const_diag_lrn
        if param['learn_q'] and not param['hetero_q'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_const_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=True, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_lrn'] = \
                self.process_noise_const_full_lrn
        if param['learn_q'] and param['hetero_q'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_hetero_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=False,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_ana'] = \
                self.process_noise_hetero_diag_ana
        if param['learn_q'] and param['hetero_q'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_hetero_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['het_full_ana'] = \
                self.process_noise_hetero_full_ana
        if param['learn_q'] and not param['hetero_q'] and \
                param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_ana'] = \
                self.process_noise_const_diag_ana
        if param['learn_q'] and not param['hetero_q'] and \
                not param['diagonal_covar'] and mode == 'filter' and \
                not param['learn_process'] and not lstm_no_noise or \
                mode == 'pretrain_process':
            self.process_noise_const_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_ana'] = \
                self.process_noise_const_full_ana

    def initial_from_observed(self, base_state, init_z, base_covar, init_R):
        state = tf.concat([init_z, base_state[:, 2:]], axis=-1)
        covar = \
            tf.concat([tf.concat([base_covar[:, :2, :2], init_R], axis=-1)],
                      base_covar[:, 2:, :], axis=1)
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

        Returns:
            loss: the total loss for training the filtering application
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        particles, weights, states, covars, _, _, z, r, q = prediction
        states = tf.reshape(states, [self.batch_size, -1, self.dim_x])
        covars = tf.reshape(covars, [self.batch_size, -1, self.dim_x,
                                     self.dim_x])
        seq_label, q_label, vis_label = label
        vis_label = tf.reshape(tf.cast(vis_label, tf.float32), [-1, 1])

        diff = seq_label - states

        # get the likelihood
        if self.param['filter'] == 'pf' and self.param['mixture_likelihood']:
            num = particles.get_shape()[2].value
            seq_label_tiled = tf.tile(seq_label[:, :, None, :], [1, 1, num, 1])
            particle_diff = seq_label_tiled - particles
            likelihood = self._mixture_likelihood(particle_diff, weights)
        else:
            likelihood = self._likelihood(diff, covars, reduce_mean=False)
        # compensate for scaling
        offset = tf.ones_like(likelihood)*tf.math.log(self.scale)*2*self.dim_x
        likelihood += offset

        # compute the errors of the predicted state
        mses = []
        dists = []
        for k in range(self.dim_x):
            mse, dist = self._mse(diff[:, :, k:k+1], reduce_mean=False)
            mses += [mse*self.scale**2]
            dists += [dist*self.scale]
        mse = tf.add_n(mses)
        dist = tf.add_n(dists)

        # compute the error in the predicted observations (only for monitoring)
        diff_obs = seq_label[:, :, :2] - z
        mse_x_obs, dist_x_obs = \
            self._mse(diff_obs[:, :, :1], reduce_mean=False)
        mse_y_obs, dist_y_obs = \
            self._mse(diff_obs[:, :, 1:], reduce_mean=False)
        dist_obs = (dist_x_obs + dist_y_obs) * self.scale

        # compute the error of the predicted process noise
        if len(q_label.get_shape()) == 3:
            q_label_m = tf.linalg.diag(tf.square(q_label))
            dist_q = self._bhattacharyya(q*self.scale**2, q_label_m)
        else:
            dist_q = self._bhattacharyya(q*self.scale**2, q_label)

        # compute the correlation between predicted observation noise and
        # the number of visible pixels of the red disc
        diag_r = tf.linalg.diag_part(r)
        diag_r = tf.sqrt(tf.abs(diag_r + 1e-5))
        diag_r = tf.reshape(diag_r, [-1, self.dim_z])
        corr_rx = tfp.stats.correlation(diag_r[:, 0:1], vis_label,
                                        sample_axis=0, event_axis=-1)
        corr_ry = tfp.stats.correlation(diag_r[:, 1:2], vis_label,
                                        sample_axis=0, event_axis=-1)
        corr_r = (corr_rx + corr_ry)/2.

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

        total_tracking = tf.reduce_mean(mse)
        if self.loss == 'like':
            total_loss = tf.reduce_mean(likelihood)
        elif self.loss == 'error':
            total_loss = total_tracking
        elif self.loss == 'mixed':
            total_loss = (total_tracking + tf.reduce_mean(likelihood)) / 2.
        elif self.loss == 'mixed_error':
            total_loss = total_tracking * 0.75 + \
                tf.reduce_mean(likelihood) * 0.25
        elif self.loss == 'mixed_like':
            total_loss = total_tracking * 0.25 + \
                tf.reduce_mean(likelihood) * 0.75
        elif self.loss == 'mixed_curr':
            total_loss = tf.cond(tf.less(step, self.epoch_size * 3),
                                 lambda: total_tracking,
                                 lambda: tf.reduce_mean(likelihood))

        if self.loss == 'mixed_curr':
            total_loss_val = tf.reduce_mean(likelihood)
        else:
            total_loss_val = total_loss

        total = tf.cond(training,
                        lambda: 100 * total_loss + wd, lambda: total_loss_val)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood', tf.reduce_mean(likelihood))
        tf.summary.scalar('loss/tracking', total_tracking)
        for k in range(self.dim_x):
            tf.summary.scalar('tracking_loss/' + self.x_names[k],
                              tf.reduce_mean(dists[k]))
        tf.summary.scalar('observation_loss/dist_x',
                          tf.reduce_mean(dist_x_obs)*self.scale)
        tf.summary.scalar('observation_loss/dist_y',
                          tf.reduce_mean(dist_y_obs)*self.scale)
        tf.summary.scalar('noise_loss/vis', tf.reduce_mean(vis_label))
        tf.summary.scalar('noise_loss/dist_q', tf.reduce_mean(dist_q))
        tf.summary.scalar('noise_loss/corr_r', tf.reduce_mean(corr_r))
        return total, [likelihood, dist, dist_obs, mse] + dists + \
            [dist_x_obs, dist_y_obs, dist_q, vis_label, diag_r, wd], \
            ['likelihood', 'dist', 'dist_obs', 'mse', 'x', 'y', 'vx', 'vy',
             'x_obs', 'y_obs', 'q', 'vis', 'r_pred', 'wd']

    def _bhattacharyya(self, pred, label):
        mean = (pred + label) / 2.
        det_mean = tf.linalg.det(mean)
        det_pred = tf.linalg.det(pred)
        det_label = tf.linalg.det(label)
        dist = det_mean/(tf.sqrt(det_pred*det_label))
        dist = 0.5 * tf.math.log(dist)
        return dist

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

        label, vis_label = label
        vis_label = tf.cast(vis_label, tf.float32)[:, None]
        diff = label - z

        likelihood_const_diag = \
            self._likelihood(diff, R_const_diag, reduce_mean=False)
        likelihood_const_tri = \
            self._likelihood(diff, R_const_tri, reduce_mean=False)
        likelihood_het_diag = \
            self._likelihood(diff, R_het_diag, reduce_mean=False)
        likelihood_het_tri = \
            self._likelihood(diff, R_het_tri, reduce_mean=False)

        _, dist = self._mse(diff, reduce_mean=False)

        likelihood = (likelihood_const_diag + likelihood_const_tri +
                      likelihood_het_diag + likelihood_het_tri) / 4.
        # compensate for scaling
        offset = tf.ones_like(likelihood)*tf.math.log(self.scale)*2*self.dim_z
        likelihood += offset

        # compute the errors of the predicted observations
        mse_x, dist_x = self._mse(diff[:, 0:1], reduce_mean=False)
        mse_y, dist_y = self._mse(diff[:, 1:], reduce_mean=False)
        # compensate for scaling
        mse_x *= self.scale**2
        mse_y *= self.scale**2
        dist_x *= self.scale
        dist_y *= self.scale

        # compute the loss for the learned likelihood model of the pf
        good_loss = tf.reduce_mean(-tf.math.log(tf.maximum(like_good, 1e-6)))
        bad_loss = tf.reduce_mean(-tf.math.log(tf.maximum(1.0 - like_bad,
                                                          1e-6)))
        like_loss = good_loss + bad_loss

        # compute the correlation between predicted observation noise and
        # the number of visible pixels of the red disc
        # this only makes sense for the heteroscedastic noise
        diag_r_het_diag = tf.linalg.diag_part(R_het_diag)
        diag_r_het_diag = tf.sqrt(tf.abs(diag_r_het_diag + 1e-5))
        diag_r_het_diag = tf.reshape(diag_r_het_diag, [-1, self.dim_z])
        corr_rx_het_diag = \
            tfp.stats.correlation(diag_r_het_diag[:, 0:1], vis_label,
                                  sample_axis=0, event_axis=-1)
        corr_ry_het_diag = \
            tfp.stats.correlation(diag_r_het_diag[:, 1:2], vis_label,
                                  sample_axis=0, event_axis=-1)
        corr_r_het_diag = (corr_rx_het_diag + corr_ry_het_diag)/2.

        diag_r_het_tri = tf.linalg.diag_part(R_het_tri)
        diag_r_het_tri = tf.sqrt(tf.abs(diag_r_het_tri + 1e-5))
        diag_r_het_tri = tf.reshape(diag_r_het_tri, [-1, self.dim_z])
        corr_rx_het_tri = \
            tfp.stats.correlation(diag_r_het_tri[:, 0:1], vis_label,
                                  sample_axis=0, event_axis=-1)
        corr_ry_het_tri = \
            tfp.stats.correlation(diag_r_het_tri[:, 1:2], vis_label,
                                  sample_axis=0, event_axis=-1)
        corr_r_het_tri = (corr_rx_het_tri + corr_ry_het_tri)/2.

        wd = []
        for la in self.observation_models.values():
            wd += la.losses
        for la in self.observation_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        vis_mask = tf.cast(tf.greater(vis_label, 0), tf.float32)
        vis_sum = tf.maximum(tf.reduce_sum(vis_mask), 1)
        obs = (mse_x + mse_y) * vis_mask
        obs = tf.reduce_sum(obs) / vis_sum
        # obs = tf.reduce_mean(mse_x) + tf.reduce_mean(mse_y)

        total_loss = \
            tf.cond(tf.less(step, self.epoch_size*3),
                    lambda: 100 * tf.reduce_mean(likelihood) + 10 * obs +
                    like_loss + wd,
                    lambda: 100 * tf.reduce_mean(likelihood) + obs +
                    10 * like_loss + wd)

        total_train = total_loss
        total_val = 1000 + tf.reduce_mean(likelihood) + 10 * like_loss + obs

        total = tf.cond(training,
                        lambda: total_train, lambda: total_val)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/observations',
                          tf.reduce_mean(mse_x) + tf.reduce_mean(mse_y))
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood_const_diag',
                          tf.reduce_mean(likelihood_const_diag + offset))
        tf.summary.scalar('loss/likelihood_const_tri',
                          tf.reduce_mean(likelihood_const_tri + offset))
        tf.summary.scalar('loss/likelihood_het_diag',
                          tf.reduce_mean(likelihood_het_diag + offset))
        tf.summary.scalar('loss/likelihood_het_tri',
                          tf.reduce_mean(likelihood_het_tri + offset))
        tf.summary.scalar('observation_loss/dist_x', tf.reduce_mean(dist_x))
        tf.summary.scalar('observation_loss/dist_y', tf.reduce_mean(dist_y))
        tf.summary.scalar('loss/like_good', good_loss)
        tf.summary.scalar('loss/like_bad', bad_loss)
        tf.summary.scalar('loss/like_loss', like_loss)

        tf.summary.scalar('noise_loss/corr_r_het_diag',
                          tf.reduce_mean(corr_r_het_diag))
        tf.summary.scalar('noise_loss/corr_r_het_tri',
                          tf.reduce_mean(corr_r_het_tri))
        return total, [likelihood_const_diag, likelihood_const_tri,
                       likelihood_het_diag, likelihood_het_tri,
                       dist_x, dist_y, like_loss,
                       vis_label, diag_r_het_diag, diag_r_het_tri, wd], \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri', 'x', 'y', 'like',
             'vis', 'r_het_diag', 'r_het_tri', 'wd']

    def get_process_loss(self, prediction, labels, step, training):
        """
        Compute the loss for the process functions - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step

        Returns:
            loss: the total loss for training the process model
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        state, Q_const_diag, Q_const_tri, Q_het_diag, Q_het_tri, \
            state_ana, Q_const_diag_ana, Q_const_tri_ana, Q_het_diag_ana, \
            Q_het_tri_ana = prediction

        label, q_label = labels

        diff = label - state
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
        likelihood_const_diag_ana = \
            self._likelihood(diff_ana, Q_const_diag_ana, reduce_mean=False)
        likelihood_const_tri_ana = self._likelihood(diff_ana, Q_const_tri_ana,
                                                    reduce_mean=False)
        likelihood_het_diag_ana = self._likelihood(diff_ana, Q_het_diag_ana,
                                                   reduce_mean=False)
        likelihood_het_tri_ana = self._likelihood(diff_ana, Q_het_tri_ana,
                                                  reduce_mean=False)

        likelihood_ana = \
            (likelihood_const_diag_ana + likelihood_const_tri_ana +
             likelihood_het_diag_ana + likelihood_het_tri_ana) / 4.

        # compensate for scaling
        offset = tf.ones_like(likelihood)*tf.math.log(self.scale)*2*self.dim_z
        likelihood += offset
        likelihood_ana += offset

        # compute the errors of the predicted observations
        mse_x, dist_x = self._mse(diff[:, 0:1], reduce_mean=False)
        mse_y, dist_y = self._mse(diff[:, 1:2], reduce_mean=False)
        mse_vx, dist_vx = self._mse(diff[:, 2:3], reduce_mean=False)
        mse_vy, dist_vy = self._mse(diff[:, 3:4], reduce_mean=False)
        mse = mse_x + mse_y + mse_vx + mse_vy

        _, dist_x_ana = self._mse(diff_ana[:, 0:1], reduce_mean=False)
        _, dist_y_ana = self._mse(diff_ana[:, 1:2], reduce_mean=False)
        _, dist_vx_ana = self._mse(diff_ana[:, 2:3], reduce_mean=False)
        _, dist_vy_ana = self._mse(diff_ana[:, 3:4], reduce_mean=False)

        # compensate for scaling
        mse *= self.scale**2
        dist_x *= self.scale
        dist_y *= self.scale
        dist_vx *= self.scale
        dist_vy *= self.scale
        dist_x_ana *= self.scale
        dist_y_ana *= self.scale
        dist_vx_ana *= self.scale
        dist_vy_ana *= self.scale

        # compute the error of the predicted process noise
        if len(q_label.get_shape()) == 2:
            q_label_m = tf.linalg.diag(tf.square(q_label))
            dist_q_const_diag = \
                self._bhattacharyya(Q_const_diag*self.scale**2, q_label_m)
            dist_q_const_tri = \
                self._bhattacharyya(Q_const_tri*self.scale**2, q_label_m)
            dist_q_het_diag = \
                self._bhattacharyya(Q_het_diag*self.scale**2, q_label_m)
            dist_q_het_tri = \
                self._bhattacharyya(Q_het_tri*self.scale**2, q_label_m)

            dist_q_const_diag_ana = \
                self._bhattacharyya(Q_const_diag_ana*self.scale**2, q_label_m)
            dist_q_const_tri_ana = \
                self._bhattacharyya(Q_const_tri_ana*self.scale**2, q_label_m)
            dist_q_het_diag_ana = \
                self._bhattacharyya(Q_het_diag_ana*self.scale**2, q_label_m)
            dist_q_het_tri_ana = \
                self._bhattacharyya(Q_het_tri_ana*self.scale**2, q_label_m)
        else:
            dist_q_const_diag = self._bhattacharyya(Q_const_diag*self.scale**2,
                                                    q_label)
            dist_q_const_tri = self._bhattacharyya(Q_const_tri*self.scale**2,
                                                   q_label)
            dist_q_het_diag = self._bhattacharyya(Q_het_diag*self.scale**2,
                                                  q_label)
            dist_q_het_tri = self._bhattacharyya(Q_het_tri*self.scale**2,
                                                 q_label)
            dist_q_const_diag_ana = \
                self._bhattacharyya(Q_const_diag_ana*self.scale**2, q_label)
            dist_q_const_tri_ana = \
                self._bhattacharyya(Q_const_tri_ana*self.scale**2, q_label)
            dist_q_het_diag_ana = \
                self._bhattacharyya(Q_het_diag_ana*self.scale**2, q_label)
            dist_q_het_tri_ana = \
                self._bhattacharyya(Q_het_tri_ana*self.scale**2, q_label)

        wd = []
        for la in self.process_models.values():
            wd += la.losses
        for la in self.process_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        total_loss = \
            tf.cond(tf.less(step, self.epoch_size*10),
                    lambda: tf.reduce_mean(mse) * 100,
                    lambda: tf.reduce_mean(likelihood) + wd +
                    tf.reduce_mean(likelihood_ana) + 100 * tf.reduce_mean(mse))

        total = \
            tf.cond(training,
                    lambda: total_loss,
                    lambda: tf.reduce_mean(likelihood) + 1000 +
                    tf.reduce_mean(likelihood_ana) + tf.reduce_mean(mse) / 10.)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/tracking', tf.reduce_mean(mse))
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood_const_diag',
                          tf.reduce_mean(likelihood_const_diag + offset))
        tf.summary.scalar('loss/likelihood_const_tri',
                          tf.reduce_mean(likelihood_const_tri + offset))
        tf.summary.scalar('loss/likelihood_het_diag',
                          tf.reduce_mean(likelihood_het_diag + offset))
        tf.summary.scalar('loss/likelihood_het_tri',
                          tf.reduce_mean(likelihood_het_tri + offset))
        tf.summary.scalar('loss/likelihood_const_diag_ana',
                          tf.reduce_mean(likelihood_const_diag_ana + offset))
        tf.summary.scalar('loss/likelihood_const_tri_ana',
                          tf.reduce_mean(likelihood_const_tri_ana + offset))
        tf.summary.scalar('loss/likelihood_het_diag_ana',
                          tf.reduce_mean(likelihood_het_diag_ana + offset))
        tf.summary.scalar('loss/likelihood_het_tri_ana',
                          tf.reduce_mean(likelihood_het_tri_ana + offset))
        tf.summary.scalar('tracking_loss/dist_x', tf.reduce_mean(dist_x))
        tf.summary.scalar('tracking_loss/dist_y', tf.reduce_mean(dist_y))
        tf.summary.scalar('tracking_loss/dist_vx', tf.reduce_mean(dist_vx))
        tf.summary.scalar('tracking_loss/dist_vy', tf.reduce_mean(dist_vy))
        tf.summary.scalar('tracking_loss/dist_x_ana',
                          tf.reduce_mean(dist_x_ana))
        tf.summary.scalar('tracking_loss/dist_y_ana',
                          tf.reduce_mean(dist_y_ana))
        tf.summary.scalar('tracking_loss/dist_vx_ana',
                          tf.reduce_mean(dist_vx_ana))
        tf.summary.scalar('tracking_loss/dist_vy_ana',
                          tf.reduce_mean(dist_vy_ana))
        tf.summary.scalar('noise_loss/dist_q_const_diag',
                          tf.reduce_mean(dist_q_const_diag))
        tf.summary.scalar('noise_loss/dist_q_const_tri',
                          tf.reduce_mean(dist_q_const_tri))
        tf.summary.scalar('noise_loss/dist_q_het_diag',
                          tf.reduce_mean(dist_q_het_diag))
        tf.summary.scalar('noise_loss/dist_q_het_tri',
                          tf.reduce_mean(dist_q_het_tri))
        tf.summary.scalar('noise_loss/dist_q_const_diag_ana',
                          tf.reduce_mean(dist_q_const_diag_ana))
        tf.summary.scalar('noise_loss/dist_q_const_tri_ana',
                          tf.reduce_mean(dist_q_const_tri_ana))
        tf.summary.scalar('noise_loss/dist_q_het_diag_ana',
                          tf.reduce_mean(dist_q_het_diag_ana))
        tf.summary.scalar('noise_loss/dist_q_het_tri',
                          tf.reduce_mean(dist_q_het_tri_ana))

        return total, [likelihood_const_diag, likelihood_const_tri,
                       likelihood_het_diag, likelihood_het_tri,
                       likelihood_const_diag_ana, likelihood_const_tri_ana,
                       likelihood_het_diag_ana, likelihood_het_tri_ana,
                       dist_x, dist_y, dist_vx, dist_vy,
                       dist_x_ana, dist_y_ana, dist_vx_ana, dist_vy_ana,
                       dist_q_const_diag, dist_q_const_tri, dist_q_het_diag,
                       dist_q_het_tri, dist_q_const_diag_ana,
                       dist_q_const_tri_ana, dist_q_het_diag_ana,
                       dist_q_het_tri_ana, wd], \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri',
             'likelihood_const_diag_ana', 'likelihood_const_tri_ana',
             'likelihood_het_diag_ana', 'likelihood_het_tri_ana',
             'x', 'y', 'vx', 'vy', 'x_ana', 'y_ana', 'vx_ana', 'vy_ana',
             'dist_q_const_diag', 'dist_q_const_tri',
             'dist_q_het_diag', 'dist_q_het_tri', 'dist_q_const_diag_ana',
             'dist_q_const_tri_ana', 'dist_q_het_diag_ana',
             'dist_q_het_tri_ana', 'wd']

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        keys = ['start_image', 'start_state', 'image', 'state', 'q',
                'visible']

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
            state = features['state']/self.scale
            label = state[:, :2]
            im = features['image']
            vis = features['visible']

            # we use several steps of the sequence
            start_inds = np.random.randint(0, im.get_shape()[0].value-1, 30)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            ims = []
            viss = []
            labels = []
            good_zs = []
            bad_zs = []
            for si in start_inds:
                ims += [im[si]]
                viss += [vis[si]]
                labels += [label[si]]
                good_noise = np.random.normal(loc=0, scale=2, size=(6, 2))
                good_noise[0, :] = 0
                good_zs += [tf.tile(label[si:si+1], [6, 1]) +
                            good_noise/self.scale]
                bad_noise = np.random.normal(loc=40, scale=20, size=(6, 2))
                bad_noise[3:] = np.random.normal(loc=-40, scale=20,
                                                 size=(3, 2))
                bad_zs += [tf.tile(label[si:si+1], [6, 1]) +
                           bad_noise/self.scale]

            values = [tf.stack(ims), tf.stack(labels), tf.stack(good_zs),
                      tf.stack(bad_zs)]
            labels = [tf.stack(labels), tf.stack(viss)]
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

            state = features['state']/self.scale
            label = state[:, :2]
            im = features['image']
            vis = features['visible']

            ims = []
            viss = []
            labels = []
            good_zs = []
            bad_zs = []
            # use every third data point
            start_inds = np.arange(0, im.get_shape()[0].value-1, 3)
            for si in start_inds:
                ims += [im[si]]
                viss += [vis[si]]
                labels += [label[si]]
                good_noise = np.random.normal(loc=0, scale=2, size=(6, 2))
                good_noise[0, :] = 0
                good_zs += [tf.tile(label[si:si+1], [6, 1]) +
                            good_noise/self.scale]
                bad_noise = np.random.normal(loc=40, scale=20, size=(6, 2))
                bad_noise[3:] = np.random.normal(loc=-40, scale=20,
                                                 size=(3, 2))
                bad_zs += [tf.tile(label[si:si+1], [6, 1]) +
                           bad_noise/self.scale]

            values = [tf.stack(ims), tf.stack(labels), tf.stack(good_zs),
                      tf.stack(bad_zs)]
            labels = [tf.stack(labels), tf.stack(viss)]
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
            state = features['state']/self.scale
            q = features['q']
            actions = state[1:, 2:] - state[:-1, 2:]

            # we use several steps of the sequence
            start_inds = np.random.randint(1, state.get_shape()[0].value-1, 30)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            starts = []
            qs = []
            labels = []
            acs = []
            for si in start_inds:
                starts += [state[si-1]]
                qs += [q[si-1]]
                labels += [state[si]]
                acs += [actions[si-1]]

            values = [tf.stack(starts), tf.stack(acs)]
            labels = [tf.stack(labels), tf.stack(qs)]
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

            state = features['state']/self.scale
            q = features['q']
            actions = state[1:, 2:] - state[:-1, 2:]

            # use every fith data point
            start_inds = np.arange(1, state.get_shape()[0].value-1, 5)
            starts = []
            qs = []
            labels = []
            acs = []
            for si in start_inds:
                starts += [state[si-1]]
                qs += [q[si-1]]
                labels += [state[si]]
                acs += [actions[si-1]]

            values = [tf.stack(starts), tf.stack(acs)]
            labels = [tf.stack(labels), tf.stack(qs)]
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
            state = features['state']/self.scale
            start = features['start_state']/self.scale
            im = features['image']
            q = features['q']
            vis = features['visible']
            actions = state[1:, 2:] - state[:-1, 2:]
            actions = tf.concat([state[0:1, 2:] - start[None, 2:], actions],
                                axis=0)

            length = im.get_shape()[0].value

            if self.sl > length:
                raise ValueError('Desired training sequence length is ' +
                                 'longer than dataset sequence length: ' +
                                 'Desired: ' + str(self.sl) + ', data: ' +
                                 str(length))

            if self.sl == 50:
                start_inds = [0]
            else:
                # we use several sub-sequences of the testsequence, such that
                # the overall amount of data stays teh same between sequence
                # lengths (maximum length is 50)
                num = 50 // self.sl
                start_inds = \
                    np.random.randint(0, length-self.sl-1,
                                      num)
                self.train_multiplier = num

            # prepare the lists of output tensors
            ims = []
            starts = []
            im_starts = []
            states = []
            qs = []
            viss = []
            acs = []
            for si in start_inds:
                end = si+self.sl
                ims += [im[si:end]]
                if si > 0:
                    im_starts += [im[si-1]]
                    starts += [state[si-1]]
                else:
                    im_starts += [features['start_image']]
                    starts += [start]
                states += [state[si:end]]
                acs += [actions[si:end]]
                qs += [q[si:end]]
                viss += [vis[si:end]]

            ims = tf.stack(ims)
            # observations, actions, initial observations, initial state,
            # info
            values = [ims, tf.stack(acs), tf.stack(im_starts),
                      tf.stack(starts), tf.zeros([ims.get_shape()[0]])]
            labels = [tf.stack(states), tf.stack(qs), tf.stack(viss)]
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
            state = features['state']/self.scale
            start = features['start_state']/self.scale
            im = features['image']
            q = features['q']
            vis = features['visible']
            actions = state[1:, 2:] - state[:-1, 2:]
            actions = tf.concat([state[0:1, 2:] - start[None, 2:],
                                 actions], axis=0)

            length = im.get_shape()[0].value

            if self.sl > length:
                raise ValueError('Desired training sequence length is ' +
                                 'longer than dataset sequence length: ' +
                                 'Desired: ' + str(self.sl) + ', data: ' +
                                 str(length))

            if self.sl == 50:
                start_inds = [-1]
            else:
                # we use several sub-sequences of the testsequence, such that
                # the overall amount of data stays teh same between sequence
                # lengths (maximum length is 50)
                num = 50 // self.sl
                # we use several sub-sequences of the testsequence
                start_inds = \
                    np.arange(0, length-self.sl-1, (self.sl+1)//2)
                start_inds = start_inds[:num]

            # prepare the lists of output tensors
            ims = []
            starts = []
            states = []
            im_starts = []
            qs = []
            viss = []
            acs = []
            for si in start_inds:
                if si >= 0:
                    end = si+self.sl+1
                    ims += [im[si+1:end]]
                    im_starts += [im[si]]
                    states += [state[si+1:end]]
                    starts += [state[si]]
                    acs += [actions[si+1:end]]
                    qs += [q[si+1:end]]
                    viss += [vis[si+1:end]]
                else:
                    end = si+self.sl+1
                    ims += [im[si+1:end]]
                    im_starts += [features['start_image']]
                    states += [state[si+1:end]]
                    starts += [start]
                    acs += [actions[si+1:end]]
                    qs += [q[si+1:end]]
                    viss += [vis[si+1:end]]

            ims = tf.stack(ims)
            # observations, actions, initial observations, initial state,
            # info
            values = [ims, tf.stack(acs), tf.stack(im_starts),
                      tf.stack(starts), tf.zeros([ims.get_shape()[0]])]
            labels = [tf.stack(states), tf.stack(qs), tf.stack(viss)]
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

            state = features['state']/self.scale
            start = features['start_state']/self.scale
            im = features['image']
            q = features['q']
            vis = features['visible']
            actions = state[1:, 2:] - state[:-1, 2:]
            actions = tf.concat([state[0:1, 2:] - start[None, 2:],
                                 actions], axis=0)
            # state = tf.concat([start[None, :], state], axis=0)
            # im = tf.concat([features['start_image'][None, :, :, :], im],
            #                axis=0)

            length = im.get_shape()[0].value

            if self.sl > length:
                raise ValueError('Desired testing sequence length is ' +
                                 'longer than dataset sequence length: ' +
                                 'Desired: ' + str(self.sl) + ', data: ' +
                                 str(length))

            if self.sl > features['state'].get_shape()[0].value//2:
                states = [state[:self.sl]]
                starts = [start]
                im_starts = [features['start_image']]
                ims = [im[:self.sl]]
                qs = [q[:self.sl]]
                viss = [vis[:self.sl]]
                acs = [actions[:self.sl]]
                num = 1
            else:
                # prepend the initial state and image
                state = tf.concat([start[None, :], state], axis=0)
                im = tf.concat([features['start_image'][None, :, :, :], im],
                               axis=0)

                # we use several sub-sequences of the testsequence
                start_inds = \
                    np.arange(0, im.get_shape()[0].value-self.sl-1, self.sl//2)
                num = len(start_inds)
                # prepare the lists of output tensors
                ims = []
                starts = []
                states = []
                im_starts = []
                qs = []
                viss = []
                acs = []
                for si in start_inds:
                    end = si+self.sl+1
                    ims += [im[si+1:end]]
                    im_starts += [im[si]]
                    states += [state[si+1:end]]
                    starts += [state[si]]
                    acs += [actions[si+1:end]]
                    qs += [q[si+1:end]]
                    viss += [vis[si+1:end]]
            self.test_multiplier = num

            ims = tf.stack(ims)
            # observations, actions, initial observations, initial state,
            # info
            values = [ims, tf.stack(acs), tf.stack(im_starts),
                      tf.stack(starts), tf.zeros([ims.get_shape()[0]])]
            labels = [tf.stack(states), tf.stack(qs), tf.stack(viss)]
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
            keys = ['noise_num', 'likelihood', 'likelihood_std',
                    'dist', 'dist_std', 'dist_obs', 'dist_obs_std',
                    'x', 'x_std', 'y', 'y_std', 'vx', 'vx_std', 'vy', 'vy_std',
                    'x_obs', 'x_obs_std', 'y_obs', 'y_obs_std',
                    'q', 'q_std', 'corr_r']

            log_file = open(os.path.join(out_dir, str(step) + '_res.csv'), 'a')
            log = csv.DictWriter(log_file, keys)
            if num == 0:
                log.writeheader()

            row = {}
            for k, v in log_dict.items():
                if k in keys and \
                        type(v[0]) not in [str, bool, np.str, np.bool]:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)
            # corr_r cannot be properly evaluated per-example when batch size
            # is 1, so we have to evaluate it here before outputting it
            r_pred = log_dict['r_pred'].reshape(-1, self.dim_z).T
            vis = log_dict['vis'].reshape(-1, 1).T
            corr_r = np.corrcoef(r_pred[0:1], vis)[0, 1]
            corr_r += np.corrcoef(r_pred[1:], vis)[0, 1]
            row['corr_r'] = corr_r / 2
            row['noise_num'] = num
            log.writerow(row)
            log_file.close()
        else:
            row = {}
            for k, v in log_dict.items():
                if type(v[0]) not in [str, bool, np.str, np.bool]:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)

            if mode == 'pretrain_obs':
                # corr_r cannot be evaluated per-example when the batch size
                # is 1, so we have to evaluate it here before outputting it
                r_het_diag = log_dict['r_het_diag'].reshape(-1, self.dim_z).T
                r_het_tri = log_dict['r_het_tri'].reshape(-1, self.dim_z).T
                vis = log_dict['vis'].reshape(-1, 1).T
                corr_het_diag = np.corrcoef(r_het_diag[0:1], vis)[0, 1]
                corr_het_diag += np.corrcoef(r_het_diag[1:], vis)[0, 1]
                row['corr_r_het_diag'] = corr_het_diag / 2
                corr_het_tri = np.corrcoef(r_het_tri[0:1], vis)[0, 1]
                corr_het_tri += np.corrcoef(r_het_tri[1:], vis)[0, 1]
                row['corr_r_het_tri'] = corr_het_tri / 2

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

    def plot_tracking(self, seq_pred, cov_pred, z, seq, q_pred, q, r_pred, vis,
                      out_dir, num):
        pos_pred = np.squeeze(seq_pred[:, :2])
        v_pred = np.squeeze(seq_pred[:, 2:])

        if z is not None:
            x_obs = np.squeeze(z[:, 0])
            y_obs = np.squeeze(z[:, 1])

        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)
            q_pred = q_pred.reshape(self.sl, self.dim_x, self.dim_x)
            r_pred = r_pred.reshape(self.sl, self.dim_z, self.dim_z)
            pos_cov_x = np.sqrt(np.squeeze(cov_pred[:, 0, 0]))
            pos_cov_y = np.sqrt(np.squeeze(cov_pred[:, 1, 1]))
            v_cov_x = np.sqrt(np.squeeze(cov_pred[:, 2, 2]))
            v_cov_y = np.sqrt(np.squeeze(cov_pred[:, 3, 3]))
            if q.size == self.sl * self.dim_x:
                q = q.reshape(self.sl, self.dim_x)
                qx = np.squeeze(q[:, 0])
                qy = np.squeeze(q[:, 1])
                qvx = np.squeeze(q[:, 2])
                qvy = np.squeeze(q[:, 3])
                qxy = np.zeros_like(qx)
                qxvx = np.zeros_like(qx)
                qxvy = np.zeros_like(qx)
                qyvx = np.zeros_like(qx)
                qyvy = np.zeros_like(qx)
                qvxvy = np.zeros_like(qx)
            else:
                q = q.reshape(self.sl, self.dim_x, self.dim_x)
                qx = np.sqrt(np.squeeze(q[:, 0, 0]))
                qy = np.sqrt(np.squeeze(q[:, 1, 1]))
                qvx = np.sqrt(np.squeeze(q[:, 2, 2]))
                qvy = np.sqrt(np.squeeze(q[:, 3, 3]))
                qxy = np.sqrt(np.squeeze(q[:, 0, 1]))
                qxvx = np.sqrt(np.squeeze(q[:, 0, 2]))
                qxvy = np.sqrt(np.squeeze(q[:, 0, 3]))
                qy = np.sqrt(np.squeeze(q[:, 1, 1]))
                qyvx = np.sqrt(np.squeeze(q[:, 1, 2]))
                qyvy = np.sqrt(np.squeeze(q[:, 1, 3]))
                qvx = np.sqrt(np.squeeze(q[:, 2, 2]))
                qvxvy = np.sqrt(np.squeeze(q[:, 2, 3]))

            if self.param['diagonal_covar']:
                qx_p = np.sqrt(np.squeeze(q_pred[:, 0, 0]))
                qy_p = np.sqrt(np.squeeze(q_pred[:, 1, 1]))
                qvx_p = np.sqrt(np.squeeze(q_pred[:, 2, 2]))
                qvy_p = np.sqrt(np.squeeze(q_pred[:, 3, 3]))
                rx_p = np.sqrt(np.squeeze(r_pred[:, 0, 0]))
                ry_p = np.sqrt(np.squeeze(r_pred[:, 1, 1]))
            else:
                qx_p = np.squeeze(q_pred[:, 0, 0])
                qy_p = np.squeeze(q_pred[:, 1, 1])
                qvx_p = np.squeeze(q_pred[:, 2, 2])
                qvy_p = np.squeeze(q_pred[:, 3, 3])
                qxy_p = np.squeeze(q_pred[:, 0, 1])
                qxvx_p = np.squeeze(q_pred[:, 0, 2])
                qxvy_p = np.squeeze(q_pred[:, 0, 3])
                qy_p = np.squeeze(q_pred[:, 1, 1])
                qyvx_p = np.squeeze(q_pred[:, 1, 2])
                qyvy_p = np.squeeze(q_pred[:, 1, 3])
                qvx_p = np.squeeze(q_pred[:, 2, 2])
                qvxvy_p = np.squeeze(q_pred[:, 2, 3])
                rx_p = np.sqrt(np.squeeze(r_pred[:, 0, 0]))
                rxy_p = np.sqrt(np.squeeze(r_pred[:, 0, 1]))
                ry_p = np.sqrt(np.squeeze(r_pred[:, 1, 1]))
            vis = 100 * vis / np.max(vis)

        fig, ax = plt.subplots(5, figsize=[12, 25])
        ts = np.arange(pos_pred.shape[0])
        ax[0].plot(ts, pos_pred[:, 0], '-r', label='x predicted')
        ax[0].plot(ts, seq[:, 0], '--g', label='x true')
        ax[0].plot(ts, x_obs, 'kx', label='x observed')
        ax[0].plot(ts, pos_pred[:, 1], '-m', label='y predicted')
        ax[0].plot(ts, seq[:, 1], '--c', label='y true')
        ax[0].plot(ts, y_obs, 'ko', label=' y observed')
        ax[0].set_title('position')
        ax[0].legend()
        ax[1].plot(ts, v_pred[:, 0], '-r', label='vx predicted')
        ax[1].plot(ts, seq[:, 2], '--g', label='vx true')
        ax[1].plot(ts, v_pred[:, 1], '-m', label='vy predicted')
        ax[1].plot(ts, seq[:, 3], '--c', label='vy true')
        ax[1].set_title('velocity')
        ax[1].legend()
        ax[2].plot(ts, qx_p, '-r', label='qx predicted')
        ax[2].plot(ts, qx, '--g', label='qx true')
        ax[2].plot(ts, qy_p, '-m', label='qy predicted')
        ax[2].plot(ts, qy, '--c', label='qy true')
        ax[2].set_title('q position')
        ax[2].legend()
        ax[3].plot(ts, qvx_p, '-r', label='q vx predicted')
        ax[3].plot(ts, qvx, '--g', label='q vx true')
        ax[3].plot(ts, qvy_p, '-m', label='q vy predicted')
        ax[3].plot(ts, qvy, '--c', label='q vy true')
        ax[3].set_title('q velocity')
        ax[3].legend()
        ax[4].plot(ts, rx_p, '-r', label='r x predicted')
        ax[4].plot(ts, ry_p, '-m', label='r y predicted')
        ax[4].plot(ts, vis, '--g', label='percent visible')
        ax[4].set_title('r')
        ax[4].legend()

        if cov_pred is not None:
            ax[0].fill_between(ts, pos_pred[:, 0] - 2 * pos_cov_x,
                               pos_pred[:, 0] + 2 * pos_cov_x,
                               color="lightblue")
            ax[0].fill_between(ts, pos_pred[:, 1] - 2 * pos_cov_y,
                               pos_pred[:, 1] + 2 * pos_cov_y,
                               color="lightblue")
            ax[1].fill_between(ts, v_pred[:, 0] - 2 * v_cov_x,
                               v_pred[:, 0] + 2 * v_cov_x,
                               color="lightblue")
            ax[1].fill_between(ts, v_pred[:, 1] - 2 * v_cov_y,
                               v_pred[:, 1] + 2 * v_cov_y,
                               color="lightblue")

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85,
                            wspace=0.1, hspace=0.3)

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking"),
                    bbox_inches="tight")

        log_file = open(os.path.join(out_dir, str(num) + '_seq.csv'), 'w')

        if self.param['diagonal_covar']:
            keys = ['t', 'x', 'y', 'vx', 'vy', 'vis', 'qx', 'qy', 'qvx', 'qvy',
                    'x_p', 'y_p', 'vx_p', 'vy_p']
        else:
            keys = ['t', 'x', 'y', 'vx', 'vy', 'vis', 'qx', 'qxy', 'qxvx',
                    'qxvy', 'qy', 'qyvx', 'qyvy', 'qvx', 'qvxvy', 'qvy',
                    'x_p', 'y_p', 'vx_p', 'vy_p']
        if cov_pred is not None and z is not None:
            if self.param['diagonal_covar']:
                keys += ['x_c', 'y_c', 'vx_c', 'vy_c', 'x_ob', 'y_ob', 'qx_p',
                         'qy_p', 'qvx_p', 'qvy_p', 'rx_p', 'ry_p']
            else:
                keys += ['x_c', 'y_c', 'vx_c', 'vy_c', 'x_ob', 'y_ob', 'qx_p',
                         'qxy_p', 'qxvx_p', 'qxvy_p', 'qy_p', 'qyvx_p',
                         'qyvy_p', 'qvx_p', 'qvxvy_p', 'qvy_p',
                         'rx_p', 'rxy_p', 'ry_p']
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                if self.param['diagonal_covar']:
                    row = {'x': seq[t, 0], 'y': seq[t, 1], 'vx': seq[t, 2],
                           'vy': seq[t, 3], 'vis': vis[t], 'qx': qx[t],
                           'qy': qy[t], 'qvx': qvx[t], 'qvy': qvy[t],
                           'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                           'vx_p': seq_pred[t, 2], 'vy_p': seq_pred[t, 3],
                           'x_c': pos_cov_x[t], 'y_c': pos_cov_x[t],
                           'vx_c': v_cov_x[t], 'vy_c': v_cov_y[t],
                           'x_ob': x_obs[t], 'y_ob': y_obs[t],
                           'qx_p': qx_p[t], 'qy_p': qy_p[t], 'qvx_p': qvx_p[t],
                           'qvy_p': qvy_p[t], 'rx_p': rx_p[t], 'ry_p': ry_p[t]}
                else:
                    row = {'x': seq[t, 0], 'y': seq[t, 1], 'vx': seq[t, 2],
                           'vy': seq[t, 3], 'vis': vis[t], 'qx': qx[t],
                           'qxy': qxy[t], 'qxvx': qxvx[t], 'qxvy': qxvy[t],
                           'qy': qy[t], 'qyvx': qyvx[t], 'qyvy': qyvy[t],
                           'qvx': qvx[t], 'qvxvy': qvxvy[t], 'qvy': qvy[t],
                           'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                           'vx_p': seq_pred[t, 2], 'vy_p': seq_pred[t, 3],
                           'x_c': pos_cov_x[t], 'y_c': pos_cov_x[t],
                           'vx_c': v_cov_x[t], 'vy_c': v_cov_y[t],
                           'x_ob': x_obs[t], 'y_ob': y_obs[t], 'qx_p': qx_p[t],
                           'qxy_p': qxy_p[t], 'qxvx_p': qxvx_p[t],
                           'qxvy_p': qxvy_p[t], 'qy_p': qy_p[t],
                           'qyvx_p': qyvx_p[t], 'qyvy_p': qyvy_p[t],
                           'qvx_p': qvx_p[t], 'qvxvy_p': qvxvy_p[t],
                           'qvy_p': qvy_p[t], 'rx_p': rx_p[t],
                           'rxy_p': rxy_p[t], 'ry_p': ry_p[t]}
                log.writerow(row)
        else:
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                row = {'x': seq[t, 0], 'y': seq[t, 1], 'vx': seq[t, 2],
                       'vy': seq[t, 3], 'vis': vis[t], 'qx': qx[t],
                       'qy': qy[t], 'qvx': qvx[t], 'qvy': qvy[t],
                       'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                       'vx_p': seq_pred[t, 2], 'vy_p': seq_pred[t, 3]}
                log.writerow(row)
        log_file.close()

    def plot_trajectory(self, particles, weights, seq, cov_pred, seq_pred,
                        out_dir, num):
        if particles is not None:
            particles = particles.reshape(self.sl, -1, self.dim_x)
            weights = weights.reshape(self.sl, -1)
        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)

        pos_pred = np.squeeze(seq_pred[:, :2])
        fig, ax = plt.subplots(figsize=[35, 35])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
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
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')

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
            # plot the particles with colour depending on weight
            ax.scatter(par[:20, 0], par[:20, 1],
                       c=wei[:20], cmap='jet', marker='x', alpha=0.5)

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking_2d"),
                    bbox_inches="tight")


class SensorModel(BaseLayer):
    def __init__(self, batch_size, summary, trainable):
        super(SensorModel, self).__init__()
        self.summary = summary
        self.batch_size = batch_size

        self.c1 = self._conv_layer('conv1', 9, 4, stride=[2, 2],
                                   trainable=trainable)
        self.c2 = self._conv_layer('conv2', 9, 8, stride=[2, 2],
                                   trainable=trainable)

        self.fc1 = self._fc_layer('fc1', 16, trainable=trainable)
        self.fc2 = self._fc_layer('fc2', 32, trainable=trainable)
        self.pos = self._fc_layer('fc_pos', 2, trainable=trainable,
                                  activation=None)

    def call(self, images, training):
        # unpack the inputs
        if self.summary:
            tf.summary.image('im', images)

        conv1 = self.c1(images)
        conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')

        if self.summary:
            tf.summary.image('conv1_im',
                             tf.slice(conv1, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv1_out', conv1)

        # conv 2
        conv2 = self.c2(conv1)
        conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')
        if self.summary:
            tf.summary.image('conv2_im',
                             tf.slice(conv2, [0, 0, 0, 0], [1, -1, -1, 1]))
            tf.summary.histogram('conv2_out', conv2)

        input_data = tf.reshape(conv2, [self.batch_size, -1])
        fc1 = self.fc1(input_data)
        fc2 = self.fc2(fc1)
        pos = self.pos(fc2)

        if self.summary:
            tf.summary.histogram('fc1_out', fc1)
            tf.summary.histogram('fc2_out', fc2)
            tf.summary.histogram('fc_x_out', pos[:, 0:1])
            tf.summary.histogram('fc_y_out', pos[:, 1:])

        return pos, fc2


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
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(0.),
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
                [tf.tile(np.array([[[1, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size, 1, 1]),
                 tf.tile(np.array([[[0, 1, 0, 0]]], dtype=np.float32),
                         [self.batch_size, 1, 1])], axis=1)

        z_pred = tf.matmul(tf.tile(H, [bs, 1, 1]),
                           tf.expand_dims(inputs, -1))
        z_pred = tf.reshape(z_pred, [bs*self.batch_size, self.dim_z])
        return z_pred, H


class ProcessModel(BaseLayer):
    def __init__(self, batch_size, dim_x, dim_u, sf, df, scale,
                 learned, jacobian, trainable, summary):
        super(ProcessModel, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.learned = learned
        self.spring_force = sf
        self.drag_force = df
        self.trainable = trainable
        self.jacobian = jacobian
        self.scale = scale

    def build(self, input_shape):
        if self.learned:
            self.fc1 = self._fc_layer('fc1', 32, trainable=self.trainable)
            self.fc2 = self._fc_layer('fc2', 64, trainable=self.trainable)
            self.fc3 = self._fc_layer('fc3', 64, trainable=self.trainable)
            self.update = self._fc_layer('fc4', self.dim_x, activation=None,
                                         trainable=self.trainable)

    def call(self, inputs, training):
        # unpack the inputs
        last_state, actions = inputs
        if self.learned:
            fc1 = self.fc1(last_state)
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
            # split the state into parts and undo the scaling
            x = tf.slice(last_state, [0, 0], [-1, 1]) * self.scale
            y = tf.slice(last_state, [0, 1], [-1, 1]) * self.scale
            vx = tf.slice(last_state, [0, 2], [-1, 1]) * self.scale
            vy = tf.slice(last_state, [0, 3], [-1, 1]) * self.scale

            x_pred = x + vx
            y_pred = y + vy
            vx_pred = vx - self.spring_force * x - \
                self.drag_force * vx**2 * tf.sign(vx)
            vy_pred = vy - self.spring_force * y - \
                self.drag_force * vy**2 * tf.sign(vy)

            new_state = tf.concat([x_pred, y_pred, vx_pred, vy_pred], axis=1)
            new_state /= self.scale

            if self.jacobian:
                one = tf.ones_like(x)
                zero = tf.zeros_like(x)
                dvxdx = - self.spring_force * one
                dvx = one - 2 * self.drag_force * vx * tf.sign(vx)
                dvydy = - self.spring_force * one
                dvy = one - 2 * self.drag_force * vy * tf.sign(vy)
                F = tf.concat(
                    [tf.concat([one, zero, one, zero], axis=1)[:, None, :],
                     tf.concat([zero, one, zero, one], axis=1)[:, None, :],
                     tf.concat([dvxdx, zero, dvx, zero], axis=1)[:, None, :],
                     tf.concat([zero, dvydy, zero, dvy], axis=1)[:, None, :]],
                    axis=1)
            else:
                F = None
            fc3 = None
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
        init_const = np.ones(self.dim_x) * 1e-3/self.scale**2
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
                                initializer=tf.constant_initializer(0.),
                                regularizer=tf.keras.regularizers.l2(l=wd),
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
                                initializer=tf.constant_initializer(0.),
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                trainable=self.trainable)
            self.const_full_ana_init_bias = \
                self.add_weight(name='const_full_ana_init_bias',
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        old_state, actions = inputs
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
