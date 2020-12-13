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
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pickle


from differentiable_filters import base_context as base
from differentiable_filters.base_layer import BaseLayer
from differentiable_filters import recordio as tfr
from differentiable_filters import push_utils as utils


class Context(base.BaseContext):
    def __init__(self, param, mode):
        base.BaseContext.__init__(self, param)

        if 'normalize' in param.keys():
            self.normalize = param['normalize']
        else:
            self.normalize = 'layer'

        # the state size
        self.dim_x = 10
        self.dim_u = 2
        self.dim_z = 8

        # dimension names
        self.x_names = ['x', 'y', 'theta', 'l', 'mu', 'rx', 'ry', 'nx', 'ny',
                        's']
        self.z_names = ['x', 'y', 'theta', 'rx', 'ry', 'nx', 'ny', 's']

        self.sl = param['sequence_length']
        self.scale = param['scale']

        # load the points on the outline of the butter object for visualization
        path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(path, 'butter_points.pkl'), 'rb') as bf:
            butter_points = pickle.load(bf)
        self.butter_points = np.array(butter_points)

        # for state in mm/deg,
        # c = np.array([50, 50, 1e-2, 5, 5, 50, 50, 0.5, 0.5, 0.5])
        self.noise_list = \
            [np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([49.8394116, -2.3510439, 0, 2.5196417, 1.93745247,
                       27.6656989, 67.1287098, 0.03124815, -0.18917632,
                       -0.14730855]),
             np.array([27.9914853, -30.3366791, 0, -4.6963326, -2.96631439,
                       3.6698755, -14.5376077, -0.49956926, 0.56362964,
                       0.54478971])]

        for i, n in enumerate(self.noise_list):
            self.noise_list[i] = n.astype(np.float32)

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
            # don't train the segmentation model is we use a pretrained
            # sensor network
            self.segmentation_layer = \
                SegmentationLayer(self.batch_size, self.normalize, summary,
                                  train_sensor_model)
            self.sensor_model_layer = \
                SensorLayer(self.batch_size, self.normalize, self.scale,
                            summary, train_sensor_model)
            self.observation_model_layer = ObservationModel(self.dim_z,
                                                            self.batch_size)
            # group the layers for easier checkpoint restoring
            self.observation_models = {'sensor': [self.segmentation_layer,
                                                  self.sensor_model_layer],
                                       'obs': self.observation_model_layer}
            self.update_ops += self.segmentation_layer.updateable
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
                                 self.scale,
                                 hetero=True, diag=True, trainable=train_r,
                                 summary=summary)
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
                ProcessModel(self.batch_size, self.dim_x, self.scale,
                             learned=True, jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['learned'] = self.process_model_learned_layer
        if mode == 'filter' and not lstm_unstructured and \
                not param['learn_process'] or mode == 'pretrain_process':
            self.process_model_analytical_layer = \
                ProcessModel(self.batch_size, self.dim_x, self.scale,
                             learned=False, jacobian=param['filter'] == 'ekf',
                             trainable=train_process_model, summary=summary)
            self.process_models['ana'] = self.process_model_analytical_layer

        self.process_noise_models = {}
        process_noise = (param['learn_q'] and not lstm_no_noise and
                         mode == 'filter')

        if process_noise and param['learn_process'] and param['hetero_q'] and \
                param['diagonal_covar'] or mode == 'pretrain_process':
            self.process_noise_hetero_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_lrn'] = \
                self.process_noise_hetero_diag_lrn
        if process_noise and param['learn_process'] and param['hetero_q'] and \
                not param['diagonal_covar'] or mode == 'pretrain_process':
            self.process_noise_hetero_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_full_lrn'] = \
                self.process_noise_hetero_full_lrn
        if process_noise and param['learn_process'] and \
                not param['hetero_q'] and param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True, learned=True,
                             trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_lrn'] = \
                self.process_noise_const_diag_lrn
        if process_noise and param['learn_process'] and \
                not param['hetero_q'] and not param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_const_full_lrn = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=True, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_lrn'] = \
                self.process_noise_const_full_lrn

        if process_noise and not param['learn_process'] and \
                param['hetero_q'] and param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_hetero_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=True, learned=False,
                             trainable=train_q, summary=summary)
            self.process_noise_models['het_diag_ana'] = \
                self.process_noise_hetero_diag_ana
        if process_noise and not param['learn_process'] and \
                param['hetero_q'] and not param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_hetero_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=True, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['het_full_ana'] = \
                self.process_noise_hetero_full_ana
        if process_noise and not param['learn_process'] and \
                not param['hetero_q'] and param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_const_diag_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=True,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_diag_ana'] = \
                self.process_noise_const_diag_ana
        if process_noise and not param['learn_process'] and \
                not param['hetero_q'] and not param['diagonal_covar'] or \
                mode == 'pretrain_process':
            self.process_noise_const_full_ana = \
                ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                             self.scale, hetero=False, diag=False,
                             learned=False, trainable=train_q, summary=summary)
            self.process_noise_models['const_full_ana'] = \
                self.process_noise_const_full_ana

    ###########################################################################
    # observation models
    ###########################################################################
    def sensor_model(self, raw_observations, training):
        """
        Process raw observations and return an encoding and
        predicted observations z for the filter
        """
        images, tip_pos, tip_pos_pix, tip_end_pix, start_glimpse = \
            raw_observations
        seg_out, pix = self.segmentation_layer(images, training)
        z, enc = self.sensor_model_layer([images, tip_pos, tip_pos_pix,
                                          tip_end_pix, start_glimpse] +
                                         seg_out, training)
        enc = list(enc) + [pix]
        return z, enc

    def process_model(self, old_state, action, learned, training):
        """
        Predict the next state from the old state and the action and returns
        the jacobian
        """
        if learned:
            new_state, last_layer, F = \
                self.process_model_learned_layer([old_state, action, self.ob],
                                                 training)
        else:
            new_state, last_layer, F = \
                self.process_model_analytical_layer([old_state, action,
                                                     self.ob], training)
        new_state = self.correct_state(new_state, diff=False)
        return new_state, last_layer, F

    def get_initial_glimpse(self, image, training):
        """
        Process the observations for the initial state and return a segmented
        glimpse of the object in its initial position
        """
        seg_out, pix = self.segmentation_layer(image, training)
        mask, pos, glimpse_rot = seg_out
        return glimpse_rot, pix, mask

    def initial_from_observed(self, base_state, init_z, base_covar, init_R):
        state = tf.concat([init_z[:, :3], base_state[:, 3:5], init_z[:, 3:]],
                          axis=-1)
        covar = \
            tf.concat([tf.concat([base_covar[:, :3, :3], init_R[:, :3, :3]],
                                 axis=-1),
                       base_covar[:, 3:5, :],
                       tf.concat([base_covar[:, 5:, 5:], init_R[:, 3:, 3:]],
                                 axis=-1)],
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

        seq_label, mv_tr, mv_rot, vis = label

        diff = seq_label - states
        diff = self.correct_state(diff)

        # version that excludes l and mu from the loss
        diff_red = tf.concat([diff[:, :, :3], diff[:, :, 5:]], axis=-1)
        covar_red = tf.concat([tf.concat([covars[:, :, :3, :3],
                                          covars[:, :, :3, 5:]], axis=-1),
                               tf.concat([covars[:, :, 5:, :3],
                                          covars[:, :, 5:, 5:]], axis=-1)],
                              axis=-2)

        # get the likelihood
        if self.param['filter'] == 'pf' and self.param['mixture_likelihood']:
            num = particles.get_shape()[2].value
            seq_label_tiled = tf.tile(seq_label[:, :, None, :], [1, 1, num, 1])

            particle_diff = self.correct_state(seq_label_tiled - particles)
            particle_diff_red = tf.concat([particle_diff[:, :, :, :3],
                                           particle_diff[:, :, :, 5:]],
                                          axis=-1)
            likelihood = self._mixture_likelihood(particle_diff, weights)
            likelihood_red = self._mixture_likelihood(particle_diff_red,
                                                      weights)
        else:
            likelihood = self._likelihood(diff, covars, reduce_mean=False)
            likelihood_red = self._likelihood(diff_red, covar_red,
                                              reduce_mean=False)

        # compute the errors of the predicted states
        mses = []
        mses_red = []
        dists = []
        for i in range(self.dim_x):
            mse, dist = self._mse(diff[:, :, i:i+1], reduce_mean=False)
            # undo the overall scaling for dist and mse
            mses += [mse*self.scale**2]
            dists += [dist*self.scale]
            if i not in [3, 4]:
                mses_red += [mse*self.scale**2]
        tracking_mse = tf.add_n(mses)
        tracking_mse_red = tf.add_n(mses_red)
        tracking_dist = tf.add_n(dists)

        _, dist_tr = self._mse(diff[:, :, 0:2], reduce_mean=False)
        _, dist_rot = self._mse(diff[:, :, 2:3], reduce_mean=False)

        # compute the error in the predicted observations (only for monitoring)
        diff_obs = tf.concat([seq_label[:, :, :3] - z[:, :, 0:3],
                              seq_label[:, :, 5:] - z[:, :, 3:]], axis=-1)
        diff_obs = self.correct_observation_diff(diff_obs)
        dist_obs = []
        for i in range(self.dim_z):
            _, dist = self._mse(diff_obs[:, :, i:i+1], reduce_mean=False)
            dist = dist*self.scale
            dist_obs += [dist]
        dist_ob = tf.add_n(dist_obs)

        # compute the correlation between predicted observation noise and
        # the number of visible object pixels
        # this only makes sense for the heteroscedastic noise
        diag_r = tf.linalg.diag_part(r)
        diag_r = tf.sqrt(tf.abs(diag_r + 1e-5))
        diag_r = tf.reshape(diag_r, [-1, self.dim_z])
        corr = []
        for i in range(self.dim_z):
            corr += \
                [tfp.stats.correlation(diag_r[:, i:i+1],
                                       tf.reshape(vis, [-1, 1]),
                                       sample_axis=0, event_axis=-1)]
        corr_r = tf.add_n(corr)/self.dim_z

        # correlation between noise and contact
        corr_r_cont = []
        for i in range(self.dim_z):
            crs = \
                tfp.stats.correlation(diag_r[:, i:i+1],
                                      tf.reshape(seq_label[:, :, 9:], [-1, 1]),
                                      sample_axis=0, event_axis=-1)
            corr_r_cont += [crs]
        corr_r_cont = tf.add_n(corr_r_cont)/self.dim_z

        # same for q
        diag_q = tf.linalg.diag_part(q)
        diag_q = tf.sqrt(tf.abs(diag_q + 1e-5))
        diag_q = tf.reshape(diag_q, [-1, self.dim_x])
        corr_q = []
        for i in range(self.dim_x-1):
            cqs = \
                tfp.stats.correlation(diag_q[:, i:i+1],
                                      tf.reshape(seq_label[:, :, 9:], [-1, 1]),
                                      sample_axis=0, event_axis=-1)
            corr_q += [cqs]
        corr_q = tf.add_n(corr_q)/(self.dim_x-1)

        # compute the output metric
        m_per_tr, deg_per_deg = \
            self._output_loss(states[:, :, :3], seq_label[:, :, :3],
                              mv_tr, mv_rot)
        tf.summary.scalar('out/m_per_tr', m_per_tr)
        tf.summary.scalar('out/deg_per_deg', deg_per_deg)
        tf.summary.scalar('out/tr_total', tf.reduce_mean(mv_tr))
        tf.summary.scalar('out/rot_total', tf.reduce_mean(mv_rot))
        tf.summary.scalar('out/tr_error', tf.reduce_mean(dist_tr))
        tf.summary.scalar('out/rot_error', tf.reduce_mean(dist_rot))

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
        total_tracking = tf.reduce_mean(tracking_mse)
        total_tracking_red = tf.reduce_mean(tracking_mse_red)
        total_obs = tf.reduce_mean(dist_obs)
        if self.loss == 'like':
            total_loss = tf.reduce_mean(likelihood)
        elif self.loss == 'like_red':
            total_loss = tf.reduce_mean(likelihood_red)
        elif self.loss == 'error':
            total_loss = total_tracking
        elif self.loss == 'mixed':
            total_loss = (total_tracking + tf.reduce_mean(likelihood)) / 2.
        elif self.loss == 'error_red':
            total_loss = total_tracking_red
        elif self.loss == 'mixed_red':
            total_loss = (total_tracking_red +
                          tf.reduce_mean(likelihood_red)) / 2.
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

        if self.loss != 'error':
            total_loss_val += 1000

        total = tf.cond(training,
                        lambda: total_loss + wd, lambda: total_loss_val)

        # add summaries
        tf.summary.scalar('loss/total', total)
        tf.summary.scalar('loss/wd', wd)
        tf.summary.scalar('loss/likelihood', tf.reduce_mean(likelihood))
        tf.summary.scalar('loss/tracking', total_tracking)
        tf.summary.scalar('loss/observations', total_obs)
        tf.summary.scalar('loss/corr_r_vis', tf.squeeze(corr_r))
        tf.summary.scalar('loss/corr_r_cont', tf.squeeze(corr_r_cont))
        tf.summary.scalar('loss/corr_q_cont', tf.squeeze(corr_q))
        for i, name in enumerate(self.x_names):
            tf.summary.scalar('tracking_loss/' + name,
                              tf.reduce_mean(dists[i]))
        for i, name in enumerate(self.z_names):
            tf.summary.scalar('observation_loss/' + name,
                              tf.reduce_mean(dist_obs[i]))
        return total, [likelihood, tracking_dist, dist_ob, tracking_mse,
                       dist_tr, dist_rot, m_per_tr, deg_per_deg, vis,
                       seq_label[:, :, 9], diag_r, diag_q, wd] +\
            dists, ['likelihood', 'dist', 'dist_obs', 'mse', 'dist_tr',
                    'dist_rot', 'm_tr', 'deg_rot', 'vis', 'cont', 'r_pred',
                    'q_pred', 'wd'] + \
            self.x_names

    def _output_loss(self, pred, label, mv_tr, mv_rot):
        endpoint_error = self._compute_sq_distance(pred[:, -1, 0:2],
                                                   label[:, -1, 0:2])
        endpoint_error_rot = self._compute_sq_distance(pred[:, -1, 2:3],
                                                       label[:, -1, 2:3], True)

        m_per_tr = tf.where(tf.greater(mv_tr, 0),
                            endpoint_error**0.5/mv_tr, endpoint_error)
        deg_per_deg = tf.where(tf.greater(mv_rot, 0),
                               endpoint_error_rot**0.5/mv_rot,
                               endpoint_error_rot)

        return tf.reduce_mean(m_per_tr), tf.reduce_mean(deg_per_deg)

    def _compute_sq_distance(self, pred, label, rotation=False):
        diff = pred - label
        if rotation:
            diff = self._adapt_orientation(diff, self.ob, 1)
        diff = tf.square(diff)
        diff = tf.reduce_sum(diff, axis=-1)

        diff = tf.where(tf.greater(diff, 0), tf.sqrt(diff), diff)

        return diff

    def get_observation_loss(self, prediction, labels, step, training):
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
        z, pix_pred, seg_pred, initial_pix_pred, initial_seg_pred, \
            R_const_diag, R_const_tri, R_het_diag, R_het_tri, \
            like_good, like_bad = prediction

        label, pix_pos, initial_pix_pos, seg, initial_seg, vis = labels
        diff = self.correct_observation_diff(label - z)

        likelihood_const_diag = self._likelihood(tf.stop_gradient(diff),
                                                 R_const_diag,
                                                 reduce_mean=False)
        likelihood_const_tri = self._likelihood(tf.stop_gradient(diff),
                                                R_const_tri,
                                                reduce_mean=False)
        likelihood_het_diag = self._likelihood(diff, R_het_diag,
                                               reduce_mean=False)
        likelihood_het_tri = self._likelihood(diff, R_het_tri,
                                              reduce_mean=False)

        likelihood = (likelihood_const_diag + likelihood_const_tri +
                      likelihood_het_diag + likelihood_het_tri) / 4.

        # compute the correlation between predicted observation noise and
        # the number of visible object pixels
        # this only makes sense for the heteroscedastic noise
        diag_r_het_diag = tf.linalg.diag_part(R_het_diag)
        diag_r_het_diag = tf.sqrt(tf.abs(diag_r_het_diag + 1e-5))
        diag_r_het_diag = tf.reshape(diag_r_het_diag, [-1, self.dim_z])
        diag_r_het_tri = tf.linalg.diag_part(R_het_tri)
        diag_r_het_tri = tf.sqrt(tf.abs(diag_r_het_tri + 1e-5))
        diag_r_het_tri = tf.reshape(diag_r_het_tri, [-1, self.dim_z])
        corr_diag = []
        corr_full = []
        for i in range(self.dim_z):
            corr_diag += \
                [tfp.stats.correlation(diag_r_het_diag[:, i:i+1],
                                       tf.reshape(vis, [-1, 1]),
                                       sample_axis=0, event_axis=-1)]
            corr_full += \
                [tfp.stats.correlation(diag_r_het_tri[:, i:i+1],
                                       tf.reshape(vis, [-1, 1]),
                                       sample_axis=0, event_axis=-1)]
        corr_r_diag = tf.add_n(corr_diag)/self.dim_z
        corr_r_full = tf.add_n(corr_full)/self.dim_z

        # compute the errors of the predicted observations
        dist_obs = []
        mses = []
        cont = label[:, 7:8]
        for i in range(self.dim_z):
            mse, dist = self._mse(diff[:, i:i+1], reduce_mean=False)
            # undo the overall scaling for dist and mse, but only undo the
            # component-wise scaling for dist
            scale_dist = self.scale
            scale_mse = self.scale**2
            # mask out non-contact cases for contact point and normal
            if i in [3, 4, 5, 6]:
                dist_obs += [tf.reduce_mean(dist*scale_dist*cont)]
                mses += [tf.reduce_sum(mse*scale_mse*cont)]
            else:
                dist_obs += [tf.reduce_mean(dist*scale_dist)]
                mses += [tf.reduce_sum(mse*scale_mse)]
        mse = tf.add_n(mses)

        # segmentatuin error
        height = seg.get_shape()[1]
        width = seg.get_shape()[2]
        seg_pred = tf.image.resize(seg_pred, [height, width])
        initial_seg_pred = tf.image.resize(initial_seg_pred, [height, width])
        seg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.squeeze(seg_pred, axis=-1),
            labels=tf.squeeze(seg, axis=-1))
        seg_loss = tf.reduce_mean(tf.reduce_sum(seg_loss, axis=[1, 2]))
        seg_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.squeeze(initial_seg_pred, axis=-1),
            labels=tf.squeeze(initial_seg, axis=-1))
        seg_loss += tf.reduce_mean(tf.reduce_sum(seg_loss2, axis=[1, 2]))

        # get the pixel prediction error for the position
        pix_diff = pix_pred - pix_pos
        pix_mse, pix_dist = self._mse(pix_diff, reduce_mean=False)
        pix_mse = tf.reduce_mean(pix_mse)
        _, dist_3d = self._mse(diff[:, :2], reduce_mean=False)

        initial_pix_diff = initial_pix_pred - initial_pix_pos
        initial_pix_mse, initial_pix_dist = self._mse(initial_pix_diff,
                                                      reduce_mean=False)
        initial_pix_mse = tf.reduce_mean(initial_pix_mse)

        # compute the angle-loss of the normals
        norm_pred = z[:, 5:7]
        norm_label = label[:, 5:7]
        normal_ang = self.normal_loss(norm_pred, norm_label)

        # compute the contact loss
        contact_loss, ce = self.contact_loss(z[:, 7:8], label[:, 7:8])

        # compute the loss for the learned likelihood model of the pf
        good_loss = tf.reduce_mean(-tf.math.log(tf.maximum(like_good, 1e-6)))
        bad_loss = \
            tf.reduce_mean(-tf.math.log(tf.maximum(1.0 - like_bad, 1e-6)))
        like_loss = good_loss + bad_loss

        # add a penalty term for predicted rotation values greater than pi
        rot_pred = tf.abs(z[:, 2])
        rot_penalty = tf.where(tf.greater(rot_pred, 180),
                               tf.square(rot_pred - 180),
                               tf.zeros_like(rot_pred))
        rot_penalty = tf.reduce_mean(rot_penalty)

        wd = []
        for la in self.observation_models.values():
            wd += la.losses
        for la in self.observation_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        # start by training only the localization for two epochs
        total_train = \
            tf.cond(tf.less(step, self.epoch_size*2),
                    lambda: 10 * (pix_mse + initial_pix_mse) + seg_loss,
                    lambda: (10 * tf.add_n(mses) +
                             10 * (pix_mse + initial_pix_mse) +
                             100 * tf.reduce_mean(normal_ang) +
                             100 * tf.reduce_mean(contact_loss) +
                             1e-4 * tf.reduce_mean(likelihood) +
                             1e-3 * like_loss +
                             rot_penalty + 0.01 * seg_loss + 0.01 * wd))
        total_train = \
            tf.cond(tf.less(step, self.epoch_size*5),
                    lambda: total_train,
                    lambda: (10 * tf.add_n(mses) +
                             10 * (pix_mse + initial_pix_mse) +
                             100 * tf.reduce_mean(normal_ang) +
                             100 * tf.reduce_mean(contact_loss) +
                             0.1 * (tf.reduce_mean(likelihood) + like_loss) +
                             rot_penalty + 0.001 * seg_loss + wd))

        total_val = 10 * tf.add_n(mses) + 10 * tf.reduce_mean(normal_ang) + \
            100 * tf.reduce_mean(contact_loss) + \
            tf.reduce_mean(likelihood) + like_loss + 100

        total = tf.cond(training, lambda: total_train, lambda: total_val)

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
        for i, name in enumerate(self.z_names):
            tf.summary.scalar('label/' + name, label[0, i])
        for i, name in enumerate(self.z_names):
            tf.summary.scalar('observation_loss/' + name,
                              tf.reduce_mean(dist_obs[i]))
        for i, name in enumerate(self.z_names):
            tf.summary.scalar('noise_loss/diag_' + name,
                              tf.reduce_mean(corr_diag[i]))
            tf.summary.scalar('noise_loss/full_' + name,
                              tf.reduce_mean(corr_full[i]))
        tf.summary.scalar('noise_loss/corr_diag', tf.reduce_mean(corr_r_diag))
        tf.summary.scalar('noise_loss/corr_full', tf.reduce_mean(corr_r_full))
        tf.summary.scalar('observation_loss/normal_ang',
                          tf.reduce_mean(normal_ang))
        tf.summary.scalar('observation_loss/mean_vis',
                          tf.reduce_mean(vis))
        tf.summary.scalar('observation_loss/dist_pix',
                          tf.reduce_mean(pix_dist))
        tf.summary.scalar('observation_loss/dist_3d',
                          tf.reduce_mean(dist_3d))
        tf.summary.scalar('observation_loss/contact_cross',
                          tf.reduce_mean(ce))
        tf.summary.scalar('observation_loss/rot_penalty', rot_penalty)
        tf.summary.scalar('loss/like_good', good_loss)
        tf.summary.scalar('loss/like_bad', bad_loss)
        tf.summary.scalar('loss/like_loss', like_loss)

        tf.summary.scalar('loss/segmentation', seg_loss)
        tf.summary.image('loss/seg_label', seg)
        tf.summary.image('loss/seg_pred', seg_pred)
        tf.summary.image('loss/initial_seg_label', initial_seg)
        tf.summary.image('loss/inital_seg_pred', initial_seg_pred)

        return total, [likelihood_const_diag, likelihood_const_tri,
                       likelihood_het_diag, likelihood_het_tri,
                       mse, like_loss, tf.reduce_mean(normal_ang),
                       tf.reduce_mean(ce), tf.reshape(vis, [-1, 1]),
                       diag_r_het_diag, diag_r_het_tri, wd] + dist_obs, \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri', 'mse', 'like',
             'normal_ang', 'contact_cross', 'vis', 'r_het_diag',
             'r_het_tri', 'wd'] + self.z_names

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

        # compute the errors of the predicted states from the learned model
        mses = []
        dists = []
        for i in range(self.dim_x):
            mse, dist = self._mse(diff[:, i:i+1], reduce_mean=False)
            # undo the overall scaling for dist and mse
            mses += [tf.reduce_mean(mse*self.scale**2)]
            dists += [tf.reduce_mean(dist*self.scale)]
        mse = tf.add_n(mses)

        # compute the errors of the predicted states from the analytical model
        dists_ana = []
        for i in range(self.dim_x):
            _, dist = self._mse(diff_ana[:, i:i+1], reduce_mean=False)
            dists_ana += [tf.reduce_mean(dist*self.scale)]

        wd = []
        for la in self.process_models.values():
            wd += la.losses
        for la in self.process_noise_models.values():
            wd += la.losses
        wd = tf.add_n(wd)

        total_loss = \
            tf.cond(tf.less(step, self.epoch_size*5),
                    lambda: (1000 * tf.reduce_mean(mse) +
                             1e-5 * tf.reduce_mean(likelihood) +
                             1e-5 * tf.reduce_mean(likelihood_ana)),
                    lambda: (tf.reduce_mean(likelihood) +
                             tf.reduce_mean(likelihood_ana) +
                             1000 * tf.reduce_mean(mse)))

        total = \
            tf.cond(training,
                    lambda: total_loss + wd,
                    lambda: (tf.reduce_mean(likelihood) + 100 +
                             tf.reduce_mean(likelihood_ana) +
                             10 * tf.reduce_mean(mse)))

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
        tf.summary.scalar('loss/tracking', tf.reduce_mean(mse))
        for i, name in enumerate(self.x_names):
            tf.summary.scalar('tracking_loss/' + name,
                              tf.reduce_mean(dists[i]))
            tf.summary.scalar('tracking_loss/' + name + '_ana',
                              tf.reduce_mean(dists_ana[i]))

        for i in range(min(self.batch_size, 1)):
            tf.summary.scalar('label/x_' + str(i), label[i, 0])
            tf.summary.scalar('label/y_' + str(i), label[i, 1])
            tf.summary.scalar('label/theta_' + str(i), label[i, 2])
            tf.summary.scalar('label/l_' + str(i), label[i, 3])
            tf.summary.scalar('label/mu_' + str(i), label[i, 4])
            tf.summary.scalar('label/rx_' + str(i), label[i, 5])
            tf.summary.scalar('label/ry_' + str(i), label[i, 6])
            tf.summary.scalar('label/nx_' + str(i), label[i, 7])
            tf.summary.scalar('label/ny_' + str(i), label[i, 8])
            tf.summary.scalar('label/s_' + str(i), label[i, 9])
            tf.summary.scalar('start/x_' + str(i), start[i, 0])
            tf.summary.scalar('start/y_' + str(i), start[i, 1])
            tf.summary.scalar('start/theta_' + str(i), start[i, 2])
            tf.summary.scalar('start/l_' + str(i), start[i, 3])
            tf.summary.scalar('start/mu_' + str(i), start[i, 4])
            tf.summary.scalar('start/rx_' + str(i), start[i, 5])
            tf.summary.scalar('start/ry_' + str(i), start[i, 6])
            tf.summary.scalar('start/nx_' + str(i), start[i, 7])
            tf.summary.scalar('start/ny_' + str(i), start[i, 8])
            tf.summary.scalar('start/s_' + str(i), start[i, 9])
            tf.summary.scalar('pred/x_ana_' + str(i), state_ana[i, 0])
            tf.summary.scalar('pred/y_ana_' + str(i), state_ana[i, 1])
            tf.summary.scalar('pred/theta_ana_' + str(i), state_ana[i, 2])
            tf.summary.scalar('pred/l_ana_' + str(i), state_ana[i, 3])
            tf.summary.scalar('pred/mu_ana_' + str(i), state_ana[i, 4])
            tf.summary.scalar('pred/rx_ana_' + str(i), state_ana[i, 5])
            tf.summary.scalar('pred/ry_ana_' + str(i), state_ana[i, 6])
            tf.summary.scalar('pred/nx_ana_' + str(i), state_ana[i, 7])
            tf.summary.scalar('pred/ny_ana_' + str(i), state_ana[i, 8])
            tf.summary.scalar('pred/s_ana_' + str(i), state_ana[i, 9])
            tf.summary.scalar('pred/x_' + str(i), state[i, 0])
            tf.summary.scalar('pred/y_' + str(i), state[i, 1])
            tf.summary.scalar('pred/theta_' + str(i), state[i, 2])
            tf.summary.scalar('pred/l_' + str(i), state[i, 3])
            tf.summary.scalar('pred/mu_' + str(i), state[i, 4])
            tf.summary.scalar('pred/rx_' + str(i), state[i, 5])
            tf.summary.scalar('pred/ry_' + str(i), state[i, 6])
            tf.summary.scalar('pred/nx_' + str(i), state[i, 7])
            tf.summary.scalar('pred/ny_' + str(i), state[i, 8])
            tf.summary.scalar('pred/s_' + str(i), state[i, 9])

        return total, \
            [likelihood_const_diag, likelihood_const_tri,
             likelihood_het_diag, likelihood_het_tri,
             likelihood_const_diag_ana, likelihood_const_tri_ana,
             likelihood_het_diag_ana, likelihood_het_tri_ana, wd] + dists + \
            dists_ana, \
            ['likelihood_const_diag', 'likelihood_const_tri',
             'likelihood_het_diag', 'likelihood_het_tri',
             'likelihood_const_diag_ana', 'likelihood_const_tri_ana',
             'likelihood_het_diag_ana', 'likelihood_het_tri_ana', 'wd'] + \
            self.x_names + list(map(lambda x: x + '_ana', self.x_names))

    def normal_loss(self, pred, label, name=""):
        # normalize both
        pred_norm = tf.norm(pred, axis=-1, keep_dims=True)
        label_norm = tf.norm(label, axis=-1, keep_dims=True)
        pred = tf.nn.l2_normalize(pred, -1)
        label = tf.nn.l2_normalize(label, -1)

        # calculate the angles between them
        if len(pred.get_shape().as_list()) == 3:
            prod = tf.matmul(tf.reshape(pred, [self.batch_size, -1, 1, 2]),
                             tf.reshape(label, [self.batch_size, -1, 2, 1]))
            prod = tf.clip_by_value(prod, -0.999999999, 0.999999999)
            prod = tf.acos(tf.reshape(prod, [self.batch_size, -1, 1]))
        else:
            prod = tf.matmul(tf.reshape(pred, [self.batch_size, 1, 2]),
                             tf.reshape(label, [self.batch_size, 2, 1]))
            prod = tf.clip_by_value(prod, -0.999999999, 0.999999999)
            prod = tf.acos(tf.reshape(prod, [self.batch_size, 1]))

        # mask out invalid values and non-contact cases
        greater = tf.logical_and(tf.greater(pred_norm, 1e-6),
                                 tf.greater(label_norm, 1e-6))
        ang_mask = tf.logical_and(greater, tf.math.is_finite(prod))
        ang = tf.where(ang_mask, tf.abs(prod), tf.zeros_like(prod))

        # correct values over 180 deg.
        ang = tf.where(tf.greater(tf.abs(ang), np.pi),
                       2*np.pi - tf.abs(ang), tf.abs(ang))*180./np.pi
        return ang

    def contact_loss(self, pred, label, name=""):
        # calculate the error
        label = tf.reshape(label, [self.batch_size, -1, 1])
        pred = tf.reshape(pred, [self.batch_size, -1, 1])
        # limit pred to [0..1]
        pred = tf.clip_by_value(pred, 0, 1.)
        # slightly downweight the loss for in-contact-cases to reduce the
        # amount of false-positives
        loss = (1 - label) * -tf.math.log(tf.maximum(1 - pred, 1e-7)) + \
            label * -tf.math.log(tf.maximum(pred, 1e-7))
        ce = (1 - label) * -tf.math.log(tf.maximum(1 - pred, 1e-7)) + \
            label * -tf.math.log(tf.maximum(pred, 1e-7))

        return loss, ce

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
        sc = self.scale
        if diff:
            state = \
                tf.concat([state[:, :2],
                           self._adapt_orientation(state[:, 2:3], self.ob, sc),
                           state[:, 3:]], axis=-1)
        else:
            state = \
                tf.concat([state[:, :2],
                           self._adapt_orientation(state[:, 2:3], self.ob, sc),
                           self._adapt_fr(state[:, 3:4]),
                           self._adapt_m(state[:, 4:5]),
                           state[:, 5:7],
                           self._adapt_n(state[:, 7:9], state[:, 5:7],
                                         state[:, 0:2]),
                           self._adapt_s(state[:, 9:])], axis=-1)
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
        sc = 1 * self.scale
        diff = tf.concat([diff[:, :2],
                          self._adapt_orientation(diff[:, 2:3], self.ob, sc),
                          diff[:, 3:]], axis=-1)
        if len(shape) > 2:
            diff = tf.reshape(diff, shape[:-1] + [self.dim_z])
        return diff

    def weighted_state_mean_with_angles(self, points, weights):
        ps = tf.concat([points[:, :, :2],
                        tf.sin(points[:, :, 2:3]*self.scale*np.pi/180.0),
                        tf.cos(points[:, :, 2:3]*self.scale*np.pi/180.0),
                        points[:, :, 3:]], axis=-1)
        mult = tf.multiply(ps, weights)
        mean = tf.reduce_sum(mult, axis=1)

        ang1 = tf.math.atan2(mean[:, 2:3], mean[:, 3:4])*180.0/np.pi

        out = tf.concat([mean[:, :2], ang1/self.scale, mean[:, 4:]], axis=-1)
        return out

    def weighted_observation_mean_with_angles(self, points, weights, axis=1):
        ps = tf.concat([points[:, :, :2],
                        tf.sin(points[:, :, 2:3]*self.scale*np.pi/180.0),
                        tf.cos(points[:, :, 2:3]*self.scale*np.pi/180.0),
                        points[:, :, 3:]], axis=-1)
        mult = tf.multiply(ps, weights)
        mean = tf.reduce_sum(mult, axis=axis)

        ang = tf.math.atan2(mean[:, 2:3], mean[:, 3:4])*180.0/np.pi

        out = tf.concat([mean[:, :2], ang/self.scale, mean[:, 4:]], axis=-1)
        return out

    def _adapt_fr(self, fr):
        # prevent l from getting too small or too big
        fr = tf.clip_by_value(fr, 0.1/self.scale, 5e3/self.scale)
        return fr

    def _adapt_m(self, m):
        # prevent m from getting negative or too large
        m = tf.clip_by_value(m, 0.1/self.scale, 90./self.scale)
        return m

    def _adapt_s(self, s):
        # keep the contact indicator between 0 and 1
        s = tf.clip_by_value(s, 0., 1.)
        return s

    def _adapt_n(self, n, r, o):
        # normalize -- not good at all!
        # n_norm = tf.linalg.norm(n, axis=-1, keepdims=True)
        # n = tf.where(tf.greater(tf.squeeze(n_norm), 1e-6),  n/n_norm, n)

        # # make sure the normal points towards the object
        # dir_center = o[:, :2] - r[:, :2]
        # dir_center_norm = tf.linalg.norm(dir_center, axis=-1, keepdims=True)
        # dir_center = tf.where(tf.greater(tf.squeeze(dir_center_norm), 0.),
        #                       dir_center/dir_center_norm, dir_center)
        # prod = tf.matmul(tf.reshape(dir_center, [bs, 1, 2]),
        #                   tf.reshape(n, [bs, 2, 1]))
        # ang = tf.acos(tf.reshape(prod, [bs]))
        # # correct values over 180 deg.
        # ang = tf.where(tf.greater(tf.abs(ang), np.pi),
        #                 2*np.pi - tf.abs(ang), tf.abs(ang))*180./np.pi
        # # if the angle is greater than 90 degree, we need to flip the
        # # normal
        # n = tf.where(tf.greater(ang, np.pi/2.),  n, -1 * n)
        return n

    def _adapt_orientation(self, rot, ob, sc):
        rot = rot * sc
        # in most cases, the maximum rotation range is 180deg, but some have
        # more or fewer symmetries
        # we first apply a modulo operation to make sure that no value is
        # larger than the maximum rotation range. Then we have to deal with the
        # periodicity of the interval
        rot_max = tf.ones_like(rot) * 180

        ob = tf.squeeze(ob)
        ob = tf.strings.regex_replace(ob, "\000", "")
        ob = tf.strings.regex_replace(ob, "\00", "")
        if len(ob.get_shape()) < 1:
            rot_max = \
                tf.case({tf.equal(ob, 'ellip1'): lambda: tf.zeros_like(rot),
                         tf.equal(ob, 'rect1'): lambda: tf.ones_like(rot)*90.,
                         tf.equal(ob, 'tri1'): lambda: tf.ones_like(rot)*360.,
                         tf.equal(ob, 'tri2'): lambda: tf.ones_like(rot)*360.,
                         tf.equal(ob, 'tri3'): lambda: tf.ones_like(rot)*360.,
                         tf.equal(ob, 'hex'): lambda: tf.ones_like(rot)*60.},
                        default=lambda: rot_max, exclusive=True)

            rot_new = \
                tf.cond(tf.equal(ob, 'ellip1'), lambda: tf.zeros_like(rot),
                        lambda: tf.math.mod(tf.abs(rot), rot_max)*tf.sign(rot))

            # now make sure that the measured rotation is the smallest
            # posslibel value in the interval - rot_max/2, rot_max/2
            rot_add = tf.where(tf.greater(rot_new, rot_max/2.),
                               rot_new - rot_max, rot_new)
            rot_add = tf.where(tf.less(rot_add, -rot_max/2.),
                               rot_add + rot_max, rot_add)
        else:
            if ob.get_shape()[0].value < rot.get_shape()[0].value:
                mult = rot.get_shape()[0].value // ob.get_shape()[0].value
                ob = tf.reshape(ob, [-1, 1])
                ob = tf.reshape(tf.tile(ob, [1, mult]), [-1])
            rot_max = tf.where(tf.equal(ob, 'ellip1'), tf.zeros_like(rot),
                               rot_max)
            rot_max = tf.where(tf.equal(ob, 'rect1'), tf.ones_like(rot)*90,
                               rot_max)
            rot_max = tf.where(tf.equal(ob, 'tri1'), tf.ones_like(rot)*360,
                               rot_max)
            rot_max = tf.where(tf.equal(ob, 'tri2'), tf.ones_like(rot)*360,
                               rot_max)
            rot_max = tf.where(tf.equal(ob, 'tri3'), tf.ones_like(rot)*360,
                               rot_max)
            rot_max = tf.where(tf.equal(ob, 'hex'), tf.ones_like(rot)*60,
                               rot_max)

            rot_new = tf.where(tf.equal(ob, 'ellip1'), tf.zeros_like(rot),
                               tf.math.mod(tf.abs(rot), rot_max)*tf.sign(rot))

            # now make sure that the measured rotation is the smallest
            # posslibel value in the interval - rot_max/2, rot_max/2
            rot_add = tf.where(tf.greater(rot_new, rot_max/2.),
                               rot_new - rot_max, rot_new)
            rot_add = tf.where(tf.less(rot_add, -rot_max/2.),
                               rot_add + rot_max, rot_add)

        rot_add /= sc
        return rot_add

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=5):
        """
        Defines how to read in the data from a tf record
        """
        keys = ['pos', 'object', 'contact_point', 'normal', 'contact',
                'tip', 'friction', 'coord', 'image', 'material', 'pix_tip',
                'pix_pos', 'segmentation']

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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*(180.0/np.pi),
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000/self.scale,
                              pose[:, 1:2]*1000/self.scale,
                              ori/self.scale], axis=1)
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]*1000
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2]*1000, cp)/self.scale
            pix_tip = features['pix_tip']

            im = features['image']
            coord = features['coord']
            mask = features['segmentation']
            mask = tf.cast(tf.where(tf.greater(mask, 2.5), tf.ones_like(mask),
                                    tf.zeros_like(mask)), tf.float32)
            vis = tf.reduce_sum(mask, axis=[1, 2, 3])

            seq_len = im.get_shape()[0].value
            im = tf.concat([im, coord], axis=-1)
            pix = features['pix_pos'][:, :2]
            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])

            # we use several steps of the sequence
            start_inds = np.random.randint(2, seq_len-2, 5)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            ims = []
            start_ims = []
            start_ts = []
            tes = []
            labels = []
            good_zs = []
            bad_zs = []
            viss = []
            pixs = []
            pixts = []
            pixte = []
            start_pixs = []
            segs = []
            start_segs = []
            for si in start_inds:
                ref = np.random.randint(1, si)
                start_ts += [tips[ref]]
                start_ims += [im[ref]]
                start_pixs += [pix[ref]]
                start_segs += [mask[ref]]

                viss += [vis[si]]
                segs += [mask[si]]
                ims += [im[si]]
                tes += [tips[si]]
                pixs += [pix[si]]
                pixts += [pix_tip[si]]
                pixte += [pix_tip[si+1]]
                relative_rot = \
                    self._adapt_orientation(pose[si, 2:3] - pose[ref, 2:3], ob,
                                            self.scale)
                label = tf.concat([pose[si, :2], relative_rot, cp[si], n[si],
                                   con[si]], axis=0)
                labels += [label]

                good_noise = np.random.normal(loc=0, scale=1e-1, size=(24, 8))
                good_noise[0, :] = 0
                bad_noise = np.random.normal(loc=10, scale=5, size=(24, 8))
                bad_noise[12:] = np.random.normal(loc=-10, scale=5,
                                                  size=(12, 8))
                # downscale noise for normal and contact
                good_noise[:, 5:] /= 10
                bad_noise[:, 5:] /= 10
                # upscale for pos and or
                bad_noise[:, :2] *= 10
                bad_noise[:, 2:3] *= 2
                good_noise[:, :2] *= 10
                good_noise[:, 2:3] *= 2

                # adapt to scaling
                bad_noise /= self.scale
                good_noise /= self.scale
                bad_zs += [tf.tile(label[None, :], [24, 1]) + bad_noise]
                good_zs += [tf.tile(label[None, :], [24, 1]) + good_noise]

            ims = tf.stack(ims)
            start_ims = tf.stack(start_ims)
            start_ts = tf.stack(start_ts)
            tes = tf.stack(tes)
            pixts = tf.stack(pixts)
            pixte = tf.stack(pixte)
            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [(ims, tes, pixts, pixte), tf.stack(labels),
                      tf.stack(good_zs),
                      tf.stack(bad_zs), (start_ims, start_ts), (ob, mat)]
            labels = [tf.stack(labels), tf.stack(pixs), tf.stack(start_pixs),
                      tf.stack(segs), tf.stack(start_segs), tf.stack(viss)]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*(180.0/np.pi),
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000/self.scale,
                              pose[:, 1:2]*1000/self.scale,
                              ori/self.scale],
                             axis=1)
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]*1000
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2]*1000, cp)/self.scale
            pix_tip = features['pix_tip']

            im = features['image']
            coord = features['coord']
            mask = features['segmentation']
            mask = tf.cast(tf.where(tf.greater(mask, 2.5), tf.ones_like(mask),
                                    tf.zeros_like(mask)), tf.float32)
            vis = tf.reduce_sum(mask, axis=[1, 2, 3])

            seq_len = im.get_shape()[0].value
            im = tf.concat([im, coord], axis=-1)
            pix = features['pix_pos'][:, :2]
            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])

            # # sanity check
            # # load a plane image for reprojecting
            # path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            # path = os.path.join(path, 'plane_image_smaller.npy')
            # print('loading plane image from: ', path)

            # plane_depth = tf.convert_to_tensor(np.load(path))[none, :, :, none]

            # pix_pos = features['pix_pos'][1:2]
            # pos_3d = features['pos'][1:2, :3]
            # projected1 = utils._to_3d(pix_pos, im[1:2, :, :, -1:])
            # projected2 = utils._to_3d(pix_pos, plane_depth)
            # pix_pro = utils._to_2d(pos_3d)

            # cp = tf.print(cp, [pix_pos, pix_pro],
            #               summarize=1000, message='pix, pix_pro\n')
            # cp = tf.print(cp, [pos_3d, projected1, projected2],
            #               summarize=1000, message='3d, pro_d, pro_plane \n')

            # use every eighth data point
            start_inds = np.arange(2, seq_len-2, 8)
            num = len(start_inds)

            # prepare the lists of output tensors
            viss = []
            ims = []
            start_ims = []
            start_ts = []
            tes = []
            labels = []
            good_zs = []
            bad_zs = []
            pixs = []
            pixts = []
            pixte = []
            start_pixs = []
            segs = []
            start_segs = []
            for si in start_inds:
                start_ts += [tips[1]]
                start_ims += [im[1]]
                start_pixs += [pix[1]]
                start_segs += [mask[1]]
                viss += [vis[si]]
                segs += [mask[si]]
                ims += [im[si]]
                pixs += [pix[si]]
                pixts += [pix_tip[si]]
                pixte += [pix_tip[si+1]]
                tes += [tips[si]]
                relative_rot = \
                    self._adapt_orientation(pose[si, 2:3] - pose[1, 2:3], ob,
                                            self.scale)
                label = tf.concat([pose[si, :2], relative_rot, cp[si], n[si],
                                   con[si]], axis=0)
                labels += [label]
                good_noise = np.random.normal(loc=0, scale=1e-1, size=(24, 8))
                good_noise[0, :] = 0
                bad_noise = np.random.normal(loc=10, scale=5, size=(24, 8))
                bad_noise[12:] = np.random.normal(loc=-10, scale=5,
                                                  size=(12, 8))
                # downscale noise for normal and contact
                good_noise[:, 5:] /= 10
                bad_noise[:, 5:] /= 10
                # upscale for pos and or
                bad_noise[:, :2] *= 10
                bad_noise[:, 2:3] *= 2
                good_noise[:, :2] *= 10
                good_noise[:, 2:3] *= 2

                # adapt to scaling
                bad_noise /= self.scale
                good_noise /= self.scale
                bad_zs += [tf.tile(label[None, :], [24, 1]) + bad_noise]
                good_zs += [tf.tile(label[None, :], [24, 1]) + good_noise]

            ims = tf.stack(ims)
            start_ims = tf.stack(start_ims)
            start_ts = tf.stack(start_ts)
            tes = tf.stack(tes)
            pixts = tf.stack(pixts)
            pixte = tf.stack(pixte)
            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [(ims, tes, pixts, pixte), tf.stack(labels),
                      tf.stack(good_zs),
                      tf.stack(bad_zs), (start_ims, start_ts), (ob, mat)]
            labels = [tf.stack(labels), tf.stack(pixs), tf.stack(start_pixs),
                      tf.stack(segs), tf.stack(start_segs), tf.stack(viss)]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*180./np.pi,
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000, pose[:, 1:2]*1000, ori],
                             axis=1)/self.scale
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2], cp)*1000/self.scale
            friction = \
                tf.square(tf.reshape(features['friction'], [1]) * 1000.)
            friction = friction/(100*self.scale)
            mu = tf.atan(tf.ones([1], dtype=tf.float32) * 0.25)*180./np.pi
            mu = mu/self.scale
            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])
            seq_len = features['pos'].get_shape()[0].value

            # calculate the actions - scale them by the same amount as the
            # position
            t_end = tips[1:, :2]
            t_start = tips[:-1, :2]
            u = (t_end - t_start) * 1000./self.scale

            # we use several steps of the sequence
            start_inds = np.random.randint(2, seq_len-1, 10)
            num = len(start_inds)
            self.train_multiplier = num

            # prepare the lists of output tensors
            start_state = []
            us = []
            labels = []
            for si in start_inds:
                p_start = pose[si-1][:2]
                s_start = tf.concat([p_start, tf.zeros([1]), friction, mu,
                                     cp[si-1], n[si-1], con[si-1]], axis=0)
                start_state += [s_start]
                us += [u[si-1]]

                relative_rot = pose[si, 2:3] - pose[si-1, 2:3]
                relative_rot = \
                    self._adapt_orientation(relative_rot, ob, self.scale)
                label = tf.concat([pose[si, :2], relative_rot, friction, mu,
                                   cp[si], n[si], con[si]], axis=0)
                labels += [label]

            start_state = tf.stack(start_state)
            us = tf.stack(us)
            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [start_state, us, (ob, mat)]
            labels = [labels, start_state]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*180./np.pi,
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000, pose[:, 1:2]*1000, ori],
                             axis=1)/self.scale
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2], cp)*1000/self.scale
            friction = \
                tf.square(tf.reshape(features['friction'], [1]) * 1000.)
            friction = friction/(100*self.scale)
            mu = tf.atan(tf.ones([1], dtype=tf.float32) * 0.25)*180./np.pi
            mu = mu/self.scale
            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])
            seq_len = features['pos'].get_shape()[0].value

            # calculate the actions - scale them by the same amount as the
            # position
            t_end = tips[1:, :2]
            t_start = tips[:-1, :2]
            u = (t_end - t_start) * 1000./self.scale

            # use every eigth data point
            start_inds = np.arange(2, seq_len-1, 8)
            num = len(start_inds)
            # prepare the lists of output tensors
            start_state = []
            us = []
            labels = []
            for si in start_inds:
                p_start = pose[si-1][:2]
                s_start = tf.concat([p_start, tf.zeros([1]), friction, mu,
                                     cp[si-1], n[si-1], con[si-1]], axis=0)
                start_state += [s_start]
                us += [u[si-1]]

                relative_rot = pose[si, 2:3] - pose[si-1, 2:3]
                relative_rot = \
                    self._adapt_orientation(relative_rot, ob, self.scale)
                label = tf.concat([pose[si, :2], relative_rot, friction, mu,
                                   cp[si], n[si], con[si]], axis=0)
                labels += [label]

            start_state = tf.stack(start_state)
            us = tf.stack(us)
            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [start_state, us, (ob, mat)]
            labels = [labels, start_state]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*180./np.pi,
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000, pose[:, 1:2]*1000, ori],
                             axis=1)/self.scale
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2], cp)*1000/self.scale
            friction = \
                tf.square(tf.reshape(features['friction'], [1]) * 1000.)
            friction = friction/(100*self.scale)
            mu = tf.atan(tf.ones([1], dtype=tf.float32) * 0.25)*180./np.pi
            mu = mu/self.scale

            # calculate the actions - scale them by the same amount as the
            # position
            t_end = tips[1:, :2]
            t_start = tips[:-1, :2]
            u = (t_end - t_start) * 1000./self.scale

            im = features['image']
            coord = features['coord']
            mask = features['segmentation']
            mask = tf.cast(tf.where(tf.greater(mask, 2.5), tf.ones_like(mask),
                                    tf.zeros_like(mask)), tf.float32)
            vis = tf.reduce_sum(mask, axis=[1, 2, 3])
            im = tf.concat([im, coord], axis=-1)
            pix_tip = features['pix_tip']

            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])
            seq_len = features['pos'].get_shape()[0].value

            num = 1
            start_inds = np.random.randint(1, seq_len-self.sl-2, num)

            # prepare the lists of output tensors
            ims = []
            start_ims = []
            start_ts = []
            start_state = []
            us = []
            tes = []
            pixts = []
            pixte = []
            labels = []
            mv_trs = []
            mv_rots = []
            viss = []
            for si in start_inds:
                p_start = pose[si][:2]
                s_start = tf.concat([p_start, tf.zeros([1]), friction, mu,
                                     cp[si], n[si], con[si]], axis=0)
                start_state += [s_start]
                start_ts += [tips[si]]
                start_ims += [im[si]]

                start = si + 1
                end = si + 1 + self.sl
                ims += [im[start:end]]
                us += [u[start:end]]
                tes += [tips[start:end]]
                pixts += [pix_tip[start:end]]
                pixte += [pix_tip[start+1:end+1]]

                relative_rot = pose[start:end, 2:3] - \
                    tf.tile(pose[si:si+1, 2:3], [self.sl, 1])
                relative_rot = \
                    self._adapt_orientation(relative_rot, ob, self.scale)
                label = tf.concat([pose[start:end, :2], relative_rot,
                                   tf.tile(friction[None, :], [self.sl, 1]),
                                   tf.tile(mu[None, :], [self.sl, 1]),
                                   cp[start:end], n[start:end],
                                   con[start:end]], axis=-1)
                labels += [label]
                viss += [vis[start:end]]

                mv = pose[start:end] - pose[si:end-1]
                mv_trs += [tf.reduce_sum(tf.norm(mv[:, :2], axis=-1))]
                mvr = self._adapt_orientation(mv[:, 2], ob, self.scale)
                mv_rots += [tf.reduce_sum(tf.abs(mvr))]

            ims = tf.stack(ims)
            start_ims = tf.stack(start_ims)
            start_ts = tf.stack(start_ts)
            start_state = tf.stack(start_state)
            us = tf.stack(us)
            tes = tf.stack(tes)
            pixts = tf.stack(pixts)
            pixte = tf.stack(pixte)
            mv_trs = tf.stack(mv_trs)
            mv_rots = tf.stack(mv_rots)
            viss = tf.stack(viss)

            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [(ims, tes, pixts, pixte), us, (start_ims, start_ts),
                      start_state, (ob, mat)]
            labels = [labels, mv_trs, mv_rots, viss]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*180./np.pi,
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000, pose[:, 1:2]*1000, ori],
                             axis=1)/self.scale
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2], cp)*1000/self.scale
            friction = \
                tf.square(tf.reshape(features['friction'], [1]) * 1000.)
            friction = friction/(100*self.scale)
            mu = tf.atan(tf.ones([1], dtype=tf.float32) * 0.25)*180./np.pi
            mu = mu/self.scale

            # calculate the actions - scale them by the same amount as the
            # position
            t_end = tips[1:, :2]
            t_start = tips[:-1, :2]
            u = (t_end - t_start) * 1000./self.scale

            im = features['image']
            coord = features['coord']
            mask = features['segmentation']
            mask = tf.cast(tf.where(tf.greater(mask, 2.5), tf.ones_like(mask),
                                    tf.zeros_like(mask)), tf.float32)
            vis = tf.reduce_sum(mask, axis=[1, 2, 3])
            im = tf.concat([im, coord], axis=-1)
            pix_tip = features['pix_tip']

            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])
            seq_len = features['pos'].get_shape()[0].value

            num = 1
            # we use several sub-sequences of the validation sequence
            start_inds = np.arange(1, seq_len-self.sl-2, (self.sl+1)//2)
            start_inds = start_inds[:num]

            # prepare the lists of output tensors
            ims = []
            start_ims = []
            start_ts = []
            start_state = []
            us = []
            tes = []
            pixts = []
            pixte = []
            labels = []
            mv_trs = []
            mv_rots = []
            viss = []
            for si in start_inds:
                p_start = pose[si][:2]
                s_start = tf.concat([p_start, tf.zeros([1]), friction, mu,
                                     cp[si], n[si], con[si]], axis=0)
                start_state += [s_start]
                start_ts += [tips[si]]
                start_ims += [im[si]]

                start = si + 1
                end = si + 1 + self.sl
                ims += [im[start:end]]
                us += [u[start:end]]
                tes += [tips[start:end]]
                pixts += [pix_tip[start:end]]
                pixte += [pix_tip[start+1:end+1]]

                relative_rot = pose[start:end, 2:3] - \
                    tf.tile(pose[si:si+1, 2:3], [self.sl, 1])
                relative_rot = \
                    self._adapt_orientation(relative_rot, ob, self.scale)
                label = tf.concat([pose[start:end, :2], relative_rot,
                                   tf.tile(friction[None, :], [self.sl, 1]),
                                   tf.tile(mu[None, :], [self.sl, 1]),
                                   cp[start:end], n[start:end],
                                   con[start:end]], axis=-1)
                labels += [label]
                viss += [vis[start:end]]

                mv = pose[start:end] - pose[si:end-1]
                mv_trs += [tf.reduce_sum(tf.norm(mv[:, :2], axis=-1))]
                mvr = self._adapt_orientation(mv[:, 2], ob, self.scale)
                mv_rots += [tf.reduce_sum(tf.abs(mvr))]

            ims = tf.stack(ims)
            start_ims = tf.stack(start_ims)
            start_ts = tf.stack(start_ts)
            start_state = tf.stack(start_state)
            us = tf.stack(us)
            tes = tf.stack(tes)
            pixts = tf.stack(pixts)
            pixte = tf.stack(pixte)
            mv_trs = tf.stack(mv_trs)
            mv_rots = tf.stack(mv_rots)

            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [(ims, tes, pixts, pixte), us, (start_ims, start_ts),
                      start_state, (ob, mat)]
            labels = [labels, mv_trs, mv_rots, tf.stack(viss)]
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
            pose = features['pos']
            ori = self._adapt_orientation(pose[:, 3:]*180./np.pi,
                                          features['object'], 1)
            pose = tf.concat([pose[:, 0:1]*1000, pose[:, 1:2]*1000, ori],
                             axis=1)/self.scale
            n = tf.squeeze(features['normal'])/self.scale
            con = tf.cast(features['contact'], tf.float32)
            con = tf.reshape(con, [-1, 1])/self.scale
            tips = features['tip']
            cp = features['contact_point'][:, :2]
            con_norm = tf.linalg.norm(cp, axis=-1)
            cp = tf.where(tf.less(con_norm, 1e-6),
                          tips[:, :2], cp)*1000/self.scale
            friction = \
                tf.square(tf.reshape(features['friction'], [1]) * 1000.)
            friction = friction/(100*self.scale)
            mu = tf.atan(tf.ones([1], dtype=tf.float32) * 0.25)*180./np.pi
            mu = mu/self.scale

            # calculate the actions - scale them by the same amount as the
            # position
            t_end = tips[1:, :2]
            t_start = tips[:-1, :2]
            u = (t_end - t_start) * 1000./self.scale

            im = features['image']
            coord = features['coord']
            mask = features['segmentation']
            mask = tf.cast(tf.where(tf.greater(mask, 2.5), tf.ones_like(mask),
                                    tf.zeros_like(mask)), tf.float32)
            vis = tf.reduce_sum(mask, axis=[1, 2, 3])
            im = tf.concat([im, coord], axis=-1)
            pix_tip = features['pix_tip']

            ob = tf.reshape(features['object'], [1])
            mat = tf.reshape(features['material'], [1])
            seq_len = features['pos'].get_shape()[0].value

            if self.sl > seq_len//2:
                start_inds = [1]
            else:
                # we use several sub-sequences of the testsequence
                start_inds = np.arange(1, seq_len-self.sl-2, 20)
            num = len(start_inds)
            self.test_multiplier = num

            # prepare the lists of output tensors
            ims = []
            start_ims = []
            start_ts = []
            start_state = []
            us = []
            tes = []
            pixts = []
            pixte = []
            labels = []
            mv_trs = []
            mv_rots = []
            viss = []
            for si in start_inds:
                p_start = pose[si][:2]
                s_start = tf.concat([p_start, tf.zeros([1]), friction, mu,
                                     cp[si], n[si], con[si]], axis=0)
                start_state += [s_start]
                start_ts += [tips[si]]
                start_ims += [im[si]]

                start = si + 1
                end = si + 1 + self.sl
                ims += [im[start:end]]
                us += [u[start:end]]
                tes += [tips[start:end]]
                pixts += [pix_tip[start:end]]
                pixte += [pix_tip[start+1:end+1]]

                relative_rot = pose[start:end, 2:3] - \
                    tf.tile(pose[si:si+1, 2:3], [self.sl, 1])
                relative_rot = \
                    self._adapt_orientation(relative_rot, ob, self.scale)
                label = tf.concat([pose[start:end, :2], relative_rot,
                                   tf.tile(friction[None, :], [self.sl, 1]),
                                   tf.tile(mu[None, :], [self.sl, 1]),
                                   cp[start:end], n[start:end],
                                   con[start:end]], axis=1)
                labels += [label]
                viss += [vis[start:end]]

                mv = pose[start:end] - pose[si:end-1]
                mv_trs += [tf.reduce_sum(tf.norm(mv[:, :2], axis=-1))]
                mvr = self._adapt_orientation(mv[:, 2], ob, self.scale)
                mv_rots += [tf.reduce_sum(tf.abs(mvr))]

            ims = tf.stack(ims)
            start_ims = tf.stack(start_ims)
            start_ts = tf.stack(start_ts)
            start_state = tf.stack(start_state)
            us = tf.stack(us)
            tes = tf.stack(tes)
            pixts = tf.stack(pixts)
            pixte = tf.stack(pixte)
            mv_trs = tf.stack(mv_trs)
            mv_rots = tf.stack(mv_rots)

            ob = tf.tile(ob, [num])
            mat = tf.tile(mat, [num])

            values = [(ims, tes, pixts, pixte), us, (start_ims, start_ts),
                      start_state, (ob, mat)]
            labels = [labels, mv_trs, mv_rots, tf.stack(viss)]
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
        elif train_mode == 'pretrain_obs':
            if data_mode == 'train':
                dataset = dataset.map(_parse_function_obs_train,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'val' or data_mode == 'test':
                dataset = dataset.map(_parse_function_obs_val,
                                      num_parallel_calls=num_threads)
        elif train_mode == 'pretrain_process':
            if data_mode == 'train':
                dataset = dataset.map(_parse_function_process_train,
                                      num_parallel_calls=num_threads)
            elif data_mode == 'val' or data_mode == 'test':
                dataset = dataset.map(_parse_function_process_val,
                                      num_parallel_calls=num_threads)
        else:
            self.log.error('unknown training mode: ' + train_mode)

        dataset = \
            dataset.flat_map(lambda x, y:
                             tf.data.Dataset.from_tensor_slices((x, y)))

        return dataset

    ######################################
    # Evaluation
    ######################################
    def save_log(self, log_dict, out_dir, step, num, mode):
        if mode == 'filter':
            keys = ['noise_num', 'likelihood', 'likelihood_std', 'dist_tr',
                    'dist_tr_std', 'dist_rot', 'dist_rot_std', 'corr_r_vis',
                    'corr_r_cont', 'corr_q_cont',
                    'm_tr', 'm_tr_std', 'deg_rot', 'deg_rot_std', 'dist',
                    'dist_std', 'dist_obs', 'dist_obs_std']
            keys += self.x_names + list(map(lambda x: x + '_std',
                                            self.x_names))

            keys_corr = ['noise_num']
            keys_corr += list(map(lambda x: 'cq_cont_' + x, self.x_names))
            keys_corr += list(map(lambda x: 'cr_cont_' + x, self.z_names))
            keys_corr += list(map(lambda x: 'cr_vis_' + x, self.z_names))

            log_file = open(os.path.join(out_dir, str(step) + '_res.csv'), 'a')
            log = csv.DictWriter(log_file, keys)
            if num == 0:
                log.writeheader()

            log_file_corr = open(os.path.join(out_dir,
                                              str(step) + '_corr.csv'), 'a')
            log_corr = csv.DictWriter(log_file_corr, keys_corr)
            if num == 0:
                log_corr.writeheader()

            row = {}
            for k, v in log_dict.items():
                if k in keys and type(v[0]) not in [str, bool, np.str,
                                                    np.bool]:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)

            # corr_r cannot be properly evaluated per-example when batch size
            # is 1, so we have to evaluate it here before outputting it
            row_corr = {}
            r_pred = log_dict['r_pred'].reshape(-1, self.dim_z).T
            vis = log_dict['vis'].reshape(-1, 1).T
            cont = log_dict['cont'].reshape(-1, 1).T
            corr_vis = []
            corr_cont = []
            for i, n in enumerate(self.z_names):
                r_c = np.corrcoef(r_pred[i:i+1], cont)[0, 1]
                r_v = np.corrcoef(r_pred[i:i+1], vis)[0, 1]
                corr_vis += [r_v]
                corr_cont += [r_c]
                row_corr['cr_cont_' + n] = r_c
                row_corr['cr_vis_' + n] = r_v
            row['corr_r_vis'] = np.mean(corr_vis)
            row['corr_r_cont'] = np.mean(corr_cont)

            q_pred = log_dict['q_pred'].reshape(-1, self.dim_x).T
            corr_cont = []
            for i, n in enumerate(self.x_names):
                q_c = np.corrcoef(q_pred[i:i+1], cont)[0, 1]
                corr_cont += [q_c]
                row_corr['cq_cont_' + n] = q_c
            row['corr_q_cont'] = np.mean(corr_cont)

            row['noise_num'] = num
            log.writerow(row)
            log_file.close()

            row_corr['noise_num'] = num
            log_corr.writerow(row_corr)
            log_file_corr.close()
        else:
            row = {}
            for k, v in log_dict.items():
                if type(v[0]) not in [str, bool, np.str, np.bool]:
                    row[k] = np.mean(v)
                    row[k + '_std'] = np.std(v)

            if mode == 'pretrain_obs':
                # corr_r cannot be properly evaluated per-example when batch
                # size is 1, so we have to evaluate it here
                r_het_diag = log_dict['r_het_diag'].reshape(-1, self.dim_z).T
                r_het_tri = log_dict['r_het_tri'].reshape(-1, self.dim_z).T
                vis = log_dict['vis'].reshape(-1, 1).T
                corr_diags = []
                corr_fulls = []
                for i in range(self.dim_z):
                    corr_diags += [np.corrcoef(r_het_diag[i:i+1], vis)[0, 1]]
                    corr_fulls += [np.corrcoef(r_het_tri[i:i+1], vis)[0, 1]]

                row['corr_r_het_diag'] = np.mean(corr_diags)
                row['corr_r_het_tri'] = np.mean(corr_fulls)

                for i, n in enumerate(self.z_names):
                    row['corr_' + n + '_diag'] = corr_diags[i]
                    row['corr_' + n + '_full'] = corr_fulls[i]

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

    def plot_tracking(self, seq_pred, cov_pred, z, seq, q_pred, r_pred, vis,
                      out_dir, num, diffs, likes, actions, ob, init,
                      full_out=False):
        pos_pred = np.squeeze(seq_pred[:, :2])
        or_pred = np.squeeze(seq_pred[:, 2])
        l_pred = np.squeeze(seq_pred[:, 3])
        mu_pred = np.squeeze(seq_pred[:, 4])
        cp_pred = np.squeeze(seq_pred[:, 5:7])
        n_pred = np.squeeze(seq_pred[:, 7:9])
        s_pred = np.squeeze(seq_pred[:, 9])
        vis = vis / np.max(vis)

        if z is not None:
            pos_obs = np.squeeze(z[:, :2])
            or_obs = np.squeeze(z[:, 2])
            r_obs = np.squeeze(z[:, 3:5])
            n_obs = np.squeeze(z[:, 5:7])
            s_obs = np.squeeze(z[:, 7])

        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)
            cx = np.sqrt(np.squeeze(cov_pred[:, 0, 0]))
            cy = np.sqrt(np.squeeze(cov_pred[:, 1, 1]))
            ct = np.sqrt(np.squeeze(cov_pred[:, 2, 2]))
            cl = np.sqrt(np.squeeze(cov_pred[:, 3, 3]))
            cmu = np.sqrt(np.squeeze(cov_pred[:, 4, 4]))
            crx = np.sqrt(np.squeeze(cov_pred[:, 5, 5]))
            cry = np.sqrt(np.squeeze(cov_pred[:, 6, 6]))
            cnx = np.sqrt(np.squeeze(cov_pred[:, 7, 7]))
            cny = np.sqrt(np.squeeze(cov_pred[:, 8, 8]))
            cs = np.sqrt(np.squeeze(cov_pred[:, 9, 9]))
            q_pred = q_pred.reshape(self.sl, self.dim_x, self.dim_x)
            r_pred = r_pred.reshape(self.sl, self.dim_z, self.dim_z)
            qx = np.sqrt(np.squeeze(q_pred[:, 0, 0]))
            qy = np.sqrt(np.squeeze(q_pred[:, 1, 1]))
            qt = np.sqrt(np.squeeze(q_pred[:, 2, 2]))
            ql = np.sqrt(np.squeeze(q_pred[:, 3, 3]))
            qmu = np.sqrt(np.squeeze(q_pred[:, 4, 4]))
            qrx = np.sqrt(np.squeeze(q_pred[:, 5, 5]))
            qry = np.sqrt(np.squeeze(q_pred[:, 6, 6]))
            qnx = np.sqrt(np.squeeze(q_pred[:, 7, 7]))
            qny = np.sqrt(np.squeeze(q_pred[:, 8, 8]))
            qs = np.sqrt(np.squeeze(q_pred[:, 9, 9]))
            rx = np.sqrt(np.squeeze(r_pred[:, 0, 0]))
            ry = np.sqrt(np.squeeze(r_pred[:, 1, 1]))
            rt = np.sqrt(np.squeeze(r_pred[:, 2, 2]))
            rrx = np.sqrt(np.squeeze(r_pred[:, 3, 3]))
            rry = np.sqrt(np.squeeze(r_pred[:, 4, 4]))
            rnx = np.sqrt(np.squeeze(r_pred[:, 5, 5]))
            rny = np.sqrt(np.squeeze(r_pred[:, 6, 6]))
            rs = np.sqrt(np.squeeze(r_pred[:, 7, 7]))

        fig, ax = plt.subplots(2, 3, figsize=[20, 15])
        ts = np.arange(pos_pred.shape[0])
        ax[0, 0].plot(ts, pos_pred[:, 0], '-r', label='x predicted')
        ax[0, 0].plot(ts, seq[:, 0], '--g', label='x true')
        ax[0, 0].plot(ts, pos_obs[:, 0], 'kx', label='x observed')
        ax[0, 0].plot(ts, pos_pred[:, 1], '-m', label='y predicted')
        ax[0, 0].plot(ts, seq[:, 1], '--c', label='y true')
        ax[0, 0].plot(ts, pos_obs[:, 1], 'ko', label='y observed')
        ax[0, 0].set_title('position')
        ax[0, 0].legend()
        ax[0, 1].plot(ts, or_pred, '-r', label='predicted')
        ax[0, 1].plot(ts, seq[:, 2], '--g', label='true')
        ax[0, 1].plot(ts, or_obs, 'kx', label='observed')
        ax[0, 1].set_title('heading')
        ax[0, 1].legend()
        ax[0, 2].plot(ts, cp_pred[:, 0], '-r', label='x predicted')
        ax[0, 2].plot(ts, seq[:, 5], '--g', label='x true')
        ax[0, 2].plot(ts, r_obs[:, 0], 'kx', label='x observed')
        ax[0, 2].plot(ts, cp_pred[:, 1], '-m', label='y predicted')
        ax[0, 2].plot(ts, seq[:, 6], '--c', label='y true')
        ax[0, 2].plot(ts, r_obs[:, 1], 'ko', label='y observed')
        ax[0, 2].set_title('contact point')
        ax[0, 2].legend()

        ax[1, 2].plot(ts, n_pred[:, 0], '-r', label='x predicted')
        ax[1, 2].plot(ts, seq[:, 7], '--g', label='x true')
        ax[1, 2].plot(ts, n_obs[:, 0], 'kx', label='x observed')
        ax[1, 2].plot(ts, n_pred[:, 1], '-m', label='y predicted')
        ax[1, 2].plot(ts, seq[:, 8], '--c', label='y true')
        ax[1, 2].plot(ts, n_obs[:, 1], 'ko', label='y observed')
        ax[1, 2].set_title('normal')
        ax[1, 2].legend()

        ax[1, 0].plot(ts, mu_pred, '-r', label='mu predicted')
        ax[1, 0].plot(ts, seq[:, 4], '--g', label='mu true')
        ax[1, 0].plot(ts, l_pred, '-m', label='l predicted')
        ax[1, 0].plot(ts, seq[:, 3], '--c', label='l true')
        ax[1, 0].set_title('friction')
        ax[1, 0].legend()
        ax[1, 1].plot(ts, s_pred, '-r', label='predicted')
        ax[1, 1].plot(ts, seq[:, 9], '--g', label='true')
        ax[1, 1].plot(ts, s_obs, 'kx', label='observed')
        ax[1, 1].plot(ts, vis, '-b', label='visibility')
        ax[1, 1].set_title('contact')
        ax[1, 1].legend()

        if cov_pred is not None:
            ax[0, 0].fill_between(ts, pos_pred[:, 0] - cx,
                                  pos_pred[:, 0] + cx, color="lightblue")
            ax[0, 0].fill_between(ts, pos_pred[:, 1] - cy,
                                  pos_pred[:, 1] + cy, color="lightblue")
            ax[0, 1].fill_between(ts, (or_pred - ct), (or_pred + ct),
                                  color="lightblue")
            ax[0, 2].fill_between(ts, cp_pred[:, 0] - crx,
                                  cp_pred[:, 0] + crx, color="lightblue")
            ax[0, 2].fill_between(ts, cp_pred[:, 1] - cry,
                                  cp_pred[:, 1] + cry, color="lightblue")
            ax[1, 0].fill_between(ts, (l_pred - cl),
                                  (l_pred + cl), color="lightblue")
            ax[1, 0].fill_between(ts, mu_pred - cmu,
                                  mu_pred + cmu, color="lightblue")
            ax[1, 1].fill_between(ts, (s_pred - cs), (s_pred + cs),
                                  color="lightblue")
            ax[1, 2].fill_between(ts, n_pred[:, 0] - cnx,
                                  n_pred[:, 0] + cnx, color="lightblue")
            ax[1, 2].fill_between(ts, n_pred[:, 1] - cny,
                                  n_pred[:, 1] + cny, color="lightblue")

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85,
                            wspace=0.1, hspace=0.3)

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking"),
                    bbox_inches="tight")

        # plot the noise estimates
        fig, ax = plt.subplots(2, 3, figsize=[20, 15])
        ts = np.arange(pos_pred.shape[0])
        sc = np.max([np.max(qx), np.max(qy), np.max(rx), np.max(ry)])
        sc = max(1., sc)
        ax[0, 0].plot(ts, qx, '-r', label='qx')
        ax[0, 0].plot(ts, rx, '--g', label='rx')
        ax[0, 0].plot(ts, qy, '-m', label='qy')
        ax[0, 0].plot(ts, ry, '--c', label='ry')
        ax[0, 0].plot(ts, vis*sc, '-b', label='visibility')
        ax[0, 0].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[0, 0].set_title('position')
        ax[0, 0].legend()

        sc = np.max([np.max(qt), np.max(rt)])
        sc = max(1., sc)
        ax[0, 1].plot(ts, qt, '-r', label='q')
        ax[0, 1].plot(ts, rt, '--g', label='r')
        ax[0, 1].plot(ts, vis*sc, '-b', label='visibility')
        ax[0, 1].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[0, 1].set_title('heading')
        ax[0, 1].legend()

        sc = np.max([np.max(qrx), np.max(qry), np.max(rrx), np.max(rry)])
        sc = max(1., sc)
        ax[0, 2].plot(ts, qrx, '-r', label='qx')
        ax[0, 2].plot(ts, rrx, '--g', label='rx')
        ax[0, 2].plot(ts, qry, '-m', label='qy')
        ax[0, 2].plot(ts, rry, '--c', label='ry')
        ax[0, 2].plot(ts, vis*sc, '-b', label='visibility')
        ax[0, 2].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[0, 2].set_title('contact point')
        ax[0, 2].legend()

        sc = np.max([np.max(qnx), np.max(qny), np.max(rnx), np.max(rny)])
        sc = max(1., sc)
        ax[1, 2].plot(ts, qnx, '-r', label='qx')
        ax[1, 2].plot(ts, rnx, '--g', label='rx')
        ax[1, 2].plot(ts, qny, '-m', label='qy')
        ax[1, 2].plot(ts, rny, '--c', label='ry')
        ax[1, 2].plot(ts, vis*sc, '-b', label='visibility')
        ax[1, 2].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[1, 2].set_title('normal')
        ax[1, 2].legend()

        sc = np.max([np.max(qmu), np.max(ql)])
        sc = max(1., sc)
        ax[1, 0].plot(ts, qmu, '-r', label='qmu')
        ax[1, 0].plot(ts, ql, '-m', label='ql')
        ax[1, 0].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[1, 0].set_title('friction')
        ax[1, 0].legend()

        sc = np.max([np.max(qs), np.max(rs)])
        sc = max(1., sc)
        ax[1, 1].plot(ts, qs, '-r', label='q')
        ax[1, 1].plot(ts, rs, '--g', label='r')
        ax[1, 1].plot(ts, vis*sc, '-b', label='visibility')
        ax[1, 1].plot(ts, seq[:, 9]*sc, '-k', label='contact')
        ax[1, 1].set_title('contact')
        ax[1, 1].legend()

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85,
                            wspace=0.1, hspace=0.3)

        fig.savefig(os.path.join(out_dir, str(num) + "_noise"),
                    bbox_inches="tight")

        log_file = open(os.path.join(out_dir, str(num) + '_seq.csv'), 'w')

        keys = ['t', 'x', 'y', 'or', 'l', 'mu', 'rx', 'ry', 'nx', 'ny', 's',
                'x_p', 'y_p', 'or_p', 'l_p', 'mu_p', 'rx_p', 'ry_p', 'nx_p',
                'ny_p', 's_p']
        if cov_pred is not None and z is not None:
            keys += ['x_c', 'y_c', 'or_c', 'l_c', 'mu_c', 'rx_c', 'ry_c',
                     'nx_c', 'ny_c', 's_c', 'x_ob', 'y_ob', 'or_ob', 'rx_ob',
                     'ry_ob', 'nx_ob', 'ny_ob', 's_ob']
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                row = {'x': seq[t, 0], 'y': seq[t, 1], 'or': seq[t, 2],
                       'l': seq[t, 3], 'mu': seq[t, 4], 'rx': seq[t, 5],
                       'ry': seq[t, 6], 'nx': seq[t, 7], 'ny': seq[t, 8],
                       's': seq[t, 9],
                       'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                       'or_p': seq_pred[t, 2], 'l_p': seq_pred[t, 3],
                       'mu_p': seq_pred[t, 4], 'rx_p': seq_pred[t, 5],
                       'ry_p': seq_pred[t, 6], 'nx_p': seq_pred[t, 7],
                       'ny_p': seq_pred[t, 8], 's_p': seq_pred[t, 9],
                       'x_c': cx[t], 'y_c': cy[t], 'or_c': ct[t], 'l_c': cl[t],
                       'mu_c': cmu[t], 'rx_c': crx[t], 'ry_c': cry[t],
                       'nx_c': cnx[t], 'ny_c': cny[t], 's_c': cs[t],
                       'x_ob': pos_obs[t, 0], 'y_ob': pos_obs[t, 1],
                       'or_ob': or_obs[t], 'rx_ob': r_obs[t, 0],
                       'ry_ob': r_obs[t, 1], 'nx_ob': n_obs[t, 0],
                       'ny_ob': n_obs[t, 1], 's_ob': s_obs[t]}
                log.writerow(row)
        else:
            log = csv.DictWriter(log_file, keys)
            log.writeheader()
            for t in ts:
                row = {'x': seq[t, 0], 'y': seq[t, 1], 'or': seq[t, 2],
                       'l': seq[t, 3], 'mu': seq[t, 4], 'rx': seq[t, 5],
                       'ry': seq[t, 6], 'nx': seq[t, 7], 'ny': seq[t, 8],
                       's': seq[t, 9],
                       'x_p': seq_pred[t, 0], 'y_p': seq_pred[t, 1],
                       'or_p': seq_pred[t, 2], 'l_p': seq_pred[t, 3],
                       'mu_p': seq_pred[t, 4], 'rx_p': seq_pred[t, 5],
                       'ry_p': seq_pred[t, 6], 'nx_p': seq_pred[t, 7],
                       'ny_p': seq_pred[t, 8], 's_p': seq_pred[t, 9]}
                log.writerow(row)
        log_file.close()

        # save debug output
        if full_out:
            name = os.path.join(out_dir, str(num))
            np.save(name + '_init', init)
            np.save(name + '_true', seq)
            np.save(name + '_pred', seq_pred)
            np.save(name + '_obs', z)
            np.save(name + '_c', cov_pred)
            np.save(name + '_q', q_pred)
            np.save(name + '_r', r_pred)
            np.save(name + '_vis', vis)
            np.save(name + '_u', actions)
            np.save(name + '_ob', ob)

    def plot_trajectory(self, particles, weights, seq, cov_pred, seq_pred,
                        ob, out_dir, num):
        if particles is not None:
            particles = particles.reshape(self.sl, -1, self.dim_x)
            weights = weights.reshape(self.sl, -1)
        if cov_pred is not None:
            cov_pred = cov_pred.reshape(self.sl, self.dim_x, self.dim_x)

        # get the object shape (deal with some encoding problems)
        ob = np.asscalar(ob).decode("utf-8").replace('\0', '')
        if 'rect' in ob:
            # c-----d
            # |     |
            # a-----b
            # get the positions of the corner points
            if '1' in ob:
                points = [[-0.045, -0.045], [0.045, -0.045],
                          [0.045, 0.045], [-0.045, 0.045]]
            if '2' in ob:
                points = [[-0.044955, -0.05629], [0.044955, -0.05629],
                          [0.044955, 0.05629], [-0.044955, 0.05629]]
            if '3' in ob:
                points = [[-0.067505, -0.04497], [0.067505, -0.04497],
                          [0.067505, 0.04497], [-0.067505, 0.04497]]
        elif 'tri' in ob:
            # b ----- a
            #         |
            #         |
            #         c
            # get the positions of the points
            if '1' in ob:
                points = [[0.045, 0.045], [-0.0809, 0.045], [0.045, -0.08087]]
            if '2' in ob:
                points = [[0.045, 0.045], [-0.106, 0.045], [0.045, -0.08087]]
            if '3' in ob:
                points = [[0.045, 0.045], [-0.1315, 0.045], [0.045, -0.08061]]
        elif 'ellip' in ob:
            if '1' in ob:
                a = 0.0525
                b = 0.0525
            elif '2' in ob:
                a = 0.0525
                b = 0.065445
            elif '3' in ob:
                a = 0.0525
                b = 0.0785
        elif 'hex' in ob:
            points = []
            for i in range(6):
                theta = (np.pi/3)*i
                points += [[0.06050*np.cos(theta),
                            0.06050*np.sin(theta)]]
        elif 'butter' in ob:
            points = self.butter_points[:]

        pos_pred = np.squeeze(seq_pred[:, :2])
        minx = min(np.min(seq[:, 0]), np.min(pos_pred[:, 0]))
        miny = min(np.min(seq[:, 1]), np.min(pos_pred[:, 1]))
        maxx = max(np.max(seq[:, 0]), np.max(pos_pred[:, 0]))
        maxy = max(np.max(seq[:, 1]), np.max(pos_pred[:, 1]))

        fig, ax = plt.subplots(figsize=[15, 15])
        ax.set_aspect('equal')
        fig2, ax2 = plt.subplots(figsize=[17, 17])
        ax2.set_aspect('equal')
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
                ax2.plot(seq[i, 0], seq[i, 1], 'cx', markersize=15.,
                         label='start')
                # plot the mean trajectory
                ax2.plot([pos_pred[i, 0], pos_pred[i+1, 0]],
                         [pos_pred[i, 1], pos_pred[i+1, 1]], '-r',
                         label='predicted')

                # plot the real trajectory
                ax2.plot([seq[i, 0], seq[i+1, 0]], [seq[i, 1], seq[i+1, 1]],
                         '-g', label='true')
            else:
                # plot the mean trajectory
                ax.plot([pos_pred[i, 0], pos_pred[i+1, 0]],
                        [pos_pred[i, 1], pos_pred[i+1, 1]], '-r')

                # plot the real trajectory
                ax.plot([seq[i, 0], seq[i+1, 0]],
                        [seq[i, 1], seq[i+1, 1]], '-g')
                # plot the mean trajectory
                ax2.plot([pos_pred[i, 0], pos_pred[i+1, 0]],
                         [pos_pred[i, 1], pos_pred[i+1, 1]], '-r')

                # plot the real trajectory
                ax2.plot([seq[i, 0], seq[i+1, 0]],
                         [seq[i, 1], seq[i+1, 1]], '-g')
            # plot the mean trajectory
            ax.plot(pos_pred[i, 0], pos_pred[i, 1], 'ro')
            ax.plot(seq[i, 0], seq[i, 1], 'go')

            if i % 5 == 0:
                if 'ellip' in ob:
                    ax2.add_artist(Ellipse((pos_pred[i, 0], pos_pred[i, 1]),
                                           2*a*1000, 2*b*1000, seq_pred[i, 2],
                                           alpha=0.1, facecolor='r',
                                           edgecolor='r'))
                    ax2.add_artist(Ellipse((seq[i, 0], seq[i, 1]),
                                           2*a*1000, 2*b*1000, seq[i, 2],
                                           alpha=0.1, facecolor='g',
                                           edgecolor='g'))
                else:
                    r_p = np.zeros((2, 2))
                    r_pred = seq_pred[i, 2]*np.pi/180.
                    r_p[0, 0] = np.cos(r_pred)
                    r_p[0, 1] = -np.sin(r_pred)
                    r_p[1, 0] = np.sin(r_pred)
                    r_p[1, 1] = np.cos(r_pred)
                    r_l = np.zeros((2, 2))
                    r_la = seq[i, 2]*np.pi/180.
                    r_l[0, 0] = np.cos(r_la)
                    r_l[0, 1] = -np.sin(r_la)
                    r_l[1, 0] = np.sin(r_la)
                    r_l[1, 1] = np.cos(r_la)
                    points_p = []
                    points_l = []
                    for p in points:
                        # rotate and translate the points according to the
                        # object's pose
                        pt = np.array(p).reshape(2, 1) * 1000
                        points_p += [np.dot(r_p, pt).reshape(2)+pos_pred[i]]
                        points_l += [np.dot(r_l, pt).reshape(2)+seq[i, :2]]
                    ax2.add_artist(Polygon(points_p, alpha=0.1, facecolor='r',
                                           edgecolor='r'))
                    ax2.add_artist(Polygon(points_l, alpha=0.1, facecolor='g',
                                           edgecolor='g'))

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
            # plot the particles with colour depending on weight
            ax.scatter(par[:20, 0], par[:20, 1],
                       c=wei[:20], cmap='jet', marker='x', alpha=0.5)

        fig.savefig(os.path.join(out_dir, str(num) + "_tracking_2d"),
                    bbox_inches="tight")
        ax2.set_xlim([minx-100, maxx+100])
        ax2.set_ylim([miny-100, maxy+100])
        fig2.savefig(os.path.join(out_dir, str(num) + "_tracking_vis"),
                     bbox_inches="tight")


class SegmentationLayer(BaseLayer):
    def __init__(self, batch_size, normalize, summary, trainable):
        super(SegmentationLayer, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.normalize = normalize

        # load a plane image for reprojecting
        path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(path, 'plane_image_smaller.npy')

        self.plane_depth = \
            tf.convert_to_tensor(np.load(path))[None, :, :, None]
        self.plane_depth = tf.tile(self.plane_depth,
                                   [self.batch_size, 1, 1, 1])

        # segmenting the image
        self.im_c1 = self._conv_layer('segment/conv1', 7, 8,
                                      trainable=trainable)
        self.im_c2 = self._conv_layer('segment/conv2', 5, 16,
                                      trainable=trainable)
        self.im_c3 = self._conv_layer('segment/conv3', 3, 32,
                                      trainable=trainable)
        self.im_d1 = self._deconv_layer('segment/deconv1', 13, 16,
                                        trainable=trainable)
        self.im_d2 = self._deconv_layer('segment/deconv2', 3, 8,
                                        trainable=trainable)
        self.im_d3 = self._deconv_layer('segment/deconv3', 3, 1,
                                        activation=None, trainable=trainable)

        if self.normalize == 'layer':
            self.im_n1 =\
                tf.keras.layers.LayerNormalization(name='segment/norm1',
                                                   trainable=trainable)
            self.im_n2 =\
                tf.keras.layers.LayerNormalization(name='segment/norm2',
                                                   trainable=trainable)
            self.im_n3 =\
                tf.keras.layers.LayerNormalization(name='segment/norm3',
                                                   trainable=trainable)
            self.im_n4 = \
                tf.keras.layers.LayerNormalization(name='segment/norm4',
                                                   trainable=trainable)
            self.im_n5 = \
                tf.keras.layers.LayerNormalization(name='segment/norm5',
                                                   trainable=trainable)
        elif self.normalize == 'batch':
            self.im_n1 =\
                tf.keras.layers.BatchNormalization(name='segment/norm1',
                                                   trainable=trainable)
            self.im_n2 =\
                tf.keras.layers.BatchNormalization(name='segment/norm2',
                                                   trainable=trainable)
            self.im_n3 =\
                tf.keras.layers.BatchNormalization(name='segment/norm3',
                                                   trainable=trainable)
            self.im_n4 = \
                tf.keras.layers.BatchNormalization(name='segment/norm4',
                                                   trainable=trainable)
            self.im_n5 = \
                tf.keras.layers.BatchNormalization(name='segment/norm5',
                                                   trainable=trainable)
            self.updateable = [self.im_n1, self.im_n2, self.im_n3, self.im_n4,
                               self.im_n5]

    def call(self, inputs, training):
        # unpack the inputs
        images = inputs[:, :, :, 0:3]
        coords = inputs[:, :, :, 3:]

        height = images.get_shape()[1].value
        width = images.get_shape()[2].value
        # disable the topmost name scope so that the summaries don't end up all
        # under one tab in tensorbaord
        with tf.name_scope(""):
            # segment the image
            with tf.name_scope('segment'):
                conv1 = self.im_c1(inputs)
                conv1 = tf.nn.max_pool2d(conv1, 3, 2, padding='SAME')
                if self.normalize == 'layer':
                    conv1 = self.im_n1(conv1)
                elif self.normalize == 'batch':
                    conv1 = self.im_n1(conv1, training)
                conv2 = self.im_c2(conv1)
                conv2 = tf.nn.max_pool2d(conv2, 3, 2, padding='SAME')
                if self.normalize == 'layer':
                    conv2 = self.im_n2(conv2)
                elif self.normalize == 'batch':
                    conv2 = self.im_n2(conv2, training)
                conv3 = self.im_c3(conv2)
                conv3 = tf.nn.max_pool2d(conv3, 5, 4, padding='SAME')
                if self.normalize == 'layer':
                    conv3 = self.im_n3(conv3)
                elif self.normalize == 'batch':
                    conv3 = self.im_n3(conv3, training)
                deconv1 = self.im_d1(conv3)
                deconv1 = tf.image.resize(deconv1, conv2.get_shape()[1:3])
                deconv1 = deconv1 + conv2
                if self.normalize == 'layer':
                    deconv1 = self.im_n4(deconv1)
                elif self.normalize == 'batch':
                    deconv1 = self.im_n4(deconv1, training)
                deconv2 = self.im_d2(deconv1)
                deconv2 = tf.image.resize(deconv2, [height // 2, width // 2])
                if self.normalize == 'layer':
                    deconv2 = self.im_n5(deconv2)
                elif self.normalize == 'batch':
                    deconv2 = self.im_n5(deconv2, training)
                mask_out = self.im_d3(deconv2)
                mask = tf.image.resize(mask_out, [height, width])
                if self.summary:
                    if self.normalize == 'batch':
                        tf.summary.histogram('n1_mean', self.im_n1.moving_mean)
                        tf.summary.histogram('n1_var',
                                             self.im_n1.moving_variance)
                    tf.summary.image('rgb', images[:, :, :, :3])
                    tf.summary.image('depth', coords[:, :, :, -1:])
                    tf.summary.image('conv1_im', conv1[0:1, :, :, 0:1])
                    tf.summary.histogram('conv1_out', conv1)
                    tf.summary.image('conv2_im', conv2[0:1, :, :, 0:1])
                    tf.summary.histogram('conv2_out', conv2)
                    tf.summary.image('conv3_im', conv3[0:1, :, :, 0:1])
                    tf.summary.histogram('conv3_out', conv3)
                    tf.summary.image('deconv1_im', deconv1[0:1, :, :, 0:1])
                    tf.summary.histogram('deconv1_out', deconv1)
                    tf.summary.image('deconv2_im', deconv2[0:1, :, :, 0:1])
                    tf.summary.histogram('deconv2_out', deconv2)
                    tf.summary.image('mask', mask_out[0:1])

            # predict the object position
            pos_pix = self._spatial_softmax(mask, 'pos', scale=1.,
                                            method='softmax',
                                            summary=self.summary)
            pos_pix = tf.reshape(pos_pix, [self.batch_size, 2])
            pos = utils._to_3d(pos_pix, self.plane_depth)

            # extract the glimpses for rotation estimation and parameter
            # estimation
            coords_rot = tf.concat([pos_pix[:, 1:2] * 2, pos_pix[:, 0:1] * 2],
                                   axis=1)
            glimpse_rot = \
                tf.image.extract_glimpse(images, size=[72, 72],
                                         offsets=coords_rot,
                                         centered=True, normalized=False)

        return [mask_out, pos, glimpse_rot], pos_pix


class SensorLayer(BaseLayer):
    def __init__(self, batch_size, normalize, scale, summary, trainable):
        super(SensorLayer, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.scale = scale
        self.normalize = normalize

        # load a plane image for reprojecting
        path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(path, 'plane_image_smaller.npy')

        self.plane_depth = \
            tf.convert_to_tensor(np.load(path))[None, :, :, None]
        self.plane_depth = tf.tile(self.plane_depth,
                                   [self.batch_size, 1, 1, 1])

        # processing the glimpse
        self.g_c1 = self._conv_layer('glimpse/conv1', 3, 8,
                                     trainable=trainable)
        self.g_c2 = self._conv_layer('glimpse/conv2', 3, 16,
                                     trainable=trainable)
        self.g_c3 = self._conv_layer('glimpse/conv2', 3, 32,
                                     trainable=trainable)

        self.g_fc1 = self._fc_layer('glimpse/r_fc1', 128, trainable=trainable)
        self.g_rfc2 = self._fc_layer('glimpse/r_fc2', 64, trainable=trainable)
        self.g_r = self._fc_layer('glimpse/r', 2, activation=None,
                                  trainable=trainable)
        self.g_nfc2 = self._fc_layer('glimpse/n_fc2', 64, trainable=trainable)
        self.g_n = self._fc_layer('glimpse/n', 2, activation=None,
                                  trainable=trainable)
        self.g_s = self._fc_layer('glimpse/s', 1, activation=None, bias=-0.1,
                                  trainable=trainable)

        # get the rotation
        self.r_c1 = self._conv_layer('rot/conv1', 3, 32, trainable=trainable)
        self.r_c2 = self._conv_layer('rot/conv2', 3, 64, trainable=trainable)
        self.r_fc1 = self._fc_layer('rot/fc1', 128, trainable=trainable)
        self.r_fc2 = self._fc_layer('rot/fc2', 64, trainable=trainable)
        self.r_rot = self._fc_layer('rot/rot', 1, activation=None,
                                    trainable=trainable)

        if self.normalize == 'layer':
            self.g_n1 = \
                tf.keras.layers.LayerNormalization(name='glimpse/norm1',
                                                   trainable=trainable)
            self.g_n2 = \
                tf.keras.layers.LayerNormalization(name='glimpse/norm2',
                                                   trainable=trainable)
            self.g_n3 = \
                tf.keras.layers.LayerNormalization(name='glimpse/norm3',
                                                   trainable=trainable)
            self.r_n1 = \
                tf.keras.layers.LayerNormalization(name='rot/norm1',
                                                   trainable=trainable)
            self.r_n2 = \
                tf.keras.layers.LayerNormalization(name='rot/norm2',
                                                   trainable=trainable)
        elif self.normalize == 'batch':
            self.g_n1 = \
                tf.keras.layers.BatchNormalization(name='glimpse/norm1',
                                                   trainable=trainable)
            self.g_n2 = \
                tf.keras.layers.BatchNormalization(name='glimpse/norm2',
                                                   trainable=trainable)
            self.g_n3 = \
                tf.keras.layers.BatchNormalization(name='glimpse/norm3',
                                                   trainable=trainable)
            self.r_n1 = \
                tf.keras.layers.BatchNormalization(name='rot/norm1',
                                                   trainable=trainable)
            self.r_n2 = \
                tf.keras.layers.BatchNormalization(name='rot/norm2',
                                                   trainable=trainable)
            self.updateable = [self.g_n1, self.g_n2, self.g_n3, self.r_n1,
                               self.r_n2]

    def call(self, inputs, training):
        # unpack the inputs
        pc, tip_pos, tip_pix, tip_pix_end, start_glimpse, mask, pos, \
            glimpse_rot = inputs
        # unpack the inputs
        image = pc[:, :, :, 0:3]
        coord = pc[:, :, :, 3:]

        # disable the topmost name scope so that the summaries don't end up all
        # under one tab in tensorbaord
        with tf.name_scope(""):
            # predict the orientation
            with tf.name_scope('rot'):
                # in_data = tf.concat([glimpse_rot, start_glimpse], axis=-1)
                in_data = start_glimpse - glimpse_rot
                rot_conv1 = self.r_c1(in_data)
                if self.normalize == 'layer':
                    rot_conv1 = self.r_n1(rot_conv1)
                elif self.normalize == 'batch':
                    rot_conv1 = self.r_n1(rot_conv1, training)
                rot_conv1 = tf.nn.max_pool2d(rot_conv1, 3, 2, padding='VALID')

                rot_conv2 = self.r_c2(rot_conv1)
                if self.normalize == 'layer':
                    rot_conv2 = self.r_n2(rot_conv2)
                elif self.normalize == 'batch':
                    rot_conv2 = self.r_n2(rot_conv2, training)
                rot_conv2 = tf.nn.max_pool2d(rot_conv2, 3, 2, padding='VALID')
                rot_fc1 = self.r_fc1(tf.reshape(rot_conv2,
                                                [self.batch_size, -1]))
                rot_fc2 = self.r_fc2(rot_fc1)
                rot = self.r_rot(rot_fc2)
                if self.summary:
                    tf.summary.image('glimpse_rot',
                                     glimpse_rot[0:1, :, :, :3])
                    tf.summary.image('glimpse_start',
                                     start_glimpse[0:1, :, :, :3])
                    tf.summary.image('conv1_im', rot_conv1[0:1, :, :, 0:1])
                    tf.summary.histogram('conv1_out', rot_conv1)
                    tf.summary.image('conv2_im', rot_conv2[0:1, :, :, 0:1])
                    tf.summary.histogram('conv2_out', rot_conv2)
                    tf.summary.histogram('fc1_out', rot_fc1)
                    tf.summary.histogram('fc2_out', rot_fc2)
                    tf.summary.histogram('rot_out', rot)

            # process the glimpse
            with tf.name_scope('glimpse'):
                tip_pix_x = tf.slice(tip_pix, [0, 0], [-1, 1]) * 2
                tip_pix_y = tf.slice(tip_pix, [0, 1], [-1, 1]) * 2
                coords = tf.concat([tip_pix_y, tip_pix_x], axis=1)
                glimpse = \
                    tf.image.extract_glimpse(coord, size=[64, 64],
                                             offsets=coords,
                                             centered=True, normalized=False)
                im_glimpse = \
                    tf.image.extract_glimpse(image, size=[64, 64],
                                             offsets=coords,
                                             centered=True, normalized=False)
                # subtract the tip pose to normalize the z coordinates
                glimpse -= tip_pos[:, None, None, :]
                in_g = tf.concat([im_glimpse, glimpse], axis=-1)

                g_conv1 = self.g_c1(in_g)
                g_conv1 = tf.nn.max_pool2d(g_conv1, 3, 2, padding='VALID')
                if self.normalize == 'layer':
                    g_conv1 = self.g_n1(g_conv1)
                elif self.normalize == 'batch':
                    g_conv1 = self.g_n1(g_conv1, training)
                g_conv2 = self.g_c2(g_conv1)
                g_conv2 = tf.nn.max_pool2d(g_conv2, 3, 2, padding='VALID')
                if self.normalize == 'layer':
                    g_conv2 = self.g_n2(g_conv2)
                elif self.normalize == 'batch':
                    g_conv2 = self.g_n2(g_conv2, training)
                g_conv3 = self.g_c3(g_conv2)
                # g_conv3 = tf.nn.max_pool2d(g_conv3, 3, 2, padding='VALID')
                if self.normalize == 'layer':
                    g_conv3 = self.g_n3(g_conv3)
                elif self.normalize == 'batch':
                    g_conv3 = self.g_n3(g_conv3, training)
                glimpse_encoding = tf.reshape(g_conv3, [self.batch_size, -1])

                # add the action
                pix_u = tf.concat([tip_pix_end - tip_pix, tip_pix], axis=1)
                glimpse_encoding = tf.concat([glimpse_encoding, pix_u],
                                             axis=-1)

                # extract contact point and push velocity from the glimpse
                g_fc1 = self.g_fc1(glimpse_encoding)
                g_rfc2 = self.g_rfc2(g_fc1)
                r_pix = self.g_r(g_rfc2)
                # add the tip's global postition to the local estimate and
                # transform to 2d (using the tip's depth if necessary)
                r_pix = r_pix + tip_pix
                # r = utils._to_3d(r_pix, self.plane_depth)
                r = utils._to_3d_d(r_pix, coord[:, :, :, -1:], tip_pos)

                g_nfc2 = self.g_nfc2(g_fc1)
                n_pix = self.g_n(g_nfc2)
                # calculate the pixel end point to get the z-value
                # for projecting the predicted normal from pixels to 3d
                n_end_pix = tf.stop_gradient(r_pix) + n_pix
                # n_end = utils._to_3d(n_end_pix, self.plane_depth)
                n_end = utils._to_3d_d(n_end_pix, coord[:, :, :, -1:],
                                       tip_pos)
                n = n_end - tf.stop_gradient(r)

                # get the contact annotation
                s = self.g_s(glimpse_encoding)
                s = tf.nn.sigmoid(s)
                # here we have to adapt the observations to the scale, since
                # the network can't learn it itself due to the sigmoid
                s = s / self.scale

                if self.summary:
                    tf.summary.image('glimpse_z', glimpse[0:1, :, :, -1:])
                    tf.summary.image('glimpse_rgb', im_glimpse[0:1])
                    tf.summary.image('conv1_im', g_conv1[0:1, :, :, 0:1])
                    tf.summary.histogram('conv1_out', g_conv1)
                    tf.summary.image('conv2_im', g_conv2[0:1, :, :, 0:1])
                    tf.summary.histogram('conv2_out', g_conv2)
                    tf.summary.image('conv3_im', g_conv3[0:1, :, :, 0:1])
                    tf.summary.histogram('g_fc1_out', g_fc1)
                    tf.summary.histogram('g_rfc2_out', g_rfc2)
                    tf.summary.histogram('r_pix_out', r_pix)
                    tf.summary.histogram('g_nfc2_out', g_nfc2)
                    tf.summary.histogram('n_pix_out', n_pix)
                    tf.summary.histogram('n_end_pix_out', n_end_pix)

            # assemble the observations: remove the z(up) coordinates,
            # convert to centimeter, normalize
            n_norm = tf.linalg.norm(n[:, :2], axis=1, keepdims=True)
            n = tf.where(tf.greater(tf.squeeze(n_norm), 1e-5),
                         n[:, :2] / n_norm, n[:, :2])
            n = tf.where(tf.greater_equal(tf.tile(s, [1, 2]), 0.5), n, 0 * n)

            # we only care for the position in the table plane
            r = r[:, :2] * 1000. / self.scale
            n = n[:, :2] / self.scale
            pos = pos[:, :2] * 1000. / self.scale
            z = tf.concat([pos, rot, r, n, s], axis=-1)

        if self.summary:
            tf.summary.scalar('r_x', r[0, 0])
            tf.summary.scalar('r_y', r[0, 1])
            tf.summary.scalar('n_x', n[0, 0])
            tf.summary.scalar('n_y', n[0, 1])
            tf.summary.scalar('o_x', pos[0, 0])
            tf.summary.scalar('o_y', pos[0, 1])
            tf.summary.scalar('t_x', tip_pos[0, 0])
            tf.summary.scalar('t_y', tip_pos[0, 1])
            tf.summary.scalar('s', s[0, 0])
            tf.summary.scalar('rot', rot[0, 0])

        return z, [mask, rot_fc2, g_fc1]


class ObservationNoise(BaseLayer):
    def __init__(self, batch_size, dim_z, r_diag, scale, hetero, diag,
                 trainable, summary):
        super(ObservationNoise, self).__init__()
        self.hetero = hetero
        self.diag = diag
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.scale = scale
        self.r_diag = r_diag
        self.summary = summary
        self.trainable = trainable

    def build(self, input_shape):
        init_const = np.ones(self.dim_z) * 1e-3 // self.scale**2
        init = np.sqrt(np.maximum(np.square(self.r_diag) - init_const, 0))
        # the constant bias keeps the predicted covariance away from zero
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_z],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))
        num = self.dim_z * (self.dim_z + 1) / 2
        wd = 1e-3 * self.scale**2

        if self.hetero and self.diag:
            # for heteroscedastic noise with diagonal covariance matrix
            # position
            self.het_diag_pos_c1 = self._conv_layer('het_diag_pos_c1', 5, 16,
                                                    stride=[2, 2],
                                                    trainable=self.trainable)
            self.het_diag_pos_c2 = self._conv_layer('het_diag_pos_c2', 3, 32,
                                                    stride=[2, 2],
                                                    trainable=self.trainable)

            self.het_diag_pos_fc1 = self._fc_layer('het_diag_pos_fc1', 64,
                                                   trainable=self.trainable)
            self.het_diag_pos_fc2 = self._fc_layer('het_diag_pos_fc2', 2,
                                                   mean=0, std=1e-3,
                                                   activation=None,
                                                   trainable=self.trainable)
            # rotation, normal, contact point and contact
            self.het_diag_rot_fc = self._fc_layer('het_diag_rot_fc', 1,
                                                  mean=0, std=1e-3,
                                                  activation=None,
                                                  trainable=self.trainable)
            self.het_diag_fc1 = self._fc_layer('het_diag_fc1', 64, std=1e-4,
                                               trainable=self.trainable)
            self.het_diag_fc2 = self._fc_layer('het_diag_fc2', 32, std=1e-3,
                                               trainable=self.trainable)
            self.het_diag_fc3 = self._fc_layer('het_diag_fc3', 5, std=1e-2,
                                               activation=None,
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
            self.het_full_pos_c1 = self._conv_layer('het_full_pos_c1', 5, 16,
                                                    stride=[2, 2],
                                                    trainable=self.trainable)
            self.het_full_pos_c2 = self._conv_layer('het_full_pos_c2', 3, 32,
                                                    stride=[2, 2],
                                                    trainable=self.trainable)

            self.het_full_pos_fc = self._fc_layer('het_full_pos_fc',
                                                  self.dim_z,
                                                  trainable=self.trainable)
            # rotation, normal, contact point and contact
            self.het_full_rot_fc = self._fc_layer('het_full_rot_fc',
                                                  self.dim_z,
                                                  trainable=self.trainable)
            self.het_full_g_fc1 = self._fc_layer('het_full_g_fc1', 64,
                                                 std=1e-3,
                                                 trainable=self.trainable)
            self.het_full_g_fc2 = self._fc_layer('het_full_g_f2', 32,
                                                 trainable=self.trainable)

            self.het_full_fc1 = self._fc_layer('het_full_fc1', 64, std=1e-3,
                                               trainable=self.trainable)
            self.het_full_fc2 = \
                self._fc_layer('het_full_fc2', num,
                               activation=None, trainable=self.trainable)

            self.het_full_init_bias = \
                self.add_weight(name='het_full_init_bias',
                                shape=[self.dim_z], trainable=self.trainable,
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
        mask, rot_encoding, glimpse_encoding, pix = inputs
        if self.hetero and self.diag:
            het_diag_pos_c1 = self.het_diag_pos_c1(mask)
            het_diag_pos_c2 = self.het_diag_pos_c2(het_diag_pos_c1)
            het_diag_pos_c2 = tf.reshape(het_diag_pos_c2,
                                         [self.batch_size, -1])
            het_diag_pos_fc1 = self.het_diag_pos_fc1(het_diag_pos_c2)
            het_diag_pos = self.het_diag_pos_fc2(het_diag_pos_fc1)

            # rotation, normal, contact point and contact
            het_diag_rot = self.het_diag_rot_fc(rot_encoding)
            het_diag_fc1 = self.het_diag_fc1(glimpse_encoding)
            het_diag_fc2 = self.het_diag_fc2(het_diag_fc1)
            het_diag_rns = self.het_diag_fc3(het_diag_fc2)

            diag = tf.concat([het_diag_pos, het_diag_rot, het_diag_rns],
                             axis=-1)
            if self.summary:
                tf.summary.image('het_diag_pos_c1_im',
                                 het_diag_pos_c1[0:1, :, :, 0:1])
                tf.summary.histogram('het_diag_pos_c1_out', het_diag_pos_c1)
                tf.summary.histogram('het_diag_pos_c2_out', het_diag_pos_c2)
                tf.summary.histogram('het_diag_pos_fc1_out', het_diag_pos_fc1)
                tf.summary.histogram('het_diag_pos_fc2_out', het_diag_pos)
                tf.summary.histogram('het_diag_rot_fc_out', het_diag_rot)
                tf.summary.histogram('het_diag_rns_fc1_out', het_diag_fc1)
                tf.summary.histogram('het_diag_rns_fc2_out', het_diag_fc2)
                tf.summary.histogram('het_diag_rns_fc3_out', het_diag_rns)
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
            het_full_pos_c1 = self.het_full_pos_c1(mask)
            het_full_pos_c2 = self.het_full_pos_c2(het_full_pos_c1)
            het_full_pos_c2 = tf.reshape(het_full_pos_c2,
                                         [self.batch_size, -1])
            het_full_pos = self.het_full_pos_fc(het_full_pos_c2)
            # rotation, normal, contact point and contact
            het_full_rot = self.het_full_rot_fc(rot_encoding)
            het_full_g1 = self.het_full_g_fc1(glimpse_encoding)
            het_full_g2 = self.het_full_g_fc2(het_full_g1)

            input_data = tf.concat([het_full_pos, het_full_rot, het_full_g2],
                                   axis=-1)
            het_full_fc1 = self.het_full_fc1(input_data)
            tri = self.het_full_fc2(het_full_fc1)
            if self.summary:
                tf.summary.image('het_full_pos_c1_im',
                                 het_full_pos_c1[0:1, :, :, 0:1])
                tf.summary.histogram('het_full_pos_c1_out', het_full_pos_c1)
                tf.summary.histogram('het_full_pos_c2_out', het_full_pos_c2)
                tf.summary.histogram('het_full_pos_fc_out', het_full_pos)
                tf.summary.histogram('het_full_rot_fc_out', het_full_rot)
                tf.summary.histogram('het_full_g_fc1_out', het_full_g1)
                tf.summary.histogram('het_full_g_fc2_out', het_full_g2)
                tf.summary.histogram('het_full_fc1_out', het_full_fc1)
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

        self.like_pos_c1 = self._conv_layer('like_pos_c1', 5, 16,
                                            stride=[2, 2],
                                            trainable=self.trainable)
        self.like_pos_c2 = self._conv_layer('like_pos_c2', 3, 32,
                                            trainable=self.trainable)

        self.like_pos_fc = self._fc_layer('like_pos_fc', 2*self.dim_z,
                                          trainable=self.trainable)
        # rotation, normal, contact point and contact
        self.like_rot_fc = self._fc_layer('like_rot_fc', 1, self.dim_z,
                                          trainable=self.trainable)
        self.like_rns_fc1 = self._fc_layer('like_rns_fc1', 128,
                                           trainable=self.trainable)
        self.like_rns_fc2 = self._fc_layer('like_rn2_fc2', 5*self.dim_z,
                                           trainable=self.trainable)

        self.fc1 = self._fc_layer('fc1', 128, trainable=trainable)
        self.fc2 = self._fc_layer('fc2', 128, trainable=trainable)
        self.fc3 = self._fc_layer('fc3', 1, trainable=trainable,
                                  activation=tf.nn.sigmoid)

    def call(self, inputs, training):
        # unpack the inputs
        particles, encoding = inputs
        bs = particles.get_shape()[0].value
        num_pred = particles.get_shape()[1].value
        # diff, encoding = inputs
        mask, rot_encoding, glimpse_encoding, pix = encoding

        # preprocess the encodings
        # mask
        pos_c1 = self.like_pos_c1(mask)
        pos_c2 = self.like_pos_c2(pos_c1)
        pos_c2 = tf.reshape(pos_c2, [bs, -1])
        pos_fc = self.like_pos_fc(pos_c2)

        # rotation, normal, contact point and contact
        rot_fc = self.like_rot_fc(rot_encoding)
        rns_fc1 = self.like_rns_fc1(glimpse_encoding)
        rns_fc2 = self.like_rns_fc2(rns_fc1)

        # concatenate and tile the preprocessed encoding
        encoding = tf.concat([pos_fc, rot_fc, rns_fc2], axis=-1)
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
            tf.summary.histogram('pos_c1_out', pos_c1)
            tf.summary.histogram('pos_c2_out', pos_c2)
            tf.summary.histogram('pos_fc_out', pos_fc)
            tf.summary.histogram('rot_fc_out', rot_fc)
            tf.summary.histogram('rns_fc1_out', rns_fc1)
            tf.summary.histogram('rns_fc2_out', rns_fc2)
            tf.summary.histogram('fc1_out', fc1)
            tf.summary.histogram('fc2_out', fc2)
            tf.summary.histogram('like', like)

        return like


class ObservationModel(BaseLayer):
    def __init__(self, dim_z, batch_size):
        super(ObservationModel, self).__init__()
        self.dim_z = dim_z
        self.batch_size = batch_size

    def call(self, inputs, training):
        H = tf.concat(
            [tf.tile(np.array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]],
                              dtype=np.float32), [self.batch_size, 1, 1]),
             tf.tile(np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]],
                              dtype=np.float32), [self.batch_size, 1, 1])],
            axis=1)

        z_pred = tf.concat([inputs[:, :3], inputs[:, 5:]], axis=1)
        return z_pred, H


class ProcessModel(BaseLayer):
    def __init__(self, batch_size, dim_x, scale, learned, jacobian,
                 trainable, summary):
        super(ProcessModel, self).__init__()
        self.summary = summary
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.learned = learned
        self.jacobian = jacobian
        self.scale = scale

        if learned:
            self.fc1 = self._fc_layer('fc1', 256, std=1e-4,
                                      trainable=trainable)
            self.fc2 = self._fc_layer('fc2', 128, trainable=trainable)
            self.fc3 = self._fc_layer('fc3', 128, trainable=trainable)
            self.update = self._fc_layer('fc4', self.dim_x, activation=None,
                                         trainable=trainable)

    def call(self, inputs, training):
        # unpack the inputs
        last_state, actions, ob = inputs

        if self.learned:
            fc1 = self.fc1(tf.concat([last_state, actions[:, :2]], axis=-1))
            fc2 = self.fc2(fc1)
            fc3 = self.fc3(fc2)
            update = self.update(fc3)

            # for the circular object, the orientation is always zero,
            # so we have to set the prediction to 0 and adapt the
            # jacobian
            ob = tf.reshape(ob, [self.batch_size, 1])
            bs = last_state.get_shape()[0]
            ob = tf.tile(ob, [1, bs // self.batch_size])
            ob = tf.reshape(ob, [-1])
            ob = tf.strings.regex_replace(ob, "\000", "")
            ob = tf.strings.regex_replace(ob, "\00", "")

            rot_pred = update[:, 2:3]
            rot_pred = tf.where(tf.equal(ob, 'ellip1'),
                                tf.zeros_like(rot_pred), rot_pred)
            update = tf.concat([update[:, :2], rot_pred, update[:, 3:]],
                               axis=-1)

            new_state = last_state + update
            if self.summary:
                tf.summary.histogram('fc1_out', fc1)
                tf.summary.histogram('fc2_out', fc2)
                tf.summary.histogram('fc3_out', fc3)
                tf.summary.histogram('update_out', update)
            if self.jacobian:
                F = self._compute_jacobian(new_state, last_state)
            else:
                F = None
        else:
            if self.jacobian:
                # with tf.GradientTape() as tape:
                #     tape.watch(last_state)
                #     # split the state into parts and undo the scaling
                #     last_state *= self.scale
                #     pos = last_state[:, :2]
                #     ori = last_state[:, 2:3]
                #     fr = last_state[:, 3:4]
                #     fr_mu = last_state[:, 4:5]
                #     cp = last_state[:, 5:7]
                #     n = last_state[:, 7:9]
                #     s = last_state[:, 9:]

                #     # undo the scaling for the actions as well
                #     actions *= self.scale

                #     # apply the analytical model to get predicted translation
                #     # and rotation
                #     tr_pred, rot_pred, keep_contact = \
                #         utils.physical_model(pos, cp, n, actions, fr, fr_mu, s)

                #     pos_pred = pos + tr_pred
                #     ori_pred = ori + rot_pred * 180.0/np.pi
                #     fr_pred = fr
                #     fr_mu_pred = fr_mu
                #     cp_pred = cp + actions
                #     keep_contact = tf.cast(keep_contact, tf.float32)
                #     n_pred = n * keep_contact
                #     s_pred = s * keep_contact

                #     # piece together the new state and apply scaling again
                #     new_state = \
                #         tf.concat([pos_pred, ori_pred, fr_pred,
                #                     fr_mu_pred, cp_pred, n_pred, s_pred],
                #                   axis=1) / self.scale
                # # block vectorization to avoid excessive memory usage for
                # # long sequences
                # F = tape.batch_jacobian(new_state, last_state,
                #                           experimental_use_pfor=False)

                # split the state into parts and undo the scaling
                last_state *= self.scale
                pos = last_state[:, :2]
                ori = last_state[:, 2:3]
                fr = last_state[:, 3:4]
                fr_mu = last_state[:, 4:5]
                cp = last_state[:, 5:7]
                n = last_state[:, 7:9]
                s = last_state[:, 9:]

                # undo the scaling for the actions as well
                actions *= self.scale

                # apply the analytical model to get predicted translation and
                # rotation
                tr_pred, rot_pred, keep_contact, dx, dy, dom = \
                    utils.physical_model_derivative(pos, cp, n, actions, fr,
                                                    fr_mu, s)

                # for the circular object, the orientation is always zero,
                # so we have to set the prediction to 0 and adapt the
                # jacobian
                ob = tf.squeeze(ob)
                ob = tf.strings.regex_replace(ob, "\000", "")
                ob = tf.strings.regex_replace(ob, "\00", "")

                rot_pred = tf.where(tf.equal(ob, 'ellip1'),
                                    tf.zeros_like(rot_pred), rot_pred)
                dom = tf.where(tf.equal(ob, 'ellip1'),
                               tf.zeros_like(dom), dom)

                pos_pred = pos + tr_pred
                ori_pred = ori + rot_pred * 180.0 / np.pi
                fr_pred = fr
                fr_mu_pred = fr_mu
                cp_pred = cp + actions
                keep_contact = tf.cast(keep_contact, tf.float32)
                n_pred = n * keep_contact
                s_pred = s * keep_contact

                # piece together the new state and apply scaling again
                new_state = \
                    tf.concat([pos_pred, ori_pred, fr_pred,
                               fr_mu_pred, cp_pred, n_pred, s_pred],
                              axis=1) / self.scale

                # piece together the jacobian (I found this to work better than
                # getting the whole jacobian from tensorflow)
                dom *= 180.0 / np.pi
                dnx = tf.concat([tf.zeros([self.batch_size, 7]),
                                 tf.cast(keep_contact, tf.float32),
                                 tf.zeros([self.batch_size, 2])],
                                axis=-1)
                dny = tf.concat([tf.zeros([self.batch_size, 8]),
                                 tf.cast(keep_contact, tf.float32),
                                 tf.zeros([self.batch_size, 1])],
                                axis=-1)
                ds = tf.concat([tf.zeros([self.batch_size, 9]),
                                tf.cast(keep_contact, tf.float32)],
                               axis=-1)

                F = tf.concat(
                    [dx + np.array([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0.]]],
                                   dtype=np.float32),
                     dy + np.array([[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0.]]],
                                   dtype=np.float32),
                     dom + np.array([[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.]]],
                                    dtype=np.float32),
                     tf.tile(np.array([[[0, 0, 0, 1, 0, 0, 0, 0, 0, 0.]]],
                                      dtype=np.float32),
                             [self.batch_size, 1, 1]),
                     tf.tile(np.array([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0.]]],
                                      dtype=np.float32),
                             [self.batch_size, 1, 1]),
                     tf.tile(np.array([[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
                                      dtype=np.float32),
                             [self.batch_size, 1, 1]),
                     tf.tile(np.array([[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]],
                                      dtype=np.float32),
                             [self.batch_size, 1, 1]),
                     tf.reshape(dnx, [-1, 1, self.dim_x]),
                     tf.reshape(dny, [-1, 1, self.dim_x]),
                     tf.reshape(ds, [-1, 1, self.dim_x])], axis=1)
            else:
                # split the state into parts and undo the scaling
                last_state *= self.scale
                pos = last_state[:, :2]
                ori = last_state[:, 2:3]
                fr = last_state[:, 3:4]
                fr_mu = last_state[:, 4:5]
                cp = last_state[:, 5:7]
                n = last_state[:, 7:9]
                s = last_state[:, 9:]

                # undo the scaling for the actions as well
                actions *= self.scale

                # apply the analytical model to get predicted translation and
                # rotation
                tr_pred, rot_pred, keep_contact = \
                    utils.physical_model(pos, cp, n, actions, fr, fr_mu, s)

                pos_pred = pos + tr_pred
                ori_pred = ori + rot_pred * 180.0 / np.pi
                fr_pred = fr
                fr_mu_pred = fr_mu
                cp_pred = cp + actions
                keep_contact = tf.cast(keep_contact, tf.float32)
                n_pred = n * keep_contact
                s_pred = s * keep_contact

                # piece together the new state and apply scaling again
                new_state = \
                    tf.concat([pos_pred, ori_pred, fr_pred,
                               fr_mu_pred, cp_pred, n_pred, s_pred],
                              axis=1) / self.scale
                F = None

            fc3 = None

        if self.jacobian:
            F = tf.stop_gradient(F)
        return new_state, fc3, F


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
        init_const = np.ones(self.dim_x) * 1e-5 / self.scale**2
        init = np.sqrt(np.square(self.q_diag) - init_const)
        # the constant bias keeps the predicted covariance away from zero
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_x],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))
        num = self.dim_x * (self.dim_x + 1) / 2
        wd = 1e-3 * self.scale**2

        if self.hetero and self.diag and self.learned:
            # for heteroscedastic noise with diagonal covariance matrix
            self.het_diag_lrn_fc1 = self._fc_layer('het_diag_lrn_fc1', 128,
                                                   trainable=self.trainable)
            self.het_diag_lrn_fc2 = self._fc_layer('het_diag_lrn_fc2', 64,
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
            self.het_full_lrn_fc1 = self._fc_layer('het_full_lrn_fc1', 128,
                                                   trainable=self.trainable)
            self.het_full_lrn_fc2 = self._fc_layer('het_full_lrn_fc2', 64,
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
            self.het_diag_ana_fc1 = self._fc_layer('het_diag_ana_fc1', 128,
                                                   std=1e-3,
                                                   trainable=self.trainable)
            self.het_diag_ana_fc2 = self._fc_layer('het_diag_ana_fc2', 64,
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
            self.het_full_ana_fc1 = self._fc_layer('het_full_ana_fc1', 128,
                                                   std=1e-3,
                                                   trainable=self.trainable)
            self.het_full_ana_fc2 = self._fc_layer('het_full_ana_fc2', 64,
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
                                shape=[self.dim_x], trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        old_state, actions = inputs

        # exclude l from the inputs for stability
        input_data = tf.concat([old_state[:, :3], old_state[:, 4:], actions],
                               axis=-1)
        # input_data = tf.concat([old_state, actions], axis=-1)
        if self.learned:
            if self.hetero and self.diag:
                fc1 = self.het_diag_lrn_fc1(input_data)
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
                fc1 = self.het_full_lrn_fc1(input_data)
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
                fc1 = self.het_diag_ana_fc1(input_data)
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
                fc1 = self.het_full_ana_fc1(input_data)
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
