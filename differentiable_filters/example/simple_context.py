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
        base.BaseContext.__init__(self, param)

        # the state size
        self.dim_x = 4
        self.dim_u = 0
        self.dim_z = 2

        self.x_names = ['x', 'y', 'vx', 'vy']
        self.z_names = ['x', 'y']

        self.spring_force = 0.05
        self.drag_force = 0.0075

        # define initial values for the process noise q and observation noise r
        # diagonals
        # Important: All values are standard-deviations, so they are
        # squared for forming the covariance matrices
        self.q_diag = self.q_diag.astype(np.float32)
        self.r_diag = self.r_diag.astype(np.float32)

        # if the noise matrices are not learned, we construct the fixed
        # covariance matrices here
        q = np.diag(np.square(self.q_diag))
        self.Q = tf.convert_to_tensor(q, dtype=tf.float32)
        r = np.diag(np.square(self.r_diag))
        self.R = tf.convert_to_tensor(r, dtype=tf.float32)

        # initial uncertatinty about the state and fixed initial noise values
        # for testing
        self.initial_covariance = [5., 5., 5., 5.]
        self.noise_list = \
            [np.array([0., 0., 0., 0.]).astype(np.float32),
             np.array([-5.587, 3.782, -5.895, -3.441]).astype(np.float32),
             np.array([-10.428, 1.034, 9.516, -1.171]).astype(np.float32),
             np.array([3.053, 4.262, -6.058, 4.927]).astype(np.float32),
             np.array([2.631, 12.652, 7.648, 5.688]).astype(np.float32)]
        self.noise_list = list(map(lambda x: x/5., self.noise_list))

        # all layers used in the context need to be instantiated here
        self.sensor_model_layer = SensorModel(self.batch_size)
        self.observation_model_layer = \
            ObservationModel(self.dim_z, self.batch_size)
        self.observation_noise_layer = \
            ObservationNoise(self.batch_size, self.dim_z, self.r_diag,
                             hetero=param['hetero_r'])
        self.process_model_layer = \
            ProcessModel(self.batch_size, self.dim_x, self.dim_u,
                         self.spring_force, self.drag_force,
                         learned=param['learn_process'],
                         jacobian=param['filter'] == 'ekf')

        self.process_noise_layer = \
            ProcessNoise(self.batch_size, self.dim_x, self.q_diag,
                         hetero=param['hetero_q'])

        # group for easier access
        self.layers = [self.sensor_model_layer, self.observation_model_layer,
                       self.observation_noise_layer,
                       self.process_noise_layer, self.process_noise_layer]

    ###########################################################################
    # loss function
    ###########################################################################
    def get_loss(self, prediction, label, step):
        """
        Compute the loss for the filtering application

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step

        Returns:
            loss: the total loss for training the filtering application
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        particles, weights, states, covars, z, r, q = prediction
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

        # compute the mean squared error and euclidean error of the predicted
        # states
        total_mse, total_dist = self._mse(diff, reduce_mean=False)

        # compute component-wise distances
        dists = []
        for i in range(self.dim_x):
            _, dist = self._mse(diff[:, :, i:i+1], reduce_mean=False)
            dists += [dist]

        # compute the error in the predicted observations (only for monitoring)
        diff_obs = seq_label[:, :, :2] - z
        mse_x_obs, dist_x_obs = \
            self._mse(diff_obs[:, :, :1], reduce_mean=False)
        mse_y_obs, dist_y_obs = \
            self._mse(diff_obs[:, :, 1:], reduce_mean=False)
        _, dist_obs = self._mse(diff_obs, reduce_mean=False)

        # compute the error of the predicted process noise
        if len(q_label.get_shape()) == 3:
            q_label_m = tf.linalg.diag(tf.square(q_label))
            dist_q = self._bhattacharyya(q, q_label_m)
        else:
            dist_q = self._bhattacharyya(q, q_label)

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

        total_tracking = tf.reduce_mean(total_mse)
        if self.loss == 'like':
            total_loss = tf.reduce_mean(likelihood)
        elif self.loss == 'mse':
            total_loss = total_tracking
        elif self.loss == 'mixed':
            total_loss = (total_tracking + tf.reduce_mean(likelihood)) / 2.

        # get the weight decay
        wd = []
        for la in self.layers:
            wd += la.losses
        total = 100 * total_loss + wd

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
        return total, [likelihood, total_dist, dist_obs, total_mse] + dists + \
            [dist_x_obs, dist_y_obs, dist_q, vis_label, diag_r, wd], \
            ['likelihood', 'dist', 'dist_obs', 'mse', 'x', 'y', 'vx', 'vy',
             'x_obs', 'y_obs', 'q', 'vis', 'r_pred', 'wd']

    def _bhattacharyya(self, pred, label):
        """
        Calculate the bhattacharyya distance between two covariance matrices

        Args:
            pred: predicted covariance matrix
            label: true covariance matrix

        Returns:
            dist: tbhattacharyya distance
        """
        mean = (pred + label) / 2.
        det_mean = tf.linalg.det(mean)
        det_pred = tf.linalg.det(pred)
        det_label = tf.linalg.det(label)
        dist = det_mean/(tf.sqrt(det_pred*det_label))
        dist = 0.5 * tf.math.log(dist)
        return dist

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, sequence_length,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record

        Parameters
        ----------
        path : str
            path to the directory that contains the dataset
        name : str
            name of the dataset
        dataset : tf.data.Dataset
            tensorflow file dataset that contains a list of the tf.record files
            to process
        data_mode : str
            "train", "val", or "test". Decides if the data is read from the
            training, validation or test split of the dataset
        sequence_length : int
            desired length of the sequences to be extracted. Must not be longer
            than the sequence length in the file
        num_threads : int, optional
            Number of threads used for preprocessing the data. The default is 3.

        Raises
        ------
        ValueError
            If the desired sequence length is greater than the sequence length
            on file

        Returns
        -------
        tf.data.Dataset
            The preprocessed data set

        """
        keys = ['start_image', 'start_state', 'image', 'state', 'q',
                'visible']

        record_meta = tfr.RecordMeta.load(path, name + '_' + data_mode + '_')

        def _parse_function(example_proto):
            features = {}
            for key in keys:
                record_meta.add_tf_feature(key, features)

            parsed_features = tf.io.parse_single_example(example_proto,
                                                         features)
            for key in keys:
                features[key] = record_meta.reshape_and_cast(key,
                                                             parsed_features)
            state = features['state']
            start = features['start_state']
            im = features['image']
            q = features['q']
            vis = features['visible']
            actions = state[1:, 2:] - state[:-1, 2:]
            actions = tf.concat([state[0:1, 2:] - start[None, 2:], actions],
                                axis=0)

            length = im.get_shape()[0].value

            if sequence_length > length:
                raise ValueError('Desired sequence length is ' +
                                 'longer than dataset sequence length: ' +
                                 'Desired: ' + str(sequence_length) +
                                 ', data: ' + str(length))

            # we use several sub-sequences of the full sequences if the
            # desired sequence length is significantly shorter than
            # the lenght in the file
            if sequence_length < length // 2:
                num = 3
                start_inds = \
                    np.random.randint(0, length-sequence_length-1, num)
                self.train_multiplier = num
            else:
                start_inds = [-1]

            # prepare the lists of output tensors
            ims = []
            starts = []
            im_starts = []
            states = []
            qs = []
            viss = []
            acs = []
            for si in start_inds:
                end = si+sequence_length
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

        dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
        dataset = \
            dataset.flat_map(lambda x, y:
                             tf.data.Dataset.from_tensor_slices((x, y)))
        return dataset


class SensorModel(BaseLayer):
    def __init__(self, batch_size):
        super(SensorModel, self).__init__()
        self.batch_size = batch_size

        self.c1 = self._conv_layer('conv1', 9, 4, stride=[2, 2])
        self.c2 = self._conv_layer('conv2', 9, 8, stride=[2, 2])

        self.fc1 = self._fc_layer('fc1', 16)
        self.fc2 = self._fc_layer('fc2', 32)
        self.pos = self._fc_layer('fc_pos', 2, activation=None)

    def call(self, images, training):
        conv1 = self.c1(images)
        conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')

        conv2 = self.c2(conv1)
        conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')

        input_data = tf.reshape(conv2, [self.batch_size, -1])
        fc1 = self.fc1(input_data)
        fc2 = self.fc2(fc1)
        pos = self.pos(fc2)

        return pos, fc2


class ObservationNoise(BaseLayer):
    def __init__(self, batch_size, dim_z, r_diag, hetero):
        super(ObservationNoise, self).__init__()

        self.hetero = hetero
        self.learned = learned
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.r_diag = r_diag

    def build(self, input_shape):
        init_const = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - init_const)

        num = self.dim_z * (self.dim_z + 1) / 2
        wd = 1e-3

        # the constant bias keeps the predicted covariance away from zero
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_z],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))
        # the learned bias alllows to initialize the prediction near a given
        # value
        self.bias_learned = \
                self.add_weight(name='bias_learned',
                                shape=[self.dim_z],
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))


        if self.hetero:
            # for heteroscedastic noise with diagonal covariance matrix
            self.het_fc = self._fc_layer('het_fc', self.dim_z, mean=0,
                                         std=1e-3, activation=None)

        else:
            # for constant noise with
            self.const_diag = \
                self.add_weight(name='const_diag',
                                shape=[self.dim_z],
                                trainable=self.trainable,
                                regularizer=tf.keras.regularizers.l2(l=wd),
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        if self.hetero:
            diag = self.het_diag_fc(inputs)
            diag = tf.square(diag + self.het_diag_init_bias)
            diag += self.bias_fixed
            R = tf.linalg.diag(diag)
        else:
            diag = self.const_diag
            diag = tf.square(diag) + self.bias_fixed
            R = tf.linalg.tensor_diag(diag)
            R = tf.tile(R[None, :, :], [self.batch_size, 1, 1])
        return R


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
    def __init__(self, batch_size, dim_x, dim_u, sf, df,
                 learned, jacobian):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.learned = learned
        self.spring_force = sf
        self.drag_force = df
        self.jacobian = jacobian

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
            fc2 = self.fc2(fc1)
            fc3 = self.fc3(fc2)
            update = self.update(fc3)

            new_state = last_state + update
            if self.jacobian:
                F = self._compute_jacobian(new_state, last_state)
            else:
                F = None
        else:
            # split the state into parts
            x = tf.slice(last_state, [0, 0], [-1, 1])
            y = tf.slice(last_state, [0, 1], [-1, 1])
            vx = tf.slice(last_state, [0, 2], [-1, 1])
            vy = tf.slice(last_state, [0, 3], [-1, 1])

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
    def __init__(self, batch_size, dim_x, q_diag, hetero):
        super(ProcessNoise, self).__init__()
        self.hetero = hetero

        self.dim_x = dim_x
        self.q_diag = q_diag
        self.batch_size = batch_size

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
