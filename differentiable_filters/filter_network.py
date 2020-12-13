#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:24:05 2020

@author: alina
"""

import tensorflow as tf
import numpy as np


class Filter(tf.keras.Model):
    def __init__(self, param, context, **kwargs):
        """
        Tf.keras.Model that combines a differentiable filter and a problem
        context to run filtering on this problem.

        Parameters
        ----------
        param : dict
            Collection of arguments
        context : Subclass of BaseContext
            A context class that describes the problm and problem-specific
            functions such as the process or observation model

        Raises
        ------
        ValueError
            If the desired filter class (param['filter']) is not implemented

        Returns
        -------
        None.

        """
        super(Filter, self).__init__(**kwargs)
        self.param = param
        self.context = context

        # shape  related information
        self.batch_size = param['batch_size']
        self.sequence_length = param['sequence_length']
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u

        # optional scaling factor for the state-space
        self.scale = param['scale']

        # instantiate the desired filter cell
        if self.param['filter'] == 'ekf':
            from differentiable_filters.filters import ekf_cell as ekf
            self.cell = ekf.EKFCell(self.param, self.context)
        elif self.param['filter'] == 'ukf':
            from differentiable_filters.filters import ukf_cell as ukf
            self.cell = ukf.UKFCell(self.param, self.context)
        elif self.param['filter'] == 'mcukf':
            from differentiable_filters.filters import mcukf_cell as mcukf
            self.cell = mcukf.MCUKFCell(self.param, self.context)
        elif self.param['filter'] == 'pf':
            from differentiable_filters.filters import pf_cell as pf
            self.cell = pf.PFCell(self.param, self.context)
        elif self.param['filter'] == 'lstm':
            from differentiable_filters.filters import lstm_cell as lstm
            self.cell = lstm.UnstructuredCell(self.param, self.context)
        else:
            self.log.error('Unknown filter type: ' + self.param['filter'])
            raise ValueError('Unknown filter type: ' + self.param['filter'])

        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True,
                                             unroll=False)

        # define the covariance matrix for the initial state
        # The initial covariance is given as standard-deviations and
        # will be squared when the covariance matrix is constructed
        if param['initial_covar'] is not None:
            cov_string = param['initial_covar']
            self.initial_covariance = list(map(lambda x: float(x),
                                               cov_string.split(' ')))

            self.initial_covariance = \
                np.array(self.initial_covariance).astype(np.float32)
        else:
            self.initial_covariance = np.ones((self.dim_x), dtype=np.float32)
        # adapt to the scaling
        cs = []
        for k in range(self.dim_x):
            cs += [self.initial_covariance[k] / self.scale]
        self.covar_start_raw = tf.stack(cs)
        covar_start = tf.square(self.covar_start_raw)
        covar_start = tf.linalg.tensor_diag(covar_start)
        self.covar_start = tf.tile(covar_start[None, :, :],
                                   [self.batch_size, 1, 1])

    def call(self, inputs, training=True):
        """
        Run one step of prediction with the model

        Parameters
        ----------
        inputs : list of tensors
            the input tensors
        training : bool
            if the model is run in training or test mode

        Returns
        -------
        res : list of tensors
            the prediction output

        """
        raw_observations, actions, initial_observations, initial_state,  \
            info, step, noise = inputs

        if self.param['problem'] == 'pushing':
            # inform the context about the current objects (for dealing
            # with different rotational symmetries)
            self.context.ob = info[0]
            # get the initial segmented glimpse
            initial_image, initial_tip_pos = initial_observations
            initial_glimpse, initial_pix, _ = \
                self.context.get_initial_glimpse(initial_image, training)
            raw_observations = \
                tuple(list(raw_observations) +
                      [tf.tile(initial_glimpse[:, None, :, :, :],
                               [1, self.sequence_length, 1, 1, 1])])
            initial_observations = \
                tuple(list(initial_observations) +
                      [tf.tile(initial_glimpse[:, None, :, :, :],
                               [1, self.sequence_length, 1, 1, 1])])

        if self.param['add_initial_noise']:
            # rescale the noise
            ns = []
            for k in range(self.dim_x):
                ns += [noise[:, k]/self.scale]
            noise = tf.stack(ns, axis=-1)

            if self.param['problem'] == 'kitti':
                # set the position and orientation noise to zero for kitti
                noise = tf.concat([tf.zeros_like(noise[:, :3]),
                                   noise[:, 3:]], axis=-1)
            elif self.param['problem'] == 'pushing':
                # set the orientation noise to zero for pushing
                noise = tf.concat([noise[:, :2],
                                   tf.zeros_like(noise[:, 2:3]),
                                   noise[:, 3:]], axis=-1)
            initial_state = initial_state + noise
            if self.param['problem'] == 'pushing':
                initial_state = self.context.correct_state(initial_state,
                                                           diff=False)
            else:
                initial_state = self.context.correct_state(initial_state)

        # ----------------------------------------------------------------
        # # construct the initial filter state
        # ----------------------------------------------------------------
        if self.param['filter'] == 'lstm' and \
                self.param['lstm_structure'] == 'lstm2':
            init_state = \
                (tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 initial_state,
                 tf.reshape(self.covar_start, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))
        elif self.param['filter'] == 'lstm' and \
                self.param['lstm_structure'] == 'lstm1':
            init_state = \
                (tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 tf.zeros([self.batch_size, self.cell.num_units],
                          dtype=tf.float32),
                 initial_state,
                 tf.reshape(self.covar_start, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))
        elif self.param['filter'] != 'pf':
            init_state = \
                (initial_state,
                 tf.reshape(self.covar_start, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))
        else:
            particles, weights = \
                self.cell.sample_from_start(initial_state,
                                            self.covar_start_raw)
            particles = tf.reshape(particles, [-1, self.dim_x])
            if self.param['problem'] == 'pushing':
                particles = self.context.correct_state(particles,
                                                       diff=False)
            else:
                particles = self.context.correct_state(particles)
            particles = tf.reshape(particles,
                                   [self.batch_size, -1, self.dim_x])
            if self.param['problem'] == 'kitti':
                # set the position and orientation variations of particles
                # to zero for kitti
                init_pose = tf.tile(initial_state[:, None, :3],
                                    [1, self.param['num_samples'],
                                     1])
                particles = \
                    tf.concat([init_pose, particles[:, :, 3:]], axis=-1)
            elif self.param['problem'] == 'pushing':
                # set the orientation variations of particles to zero for
                # pushing
                particles = \
                    tf.concat([particles[:, :, :2],
                               tf.zeros_like(particles[:, :, 2:3]),
                               particles[:, :, 3:]], axis=-1)

            init_state = \
                (tf.reshape(particles, [self.batch_size, -1]),
                 tf.reshape(weights, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))

        # ----------------------------------------------------------------
        # run the filter
        # ----------------------------------------------------------------
        # inputs are the raw observation inputs and actions
        inputs = (raw_observations, actions)
        out = self.rnn_layer(inputs, training=training,
                             initial_state=init_state)

        # ----------------------------------------------------------------
        # collect the results
        # ----------------------------------------------------------------
        # out contains the full sequence of values for the outputs defined
        # in the cell
        if self.param['filter'] == 'lstm':
            state_sequence, covariance_sequence, z = out
            Q = tf.tile(self.context.Q[None, None, :, :],
                        [self.batch_size, self.sequence_length, 1, 1])
            R = tf.tile(self.context.R[None, None, :, :],
                        [self.batch_size, self.sequence_length, 1, 1])
            particles = tf.zeros([1, 1, self.dim_x, self.dim_x])
            weights = tf.zeros([])
        elif self.param['filter'] != 'pf':
            state_sequence, covariance_sequence, z, R, Q = out

            particles = tf.zeros([1, 1, self.dim_x, self.dim_x])
            weights = tf.zeros([])
        else:
            particles, weights, state_sequence, covariance_sequence, \
                z, R, Q = out

            particles = tf.reshape(particles,
                                   [self.batch_size, -1,
                                    self.cell.num_particles, self.dim_x])
            weights = tf.reshape(weights, [self.batch_size, -1,
                                           self.cell.num_particles])

            # weights are in log scale, to turn them into a distribution,
            # we exponentiate and normalize them == apply the softmax
            # transform
            weights = tf.nn.softmax(weights, axis=-1)
            # remove nans and infs
            weights = tf.where(tf.math.is_finite(weights), weights,
                               tf.zeros_like(weights))

        Q = tf.reshape(Q, [self.batch_size, -1, self.dim_x, self.dim_x])
        R = tf.reshape(R, [self.batch_size, -1, self.dim_z, self.dim_z])

        z = tf.reshape(z, [self.batch_size, -1, self.dim_z])
        covars = \
            tf.reshape(covariance_sequence,
                       [self.batch_size, -1, self.dim_x, self.dim_x])

        res = [particles, weights, state_sequence, covars,
               initial_state, self.covar_start, z, R, Q]

        #######################################################################
        # add summaries
        ######################################################################
        for i in range(min(self.sequence_length, 3)):
            diag_r = tf.linalg.diag_part(tf.slice(R, [0, i, 0, 0],
                                                  [1, 1, -1, -1]))
            diag_r = tf.squeeze(tf.sqrt(tf.abs(diag_r)))
            for k in range(self.dim_z):
                name = 'r/' + self.context.z_names[k] + '_' + str(i)
                tf.summary.histogram(name,
                                     diag_r[k:k+1]*self.scale)

            if self.param['diagonal_covar']:
                diag_q = tf.linalg.diag_part(tf.slice(Q, [0, i, 0, 0],
                                                      [1, 1, -1, -1]))
                diag_q = tf.squeeze(tf.sqrt(tf.abs(diag_q)))
                for k in range(self.dim_x):
                    name = 'q/' + self.context.x_names[k] + '_' + str(i)
                    tf.summary.histogram(name,
                                         diag_q[k:k+1]*self.scale)
            else:
                for k in range(self.dim_x):
                    for j in np.arange(k, self.dim_x):
                        if k != j:
                            n = 'q_sqr/' + self.context.x_names[k] + \
                                '_' + self.context.x_names[j]
                        else:
                            n = 'q_sqr/' + self.context.x_names[k]
                        tf.summary.histogram(n,
                                             Q[0, i, k, j] * self.scale**2)

            diag_c = tf.linalg.diag_part(tf.slice(covars, [0, i, 0, 0],
                                                  [1, 1, -1, -1]))
            diag_c = tf.squeeze(tf.sqrt(tf.abs(diag_c)))
            for k in range(self.dim_x):
                name = 'c/' + self.context.x_names[k] + '_' + str(i)
                tf.summary.histogram(name,
                                     diag_c[k:k+1]*self.scale)

            if self.param['filter'] == 'pf':
                tf.summary.histogram('weights/' + str(i), weights[0, i])
                tf.summary.histogram('weights/max_' + str(i),
                                     tf.reduce_max(weights[:, i, :],
                                                   axis=-1))

                # count the number of extremely small weights
                small = \
                    tf.where(tf.less(weights, 1e-7), tf.ones_like(weights),
                             tf.zeros_like(weights))
                small = tf.reduce_sum(small, axis=-1)
                small = tf.reshape(small, [self.batch_size, -1])
                tf.summary.histogram('weights/small_' + str(i),
                                     small[:, i])

        for k in range(self.dim_z):
            tf.summary.histogram('observations/start' +
                                 self.context.z_names[k],
                                 z[0, 0, k:k+1] * self.scale)
            tf.summary.histogram('observations/end' +
                                 self.context.z_names[k],
                                 z[0, -1, k:k+1] * self.scale)
            if self.param['filter'] == 'pf':
                tf.summary.histogram('weights/small_end', small[:, -1])

        diag_c = tf.squeeze(tf.linalg.diag_part(covars[0, -1]))
        diag_c = tf.sqrt(tf.abs(diag_c + 1e-3))
        for k in range(self.dim_x):
            name = 'c_end/' + self.context.x_names[k] + '_' + str(i)
            tf.summary.histogram(name, diag_c[k:k+1]*self.scale)

        return res

    def get_loss(self, prediction, label, step, training):
        return self.context.get_filter_loss(prediction, label, step, training)

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        return self.context.tf_record_map(path, name, dataset, data_mode,
                                          train_mode, num_threads)

    ###########################################################################
    # Evaluation
    ###########################################################################
    def save_log(self, log_dict, out_dir, step, num, mode):
        self.context.save_log(log_dict, out_dir, step, num, mode)

    def get_fetches(self, inputs, labels, prediction):
        particles, weights, seq_pred, cov_pred, init_s, init_c, z, r, q = \
            prediction
        state = labels[0]

        raw_observations, actions, initial_observations, initial_state,  \
            info, noise = inputs

        # undo the scaling
        ps = []
        states = []
        covars = []
        zs = []
        Rs = []
        Qs = []
        ls = []
        for k in range(self.context.dim_x):
            states += [seq_pred[:, :, k:k+1] * self.context.scale]
            ls += [state[:, :, k:k+1] * self.context.scale]
            ps += [particles[:, :, :, k:k+1] * self.context.scale]
            rows_c = []
            rows_q = []
            for j in range(self.context.dim_x):
                val_c = cov_pred[:, :, k:k+1, j:j+1]*self.context.scale**2
                rows_c += [val_c]
                rows_q += [q[:, :, k:k+1, j:j+1]*self.context.scale**2]
            covars += [tf.concat(rows_c, axis=-1)]
            Qs += [tf.concat(rows_q, axis=-1)]
        states = tf.concat(states, axis=-1)
        ls = tf.concat(ls, axis=-1)
        ps = tf.concat(ps, axis=-1)
        covars = tf.concat(covars, axis=-2)
        Qs = tf.concat(Qs, axis=-2)

        for k in range(self.context.dim_z):
            zs += [z[:, :, k:k+1] * self.context.scale]
            rows_r = []
            for j in range(self.context.dim_z):
                rows_r += [r[:, :, k:k+1, j:j+1]*self.context.scale*2]
            Rs += [tf.concat(rows_r, axis=-1)]
        Rs = tf.concat(Rs, axis=-2)
        zs = tf.concat(zs, axis=-1)

        out = {'seq_pred': states, 'cov_pred': covars, 'z': zs,
               'r_pred': Rs, 'q_pred': Qs, 'seq': ls}
        if self.param['problem'] == 'toy':
            out.update({'q': labels[1], 'vis': labels[2]})
        if self.param['problem'] == 'pushing':
            out.update({'ob': info[0], 'mat': info[1], 'vis': labels[3],
                        'actions': actions, 'init': initial_state})
        if self.param['filter'] == 'pf':
            out.update({'particles': ps, 'weights': weights})
        return out

    def evaluate(self, loss_dict, additional, out_dir, step):
        """
        Space for additional evaluations after testing
        Args:
            loss_dict: dictionary that contains a list of all values for the
                loss and all loss components
            additional: dictionary that contains a list of all values that
                were fetched through get_fetches
            out_dir: where to store results
            step: the training-step that we evaluate (use for naming)
        """

        for i in [0, 3, 8]:
            seq_pred = np.squeeze(additional['seq_pred'][i])
            cov_pred = np.squeeze(additional['cov_pred'][i])
            seq = np.squeeze(additional['seq'][i])
            z = np.squeeze(additional['z'][i])
            q_pred = np.squeeze(additional['q_pred'][i])
            r_pred = np.squeeze(additional['r_pred'][i])

            if self.param['problem'] == 'toy':
                q = np.squeeze(additional['q'][i])
                vis = np.squeeze(additional['vis'][i])
                self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                           q_pred, q, r_pred, vis, out_dir,
                                           i)
            elif self.param['problem'] == 'pushing':
                diffs = np.squeeze(loss_dict['dist_steps'][i])
                likes = np.squeeze(loss_dict['likelihood_steps'][i])
                vis = np.squeeze(additional['vis'][i])
                actions = np.squeeze(additional['actions'][i])
                ob = np.squeeze(additional['ob'][i])
                init = np.squeeze(additional['init'][i])
                self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                           q_pred, r_pred, vis, out_dir, i,
                                           diffs, likes, actions, ob,
                                           init)
            else:
                diffs = np.squeeze(loss_dict['dist_steps'][i])
                likes = np.squeeze(loss_dict['likelihood_steps'][i])
                self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                           q_pred, r_pred, out_dir, i,
                                           diffs, likes)
            if self.param['problem'] == 'pushing':
                if self.param['filter'] == 'pf':
                    particles = np.squeeze(additional['particles'][i])
                    weights = np.squeeze(additional['weights'][i])
                    self.context.plot_trajectory(particles, weights, seq,
                                                 cov_pred, seq_pred, ob,
                                                 out_dir, i)
                else:
                    self.context.plot_trajectory(None, None, seq, cov_pred,
                                                 seq_pred, ob, out_dir, i)
            else:
                if self.param['filter'] == 'pf':
                    particles = np.squeeze(additional['particles'][i])
                    weights = np.squeeze(additional['weights'][i])
                    self.context.plot_trajectory(particles, weights, seq,
                                                 cov_pred, seq_pred,
                                                 out_dir, i)
                else:
                    self.context.plot_trajectory(None, None, seq, cov_pred,
                                                 seq_pred, out_dir, i)

        # plot the worst trajectory
        worst = np.argmax(loss_dict['loss'])
        print('worst sequence: ' + str(worst))
        seq_pred = np.squeeze(additional['seq_pred'][worst])
        cov_pred = np.squeeze(additional['cov_pred'][worst])
        seq = np.squeeze(additional['seq'][worst])
        z = np.squeeze(additional['z'][worst])
        q_pred = np.squeeze(additional['q_pred'][worst])
        r_pred = np.squeeze(additional['r_pred'][worst])

        if self.param['problem'] == 'toy':
            q = np.squeeze(additional['q'][worst])
            vis = np.squeeze(additional['vis'][worst])
            self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                       q_pred, q, r_pred, vis, out_dir,
                                       worst)
        elif self.param['problem'] == 'pushing':
            diffs = np.squeeze(loss_dict['dist_steps'][worst])
            likes = np.squeeze(loss_dict['likelihood_steps'][worst])
            vis = np.squeeze(additional['vis'][worst])
            actions = np.squeeze(additional['actions'][worst])
            ob = np.squeeze(additional['ob'][worst])
            init = np.squeeze(additional['init'][worst])
            self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                       q_pred, r_pred, vis, out_dir,
                                       worst, diffs, likes, actions, ob,
                                       init, full_out=True)
        else:
            diffs = np.squeeze(loss_dict['dist_steps'][worst])
            likes = np.squeeze(loss_dict['likelihood_steps'][worst])
            self.context.plot_tracking(seq_pred, cov_pred, z, seq,
                                       q_pred, r_pred, out_dir, worst,
                                       diffs, likes)

        if self.param['problem'] == 'pushing':
            if self.param['filter'] == 'pf':
                particles = np.squeeze(additional['particles'][worst])
                weights = np.squeeze(additional['weights'][worst])
                self.context.plot_trajectory(particles, weights, seq,
                                             cov_pred, seq_pred, ob,
                                             out_dir, worst)
            else:
                self.context.plot_trajectory(None, None, seq, cov_pred,
                                             seq_pred, ob, out_dir, worst)
        else:
            if self.param['filter'] == 'pf':
                particles = np.squeeze(additional['particles'][worst])
                weights = np.squeeze(additional['weights'][worst])
                self.context.plot_trajectory(particles, weights, seq,
                                             cov_pred, seq_pred, out_dir,
                                             worst)
            else:
                self.context.plot_trajectory(None, None, seq, cov_pred,
                                             seq_pred, out_dir, worst)
        return


class PretrainObservations(tf.keras.Model):
    def __init__(self, param, context, name='pretrain_observations', **kwargs):
        """
        Tf.keras.Model for pretraining the observation-related functions of
        a filtering problem

        Parameters
        ----------
        param : dict
            Collection of arguments
        context : Subclass of BaseContext
            A context class that describes the problm and problem-specific
            functions such as the process or observation model

        Returns
        -------
        None.

        """
        super(PretrainObservations, self).__init__(name=name, **kwargs)
        self.param = param
        self.context = context

        # shape  related information
        self.batch_size = param['batch_size']
        self.sequence_length = param['sequence_length']
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u

        # optional scaling factor for the state-space
        self.scale = param['scale']

    def call(self, inputs, training=True):
        """
        Run one step of prediction with the model

        Parameters
        ----------
        inputs : list of tensors
            the input tensors
        training : bool
            if the model is run in training or test mode

        Returns
        -------
        res : list of tensors
            the prediction output

        """
        # turn off the topmost name scope to improve summary logging
        with tf.name_scope(""):
            if self.param['problem'] == 'pushing':
                raw_observations, true_z, good_z, bad_z,  \
                    initial_observations, info = inputs

                # inform the context about the current objects (for dealing
                # with different rotational symmetries)
                self.context.ob = info[0]
                # get the initial segmented glimpse
                initial_image, _ = initial_observations
                initial_glimpse, initial_pix, initial_seg = \
                    self.context.get_initial_glimpse(initial_image, training)
                raw_observations = \
                    tuple(list(raw_observations) + [initial_glimpse])
            else:
                raw_observations, true_z, good_z, bad_z = inputs

            z, encoding = self.context.sensor_model(raw_observations, training)

            if self.param['problem'] == 'pushing':
                pix = encoding[-1]
                seg = encoding[0]

            R_const_diag = \
                self.context.observation_noise(encoding, hetero=False,
                                               diag=True,
                                               training=training)
            R_const_tri = \
                self.context.observation_noise(encoding, hetero=False,
                                               diag=False,
                                               training=training)
            R_het_diag = \
                self.context.observation_noise(encoding, hetero=True,
                                               diag=True,
                                               training=training)
            R_het_tri = \
                self.context.observation_noise(encoding, hetero=True,
                                               diag=False,
                                               training=training)

            like_good = self.context.likelihood(good_z, encoding, training)
            like_bad = self.context.likelihood(bad_z, encoding, training)

            # add summaries
            tf.summary.histogram('predicted_likelihood/good', like_good)
            tf.summary.histogram('predicted_likelihood/bad', like_bad)

            diag_r_const_diag = tf.linalg.diag_part(R_const_diag[0])
            diag_r_het_diag = tf.linalg.diag_part(R_het_diag[0])
            for k in range(self.dim_z):
                tf.summary.histogram('r_const_diag/' + self.context.z_names[k],
                                     diag_r_const_diag[k:k+1]*self.scale**2)
                tf.summary.histogram('r_het_diag/' + self.context.z_names[k],
                                     diag_r_het_diag[k:k+1]*self.scale**2)

            for k in range(self.dim_z):
                for j in np.arange(k, self.dim_z):
                    tf.summary.histogram('r_const_tri/' +
                                         self.context.z_names[k] +
                                         '_' + self.context.z_names[j],
                                         R_const_tri[0, k, j]*self.scale**2)
                    tf.summary.histogram('r_het_tri/' +
                                         self.context.z_names[k] +
                                         '_' + self.context.z_names[j],
                                         R_het_tri[0, k, j]*self.scale**2)

        if self.param['problem'] == 'pushing':
            return z, pix, seg, initial_pix, initial_seg, R_const_diag, \
                R_const_tri, R_het_diag, R_het_tri, like_good, like_bad
        else:
            return z, R_const_diag, R_const_tri, R_het_diag, R_het_tri, \
                like_good, like_bad

    def get_loss(self, prediction, label, step, training):
        return self.context.get_observation_loss(prediction, label, step,
                                                 training)

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        return self.context.tf_record_map(path, name, dataset, data_mode,
                                          train_mode, num_threads)

    ###########################################################################
    # Evaluation
    ###########################################################################
    def save_log(self, log_dict, out_dir, step, num, mode):
        self.context.save_log(log_dict, out_dir, step, num, mode)

    def get_fetches(self, inputs, labels, prediction):
        out = {}
        return out

    def evaluate(self, loss_dict, additional, out_dir, step):
        return


class PretrainProcess(tf.keras.Model):
    def __init__(self, param, context, name='pretrain_process', **kwargs):
        """
        Tf.keras.Model for pretraining the dynamics-related functions of
        a filtering problem

        Parameters
        ----------
        param : dict
            Collection of arguments
        context : Subclass of BaseContext
            A context class that describes the problm and problem-specific
            functions such as the process or observation model

        Returns
        -------
        None.

        """
        super(PretrainProcess, self).__init__(name=name, **kwargs)
        self.param = param
        self.context = context

        # shape  related information
        self.batch_size = param['batch_size']
        self.sequence_length = param['sequence_length']
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u

        # optional scaling factor for the state-space
        self.scale = param['scale']

    def call(self, inputs, training=True):
        """
        Run one step of prediction with the model

        Parameters
        ----------
        inputs : list of tensors
            the input tensors
        training : bool
            if the model is run in training or test mode

        Returns
        -------
        res : list of tensors
            the prediction output

        """
        # turn off the topmost name scope to improve summary logging
        with tf.name_scope(""):
            # unpack the inputs
            if self.param['problem'] == 'pushing':
                last_state, actions, info = inputs
                # inform the context about the current objects (for dealing
                # with different rotational symmetries)
                self.context.ob = info[0]
            else:
                last_state, actions = inputs
            next_state, _ = \
                self.context.process_model(last_state, actions, True,
                                           training=training)
            next_state_ana, _ = \
                self.context.process_model(last_state, actions, False,
                                           training=training)

            # we have to cover all  cases
            Q_const_diag = \
                self.context.process_noise(last_state, actions, True, False,
                                           True, training=training)
            Q_const_tri = \
                self.context.process_noise(last_state, actions, True, False,
                                           False, training=training)
            Q_het_diag = self.context.process_noise(last_state, actions, True,
                                                    True, True,
                                                    training=training)
            Q_het_tri = self.context.process_noise(last_state, actions, True,
                                                   True, False,
                                                   training=training)

            Q_const_diag_ana = \
                self.context.process_noise(last_state, actions, False, False,
                                           True, training=training)
            Q_const_tri_ana = \
                self.context.process_noise(last_state, actions, False, False,
                                           False, training=training)
            Q_het_diag_ana = \
                self.context.process_noise(last_state, actions, False, True,
                                           True, training=training)
            Q_het_tri_ana = \
                self.context.process_noise(last_state, actions, False, True,
                                           False, training=training)

            # add summaries
            diag_q_const_diag = \
                tf.linalg.diag_part(Q_const_diag[0])*self.scale**2
            diag_q_het_diag = \
                tf.linalg.diag_part(Q_het_diag[0])*self.scale**2
            diag_q_const_diag_ana = \
                tf.linalg.diag_part(Q_const_diag_ana[0])*self.scale**2
            diag_q_het_diag_ana = \
                tf.linalg.diag_part(Q_het_diag_ana[0])*self.scale**2
            for k in range(self.dim_x):
                tf.summary.histogram('q_const_diag_lrn/' +
                                     self.context.x_names[k],
                                     diag_q_const_diag[k:k+1])
                tf.summary.histogram('q_het_diag_lrn/' +
                                     self.context.x_names[k],
                                     diag_q_het_diag[k:k+1])
                tf.summary.histogram('q_const_diag_ana/' +
                                     self.context.x_names[k],
                                     diag_q_const_diag_ana[k:k+1])
                tf.summary.histogram('q_het_diag_ana/' +
                                     self.context.x_names[k],
                                     diag_q_het_diag_ana[k:k+1])

            for k in range(self.dim_x):
                for j in np.arange(k, self.dim_x):
                    n = self.context.x_names[k] + '_' + self.context.x_names[j]
                    tf.summary.histogram('q_const_tri_lrn/' + n,
                                         Q_const_tri[0, k, j]*self.scale**2)
                    tf.summary.histogram('q_het_tri_lrn/' + n,
                                         Q_het_tri[0, k, j]*self.scale**2)
                    tf.summary.histogram('q_const_tri_ana/' + n,
                                         Q_const_tri_ana[0, k, j] *
                                         self.scale**2)
                    tf.summary.histogram('q_het_tri_ana/' + n,
                                         Q_het_tri_ana[0, k, j] *
                                         self.scale**2)

        return next_state, Q_const_diag, Q_const_tri, Q_het_diag, Q_het_tri, \
            next_state_ana, Q_const_diag_ana, Q_const_tri_ana, \
            Q_het_tri_ana, Q_het_diag_ana

    def get_loss(self, prediction, label, step, training):
        return self.context.get_process_loss(prediction, label, step,
                                             training)

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        return self.context.tf_record_map(path, name, dataset, data_mode,
                                          train_mode, num_threads)

    ###########################################################################
    # Evaluation
    ###########################################################################
    def save_log(self, log_dict, out_dir, step, num, mode):
        self.context.save_log(log_dict, out_dir, step, num, mode)

    def get_fetches(self, inputs, labels, prediction):
        out = {}
        return out

    def evaluate(self, loss_dict, additional, out_dir, step):
        return
