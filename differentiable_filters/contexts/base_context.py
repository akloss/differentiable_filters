# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:35:37 2020

@author: akloss
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import csv


class BaseContext(tf.keras.Model):
    def __init__(self, param, mode):
        """
        Abstract base class for filtering contexts. A context contains
        problem-specific information such as the state size or process and
        sensor models.

        Parameters
        ----------
        param : dict
            A dictionary of arguments
        mode : string
            determines which parts of the model are trained. Use "filter" for
            the whole model, "pretrain_obs" for pretraining the observation
            related functions of the context in isolation or "pretrain_proc"
            for pretrainign the process-related functions of the context.
        """
        super(BaseContext, self).__init__()

        # determine the loss function
        self.loss = param['loss']
        self.batch_size = param['batch_size']

        self.mixture_std = param['mixture_std']
        self.debug = param['debug']
        self.param = param

        self.update_ops = []

        # if we extract more than one training example from one record in the
        # dataset, we need to indicate this
        self.train_multiplier = 1
        self.test_multiplier = 1
        self.epoch_size = 1

        self.mode = mode

        # all layers used in the context need to be instantiated in the
        # constructor of the context to avoid
        # creating variables inside the rnn while loop of the filter
        self.sensor_model_layer = None
        self.observation_noise_hetero_diag = None
        self.observation_noise_hetero_full = None
        self.observation_noise_const_diag = None
        self.observation_noise_const_full = None
        self.likelihood_layer = None
        self.observation_model_layer = None

        self.process_model_learned_layer = None
        self.process_model_analytical_layer = None
        self.process_noise_hetero_diag_lrn = None
        self.process_noise_hetero_full_lrn = None
        self.process_noise_const_diag_lrn = None
        self.process_noise_const_full_lrn = None
        self.process_noise_hetero_diag_ana = None
        self.process_noise_hetero_full_ana = None
        self.process_noise_const_diag_ana = None
        self.process_noise_const_full_ana = None

        # group the layers for easier checkpoint restoring, i.e.
        self.observation_models = {'sensor': self.sensor_model_layer,
                                   'obs': self.observation_model_layer}
        self.observation_noise_models = \
            {'het_diag': self.observation_noise_hetero_diag,
             'het_full': self.observation_noise_hetero_full,
             'const_diag': self.observation_noise_const_diag,
             'const_full': self.observation_noise_const_full,
             'like': self.likelihood_layer}

        self.process_models = {self.process_model_learned_layer,
                               self.process_model_analytical_layer}
        self.process_noise_models = {}

    ###########################################################################
    # observation models
    ###########################################################################
    def sensor_model(self, raw_observations, training):
        """
        Process raw observations and return the predicted observations z
        for the filter and an encoding for predicting the observation noise

        Parameters
        ----------
        raw_observations : list of tensors
            Raw sensory observations
        training : boolean tensor
            flag that indicates if model is in training or test mode

        Returns
        -------
        z : tensor [batch_size, dim_z]
            Low-dimensional observations
        enc : tensor or list of tensors
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation noise or the learned observation update
            of the particle filter
        """
        z, enc = self.sensor_model_layer(raw_observations, training)
        return z, enc

    def observation_noise(self, encoding, hetero, diag, training):
        """
        Observation noise model

        Parameters
        ----------
        encoding : ensor or list of tensors
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation
        hetero : bool
            predict heteroscedastic observation noise? (else constant)
        diag : bool
            predict diagonal noise covariance matrices? (else full)
        training : bool
            training or testing?

        Returns
        -------
        tf.keras.layer
            A layer that predicts the desired observation noise

        """
        if hetero:
            if diag:
                return self.observation_noise_hetero_diag(encoding, training)
            else:
                return self.observation_noise_hetero_full(encoding, training)
        else:
            if diag:
                return self.observation_noise_const_diag(encoding, training)
            else:
                return self.observation_noise_const_full(encoding, training)

    def likelihood(self, particles, encoding, training):
        """
        Learned observation update for the particle filter.
        Consumes an encoding of the raw observatuions and the predicted
        particles and returns the likelihood of each particle

        Parameters
        ----------
        particles : tensor [batch_size, num_particles, dim_z]
            Predicted observations for the particles
        encoding : tensor or list of tensors
            An encoding of the raw observations
        training : bool
            training or testing?

        Returns
        -------
        tf.keras.layer
            A layer that predicts the likelihood of the observations under each
            particle

        """
        return self.likelihood_layer([particles, encoding], training)

    def observation_model(self, state, training):
        """
        Predicts the observations for a given state

        Parameters
        ----------
        state : tensor [batch_size, dim_x]
            the predicted state
        training : bool
            training or testing?

        Returns
        -------
        tf.keras.layer
            A layer that computes the expected observations for the input
            state and the Jacobian  of the observation model
        """
        return self.observation_model_layer(state, training)

    ###########################################################################
    # process models
    ###########################################################################
    def process_model(self, old_state, action, learned, training):
        """
        Predicts the next state given the old state and actions performed

        Parameters
        ----------
        old_state : tensor [batch_size, dim_x]
            the previous state
        action : tensor [batch_size, dim_u]
            the performed actions
        learned : bool
            use a learned process model? (else: analytical process model)
        training : bool
            training or testing?

        Returns
        -------
        new_state : tensor [batch_size, dim_x]
            the predicted next state
        F : tensor [batch_size, dim_x, dim_x]
            the jacobian of the process model

        """
        if learned:
            new_state, F = \
                self.process_model_learned_layer([old_state, action], training)
        else:
            new_state, F = \
                self.process_model_analytical_layer([old_state, action],
                                                    training)
        new_state = self.correct_state(new_state, diff=False)
        return new_state, F

    def process_noise(self, old_state, action, learned, hetero, diag,
                      training):
        """
        Consumes the old state and action and predicts the process
        noise with the desired attributs

        Parameters
        ----------
        old_state : tensor [batch_size, dim_x]
            the previous state
        action : tensor [batch_size, dim_u]
            the performed actions
        learned : bool
            use a learned process model? (else: analytical process model)
            NOTE: this is only necessary for being able to pretrain different
            noise models for the learned and analytical process model
         hetero : bool
            predict heteroscedastic process noise? (else constant)
        diag : bool
            predict diagonal noise covariance matrices? (else full)
        training : bool
            training or testing?

        Returns
        -------
        tf.keras.layer
            A layer that computes the desired process noise
        """
        if learned:
            if hetero:
                if diag:
                    return self.process_noise_hetero_diag_lrn([old_state,
                                                               action],
                                                              training)
                else:
                    return self.process_noise_hetero_full_lrn([old_state,
                                                              action],
                                                              training)
            else:
                if diag:
                    return self.process_noise_const_diag_lrn([old_state,
                                                              action],
                                                             training)
                else:
                    return self.process_noise_const_full_lrn([old_state,
                                                              action],
                                                             training)
        else:
            if hetero:
                if diag:
                    return self.process_noise_hetero_diag_ana([old_state,
                                                              action],
                                                              training)
                else:
                    return self.process_noise_hetero_full_ana([old_state,
                                                              action],
                                                              training)
            else:
                if diag:
                    return self.process_noise_const_diag_ana([old_state,
                                                              action],
                                                             training)
                else:
                    return self.process_noise_const_full_ana([old_state,
                                                              action],
                                                             training)

    ###########################################################################
    # keeping the state correct
    ###########################################################################
    def correct_state(self, state, diff=True):
        """
        Correct a state to account for e.g. angle intervals. In the easiest
        case, nothing needs to be done.

        Parameters
        ----------
        state : tensor [batch_size, dim_x]
            the state
        diff : bool, optional
            treat the state as a difference between two states?

        Returns
        -------
        state : tensor [batch_size, dim_x]
            the corrected state

        """
        return state

    def correct_observation_diff(self, diff):
        """
        Correct a difference between two observations to account for e.g.
        angle intervals. In the easiest  case, nothing needs to be done.

        Parameters
        ----------
        state : tensor [batch_size, dim_z]
            the observation difference

        Returns
        -------
        state : tensor [batch_size, dim_z]
            the corrected observation difference

        """
        return diff

    def weighted_state_mean_with_angles(self, points, weights, axis=1):
        """
        Correctly compute the mean of a set of particles or sigma points.
        In the easiest  case, nothing needs to be done. However, for states
        that contain angles a suitable method must be implemented.

        Parameters
        ----------
        points : tensor [batch_size, num_samples, dim_x]
            a set of particles or sigma points
        weights : tensor [batch_size, num_samples]
            their weights

        Returns
        -------
        mean : tensor [batch_size, dim_x]
            the mean state

        """
        mult = tf.multiply(points, weights)
        mean = tf.reduce_sum(mult, axis=axis)
        return mean

    def weighted_observation_mean_with_angles(self, points, weights, axis=1):
        """
        Correctly compute the mean of a set of observations for particles or
        sigma points.
        In the easiest case, nothing needs to be done. However, for
        observations that contain angles a suitable method must be implemented.

        Parameters
        ----------
        points : tensor [batch_size, num_samples, dim_z]
            a set of observations for particles or sigma points
        weights : tensor [batch_size, num_samples]
            their weights

        Returns
        -------
        mean : tensor [batch_size, dim_z]
            the mean observation

        """
        mult = tf.multiply(points, weights)
        mean = tf.reduce_sum(mult, axis=axis)
        return mean

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
        raise NotImplementedError("Please implement this method")

    def get_observation_loss(self, prediction, label, step, training):
        """
        Compute the loss for the observation functions - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors
            step: training step
            training: boolean tensor, indicates if we compute a loss for
            training or testing

        Returns:
            loss: the total loss for training the observation preprocessing
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        raise NotImplementedError("Please implement this method")

    def get_process_loss(self, prediction, label, step, training):
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
        raise NotImplementedError("Please implement this method")

    ###########################################################################
    # data loading
    ###########################################################################
    def tf_record_map(self, path, name, dataset, data_mode, train_mode,
                      num_threads=3):
        """
        Defines how to read in the data from a tf record
        """
        raise NotImplementedError("Please Implement this method")

    ###########################################################################
    # loss functions
    ###########################################################################
    def _likelihood(self, diff, covar, reduce_mean=False):
        """
        Compute the negative log likelihood of y under a gaussian
        with mean mu and covariance covar, where x and y are given as
        diff = y - mu

        Parameters
        ----------
        diff : tensor
            difference between y and the mean of the gaussian
        covar : tensor
            covariance matrix of the gaussian
        reduce_mean : bool, optional
            if true, return the mean likelihood loss over the complete tensor.
            The default is False.

        Returns
        -------
        likelihood : tensor
            the negative log likelihood

        """
        dim = diff.get_shape()[-1].value

        # transfer to float 64 for higher accuracy
        covar = tf.cast(covar, tf.float64)
        diff = tf.cast(diff, tf.float64)

        if dim > 1:
            if len(diff.get_shape().as_list()) > 2:
                diff = tf.reshape(diff, [self.batch_size, -1, dim, 1])
            else:
                diff = tf.reshape(diff, [self.batch_size, dim, 1])

            c_inv = tf.linalg.inv(covar)

            err = tf.matmul(tf.matmul(tf.linalg.matrix_transpose(diff),
                                      c_inv), diff)
            err = tf.reshape(err, [self.batch_size, -1])

            # the error term needs to be finite and positive
            err = tf.where(tf.math.is_finite(err), err, tf.ones_like(err)*500)
            err = tf.where(tf.greater_equal(err, 0), err,
                           tf.ones_like(err)*500)

            det = tf.reshape(tf.math.log(tf.linalg.det(covar)),
                             [self.batch_size, -1])
            # nans mostly come from too small values, so we replace them with
            # zero
            det = tf.where(tf.math.is_finite(det), det, tf.zeros_like(det))

            with tf.control_dependencies(
                [tf.debugging.assert_all_finite(det, name='det',
                 message='det')]):
                likelihood = err + det
        else:
            diff = tf.reshape(diff, [self.batch_size, -1])
            covar = tf.reshape(covar, [self.batch_size, -1])
            det = tf.math.log(covar)
            err = (diff**2)/covar

            # the error term needs to be finite and positive
            err = tf.where(tf.math.is_finite(err), err,
                           tf.ones_like(err)*500)
            err = tf.where(tf.greater_equal(err, 0), err,
                           tf.ones_like(err)*500)

            likelihood = det + err

        likelihood = 0.5 * (likelihood + dim * np.log(2*np.pi))
        likelihood = tf.reshape(likelihood, [self.batch_size, -1])
        likelihood = tf.cast(likelihood, tf.float32)

        if reduce_mean:
            likelihood = tf.reduce_mean(likelihood)

        return likelihood

    def _mixture_likelihood(self, diffs, weights, reduce_mean=False):
        """
        Compute the negative log likelihood of y under a a gaussian
        mixture model defined by a set of particles and their weights.

        Parameters
        ----------
        diffs : tensor
            difference between y and the states of the particles
        weights : tensor
            weights of the particles
        reduce_mean : bool, optional
            if true, return the mean likelihood loss over the complete tensor.
            The default is False.

        Returns
        -------
        likelihood : tensor
            the negative log likelihood

        """
        dim = diffs.get_shape()[-1].value
        num = diffs.get_shape()[-2].value

        # remove nans and infs and replace them with high values/zeros
        diffs = tf.where(tf.math.is_finite(diffs), diffs,
                         tf.ones_like(diffs)*1e5/self.scale)
        weights = tf.where(tf.math.is_finite(weights), weights,
                           tf.zeros_like(weights))
        weights /= tf.reduce_sum(weights, axis=-1, keepdims=True)

        covar = np.ones(dim, dtype=np.float32)
        for k in range(dim):
            covar[k] *= self.mixture_std/self.scale

        covar = tf.linalg.diag(tf.square(covar))
        if len(diffs.get_shape().as_list()) > 3:
            sl = diffs.get_shape()[1].value
            diffs = tf.reshape(diffs, [self.batch_size, -1, num, dim, 1])
            covar = tf.tile(covar[None, None, None, :, :],
                            [self.batch_size, sl, num, 1, 1])
        else:
            sl = 1
            diffs = tf.reshape(diffs, [self.batch_size, num, dim, 1])
            covar = tf.tile(covar[None, None, :, :],
                            [self.batch_size, num, 1, 1])

        # transfer to float 64 for higher accuracy
        covar = tf.cast(covar, tf.float64)
        diffs = tf.cast(diffs, tf.float64)
        weights = tf.cast(weights, tf.float64)

        exponent = tf.matmul(tf.matmul(tf.linalg.matrix_transpose(diffs),
                                       tf.linalg.inv(covar)), diffs)
        exponent = tf.reshape(exponent, [self.batch_size, sl, num])

        normalizer = tf.math.log(tf.linalg.det(covar)) + \
            tf.cast(dim * tf.log(2*np.pi), tf.float64)

        log_like = -0.5 * (exponent + normalizer)
        log_like = tf.reshape(log_like, [self.batch_size, sl, num])

        log_like = tf.where(tf.greater_equal(log_like, -500), log_like,
                            tf.ones_like(log_like)*-500)

        exp = tf.exp(log_like)

        # the per particle likelihoods are weighted and summed in the particle
        # dimension
        weighted = weights * exp
        weighted = tf.reduce_sum(weighted, axis=-1)

        # compute the negative logarithm and undo the bias
        likelihood = - (tf.math.log(tf.maximum(weighted, 1e-300)))

        if reduce_mean:
            likelihood = tf.reduce_mean(likelihood)

        likelihood = tf.cast(likelihood, tf.float32)

        return likelihood

    def _mse(self, diff, reduce_mean=False):
        """
        Returns the mean squared error of diff = label - pred plus their
        euclidean distance (dist)

        Parameters
        ----------
        diff : tensor
            difference between label and prediction
        reduce_mean : bool, optional
            if true, return the mean errors over the complete tensor. The
            default is False.

        Returns
        -------
        loss : tensor
            the mean squared error
        dist : tensor
            the euclidean distance

        """
        diff_a = tf.expand_dims(diff, axis=-1)
        diff_b = tf.expand_dims(diff, axis=-2)

        loss = tf.matmul(diff_b, diff_a)

        # the loss needs to be finite and positive
        loss = tf.where(tf.math.is_finite(loss), loss,
                        tf.ones_like(loss)*1e20)
        loss = tf.where(tf.greater_equal(loss, 0), loss,
                        tf.ones_like(loss)*1e20)

        loss = tf.squeeze(loss, axis=-1)
        dist = tf.sqrt(loss)

        if reduce_mean:
            loss = tf.reduce_mean(loss)
            dist = tf.reduce_mean(dist)

        return loss, dist

    ######################################
    # Evaluation
    ######################################
    def save_log(self, log_dict, out_dir, step, num=0, mode='filter'):
        """
        A helper to save the results of testing a filter on a a given problem.

        Parameters
        ----------
        log_dict : dict
            dictionary of the losses that should be logged (as lists, one loss
            per test example)
        out_dir : str
            the directory where the results are written to
        step : int
            the training step of the model that ahs been evaluated
        num : int, optional
            The number of this test run (if the model is evaluated several
                                         times)
        mode : str, optional
            flag that indicates if the context is run in filtering or
            pretraining mode.

        """
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
