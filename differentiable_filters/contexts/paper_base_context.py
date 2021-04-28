"""
Base class for the contexts as used in the paper "How to Train Your
Differentiable Filter". Contains code that is shared between all three
contexts.
"""

# this code only works with tensorflow 1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os
import csv

from differentiable_filters.contexts import base_context as base
import differentiable_filters.utils.tensorflow_compatability as compat



class PaperBaseContext(base.BaseContext):
    def __init__(self, param, mode):
        """
        Base class for the contexts used in the paper containing shared
        functions.

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
        super(base.BaseContext, self).__init__()
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
        self.scale = param['scale']
        self.sl = param['sequence_length']

    ###########################################################################
    # observation models
    ###########################################################################
    def run_sensor_model(self, raw_observations, training):
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

    def get_observation_noise(self, encoding, training):
        """
        Observation noise model

        Parameters
        ----------
        encoding : ensor or list of tensors
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation
        training : bool
            training or testing?

        Returns
        -------
        R : tensor [batch_size, dim_z, dim_z]
            Observation noise covariance matrix

        """
        if not self.param['learn_r']:
            return tf.tile(self.R[None, :, :], [self.batch_size, 1, 1])
        if self.param['hetero_r']:
            if self.param['diagonal_covar']:
                return self.observation_noise_hetero_diag(encoding, training)
            else:
                return self.observation_noise_hetero_full(encoding, training)
        else:
            if self.param['diagonal_covar']:
                return self.observation_noise_const_diag(encoding, training)
            else:
                return self.observation_noise_const_full(encoding, training)

    def get_observation_likelihood(self, particles, encoding, training):
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

    def run_observation_model(self, state, training):
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
    def run_process_model(self, old_state, action, training):
        """
        Predicts the next state given the old state and actions performed

        Parameters
        ----------
        old_state : tensor [batch_size, dim_x]
            the previous state
        action : tensor [batch_size, dim_u]
            the performed actions
        training : bool
            training or testing?

        Returns
        -------
        new_state : tensor [batch_size, dim_x]
            the predicted next state
        F : tensor [batch_size, dim_x, dim_x]
            the jacobian of the process model

        """
        if self.param['learn_process']:
            new_state, F = \
                self.process_model_learned_layer([old_state, action], training)
        else:
            new_state, F = \
                self.process_model_analytical_layer([old_state, action],
                                                    training)
        new_state = self.correct_state(new_state, diff=False)
        return new_state, F

    def get_process_noise(self, old_state, action, training):
        """
        Consumes the old state and action and predicts the process
        noise with the desired attributs

        Parameters
        ----------
        old_state : tensor [batch_size, dim_x]
            the previous state
        action : tensor [batch_size, dim_u]
            the performed actions
        training : bool
            training or testing?

        Returns
        -------
        tf.keras.layer
            A layer that computes the desired process noise
        """
        if not self.param['learn_q']:
            return tf.tile(self.Q[None, :, :], [self.batch_size, 1, 1])
        if self.param['learn_process']:
            if self.param['hetero_q']:
                if self.param['diagonal_covar']:
                    return self.process_noise_hetero_diag_lrn([old_state,
                                                               action],
                                                              training)
                else:
                    return self.process_noise_hetero_full_lrn([old_state,
                                                              action],
                                                              training)
            else:
                if self.param['diagonal_covar']:
                    return self.process_noise_const_diag_lrn([old_state,
                                                              action],
                                                             training)
                else:
                    return self.process_noise_const_full_lrn([old_state,
                                                              action],
                                                             training)
        else:
            if self.param['hetero_q']:
                if self.param['diagonal_covar']:
                    return self.process_noise_hetero_diag_ana([old_state,
                                                              action],
                                                              training)
                else:
                    return self.process_noise_hetero_full_ana([old_state,
                                                              action],
                                                              training)
            else:
                if self.param['diagonal_covar']:
                    return self.process_noise_const_diag_ana([old_state,
                                                              action],
                                                             training)
                else:
                    return self.process_noise_const_full_ana([old_state,
                                                              action],
                                                             training)

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
    # loss functions
    ###########################################################################
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
        dim = compat.get_dim_int(diffs, -1)
        num = compat.get_dim_int(diffs, -2)

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
            sl = compat.get_dim_int(diffs, 1)
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
