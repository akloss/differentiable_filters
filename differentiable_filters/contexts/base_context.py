"""
Abstract base class for filtering contexts. A context contains problem-specific
information such as the state size or process and sensor models.
"""


import numpy as np

import tensorflow as tf
import differentiable_filters.utils.tensorflow_compatability as compat


class BaseContext(tf.keras.Model):
    def __init__(self):
        """
        Abstract base class for filtering contexts. A context contains
        problem-specific information such as the state size or process and
        sensor models.

        For convenience, the class also implements methods for computing
        loss functions like likelihood, mean squared error and
        Bhattacharyya distance.

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
        encoding : tensor or list of tensors
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation noise or the learned observation update
            of the particle filter
        """
        raise NotImplementedError("Please implement this method")

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
        raise NotImplementedError("Please implement this method")

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
        likelihoods : tensor [batch_size, particles]
            The likelihood of the observations under each particle

        """
        raise NotImplementedError("Please implement this method")

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
        z_pred : tensor [batch_size, dim_z]
            The predicted observations for the current state estimate
        H : tensor [batch_size, dim_z, dim_x]
            The Jacobian of the observation model
        """
        raise NotImplementedError("Please implement this method")

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
        raise NotImplementedError("Please implement this method")

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
        Q : tensor [batch_size, dim_x, dim_x]
           The process noise covariance matrix
        """
        raise NotImplementedError("Please implement this method")

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
        dim = compat.get_dim_int(diff, -1)

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
        dim = compat.get_dim_int(diffs, -1)
        num = compat.get_dim_int(diffs, -2)

        # remove nans and infs and replace them with high values/zeros
        diffs = tf.where(tf.math.is_finite(diffs), diffs,
                         tf.ones_like(diffs)*1e5)
        weights = tf.where(tf.math.is_finite(weights), weights,
                           tf.zeros_like(weights))
        weights /= tf.reduce_sum(weights, axis=-1, keepdims=True)

        covar = np.ones(dim, dtype=np.float32)
        for k in range(dim):
            covar[k] *= self.mixture_std

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

    def _bhattacharyya(self, pred, label):
        """
        Computes the Bhattacharyya distance between two covariance matrices

        Parameters
        ----------
        pred : tensor
            the predicted covariance matrix
        label : tensor
            the true covariance matrix

        Returns
        -------
        dist : tensor
            The Bhattacharyya distance

        """
        mean = (pred + label) / 2.
        det_mean = tf.linalg.det(mean)
        det_pred = tf.linalg.det(pred)
        det_label = tf.linalg.det(label)
        dist = det_mean/(tf.sqrt(det_pred*det_label))
        dist = 0.5 * tf.math.log(dist)
        return dist
