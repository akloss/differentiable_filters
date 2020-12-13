# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:51:37 2020

@author: akloss
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np


class FilterCellBase(tf.keras.layers.AbstractRNNCell):
    def __init__(self, param, context):
        """
        Abstract base class for Filter Cells

        Parameters
        ----------
        param : dict
            parmaters
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)

        Returns
        -------
        None.

        """
        tf.keras.layers.AbstractRNNCell.__init__(self)
        self.context = context
        self.param = param

        self.epoch_size = None
        self.debug = param['debug']

        # shape  related information
        self.batch_size = param['batch_size']
        self.sequence_length = param['sequence_length']
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u

        # define if process noise q and observation noise r should be learned
        # and if the noise should be heteroscedastic (state and action
        # dependent) or constant
        self.learn_q = param['learn_q']
        self.learn_r = param['learn_r']
        self.hetero_q = param['hetero_q']
        self.hetero_r = param['hetero_r']

        # learned or analytical process model
        self.learn_process = param['learn_process']

        # diagonal or full covariance matrices
        self.diagonal_covar = param['diagonal_covar']

        # define if the initial state should be perturbed randomly
        self.add_initial_noise = param['add_initial_noise']

        # rate at which observations are taken into account
        self.update_rate = tf.ones([], dtype=tf.float32) * param['update_rate']

        self.filter_layers = []

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        raise NotImplementedError("Please implement this method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Please implement this method")

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
        raise NotImplementedError("Please implement this method")

    ###########################################################################
    # convenience functions for ensuring stability
    ###########################################################################
    def _condition_number(self, s):
        """
        Compute the condition number of a matrix based on its eigenvalues s

        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix

        Returns
        -------
        r_corrected : tensor
            the condition number of the matrix

        """
        r = s[..., 0] / s[..., -1]

        # Replace NaNs in r with infinite
        r_nan = tf.math.is_nan(r)
        r_inf = tf.fill(tf.shape(r), tf.constant(np.Inf, r.dtype))
        r_corrected = tf.where(r_nan, r_inf, r)

        return r_corrected

    def _is_invertible(self, s, epsilon=1e-6):
        """
        Check if a matrix is invertible based on its eigenvalues s

        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        epsilon : float, optional
            threshold for the condition number

        Returns
        -------
        invertible : tf.bool tensor
            true if the matrix is invertible

        """
        "c"
        # Epsilon may be smaller with tf.float64
        eps_inv = tf.cast(1. / epsilon, s.dtype)
        cond_num = self._condition_number(s)
        invertible = tf.logical_and(tf.math.is_finite(cond_num),
                                    tf.less(cond_num, eps_inv))
        return invertible

    def _make_valid(self, covar):
        """
        Trys to make a possibly degenerate covariance valid by
          - replacing nans and infs with high values/zeros
          - making the matrix symmetric
          - trying to make the matrix invertible by adding small offsets to
            the smallest eigenvalues

        Parameters
        ----------
        covar : tensor
            a covariance matrix that is possibly degenerate

        Returns
        -------
        covar_valid : tensor
            a covariance matrix that is hopefully valid

        """
        # eliminate nans and infs (replace them with high values on the
        # diagonal and zeros else)
        bs = covar.get_shape()[0].value
        dim = covar.get_shape()[-1].value
        covar = tf.where(tf.math.is_finite(covar), covar,
                         tf.eye(dim, batch_shape=[bs])*1e5)

        # make symmetric
        covar = (covar + tf.linalg.matrix_transpose(covar)) / 2.

        # add a bit of noise to the diagonal of c to prevent
        # nans in the gradient of the svd
        noise = tf.random.uniform(covar.get_shape().as_list()[:-1], minval=0,
                                  maxval=0.001/self.context.scale**2)
        s, u, v = tf.linalg.svd(covar + tf.linalg.diag(noise))
        # test if the matrix is invertible
        invertible = self._is_invertible(s)
        # test if the matrix is positive definite
        pd = tf.reduce_all(tf.greater(s, 0), axis=-1)

        # try making a valid version of the covariance matrix by ensuring that
        # the minimum eigenvalue is at least 1e-4/self.context.scale
        min_eig = s[..., -1:]
        eps = tf.tile(tf.maximum(1e-4/self.context.scale - min_eig, 0),
                      [1, s.get_shape()[-1]])
        covar_invertible = tf.matmul(u, tf.matmul(tf.linalg.diag(s + eps), v,
                                                  adjoint_b=True))

        # if the covariance matrix is valid, leave it as is, else replace with
        # the modified variant
        covar_valid = tf.where(tf.logical_and(invertible, pd),
                               covar, covar_invertible)

        # make symmetric again
        covar_valid = \
            (covar_valid + tf.linalg.matrix_transpose(covar_valid)) / 2.

        return covar_valid

    def zca_whiten(self, data):
        """
        Whiten noise sampled from a standard normal distribution

        Parameters
        ----------
        data : tensor tf.float32 [batch_size, num_samples, dim_x]
            the sampled noise

        Returns
        -------
        whitened : tensor tf.float64 [batch_size, num_samples, dim_x]
            the whitened noise samples

        """
        # input tensor is [batch_size, num_samples, dim_x]
        num = data.get_shape()[1]

        data = tf.cast(data, tf.float64)

        # center the samples
        mean = tf.reduce_mean(data, axis=1, keep_dims=True)
        centered = data - tf.tile(mean, [1, num, 1])

        # whiten
        # compute the current covariance
        diff_square = tf.matmul(centered[:, :, :, None],
                                centered[:, :, None, :])
        # sigma: [batch_size, dim_x, dim_x]
        sigma = tf.reduce_mean(diff_square, axis=1)
        # get the whitening matrix
        # s: [batch_size, dim_x], u: [batch_size, dim_x, dim_x]
        s, u, _ = tf.svd(sigma, full_matrices=True)
        s_inv = 1. / tf.sqrt(s + 1e-5)
        s_inv = tf.linalg.diag(s_inv)
        # w: [batch_size, dim_x, dim_x]
        w = tf.matmul(u, tf.matmul(s_inv, u, transpose_b=True))

        whitened = tf.matmul(centered, w)

        return whitened
