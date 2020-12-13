#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:42:19 2020

@author: akloss
"""

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import tensorflow as tf


class BaseLayer(tf.keras.layers.Layer):
    def __init__(self):
        """
        A base class providing
         - wrappers around some tensorflow layers to set default arguments
         - a function to compute jacobians
         - a spatial-softmax layer


        The list "updatable" is used to register paramaeters like moving
        averages that need to be updated during training (since this is not
        automatically handeld in tensorflow 1)
        """
        super(BaseLayer, self).__init__()
        self.updateable = []

    ###########################################################################
    # convenience functions
    ###########################################################################
    def _conv_layer(self, name, kernel_size, output_channels, stride=[1, 1],
                    padding='VALID', activation=tf.nn.relu, std=None, mean=0,
                    bias=1e-3, wd_w=1e-3, wd_b=1e-3, add_bias=True,
                    trainable=True):
        """
        Helper to create a 2d convolution layer

        Parameters
        ----------
        name : str
            name of the layer
        kernel_size : int
            kernel size
        output_channels : int
            number of output channels
        stride : int, optional
            stride for the convolution. The default is [1, 1].
        padding : int, optional
            padding ('SAME' or 'VALID'). The default is 'VALID'.
        activation : function, optional
            activation function. The default is tf.nn.relu.
        std : float, optional
            standard deviation of a truncated Gaussian for initializing
                the kernel weights. The default is None.
        mean : float, optional
            mean of the gaussian for initializing the kernel weights. The
            default is 0.
        bias : float, optional
            constant value to which to initialize the bias. The default is
            1e-3.
        wd_w : float, optional
            weight decay factor for the kernel weights. The default is 1e-3.
        wd_b : float, optional
            weight decay factor for the bias. The default is 1e-3.
        add_bias : bool, optional
            whether to add a bias. The default is True.
        trainable : bool, optional
            make this layer's variable trainable?. The default is True.

        Returns
        -------
        lay : tf.keras.layers.Conv2D
            A keras conv2D layer with the requested parameters

        """
        if std is None:
            init_w = tf.initializers.glorot_normal()
        else:
            init_w = tf.keras.initializers.truncated_normal(stddev=std,
                                                            mean=mean)
        init_b = tf.constant_initializer(bias)

        lay = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=kernel_size,
            strides=stride, padding=padding, activation=activation,
            use_bias=add_bias,
            kernel_initializer=init_w, bias_initializer=init_b,
            kernel_regularizer=tf.keras.regularizers.l2(l=wd_w),
            bias_regularizer=tf.keras.regularizers.l2(l=wd_b),
            name=name, trainable=trainable)

        return lay

    def _deconv_layer(self, name, kernel_size, output_channels, stride=[1, 1],
                      padding='VALID', activation=tf.nn.relu, std=None, mean=0,
                      bias=1e-3, wd_w=1e-3, wd_b=1e-3, add_bias=True,
                      trainable=True):
        """
        Helper to create a 2d deconvolution layer

        Parameters
        ----------
        name : str
            name of the layer
        kernel_size : int
            kernel size
        output_channels : int
            number of output channels
        stride : int, optional
            stride for the deconvolution. The default is [1, 1].
        padding : int, optional
            padding ('SAME' or 'VALID'). The default is 'VALID'.
        activation : function, optional
            activation function. The default is tf.nn.relu.
        std : float, optional
            standard deviation of a truncated Gaussian for initializing
                the kernel weights. The default is None.
        mean : float, optional
            mean of the gaussian for initializing the kernel weights. The
            default is 0.
        bias : float, optional
            constant value to which to initialize the bias. The default is
            1e-3.
        wd_w : float, optional
            weight decay factor for the kernel weights. The default is 1e-3.
        wd_b : float, optional
            weight decay factor for the bias. The default is 1e-3.
        add_bias : bool, optional
            whether to add a bias. The default is True.
        trainable : bool, optional
            make this layer's variable trainable?. The default is True.

        Returns
        -------
        lay : tf.keras.layers.Conv2DTranspose
            A keras Conv2DTranspose layer with the requested parameters

        """
        if std is None:
            init_w = tf.initializers.glorot_normal()
        else:
            init_w = tf.keras.initializers.truncated_normal(stddev=std,
                                                            mean=mean)
        init_b = tf.constant_initializer(bias)

        lay = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=kernel_size,
            strides=stride, padding=padding, activation=activation,
            use_bias=add_bias,
            kernel_initializer=init_w, bias_initializer=init_b,
            kernel_regularizer=tf.keras.regularizers.l2(l=wd_w),
            bias_regularizer=tf.keras.regularizers.l2(l=wd_b),
            name=name, trainable=trainable)

        return lay

    def _fc_layer(self, name, output_channels, activation=tf.nn.relu, std=None,
                  mean=0, bias=1e-3, wd_w=1e-3, wd_b=1e-3, add_bias=True,
                  trainable=True):
        """
        Helper to create a 2d fully connected layer

        Parameters
        ----------
        name : str
            name of the layer
        output_channels : int
            number of output channels
        activation : function, optional
            activation function. The default is tf.nn.relu.
        std : float, optional
            standard deviation of a truncated Gaussian for initializing
                the kernel weights. The default is None.
        mean : float, optional
            mean of the gaussian for initializing the kernel weights. The
            default is 0.
        bias : float, optional
            constant value to which to initialize the bias. The default is
            1e-3.
        wd_w : float, optional
            weight decay factor for the kernel weights. The default is 1e-3.
        wd_b : float, optional
            weight decay factor for the bias. The default is 1e-3.
        add_bias : bool, optional
            whether to add a bias. The default is True.
        trainable : bool, optional
            make this layer's variable trainable?. The default is True.

        Returns
        -------
        lay : tf.keras.layers.Dense
            A keras Cdense layer with the requested parameters

        """
        if std is None:
            init_w = tf.initializers.glorot_normal()
        else:
            init_w = tf.keras.initializers.truncated_normal(stddev=std,
                                                            mean=mean)
        init_b = tf.constant_initializer(bias)

        lay = tf.keras.layers.Dense(
            units=output_channels, activation=activation,
            use_bias=add_bias,
            kernel_initializer=init_w, bias_initializer=init_b,
            kernel_regularizer=tf.keras.regularizers.l2(l=wd_w),
            bias_regularizer=tf.keras.regularizers.l2(l=wd_b),
            name=name, trainable=trainable)

        return lay

    def _compute_jacobian(self, ys, xs, no_batch=False):
        """
        Helper to compute the jacobian of ys wrt. xs


        Parameters
        ----------
        ys : tensor
            tensor to be derived
        xs : tensor
            tensor with respect to whom to derive
        no_batch : TYPE, optional
            Whether the tensors have a leading batch dimension or not. The
            default is False.

        Returns
        -------
        J : tensor
            J = d ys / d xs

        """
        if no_batch:
            xs = tf.reshape(xs, [-1])
            ys = tf.reshape(ys, [-1])
            y_num = ys.get_shape()[0]
            x_num = xs.get_shape()[0]

            if y_num.value > 1:
                tmp = []
                yys = tf.unstack(ys)
                for ind, y in enumerate(yys):
                    grad = tf.gradients(y, xs)
                    if grad == [None]:
                        grad = [tf.zeros([x_num])]
                    tmp += [grad]
                tmp = tf.stack(tmp)
            else:
                tmp = tf.gradients(ys, xs)
                if tmp == [None]:
                    tmp = tf.zeros([y_num, x_num])

            tmp = tf.reshape(tmp, [y_num, x_num])
            return tmp

        ys = tf.reshape(ys, [self.batch_size, -1])
        out = []
        for b, batch in enumerate(tf.unstack(ys)):
            tmp = []
            for y in tf.unstack(batch):
                tmp += [tf.gradients(y, xs)]
            # each gradient tensor in tmp is batch_size x dim_x, but we
            # only need the parts that correspond to the bth batch
            tmp = [tf.slice(t[0], [b, 0], [1, -1]) for t in tmp]
            tmp = tf.stack(tmp)
            if len(tmp.get_shape()) > 2:
                tmp = tf.reshape(tmp, [-1, xs.get_shape()[1].value])
            out += [tmp]
        J = tf.stack(out)
        return J

    def _normalize_2d(self, bottom, name='normalizer', method='normal',
                      summary=True):
        """
        Helper to turn a 2d input into a 2d distribution

        Parameters
        ----------
        bottom : tensor [batch_size, height, width, channels]
            Dinput tensor
        name : str, optional
            name of the oberation. The default is 'normalizer'.
        method : str, optional
            What method to use for the normalization, either "normal" (i.e
            divide by the sum over the image) or  "softmax". The default is '
            normal'.
        summary : bool, optional
            write out summary information to tensorboard?. The default is True.

        Returns
        -------
        out : tensor
            the normalized output tensor

        """
        with tf.variable_scope(name + '/normalize') as scope:
            sh = bottom.get_shape().as_list()

            # the input should be of shape (batch, height, width, channels)
            # or (height, width, channels)
            image_dim = sh[-2] * sh[-3]
            channel_dim = sh[-1]
            # Assume features is of size [(N), H, W, C]
            # Transpose it to [(N), C, H, W], reshape to [N * C, H * W]
            if len(sh) == 4:
                channel_dim *= sh[0]
                features = tf.reshape(tf.transpose(bottom, [0, 3, 1, 2]),
                                      [channel_dim, image_dim])
            else:
                features = tf.reshape(tf.transpose(features, [2, 0, 1]),
                                      [channel_dim, image_dim])

            features = features - tf.reduce_min(features, reduction_indices=1,
                                                keepdims=True)

            if method == 'normal':
                normalizer = tf.reduce_sum(features, reduction_indices=1,
                                           keepdims=True)

                # check each channel
                fs = tf.unstack(features, axis=0)
                tmp = []
                for ind, chan in enumerate(tf.unstack(normalizer, axis=0)):
                    # if the normalizer is 0, we set each element to 1/#elem
                    tmp += [tf.cond(tf.squeeze(tf.equal(chan, 0.)),
                                    lambda: tf.divide(tf.ones_like(fs[ind]),
                                                      tf.cast(tf.size(fs[ind]),
                                                              tf.float32)),
                                    lambda: tf.divide(fs[ind], chan))]
                normalized = tf.stack(tmp)
                # create activation summary
                if summary and not scope.reuse:
                    if scope.name not in name:
                        tf.summary.histogram(scope.name + '/' + name,
                                             normalizer)
                    else:
                        tf.summary.histogram(name + '/normalizer', normalizer)
            elif method == 'softmax':
                normalized = tf.nn.softmax(features)

            # Reshape and transpose back to original format.
            if len(sh) == 4:
                out = tf.transpose(tf.reshape(normalized,
                                              [sh[0], sh[3], sh[1], sh[2]]),
                                   [0, 2, 3, 1])
            else:
                out = tf.transpose(tf.reshape(normalized,
                                              [sh[2], sh[0], sh[1]]),
                                   [1, 2, 0])

            return out

    def _spatial_softmax(self, bottom, name, method='softmax', summary=False):
        """
        Helper to find the pixel position of the mean of a 2d input. First
        computes the softmax of the input and then returns the 2d position
        of the mean.

        Parameters
        ----------
        bottom : tensor [batch_size, height, width, channels]
            the input tensor
        name : str
            the name of the operation
        method : str, optional
            What method to use for normalizating the input tensor, either
            "normal" (i.e divide by the sum over the image) or  "softmax".
            The default is 'softmax'.
        summary : bool, optional
            write out summary information to tensorboard?. The default is True.

        Returns
        -------
        out : tensor [batch_size, channels, 2]
            the pixel coordinates of the mean

        """
        sh = bottom.get_shape().as_list()
        dist = self._normalize_2d(bottom, name, method, summary)

        # image_coords is a tensor of size [H, W, 2] representing the
        # image coordinates of each pixel.
        x_vals = tf.expand_dims(tf.linspace(-sh[-2]/2, sh[-2]/2., sh[-2]),
                                1)
        x_t = tf.matmul(tf.ones(shape=[sh[-3], 1]),
                        tf.transpose(x_vals, [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-sh[-3]/2, sh[-3]/2.,
                                                   sh[-3]), 1),
                        tf.ones(shape=[1, sh[-2]]))

        xs = tf.expand_dims(x_t, -1)
        ys = tf.expand_dims(y_t, -1)

        image_coords = tf.concat(axis=-1, values=[xs, ys])

        # Convert the distribution to shape [N, H, W, C, 1]
        dist_ex = tf.expand_dims(dist, -1)

        # Convert image coords to shape [H, W, 1, 2]
        image_coords = tf.expand_dims(image_coords, 2)
        # Convert image coords to shape [1, H, W, 1, 2]
        image_coords = tf.expand_dims(image_coords, 0)
        # tile
        image_coords = tf.tile(image_coords,
                               [sh[0], 1, 1, sh[-1], 1])
        # Multiply (with broadcasting) and reduce over image dimensions to
        # get the result of shape [N, C, 2]
        mult = dist_ex * image_coords
        out = tf.reduce_sum(mult, reduction_indices=[1, 2])

        # create activation summary
        scope = tf.get_variable_scope()
        if summary:
            if scope.name not in name:
                tf.summary.histogram(scope.name + '/' + name, out)
                tf.summary.image(scope.name + '/' + name, dist)
            else:
                tf.summary.histogram(name, out)
                tf.summary.image(name, dist)
        return out
