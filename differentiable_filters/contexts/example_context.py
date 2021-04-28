"""
Simplified context class for the simultated disc tracking task described in the
paper "How to Train Your Differentiable Filter" to be used as example
implementation of a context.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from differentiable_filters.contexts import base_context as base


class ExampleContext(base.BaseContext):
    def __init__(self, batch_size, filter_type, loss, hetero_q, hetero_r,
                 learned_process, mixture_likelihood=True, mixture_std=0.1):
        """
        Minimal example context for the simulated disc tracking task. A context
        contains problem-specific information such as the state size or process
        and sensor models.

        Parameters
        ----------
        batch_size : int
            The batch size used.
        filter_type : str
            Which filter is used with the context.
        loss : str
            Which loss function to use.
        hetero_q : bool
            Learn heteroscedastic process noise?
        hetero_r : bool
            Learn heteroscedastic observation noise?
        learned_process : bool
            Learn the process model or use an analytical one?
        rescale : float, optional
            A factor by which the state-space has been downscaled. Default is 1.
        mixture_likelihood : bool, optional
            Only used with the particle filter. If true, the particle
            distribution is approximated with a GMM for calculating the
            nll loss. Else, a single gaussian is used. Default is True
        mixture_std : float, optional
            Only used with the particle filter if mixture_likelihood is true.
            The fixed covariance used for the individual gaussians in the GMM.
            Default is 0.1
        """
        super(ExampleContext, self).__init__()

        # mark which filter and loss function are used
        self.filter_type = filter_type
        self.loss = loss
        self.learned = learned_process
        self.mixture_likelihood = mixture_likelihood
        self.mixture_std = mixture_std

        # define the state size and name the components
        self.batch_size = batch_size

        # the state size
        self.dim_x = 4
        self.dim_u = 0
        self.dim_z = 2

        self.x_names = ['x', 'y', 'vx', 'vy']
        self.z_names = ['x', 'y']

        # parameters of the process model
        self.spring_force = 0.05
        self.drag_force = 0.0075

        # define initial values for the process noise q and observation noise r
        # diagonals
        # Important: All values are standard-deviations, so they are
        # squared for forming the covariance matrices
        self.q_diag = np.ones((self.dim_x)).astype(np.float32) * 10 / 60.
        self.q_diag = self.q_diag.astype(np.float32)

        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 30 / 60.
        self.r_diag = self.r_diag.astype(np.float32)

        # predefine all the necessary sub-models (defined below)
        # learned sensor model for processing the images
        self.sensor_model = SensorModel(self.batch_size)

        # learned observation noise
        self.observation_noise_model = \
            ObservationNoiseModel(batch_size, self.dim_z, self.r_diag,
                                  hetero_r)

        # learned process model if desired
        if self.learned:
            self.process_model = ProcessModel(self.batch_size, self.dim_x,
                                              jacobian=filter_type=='ekf')

        # learned process noise
        self.process_noise_model = \
            ProcessNoiseModel(batch_size, self.dim_x, self.q_diag, hetero_q)

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
        enc : tensor [batch_size, 32]
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation noise
        """
        return self.sensor_model(raw_observations, training)

    def get_observation_noise(self, encoding, training):
        """
        Observation noise model

        Parameters
        ----------
        encoding : ensor or list of tensors
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation
        training : boolean tensor
            flag that indicates if model is in training or test mode

        Returns
        -------
        R : tensor [batch_size, dim_z, dim_z]
            Observation noise covariance matrix

        """
        return self.observation_noise_model(encoding, training)

    def run_observation_model(self, state, training):
        """
        Predicts the observations for a given state

        Parameters
        ----------
        state : tensor [batch_size (x number of particles/sigma points), dim_x]
            the predicted state
        training : boolean tensor
            flag that indicates if model is in training or test mode

        Returns
        -------
        tf.keras.layer
            A layer that computes the expected observations for the input
            state and the Jacobian  of the observation model
        """
        bs = state.get_shape()[0] // self.batch_size
        H = tf.concat(
                [tf.tile(np.array([[[1, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size, 1, 1]),
                 tf.tile(np.array([[[0, 1, 0, 0]]], dtype=np.float32),
                         [self.batch_size, 1, 1])], axis=1)

        z_pred = tf.matmul(tf.tile(H, [bs, 1, 1]),
                           tf.expand_dims(state, -1))
        z_pred = tf.reshape(z_pred, [bs*self.batch_size, self.dim_z])
        return z_pred, H

    ###########################################################################
    # process model
    ###########################################################################
    def run_process_model(self, old_state, actions, training):
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
        if self.learned:
            new_state, F = self.process_model(old_state, training)
        else:

            # split the state into parts
            x = tf.slice(old_state, [0, 0], [-1, 1])
            y = tf.slice(old_state, [0, 1], [-1, 1])
            vx = tf.slice(old_state, [0, 2], [-1, 1])
            vy = tf.slice(old_state, [0, 3], [-1, 1])

            x_pred = x + vx
            y_pred = y + vy
            vx_pred = vx - self.spring_force * x - \
                self.drag_force * vx**2 * tf.sign(vx)
            vy_pred = vy - self.spring_force * y - \
                self.drag_force * vy**2 * tf.sign(vy)

            new_state = tf.concat([x_pred, y_pred, vx_pred, vy_pred], axis=1)

            if self.filter_type == 'ekf':
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
                F = tf.stop_gradient(F)
            else:
                F = None

        return new_state, F

    def get_process_noise(self, old_state, actions, training):
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
        return self.process_noise_model(old_state, training)

    ###########################################################################
    # loss functions
    ###########################################################################
    def get_loss(self, label, prediction):
        """
        Compute the loss for the filtering application - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors

        Returns:
            loss: the total loss for training the filtering application
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        particles, weights, states, covars, z, r, q = prediction
        seq_label, q_label, vis_label = label

        diff = seq_label - states

        # get the likelihood
        if self.filter_type == 'pf' and self.mixture_likelihood:
            num = particles.get_shape()[2]
            seq_label_tiled = tf.tile(seq_label[:, :, None, :], [1, 1, num, 1])
            particle_diff = seq_label_tiled - particles
            likelihood = self._mixture_likelihood(particle_diff, weights)
        else:
            likelihood = self._likelihood(diff, covars, reduce_mean=False)

        # compute the errors of the predicted states
        total_mse, total_dist = self._mse(diff, reduce_mean=False)

        # compute the error in the predicted observations (only for monitoring)
        diff_obs = seq_label[:, :, :2] - z
        mse_obs, dist_obs = self._mse(diff_obs, reduce_mean=False)

        # # mask out cases where the disc is not visible
        # dist_obs *= (tf.cast(tf.greater(vis_label[:, :, None], 0), tf.float32))

        # compute the error of the predicted process noise
        if len(q_label.get_shape()) == 3:
            q_label_m = tf.linalg.diag(tf.square(q_label))
            dist_q = self._bhattacharyya(q, q_label_m)
        else:
            dist_q = self._bhattacharyya(q, q_label)

        # compute the correlation between predicted observation noise and
        # the number of visible pixels of the red disc
        vis_label = tf.reshape(tf.cast(vis_label, tf.float32), [-1, 1])
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
        for la in self.layers:
            wd += la.losses
        wd = tf.add_n(wd)

        if self.loss == 'nll':
            total_loss = tf.reduce_mean(likelihood)
        elif self.loss == 'mse':
            total_loss = tf.reduce_mean(total_mse)
        elif self.loss == 'mixed':
            total_loss = (tf.reduce_mean(total_mse) +
                          tf.reduce_mean(likelihood)) / 2.
        elif self.loss == 'obs':
            total_loss = tf.reduce_mean(mse_obs)

        total = 100 * total_loss + wd
        # total = tf.reduce_mean(mse_obs) + wd

        metrics = [likelihood, total_mse, total_dist*60, dist_obs*60, dist_q,
                   corr_r, wd]
        metric_names = ['nll', 'mse', 'euclidean_distance',
                        'observation distance', 'q error',
                        'correlation r visibility', 'wd']
        return total, metrics, metric_names


class SensorModel(tf.keras.Model):
    def __init__(self, batch_size):
        super(SensorModel, self).__init__()
        self.batch_size = batch_size

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv1')
        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')

        self.flatten = tf.keras.layers.Flatten()

        self.sensor_fc1 = tf.keras.layers.Dense(
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3',
            activation=None)

    def call(self, images, training):
        conv1 = self.sensor_conv1(images)
        conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')
        conv2 = self.sensor_conv2(conv1)
        #conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')

        with tf.summary.record_if(tf.summary.experimental.get_step()%50==0):
            tf.summary.image('input_image', images[0:1],
                             step=tf.summary.experimental.get_step())
            for i in range(4):
                tf.summary.image('conv1_' + str(i), conv1[0:1, :, :, i:i+1],
                                 step=tf.summary.experimental.get_step())
            for i in range(8):
                tf.summary.image('conv2_' + str(i), conv2[0:1, :, :, i:i+1],
                                 step=tf.summary.experimental.get_step())

        input_data = self.flatten(conv2)
        fc1 = self.sensor_fc1(input_data)
        fc2 = self.sensor_fc2(fc1)
        pos = self.sensor_fc3(fc2)

        return pos, fc2


class ObservationNoiseModel(tf.keras.Model):
    def __init__(self, batch_size, dim_z, r_diag, hetero):
        super(ObservationNoiseModel, self).__init__()

        self.hetero = hetero
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.r_diag = r_diag

    def build(self, input_shape):
        init_const = np.ones(self.dim_z) * 1e-3
        init = np.sqrt(np.square(self.r_diag) - init_const)
        # the constant bias keeps the predicted covariance away from zero
        self.observation_noise_bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_z],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))

        if self.hetero:
            # for heteroscedastic noise with diagonal covariance matrix
            self.observation_noise_fc = tf.keras.layers.Dense(
                units=self.dim_z,
                activation=None,
                kernel_initializer=tf.initializers.glorot_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='observation_noise_fc2')
            self.observation_noise_bias_learned = \
                self.add_weight(name='het_diag_init_bias',
                                shape=[self.dim_z],
                                regularizer=tf.keras.regularizers.l2(l=1e-3),
                                initializer=tf.constant_initializer(init))
        else:
            # for constant noise with diagonal covariance matrix
            self.observation_noise_bias_learned = \
                self.add_weight(name='const_diag',
                                shape=[self.dim_z],
                                regularizer=tf.keras.regularizers.l2(l=1e-3),
                                initializer=tf.constant_initializer(init))

    def call(self, inputs, training):
        if self.hetero:
            diag = self.observation_noise_fc(inputs)
            diag = tf.square(diag + self.observation_noise_bias_learned)
            diag += self.observation_noise_bias_fixed
            R = tf.linalg.diag(diag)
        else:
            diag = self.observation_noise_bias_learned
            diag = tf.square(diag) + self.observation_noise_bias_fixed
            R = tf.linalg.tensor_diag(diag)
            R = tf.tile(R[None, :, :], [self.batch_size, 1, 1])

        return R


class ProcessModel(tf.keras.Model):
    def __init__(self, batch_size, dim_x, jacobian):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.jacobian = jacobian
        self.dim_x = dim_x

    def build(self, input_shape):
        self.process_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc1')
        self.process_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc2')
        self.process_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc3')

    def call(self, last_state, training):
        if self.jacobian:
            new_state, F = self.with_jacobian(last_state)
        else:
            fc1 = self.process_fc1(last_state)
            fc2 = self.process_fc2(fc1)
            update = self.process_fc3(fc2)

            new_state = last_state + update
            F = None

        return new_state, F

    @tf.function
    def with_jacobian(self, last_state):
        with tf.GradientTape() as tape:
            tape.watch(last_state)
            fc1 = self.process_fc1(last_state)
            fc2 = self.process_fc2(fc1)
            update = self.process_fc3(fc2)

            new_state = last_state + update
        F = tape.batch_jacobian(new_state, last_state)
        F = tf.stop_gradient(F)

        return new_state, F

class ProcessNoiseModel(tf.keras.Model):
    def __init__(self, batch_size, dim_x, q_diag, hetero):
        super(ProcessNoiseModel, self).__init__()

        self.hetero = hetero
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.q_diag = q_diag

    def build(self, input_shape):
        init_const = np.ones(self.dim_x) * 1e-3
        init = np.sqrt(np.square(self.q_diag) - init_const)
        # the constant bias keeps the predicted covariance away from zero
        self.process_noise_bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_x],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))

        if self.hetero:
            # for heteroscedastic noise with diagonal covariance matrix
            self.process_noise_fc1 = tf.keras.layers.Dense(
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.glorot_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='process_noise_fc1')
            self.process_noise_fc2 = tf.keras.layers.Dense(
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.glorot_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='process_noise_fc2')
            self.process_noise_fc3 = tf.keras.layers.Dense(
                units=self.dim_x,
                activation=None,
                kernel_initializer=tf.initializers.glorot_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='process_noise_fc3')
            self.process_noise_bias_learned = \
                self.add_weight(name='process_noise_bias_learned',
                                shape=[self.dim_x],
                                regularizer=tf.keras.regularizers.l2(l=1e-3),
                                initializer=tf.constant_initializer(init))
        else:
            # for constant noise with diagonal covariance matrix
            self.process_noise_bias_learned = \
                self.add_weight(name='process_noise_bias_learned',
                                shape=[self.dim_x],
                                regularizer=tf.keras.regularizers.l2(l=1e-3),
                                initializer=tf.constant_initializer(init))

    def call(self, old_state, training):
        if self.hetero:
            fc1 = self.process_noise_fc1(old_state)
            fc2 = self.process_noise_fc2(fc1)
            diag = self.process_noise_fc3(fc2)
            diag = tf.square(diag + self.process_noise_bias_learned)
            diag += self.process_noise_bias_fixed
            Q = tf.linalg.diag(diag)
        else:
            diag = self.process_noise_bias_learned
            diag = tf.square(diag) + self.process_noise_bias_fixed
            Q = tf.linalg.tensor_diag(diag)
            Q = tf.tile(Q[None, :, :], [self.batch_size, 1, 1])
        return Q
