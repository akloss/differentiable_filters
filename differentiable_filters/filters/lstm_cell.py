"""
Filtering with LSTM models.
"""

import tensorflow as tf
import numpy as np
from differentiable_filters.filters import filter_cell_base as base
from differentiable_filters.utils import base_layer as base2
import differentiable_filters.utils.tensorflow_compatability as compat


class UnstructuredCell(base.FilterCellBase, base2.BaseLayer):
    def __init__(self, context, problem, lstm_structure, num_units,
                 update_rate=1, debug=False):
        """
        Filtering with LSTM models. Depending on the class, the cell contains
        one or two layers of LSTM cells followed by fully connected layers that
        decode the LSTM state into a Gaussian satte estimate

        Parameters
        ----------
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)
        problem : str
            A string identifyer for the problem defined by the context
        lstm_structure : str
            String coding for the desired structure of the LSTM, either lstm1
            (for one lstm layer) or lstm2 (for two LSTM layers).
        num_units : int
            The number of units in the LSTM layers
        update_rate : int, optional
            The rate at which observations come in (allows simulating lower
            observation rates). Default is 1
        debug : bool, optional
            If true, the filters will print out information at each step.
            Default is False.
        """
        base.FilterCellBase.__init__(self, context, problem, update_rate,
                                     debug)
        base2.BaseLayer.__init__(self)

        self.structure = lstm_structure
        self.num_units = num_units

        if self.structure == 'lstm2':
            self.lstm1 = tf.keras.layers.LSTMCell(self.num_units)
            self.lstm2 = tf.keras.layers.LSTMCell(self.num_units)

            self.filter_layers = [self.lstm1, self.lstm2]
        elif self.structure == 'lstm1':
            self.lstm1 = tf.keras.layers.LSTMCell(self.num_units)
            self.filter_layers = [self.lstm1]

        if self.problem == 'pushing':
            self.pos_c1 = self._conv_layer('pos_c1', 5, 16,
                                           stride=[2, 2])
            self.pos_c2 = self._conv_layer('pos_c2', 3, 32)

            self.pos_fc = self._fc_layer('pos_fc', 4*self.dim_z)
            # rotation, normal, contact point and contact
            self.rot_fc = self._fc_layer('rot_fc', 1, 2*self.dim_z)
            self.rns_fc1 = self._fc_layer('rns_fc1', 128)
            self.rns_fc2 = self._fc_layer('rn2_fc2', 10*self.dim_z)

            self.filter_layers += [self.pos_c1, self.pos_c2, self.pos_fc,
                                   self.rot_fc, self.rns_fc1, self.rns_fc2]

        # all architectures share the same decoder to get the state and
        # covariance from the [lstm] hidden state
        self.decoder_fc1 = self._fc_layer('decoder_fc1', 128)
        self.decoder_fc2 = self._fc_layer('decoder_fc2', 64)
        num = self.dim_x * (self.dim_x + 1) / 2
        self.decoder_fc3 = self._fc_layer('decoder_fc3', self.dim_x + num,
                                          activation=None)

        init_const = np.ones(self.dim_x) * 1e-2/self.scale**2
        init = np.ones(self.dim_x)/self.scale**2
        self.covar_init_bias = \
            self.add_weight(name='covar_init_bias',
                            shape=[self.dim_x], trainable=True,
                            regularizer=tf.keras.regularizers.l2(l=1e-3),
                            initializer=tf.constant_initializer(init))
        self.bias_fixed = \
            self.add_weight(name='bias_fixed', shape=[self.dim_x],
                            trainable=False,
                            initializer=tf.constant_initializer(init_const))

        self.filter_layers += [self.decoder_fc1, self.decoder_fc2,
                               self.decoder_fc3, self.covar_init_bias,
                               self.bias_fixed]

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        if self.structure in ['lstm2']:
            # internal states of the first and second lstm (hidden and carry)
            # and ouput state + covariance
            return [[self.num_units], [self.num_units], [self.num_units],
                    [self.num_units], [self.dim_x], [self.dim_x*self.dim_x],
                    [1]]
        elif self.structure in ['lstm1']:
            # internal states of the lstm (hidden and carry)
            # and ouput state + covariance
            return [[self.num_units], [self.num_units], [self.dim_x],
                    [self.dim_x*self.dim_x], [1]]
        else:
            return [[self.dim_x], [self.dim_x*self.dim_x], [1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # estimated state and covariance, observations
        return ([self.dim_x], [self.dim_x*self.dim_x], [self.dim_z])

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
        # turn off the '/rnn' name scope to improve summary logging
        with tf.name_scope(""):
            if self.structure == 'lstm2':
                hidden1_old, carry1_old, hidden2_old, carry2_old, \
                    state_old, covar_old, step = states
            else:
                hidden1_old, carry1_old, state_old, covar_old, step = states

            ###################################################################
            # get the inputs
            raw_observations, actions = inputs
            covar_old = tf.reshape(covar_old,
                                   [self.batch_size, self.dim_x, self.dim_x])

            # preprocess the raw observations
            z, encoded_observations = \
                self.context.run_sensor_model(raw_observations, training)

            if self.problem == 'pushing':
                mask, rot_encoding, glimpse_encoding, _ = encoded_observations

                # preprocess the encodings
                # mask
                pos_c1 = self.pos_c1(mask)
                pos_c2 = self.pos_c2(pos_c1)
                pos_c2 = tf.reshape(pos_c2, [self.batch_size, -1])
                pos_fc = self.pos_fc(pos_c2)

                # rotation, normal, contact point and contact
                rot_fc = self.rot_fc(rot_encoding)
                rns_fc1 = self.rns_fc1(glimpse_encoding)
                rns_fc2 = self.rns_fc2(rns_fc1)

                encoded_observations = tf.concat([pos_fc, rot_fc, rns_fc2],
                                                 axis=-1)

            if self.structure == 'lstm2':
                # we actually only use the initial state as input in the first
                # step
                if self.context.param['problem'] == 'kitti':
                    theta = tf.slice(state_old, [0, 2], [-1, 1])
                    ct = tf.cos(theta*np.pi/180.)
                    st = tf.sin(theta*np.pi/180.)
                    state_in = tf.concat([ct, st, state_old[:, 3:]], axis=-1)
                else:
                    state_in = state_old

                state_in = tf.cond(tf.math.equal(tf.squeeze(step[0]), 0),
                                   lambda: state_in,
                                   lambda: tf.zeros_like(state_in))
                covar_diag = tf.linalg.diag_part(covar_old)
                covar_diag = \
                    tf.cond(tf.math.equal(tf.squeeze(step[0]), 0),
                            lambda: covar_diag,
                            lambda: tf.zeros_like(covar_diag))

                lstm_inputs = tf.concat([encoded_observations, state_in,
                                         covar_diag], axis=-1)
                # predict the next internal state
                _, out1 = self.lstm1(lstm_inputs, [hidden1_old, carry1_old],
                                     training)
                hidden1, carry1 = out1
                _, out2 = self.lstm2(hidden1, [hidden2_old, carry2_old],
                                     training)
                hidden, carry2 = out2
            elif self.structure == 'lstm1':
                if self.context.param['problem'] == 'kitti':
                    theta = tf.slice(state_old, [0, 2], [-1, 1])
                    ct = tf.cos(theta*np.pi/180.)
                    st = tf.sin(theta*np.pi/180.)
                    state_in = tf.concat([ct, st, state_old[:, 3:]], axis=-1)
                else:
                    state_in = state_old

                # we actually only use the initial state as input in the first
                # step
                state_in = tf.cond(tf.math.equal(tf.squeeze(step[0]), 0),
                                   lambda: state_in,
                                   lambda: tf.zeros_like(state_in))
                covar_diag = tf.linalg.diag_part(covar_old)
                covar_diag = \
                    tf.cond(tf.math.equal(tf.squeeze(step[0]), 0),
                            lambda: covar_diag,
                            lambda: tf.zeros_like(covar_diag))

                lstm_inputs = tf.concat([encoded_observations, state_in,
                                         covar_diag], axis=-1)
                # predict the next internal state
                _, out = self.lstm1(lstm_inputs, [hidden1_old, carry1_old],
                                    training)
                hidden, carry = out

            # predict the output state and covariance from the internal state
            decoder_fc1 = self.decoder_fc1(hidden)
            decoder_fc2 = self.decoder_fc2(decoder_fc1)
            output = self.decoder_fc3(decoder_fc2)

            if self.context.param['problem'] == 'kitti' and \
                    self.structure in ['lstm2', 'lstm1']:
                out_state = output[:, :self.dim_x+2]
                out_covar = output[:, self.dim_x+2:]
                ct = out_state[:, 2:3]
                st = out_state[:, 3:4]
                ang = tf.math.atan2(st, ct)*180/np.pi
                cd = out_state[:, 5:6]
                sd = out_state[:, 6:7]
                ang_d = tf.math.atan2(sd, cd)*180/np.pi
                out_state = tf.concat([out_state[:, :2], ang,
                                       out_state[:, 4:5], ang_d], axis=-1)
            else:
                out_state = output[:, :self.dim_x]
                out_covar = output[:, self.dim_x:]
            out_state = output[:, :self.dim_x]
            out_covar = output[:, self.dim_x:]

            if self.context.param['problem'] == 'kitti':
                out_state = state_old + out_state
            out_state = self.context.correct_state(out_state, diff=False)
            out_covar = compat.fill_triangular(out_covar)
            out_covar += tf.linalg.diag(self.covar_init_bias)
            out_covar = tf.matmul(out_covar,
                                  tf.linalg.matrix_transpose(out_covar))
            out_covar += tf.linalg.diag(self.bias_fixed)

            out_covar = tf.reshape(out_covar, [self.batch_size, -1])

            if self.structure in ['lstm2']:
                # the recurrent state contains the updated internal states of
                # both lstms,the output and updated step
                new_state = (hidden1, carry1, hidden, carry2, out_state,
                             out_covar, step + 1)
            elif self.structure in ['lstm1']:
                # the recurrent state contains the updated internal states of
                # the lstm, the output and updated step
                new_state = (hidden, carry, out_state, out_covar, step + 1)
            else:
                # the recurrent state contains the output and updated step
                new_state = (out_state, out_covar, step + 1)

            output = (out_state, out_covar, z)
            return output, new_state
