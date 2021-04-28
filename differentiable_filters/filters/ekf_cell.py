"""
RNN cell implementing a Differentiable Extended Kalman Filter
"""

import tensorflow as tf
from differentiable_filters.filters import filter_cell_base as base


class EKFCell(base.FilterCellBase):
    def __init__(self, context, problem, update_rate=1, debug=False):
        """
        RNN cell implementing a Differentiable Extended Kalman Filter

        Parameters
        ----------
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)
        problem : str
            A string identifyer for the problem defined by the context
        update_rate : int, optional
            The rate at which observations come in (allows simulating lower
            observation rates). Default is 1
        debug : bool, optional
            If true, the filters will print out information at each step.
            Default is False.
        """
        base.FilterCellBase.__init__(self, context, problem, update_rate,
                                     debug)

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # estimated state, its covariance, and the step number
        return [[self.dim_x], [self.dim_x * self.dim_x], [1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # estimated state and covariance, observations, R, Q
        return ([self.dim_x], [self.dim_x * self.dim_x],
                [self.dim_z], [self.dim_z * self.dim_z],
                [self.dim_x * self.dim_x])

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
            # get the inputs
            raw_observations, actions = inputs
            state_old, covar_old, step = states

            state_old = tf.reshape(state_old, [self.batch_size, self.dim_x])
            covar_old = tf.reshape(covar_old, [self.batch_size, self.dim_x,
                                               self.dim_x])

            print_ops = []
            if self.debug:
                print_ops += [tf.print('------------------- \n step \n ',
                                       tf.squeeze(step[0]))]
                print_ops += [tf.print('old state: \n', state_old[0],
                                       summarize=-1)]
                print_ops += [tf.print('old covar: \n', covar_old[0],
                                       summarize=-1)]
                print_ops += [tf.print('actions: \n', actions[0], summarize=-1)]

            ###################################################################
            # preproess the raw observations
            z, encoding = self.context.run_sensor_model(raw_observations,
                                                        training)

            R = self.context.get_observation_noise(encoding, training=training)
            ###################################################################
            # predict the next state
            state_pred, covar_pred, Q, F = \
                self._prediction_step(state_old, covar_old, actions,
                                      training=training)

            # and the expected observations
            z_pred, H = self.context.run_observation_model(state_pred,
                                                           training=training)
            if self.debug:
                print_ops += [tf.print('predicted state: \n', state_pred[0],
                                       summarize=-1)]
                print_ops += [tf.print('predicted covar: \n', covar_pred[0],
                                       summarize=-1)]
                print_ops += [tf.print('F: \n', F[0], summarize=-1)]
                print_ops += [tf.print('Q: \n', Q[0], summarize=-1)]
                print_ops += [tf.print('R: \n', R[0], summarize=-1)]
                print_ops += [tf.print('predicted z: \n', z_pred[0],
                                       summarize=-1)]

            ###################################################################
            # update the predictions with the observations
            state_up, covar_up = \
                self._update(state_pred, covar_pred, z_pred, H, z, R,
                             print_ops)

            # this lets us emulate not getting observations at every step
            state = tf.cond(tf.equal(tf.math.mod(step[0, 0],
                                                 self.update_rate), 0),
                            lambda: state_up, lambda: state_pred)
            covar = tf.cond(tf.equal(tf.math.mod(step[0, 0],
                                                 self.update_rate), 0),
                            lambda: covar_up, lambda: covar_pred)

            # now flatten the tensors again for output
            # the control_dependency statement ensures that the debug print
            # operations are executed in tf1. graph mode as well
            with tf.control_dependencies(print_ops):
                state = tf.reshape(state, [self.batch_size, -1])
                covar = tf.reshape(covar, [self.batch_size, -1])

            # the recurrent state contains the updated state estimate
            new_state = (state, covar, step + 1)

            # we can output additional output, but it also makes sense to add
            # the predicted state here again to have access to the full
            # sequence of states after running the network
            output = (state, covar, z,
                      tf.reshape(R, [self.batch_size, -1]),
                      tf.reshape(Q, [self.batch_size, -1]))

            return output, new_state

    def _prediction_step(self, state_old, covar_old, actions, training):
        """
        Prediction step of the EKF

        Parameters
        ----------
        state_old : tensor [batch_size, dim_x]
            the previous state estimate
        covar_old : tensor [batch_size, dim_x, dim_x]
            it's predicted covariance
        actions : tensor [batch_size, dim_u]
            the current actions
        training : bool
            training or testing?

        Returns
        -------
        state_pred : tensor [batch_size, dim_x]
            the predicted state
        covar_pred : tensor [batch_size, dim_x, dim_x]
            the predicted covariance
        Q : tensor [batch_size, dim_x, dim_x]
            the predicted process noise for this time step
        F : tensor [batch_size, dim_x, dim_x]
            The Jacobian of the process model wrt. the old state
        """
        state_pred, F = \
            self.context.run_process_model(state_old, actions, training)

        # get the process noise
        Q = self.context.get_process_noise(state_old, actions, training)

        covar_pred = \
            tf.matmul(F, tf.matmul(covar_old,
                                   tf.linalg.matrix_transpose(F))) + Q

        return state_pred, covar_pred, Q, F

    def _update(self, state_pred, covar_pred, z_pred, H, z, R, print_ops):
        """
        Update step of the EKF

        Parameters
        ----------
        state_pred : tensor [batch_size, dim_x]
            the predicted state from the process model
        covar_pred : tensor [batch_size, dim_x, dim_x]
            it's predicted covariance
        z_pred : tensor [batch_size, dim_z]
            the predicted observations for the predicted state
        H : tensor [batch_size, dim_z, dim_x]
            Jacobian of the the observation model wrt the predicted state
        z : tensor [batch_size, dim_z]
            the observations for this time step
        R : tensor [batch_size, dim_z, dim_z]
            the observation noise for this time step

        Returns
        -------
        state_up: tensor [batch_size, dim_x]
            the updated state
        covar_up: tensor  [batch_size, dim_x, dim_x]
            the updated state covariance
        """
        # Compute the Kalman gains and the innovation
        K = self._KF_gain(covar_pred, H, R)
        innovation = self.context.correct_observation_diff(z - z_pred)
        innovation = tf.expand_dims(innovation, -1)
        update = tf.squeeze(tf.matmul(K, innovation), -1)

        if self.debug:
            print_ops += [tf.print('observed z: \n', z[0], summarize=-1)]
            print_ops += [tf.print('K: \n', K[0], summarize=-1)]
            print_ops += [tf.print('state update: \n', update[0], summarize=-1)]

        # Compute the new state
        state_up = state_pred + update
        # correct the updated state, e.g.\ warping angles
        state_up = self.context.correct_state(state_up, diff=False)

        # Joseph form update: this is an alternative form of writing the update
        # of the covariance matrix that keeps the covariance matrix
        # positive semidefinite (i.e. it remains a correct covariance matrix)
        mult = tf.eye(self.dim_x) - tf.matmul(K, H)
        covar_up = \
            tf.matmul(mult, tf.matmul(covar_pred,
                                      tf.linalg.matrix_transpose(mult)))
        covar_up += tf.matmul(K, tf.matmul(R,
                                           tf.linalg.matrix_transpose(K)))

        covar_up = self._make_valid(covar_up)

        return state_up, covar_up

    def _KF_gain(self, covar, H, R):
        """
        Computing the Kalman Gain

        Parameters
        ----------
        covar : tensor [batch_size, dim_x, dim_x]
            the predicted covariance of the current state estimate
        H : tensor [batch_size, dim_z, dim_x]
            Jacobian of the the observation model wrt the predicted state
        R : tensor [batch_size, dim_z, dim_z]
            the observation noise for this time step

        Returns
        -------
        K : tensor [batch_size, dim_x, dim_x]
            the Kalman Gain

        """
        S = tf.matmul(H, tf.matmul(covar,
                                   tf.linalg.matrix_transpose(H))) + R
        S = self._make_valid(S)

        s_inv = tf.linalg.inv(S)
        K = tf.matmul(covar, tf.matmul(tf.linalg.matrix_transpose(H), s_inv))

        return K
