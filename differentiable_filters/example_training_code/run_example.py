#!/usr/bin/env python3
"""
Example code for training a differentiable filter on a simulated disc tracking
task.
"""

import tensorflow as tf
import numpy as np
import os
import argparse
import time

from differentiable_filters.data.create_disc_tracking_dataset import DiscTrackingData
from differentiable_filters.contexts.example_context import ExampleContext
from differentiable_filters.utils import recordio as tfr


def run_example(filter_type, loss, out_dir, batch_size, hetero_q, hetero_r,
                learned_process, image_size, use_gpu, debug):
    """
    Exemplary code to set up and train a differentiable filter for the
    simulated disc tracking task described in the paper "How to train your
    Differentiable FIlter"

    Parameters
    ----------
    filter_type : str
        Defines which filtering algorithm is used. Can be ekf, ukf, mcukf or pf
    loss : str
        Which loss to use for training the filter. This can be "nll" for the
        negative log likelihood, "mse" for the mean squared error or "mixed"
        for a combination of both
    out_dir : str
        Path to the directory where results and data should be written to.
    batch_size : int
        Batch size for training and testing.
    hetero_q : bool
        If true, heteroscedastic process noise is learned, else constant.
    hetero_r : bool
        If true, heteroscedastic observation noise is learned, else constant.
    learned_process : bool
        If true, a neural network is used as process model in the filter, else
        an analytical process model is used.
    image_size : int
        Width and height of the image observations
    use_gpu : bool
        If true, the training and testing is run on GPU (if one is available)
    debug : bool
        Turns on additional debug output and prints.

    Returns
    -------
    None.

    """
    if use_gpu:
        # limit tensorflows gpuy memory consumption
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        # Hide GPU from visible devices to run on cpu
        tf.config.set_visible_devices([], 'GPU')

    # prepare the output directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_dir = os.path.join(out_dir + '/train')
    data_dir = os.path.join(out_dir + '/data')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # create a small dataset (if it doesn't already exist)
    name = 'example'
    if not os.path.exists(os.path.join(data_dir, 'info_' + name + '.txt')):
        c = DiscTrackingData(name, data_dir, image_size, 1000, 30, 1000,
                             rescale=1, debug=debug)
        c.create_dataset(15, 0, 0, 3.0)
    else:
        print('data already exists')

    # create a tensorflow model that combines a differentiable filter with a
    # problem context
    model = FilterApplication(filter_type, loss, batch_size, hetero_q,
                              hetero_r, learned_process, image_size,
                              debug=debug)

    # Load training and test datasets
    # we use sequence length 10 for training and validation and sequence
    # length 30 for testing
    train_files, val_files, test_files = load_data(data_dir, name)
    train_set = tf.data.TFRecordDataset(train_files)
    train_set = model.preprocess(data_dir, name, train_set, 'train', 10)
    train_set = train_set.shuffle(500)
    train_set = train_set.batch(batch_size, drop_remainder=True)

    val_set = tf.data.TFRecordDataset(val_files)
    val_set = model.preprocess(data_dir, name, val_set, 'val', 10)
    val_set = val_set.batch(batch_size, drop_remainder=True)

    test_set = tf.data.TFRecordDataset(test_files)
    test_set = model.preprocess(data_dir, name, test_set, 'test', 30)
    test_set = test_set.batch(batch_size, drop_remainder=True)

    # prepare the training
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    epochs = 3
    step = 0

    # prepare a summary writer for logging information that can be viewed
    # with tensorboard
    train_summary_writer = tf.summary.create_file_writer(train_dir + '/' +
                                                         str(time.time()))
    tf.summary.experimental.set_step(step)

    # unfortunately, we cannot use keras model.fit here, since keras currently
    # does not support loss functions that receive multiple output tensors
    # (like mean and covariance of the filter's belief for computing the nll
    # loss) We thus write a custom training loop
    print("\n Start training with sequence length 10")
    for epoch in range(epochs):
        print("\nStart of epoch %d \n" % (epoch))
        print("Validating ...")
        evaluate(model, val_set, batch_size)

        for (x_batch_train, y_batch_train) in train_set:
            start = time.time()

            with tf.GradientTape() as tape:
                # sample a random disturbance of the initial state from the
                # initial covariance
                n_val = np.random.normal(loc=np.zeros((model.dim_x)),
                                         scale=model.initial_covariance,
                                         size=(batch_size, model.dim_x))
                x_batch_train = (*x_batch_train, n_val)
                # Run the forward pass of the model
                with train_summary_writer.as_default():
                    out = model(x_batch_train, training=False)

                # Compute the loss value for this minibatch.
                loss_value, metrics, metric_names = \
                    model.context.get_loss(y_batch_train, out)

                # log summaries of the metrics every 50 steps
                with train_summary_writer.as_default():
                    with tf.summary.record_if(step%50==0):
                        for i, name in enumerate(metric_names):
                            tf.summary.scalar('metrics/' + name,
                                              tf.reduce_mean(metrics[i]))

            # Use the gradient tape to automatically retrieve the
            # gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            end = time.time()

            # Log every 50 batches.
            if step % 50 == 0:
                print("Training loss at step %d: %.4f (took %.3f seconds) " %
                      (step, float(loss_value), float(end-start)))
            step += 1
            tf.summary.experimental.set_step(step)

    # test the trained model on the held out data
    print("\n Testing with sequence length 30")
    evaluate(model, test_set, batch_size)


def load_data(data_dir, data_name):
    """
    Collects the tf recod files that make up a dataset and returns the file
    lists for training, validation and testset

    Parameters
    ----------
    data_dir : str
        Path of the folder containing the tf record files
    data_name : str
        Identifyer of the dataset

    Returns
    -------
    train_files : list of str
        List with the filepaths of the tf record files in the training split
    val_files : list of str
        List with the filepaths of the tf record files in the validation split
    test_files : list of str
        List with the filepaths of the tf record files in the test split

    """
    files = os.listdir(data_dir)

    # gather the datafiles
    train_files = [os.path.join(data_dir, f) for f in files
                   if f.startswith(data_name + '_') and
                   '_train_' in f and '.tfrecord' in f]
    val_files = [os.path.join(data_dir, f) for f in files
                 if f.startswith(data_name + '_') and
                 '_val_' in f and '.tfrecord' in f]
    test_files = [os.path.join(data_dir, f) for f in files
                  if f.startswith(data_name + '_') and
                  '_test_' in f and '.tfrecord' in f]

    return train_files, val_files, test_files


def evaluate(model, dataset, batch_size):
    """
    Evaluates the model on the given dataset (without training)

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate
    dataset : tf.data.Dataset
        The dataset on which to evaluate the model
    batch_size : int
        The batch size used.

    Returns
    -------
    None.

    """
    outputs = {}
    # Iterate over the batches of the testset.
    for step, (x_batch, y_batch) in enumerate(dataset):
        # sample a random disturbance from the initial covariance
        # and add it to the intial state
        n_val = np.random.normal(loc=np.zeros((model.dim_x)),
                                 scale=model.initial_covariance,
                                 size=(batch_size, model.dim_x))
        x_batch = (*x_batch, n_val)

        # Run the forward pass of the layer.
        out = model(x_batch, training=False)

        # Compute the loss and metrics for this minibatch.
        loss_value, metrics, metric_names = \
            model.context.get_loss(y_batch, out)

        if step == 0:
            for ind, k in enumerate(metric_names):
                outputs[k] = [metrics[ind]]
        else:
            for ind, k in enumerate(metric_names):
                outputs[k].append(metrics[ind])

    print('Result: ')
    for ind, k in enumerate(metric_names):
        tf.print(k, ": ", tf.reduce_mean(outputs[k]))
    print('\n')


class FilterApplication(tf.keras.Model):
    def __init__(self, filter_type='ekf', loss='nll', batch_size=32,
                 hetero_q=False, hetero_r=True, learned_process=True,
                 image_size=120, debug=False, **kwargs):
        """
        Tf.keras.Model that combines a differentiable filter and a problem
        context to run filtering on this problem.

        Parameters
        ----------
        filter_type : str, optional
            A string that defines which filter to use, can be ekf, ukf, mcukf
            or pf. Default is ekf.
        batch_size : int, optional
            Batch size. Default is 32
        loss : str, optional
            The loss function to use, can be nll, mse or mixed. Default is nll.
        hetero_q : bool, optional
            Learn heteroscedastic process noise? Default is False.
        hetero_e : bool, optional
            Learn heteroscedastic observation noise? Default is True.
        learned_process : bool, optional
            Learn the process model or use an analytical one? Default is True.
        image_size : int, optional
            Width and height of the image observations. Default is 120.
        debug : bool, optional
            Print debug output? Default is False.

        Raises
        ------
        ValueError
            If the desired filter class (filter_type) is not implemented

        Returns
        -------
        None.

        """
        super(FilterApplication, self).__init__(**kwargs)

        # -------------------------- (1) --------------------------------------
        # Construct the context class that describes the problem on which
        # we want to run a differentiable filter
        #----------------------------------------------------------------------
        self.context = ExampleContext(batch_size, filter_type, loss,
                                      hetero_q, hetero_r, learned_process)

        # -------------------------- (2) --------------------------------------
        # Instantiate the desired filter cell
        #----------------------------------------------------------------------
        problem = 'simple'
        if filter_type == 'ekf':
            from differentiable_filters.filters import ekf_cell as ekf
            self.cell = ekf.EKFCell(self.context, problem, debug=debug)
        elif filter_type == 'ukf':
            from differentiable_filters.filters import ukf_cell as ukf
            self.cell = ukf.UKFCell(self.context, problem, debug=debug)
        elif filter_type == 'mcukf':
            from differentiable_filters.filters import mcukf_cell as mcukf
            self.cell = mcukf.MCUKFCell(self.context, problem, debug=debug)
        elif filter_type == 'pf':
            from differentiable_filters.filters import pf_cell as pf
            self.cell = pf.PFCell(self.context, problem, debug=debug)
        else:
            self.log.error('Unknown filter type: ' + filter_type)
            raise ValueError('Unknown filter type: ' + filter_type)


        # -------------------------- (3) --------------------------------------
        # wrap the Filter cell in a keras RNN Layer
        # ---------------------------------------------------------------------
        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True,
                                             unroll=False)

        # store some shape related information
        self.batch_size = self.context.batch_size
        self.image_size = image_size
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u
        self.filter_type = filter_type

        # -------------------------- (4) --------------------------------------
        # Define the covariance matrix for the initial belief of the filter
        # ---------------------------------------------------------------------
        self.initial_covariance = np.array([10.0, 10.0, 5.0, 5.0])/ 60.
        self.initial_covariance = self.initial_covariance.astype(np.float32)
        covar_start = tf.square(self.initial_covariance)
        covar_start = tf.linalg.tensor_diag(covar_start)
        self.covar_start = tf.tile(covar_start[None, :, :],
                                   [self.batch_size, 1, 1])



    def call(self, inputs, training=True):
        """
        Run one step of prediction with the model

        Parameters
        ----------
        inputs : list of tensors
            the input tensors include the sequence of raw sensory observations,
            the true initial satte of the system and a noise vector to perturb
            this initial state before passing it to the filter
        training : bool
            if the model is run in training or test mode

        Returns
        -------
        res : list of tensors
            the prediction output

        """
        raw_observations, initial_state, noise = inputs

        # -------------------------- (1) --------------------------------------
        # Construct the initial state of the differentiable filter RNN
        # ---------------------------------------------------------------------
        initial_state += noise
        if self.filter_type != 'pf':
            # for ekf, ukf and mcukf, the filter state contains the mean
            # and covariance of the gaussian belief as well as an indicator
            # of the step in the timeseries
            init_state = \
                (initial_state,
                 tf.reshape(self.covar_start, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))
        else:
            # for the particle filter, the belief is represented by particles
            # and their weights. The state also contains an indicator of the
            # step in the timeseries
            particles, weights = \
                self.cell.sample_from_start(initial_state,
                                            self.initial_covariance)

            init_state = \
                (tf.reshape(particles, [self.batch_size, -1]),
                 tf.reshape(weights, [self.batch_size, -1]),
                 tf.zeros([self.batch_size, 1]))

        # -------------------------- (2) --------------------------------------
        # Run the filtering algorithm for the full sequence.
        # Inputs for all filters are the raw observations and the control
        # actions. Since we do not use a control input in this example, we
        # pass a vector of zeros instead.
        # ---------------------------------------------------------------------
        fake_actions = tf.zeros([self.batch_size,
                                 raw_observations.get_shape()[1], 1])
        inputs = (raw_observations, fake_actions)
        outputs = self.rnn_layer(inputs, training=training,
                                 initial_state=init_state)

        # -------------------------- (3) --------------------------------------
        # Collect the results
        # ---------------------------------------------------------------------
        if self.filter_type != 'pf':
            state_sequence, covariance_sequence, z, R, Q = outputs
            particles = None
            weights = None
        else:
            particles, weights, state_sequence, covariance_sequence, \
                z, R, Q = outputs

            particles = tf.reshape(particles,
                                   [self.batch_size, -1,
                                    self.cell.num_particles, self.dim_x])
            weights = tf.reshape(weights, [self.batch_size, -1,
                                           self.cell.num_particles])
            # The weights are in log scale, to turn them into a distribution,
            # we exponentiate and normalize them == apply the softmax
            # transform
            weights = tf.nn.softmax(weights, axis=-1)

        # restore the proper dimension of the covariance matrices
        covars = \
            tf.reshape(covariance_sequence,
                       [self.batch_size, -1, self.dim_x, self.dim_x])
        R = tf.reshape(R, [self.batch_size, -1, self.dim_z, self.dim_z])
        Q = tf.reshape(Q, [self.batch_size, -1, self.dim_x, self.dim_x])

        res = [particles, weights, state_sequence, covars, z, R, Q]
        return res

    ###########################################################################
    # data loading
    ###########################################################################
    def preprocess(self, path, name, dataset, data_mode, sl, num_threads=1):
        """
        Converts from tf.Records to tensors and applys preprocessing

        Parameters
        ----------
        path : str
            Path to the directory that contains the tf.Record files
        name : str
            The name of the dataset.
        dataset : tf.data.Dataset
            Atf.data.TFRecordDataset object that contains the filenames of all
            tf.Records to read in.
        data_mode : str
            Defines for which data split the data is read. Can be "train",
            "val" or "test"
        sl : int
            Desired sequence length. This enables extraction of shorter
            subsequences from the original sequences.
        num_threads : int, optional
            The number of threads used to preeprocess the data. The default is
            1.

        Returns
        -------
        dataset : tf.data.Dataset
            A dataset with input and label tensors.

        """
        keys = ['start_state', 'image', 'state', 'q', 'visible']
        record_meta = tfr.RecordMeta.load(path, name + '_' + data_mode + '_')

        dataset = \
            dataset.map(lambda x: self._parse_function(x, keys, record_meta,
                                                       sl, data_mode),
                        num_parallel_calls=num_threads)
        dataset = dataset.flat_map(lambda x, y:
                                   tf.data.Dataset.from_tensor_slices((x, y)))
        return dataset

    def _decode_im(self, im):
        """
        Decode an image from a byte-string.

        Parameters
        ----------
        im : tf.String tensor
            Byte string that encodes one image

        Returns
        -------
        im : tf.float32 tensor
            The decoded image

        """
        im = tf.io.decode_image(im, channels=3, dtype=tf.dtypes.uint8)
        im = tf.cast(im, tf.float32) / 255.
        im.set_shape([self.image_size, self.image_size, 3])
        return im

    def _parse_function(self, example_proto, keys, record_meta, sl, data_mode):
        """
        This function defines how a single example from the tf.Recod data is
        parsed into tensors. Also applies preprocessing.

        Parameters
        ----------
        example_proto : tf.train.Example
            A single example from the dataset
        keys : list of str
            The names of the data that is contained in each example
        record_meta : tfr.RecordMeta
            An object that holds meta information about the the datase, such
            as tensor shapes and data types
        sl : int
            Desired sequence length. This enables extraction of shorter
            subsequences from the original sequences.
        data_mode : str
            Defines for which data split the data is read. Can be "train",
            "val" or "test"

        Raises
        ------
        ValueError
            If the desired sequence length (sl) is longer than the lenght of
            the sequences in the dataset

        Returns
        -------
        inputs : tuple of tensors
            The tensors that serve as input for the differentiable filter. In
            this case the sequence of image observations and the inital state
            of the system
        labels : tuple of tensors
            The labels. In this case the true state sequence, the true
            process nosie covariance and the number of visible target disc
            pixels in each image.

        """
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
        q = features['q']
        vis = features['visible']

        length = q.get_shape()[0]

        # reading in images from bytes
        im = features['image']
        im = tf.map_fn(self._decode_im, im, fn_output_signature=tf.float32)

        if sl > length:
            raise ValueError('Desired training sequence length is ' +
                             'longer than dataset sequence length: ' +
                             'Desired: ' + str(sl) + ', data: ' +
                             str(length))

        if sl == length or data_mode == 'test':
            start_inds = [-1]
        else:
            # we use several sub-sequences of the full sequence
            num = length // sl
            start_inds = \
                np.arange(-1, length-sl-1, (sl+1)//2)
            start_inds = start_inds[:num]

        # prepare the lists of output tensors
        ims = []
        starts = []
        states = []
        qs = []
        viss = []
        for si in start_inds:
            if si >= 0:
                starts += [state[si]]
            else:
                starts += [start]
            end = si+sl+1
            ims += [im[si+1:end]]
            states += [state[si+1:end]]
            qs += [q[si+1:end]]
            viss += [vis[si+1:end]]

        # observations, initial state,
        inputs =  tuple([tf.stack(ims), tf.stack(starts)])
        labels =  tuple([tf.stack(states), tf.stack(qs), tf.stack(viss)])
        return inputs, labels


def main():
    parser = argparse.ArgumentParser('run example')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        required=True, help='where to store results')
    parser.add_argument('--filter', dest='filter', type=str,
                        default='ekf', choices=['ekf', 'ukf', 'mcukf', 'pf'],
                        help='which filter class to use')
    parser.add_argument('--loss', dest='loss', type=str,
                        default='nll', choices=['nll', 'mse', 'mixed'],
                        help='which loss function to use')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=16, help='batch size for training')
    parser.add_argument('--image-size', dest='image_size',
                        type=int, default=120,
                        help='width and height of image observations')
    parser.add_argument('--hetero-q', dest='hetero_q', type=int,
                        choices=[0, 1], default=0,
                        help='learn heteroscedastic process noise?')
    parser.add_argument('--hetero-r', dest='hetero_r', type=int,
                        choices=[0, 1], default=1,
                        help='learn heteroscedastic observation noise?')
    parser.add_argument('--learned-process', dest='learned_process', type=int,
                        choices=[0, 1], default=1,
                        help='learn the process model or use an analytical one?')
    parser.add_argument('--gpu', dest='gpu',
                        type=int, choices=[0, 1], default=1,
                        help='if true, the code is run on gpu if one is found')
    parser.add_argument('--debug', dest='debug',
                        type=int, choices=[0, 1], default=0,
                        help='turns debugging on/off ')

    args = parser.parse_args()

    run_example(args.filter, args.loss, args.out_dir, args.batch_size,
                args.hetero_q, args.hetero_r, args.learned_process,
                args.image_size, args.gpu, args.debug)


if __name__ == "__main__":
    main()
