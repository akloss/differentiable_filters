#!/usr/bin/env python3
"""
Script and class for creating a tf.Record dataset for the simulated disc
tracking task
"""

import numpy as np
import logging
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import sys

from differentiable_filters.utils import recordio as tfr


class DiscTrackingData():
    def __init__(self, name, out_dir, width, num_examples, sequence_length,
                 file_size, rescale=False, debug=False):
        """
        Class for creating a tf.Record dataset for the simulated disc tracking
        task.

        Parameters
        ----------
        name : str
            Name of the dataset.
        out_dir : str
            Output directory
        width : int
            Width (and height) of the image observations. Note: Images are
            always generated with size [120, 120, 3]. If width is set to
            a different value, the images are rescaled accordingly.
        num_examples : int
            Maximum number of trainign examples to generate.
        sequence_length : int
            Number of timesteps in the sequence.
        file_size : int
            Maximum number of examples stored in one file.
        rescale : bool, optional
            If true, the state-space is rescaled to be in [-1, 1].
            The default is False.
        debug : bool, optional
            Turns on debugging output. The default is False.

        Returns
        -------
        None.

        """
        self.im_size = 120
        self.factor = self.im_size / width
        self.out_dir = out_dir
        self.name = name
        self.num_examples = num_examples
        self.sequence_length = sequence_length
        self.file_size = min(self.num_examples, file_size)
        self.rescale = rescale
        self.debug = debug

        # parameters of the process model
        self.spring_force = 0.05
        self.drag_force = 0.0075

        # colors for the distractor discs
        self.cols = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                     (255, 255, 0), (255, 255, 255)]

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # setup logging
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s: [%(name)s] ' +
                                      '[%(levelname)s] %(message)s')
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

        # create file handler which logs warnings errors and criticals
        if os.path.exists(os.path.join(self.out_dir,
                                       self.name + '_error.log')):
            os.remove(os.path.join(self.out_dir,
                                   self.name + '_error.log'))
        fh = logging.FileHandler(os.path.join(self.out_dir,
                                              self.name + '_error.log'))
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

    def create_dataset(self, num_distractors, hetero_q, corr_q, pos_noise):
        """
        Creates a tf.Record dataset with the desired characteristics.

        Parameters
        ----------
        num_distractors : int
            The number of distractor discs.
        hetero_q : bool
            If true, the process noise on the velecity components is
            heteroscedastic.
        corr_q : bool
            If true, the process noise is correlated. In this case, the full
            covariance matrix Q is stored, otherwise, we only store the
            diagonal.
        pos_noise : float
            Magnitude of the process noise on the position components.

        Returns
        -------
        None.

        """
        # setup directory for debug output if desired
        if self.debug:
            self.debug_dir = os.path.join(self.out_dir, 'debug', self.name)
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)

        train_count = 0
        val_count = 0
        test_count = 0

        self.keys = ['start_image', 'start_state', 'image', 'state', 'q',
                     'visible']
        train_data = {key: [] for key in self.keys}

        self.record_writer_train = \
            tfr.RecordioWriter(self.out_dir, self.file_size,
                               self.name + '_train_')
        self.record_meta_train = tfr.RecordMeta(self.name + '_train_')
        self.record_writer_val = \
            tfr.RecordioWriter(self.out_dir, self.file_size,
                               self.name + '_val_')
        self.record_meta_val = tfr.RecordMeta(self.name + '_val_')
        self.record_writer_test = \
            tfr.RecordioWriter(self.out_dir, self.file_size,
                               self.name + '_test_')
        self.record_meta_test = tfr.RecordMeta(self.name + '_test_')

        self.ct = 0
        self.log.info('Starting to generate dataset ' + self.name)
        while train_count < self.num_examples:
            values = self._get_data(num_distractors, hetero_q, corr_q,
                                    pos_noise)
            self.ct += 1
            for key in self.keys:
                train_data[key] += [values[key]]
            if len(train_data['image']) * 8 // 10 > self.file_size:
                train_size, val_size, test_size = self._save(train_data)
                train_count += train_size
                val_count += val_size
                test_count += test_size
                train_data = {key: [] for key in self.keys}

            if (len(train_data['image']) * 8 / 10 + train_count) % 250 == 0:
                num = len(train_data['image']) * 8 // 10 + train_count
                self.log.info('Done ' + str(num) + ' of ' +
                              str(self.num_examples))

        if len(train_data['image']) > 0:
            train_size, val_size, test_size = self._save(train_data)
            train_count += train_size
            val_count += val_size
            test_count += test_size

        # save the meta information
        count = train_count + val_count + test_count
        fi = open(os.path.join(self.out_dir, 'info_' + self.name + '.txt'),
                  'w')
        fi.write('Num data points: ' + str(count) + '\n')
        fi.write('Num train: ' + str(train_count) + '\n')
        fi.write('Num val: ' + str(val_count) + '\n')
        fi.write('Num test: ' + str(test_count) + '\n')
        fi.close()

        self.log.info('Done')

        self.record_writer_train.close()
        self.record_writer_test.close()
        self.record_writer_val.close()

        return

    def _get_data(self, num_distractors, hetero_q, corr_q, pos_noise):
        """
        Generates one example.

        Parameters
        ----------
        num_distractors : int
            The number of distractor discs.
        hetero_q : bool
            If true, the process noise on the velecity components is
            heteroscedastic.
        corr_q : bool
            If true, the process noise is correlated. In this case, the full
            covariance matrix Q is stored, otherwise, we only store the
            diagonal.
        pos_noise : float
            Magnitude of the process noise on the position components.

        Returns
        -------
        example : dict
            Dictionary containign the data

        """
        states = []
        images = []
        qs = []
        viss = []

        # the state consists of the red disc's position and velocity
        # draw a random position
        pos = np.random.uniform(-self.im_size//2, self.im_size//2, size=(2))
        # draw a random velocity
        vel = np.random.normal(loc=0., scale=1., size=(2)) * 3
        initial_state = np.array([pos[0], pos[1], vel[0], vel[1]])

        distractors = []
        for dist in range(num_distractors):
            # also draw a random starting positions for distractors
            pos = np.random.uniform(-self.im_size//2, self.im_size//2,
                                    size=(2))
            # draw a random velocity
            vel = np.random.normal(loc=0, scale=1, size=(2)) * 3
            # and a random radius
            rad = np.random.choice(np.arange(3, 10))
            # draw a random color
            col = np.random.choice(len(self.cols))
            distractors += [(rad, np.array([pos[0], pos[1], vel[0], vel[1]]),
                            col)]

        # generate the initial image
        initial_im, initial_vis = self._observation_model(initial_state,
                                                          distractors)
        last_state = initial_state
        for step in range(self.sequence_length):
            # get the next state
            state, q = self._process_model(last_state, hetero_q, corr_q,
                                           pos_noise)
            # also move the distractors
            new_distractors = []
            for d in distractors:
                d_new, _ = self._process_model(d[1], hetero_q, corr_q,
                                               pos_noise)
                new_distractors += [(d[0], d_new, d[2])]
            # get the new image
            im, vis = self._observation_model(state, new_distractors)

            states += [state]
            images += [im]
            qs += [q]
            viss += [vis]
            distractors = new_distractors
            last_state = state

        if self.ct < 3 and self.debug:
            for i, im in enumerate(images):
                fig, ax = plt.subplots()
                ax.set_axis_off()
                ax.imshow(im)
                ax.plot(states[i][0]+60, states[i][1]+60, 'bo')

                if i + 1 < len(images):
                    ax.plot(states[i+1][0]+60, states[i+1][1]+60, 'go')
                    ax.plot([states[i][0]+60, states[i][0] + states[i][2]+60],
                            [states[i][1]+60, states[i][1] + states[i][3]+60],
                            'g')

                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                                    wspace=0.1, hspace=0.1)

                fig.savefig(os.path.join(self.debug_dir, str(self.ct) +
                                          "_tracking_" + str(i)),
                            bbox_inches="tight")
                plt.close(fig)

        # we found it helpful to rescale the state space to be roughly in
        # [-1, 1]
        if self.rescale:
            initial_state /= self.im_size / 2
            states = np.array(states) / (self.im_size / 2)
            qs = np.array(qs) / (self.im_size / 2)
            if corr_q:
                qs /= self.im_size / 2.

        # compress the images by encoding them as png byte-strings
        for ind, im in enumerate(images):
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = cv2.imencode('.png', im)[1].tobytes()
            images[ind] = im
        initial_im = cv2.cvtColor(initial_im, cv2.COLOR_RGB2BGR)
        initial_im = cv2.imencode('.png', initial_im)[1].tobytes()

        return {'start_image': initial_im, 'start_state': initial_state,
                'image': np.array(images), 'state': states,
                'q': qs, 'visible': np.array(viss)}

    def _observation_model(self, state, distractors):
        """
        Generates an observation image for the current state.

        Parameters
        ----------
        state : np.array
            The state (position and velocity) of the target disc
        distractors : list
            List with the radius, state and color of each distractor disc

        Returns
        -------
        im : np.array
            The image data.
        vis : float
            The number of visible pixels of the target disc.

        """
        im = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)

        # draw the red disc
        cv2.circle(im, (int(state[0]+self.im_size//2),
                        int(state[1]+self.im_size//2)),
                   radius=7, color=[255, 0, 0], thickness=-1)

        # draw the other distractors
        for d in distractors:
            cv2.circle(im, (int(d[1][0]+self.im_size//2),
                            int(d[1][1]+self.im_size//2)),
                       radius=d[0], color=self.cols[d[2]], thickness=-1)

        # get the number of pixels visible from the red disc
        mask = np.logical_and(im[:, :, 0] == 255,
                              np.logical_and(im[:, :, 1] == 0,
                                             im[:, :, 2] == 0))
        vis = np.sum(mask.astype(np.float32))

        im = cv2.resize(im, (int(self.im_size/self.factor),
                             int(self.im_size/self.factor)))
        vis /= self.factor

        return im, vis

    def _process_model(self, state, hetero_q, correlated, pos_noise):
        """
        Calculates the next state of the target disc.

        Parameters
        ----------
        state : np.array
            The state (position and velocity) of the target disc
        hetero_q : bool
            If true, the process noise on the velecity components is
            heteroscedastic.
        correlated : bool
            If true, the process noise is correlated. In this case, the full
            covariance matrix Q is stored, otherwise, we only store the
            diagonal.
        pos_noise : float
            Magnitude of the process noise on the position components.

        Returns
        -------
        new_state : np.array
            The next state (position and velocity) of the target disc
        q : np.array
            The process noise used in this step

        """
        new_state = np.copy(state)
        pull_force = - self.spring_force * state[:2]
        drag_force = - self.drag_force * state[2:]**2 * np.sign(state[2:])
        new_state[0] += state[2]
        new_state[1] += state[3]
        new_state[2] += pull_force[0] + drag_force[0]
        new_state[3] += pull_force[1] + drag_force[1]

        if not correlated:
            position_noise = np.random.normal(loc=0, scale=pos_noise, size=(2))
            if hetero_q:
                if np.abs(state[0]) > self.im_size//2 - self.im_size//6 or \
                        np.abs(state[1]) > self.im_size//2 - self.im_size//6:
                    velocity_noise = np.random.normal(loc=0, scale=0.1,
                                                      size=(2))
                    q = 0.1
                elif np.abs(state[0]) > self.im_size//2 - self.im_size//3 or \
                        np.abs(state[1]) > self.im_size//2 - self.im_size//3:
                    velocity_noise = np.random.normal(loc=0, scale=1.,
                                                      size=(2))
                    q = 1.
                else:
                    velocity_noise = np.random.normal(loc=0, scale=3.,
                                                      size=(2))
                    q = 3.
            else:
                velocity_noise = np.random.normal(loc=0, scale=2.,
                                                  size=(2))
                q = 2.

            new_state[:2] += position_noise
            new_state[2:] += velocity_noise

            q = np.array([pos_noise, pos_noise, q, q])
        else:
            pn = 3.0
            cn = 2
            c1 = -0.4
            c2 = 0.2
            c3 = 0.9
            c4 = -0.1
            c5 = 0

            covar = np.array([[pn**2, c1*pn*pn, c2*pn*cn, c3*pn*cn],
                              [c1*pn*pn, pn**2, c4*pn*cn, c5*pn*cn],
                              [c2*pn*cn, c4*pn*cn, cn**2, 0],
                              [c3*pn*cn, c5*pn*cn, 0, cn**2]])

            mean = np.zeros((4))
            noise = np.random.multivariate_normal(mean, covar)
            q = covar
            new_state += noise

        return new_state, q

    def _save(self, data):
        """
        Save a portion of the data to file and splits it into training,
        validation and test data

        Parameters
        ----------
        data : dict of lists
            A dictionary containing the example data

        Returns
        -------
        train_size : int
            Number of training examples saved
        val_size : int
            Number of validation examples saved.
        test_size : int
            Number of test examples saved.

        """
        length = len(data['image'])
        # convert lists to numpy arrays
        for key in self.keys:
            if type(data[key]) == np.ndarray and data[key].dtype == np.float64:
                data[key] = np.array(data[key]).astype(np.float32)

        # shuffle the arrays together
        permutation = np.random.permutation(length)
        for key in self.keys:
            vals = np.copy(data[key])
            data[key] = vals[permutation]

        train_size = int(np.floor(length * 8. / 10.))
        val_size = int(np.floor(length * 1. / 10.))
        test_size = length - train_size - val_size

        if train_size > 0:
            train_data = {}
            for key in self.keys:
                train_data[key] = np.copy(data[key][:train_size])
            rw = self.record_writer_train
            rm = self.record_meta_train
            tfr.write_tfr(train_data, rw, rm, self.out_dir)

        if val_size > 0:
            val_data = {}
            for key in self.keys:
                val_data[key] = \
                    np.copy(data[key][train_size:train_size+val_size])
            rw = self.record_writer_val
            rm = self.record_meta_val
            tfr.write_tfr(val_data, rw, rm, self.out_dir)

        if test_size > 0:
            test_data = {}
            for key in self.keys:
                test_data[key] = np.copy(data[key][train_size+val_size:])
            rw = self.record_writer_test
            rm = self.record_meta_test
            tfr.write_tfr(test_data, rw, rm, self.out_dir)
        return train_size, val_size, test_size


def main(argv=None):
    parser = argparse.ArgumentParser('create disc tracking datset')
    parser.add_argument('--out-dir', dest='out_dir', type=str, required=True,
                        help='where to store results')
    parser.add_argument('--name', dest='name', type=str,
                        default='disc_tracking')
    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=50, help='length of the generated sequences')
    parser.add_argument('--width', dest='width', type=int, default=120,
                        help='width (= height) of the generated observations')
    parser.add_argument('--num-examples', dest='num_examples', type=int,
                        default=2000,
                        help='how many training examples should be generated')
    parser.add_argument('--file-size', dest='file_size', type=int,
                        default=500,
                        help='how many examples should be saved in one file')
    parser.add_argument('--hetero-q', dest='hetero_q', type=int,
                        default=0, choices=[0, 1],
                        help='if the process noise should be heteroscedastic '
                        + 'or contstant')
    parser.add_argument('--correlated-q', dest='correlated_q', type=int,
                        default=0, choices=[0, 1],
                        help='if the process noise should have a full or a '
                        + 'diagonal covariance matrix')
    parser.add_argument('--pos-noise', dest='pos_noise', type=float,
                        default=0.1,
                        help='sigma for the positional process noise')
    parser.add_argument('--num-distractors', dest='num_distractors', type=int,
                        default=5, help='number of distractor disc')
    parser.add_argument('--rescale', dest='rescale', type=int,
                        default=0, choices=[0, 1],
                        help='Rescale the state space to be roughly in [-1, 1]?')
    parser.add_argument('--debug', dest='debug', type=int,
                        default=0, choices=[0, 1],
                        help='Write out images for three sequences as debug ' +
                        'output')

    args = parser.parse_args(argv)

    name = args.name + '_pn=' + str(args.pos_noise) \
        + '_d=' + str(args.num_distractors)
    if args.correlated_q:
        name += '_corr'
    if args.hetero_q:
        name += '_hetero'
    else:
        name += '_const'

    if not os.path.exists(os.path.join(args.out_dir, 'info_' + name + '.txt')):
        c = DiscTrackingData(name, args.out_dir, args.width, args.num_examples,
                             args.sequence_length, args.file_size, args.debug)
        c.create_dataset(args.num_distractors, args.hetero_q,
                         args.correlated_q, args.pos_noise)
    else:
        print('A dataset with this name already exists at ' + args.out_dir)


if __name__ == "__main__":
    main()
