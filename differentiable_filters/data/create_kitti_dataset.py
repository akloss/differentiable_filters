#!/usr/bin/env python3
"""
Script and class for creating tf.Record datasets for the KITTI visual odometry
task.

Will create 11 different datasets where one sequence is held out as test
data while the other 10 sequences are used for training and validation.
"""

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle
import logging
import sys
from PIL import Image
import cv2

import differentiable_filters.utils.recordio as tfr


class KittiDataset():
    def __init__(self, param):
        """
        Class for creating tf.Record datasets for the KITTI visual odometry
        task.

        Parameters
        ----------
        param : dict
            Dictionary of parameters

        Returns
        -------
        None.

        """
        self.param = param
        self.out_dir = param.out_dir

    def create_dataset(self, files_list_train, test_file, name):
        """
        Creates a tf.Record dataset with the desired characteristics.

        Parameters
        ----------
        files_list_train : list
            List of the raw data files that belong to the sequences for
            training and validation.
        test_file : str
            Raw data file for the test sequence
        name : str
            Name of the dataset

        Returns
        -------
        None.

        """
        # setup logging
        self.log = logging.getLogger(self.param.name)
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
        if os.path.exists(os.path.join(self.param.out_dir,
                                       name + '_error.log')):
            os.remove(os.path.join(self.param.out_dir, name + '_error.log'))
        fh = logging.FileHandler(os.path.join(self.param.out_dir,
                                              name + '_error.log'))
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

        mean_diff = np.zeros(3)
        mean_rgb = np.zeros(3)

        count = 0
        train_count = 0
        val_count = 0

        self.first = True
        self.keys = ['image', 'image_diff', 'state']
        train_data = {key: [] for key in self.keys}
        test_data = {key: [] for key in self.keys}

        self.record_writer_train = \
            tfr.RecordioWriter(self.out_dir, 300, name + '_train_')
        self.record_meta_train = tfr.RecordMeta(name + '_train_')
        self.record_writer_val = \
            tfr.RecordioWriter(self.out_dir, 300, name + '_val_')
        self.record_meta_val = tfr.RecordMeta(name + '_val_')
        self.record_writer_test = \
            tfr.RecordioWriter(self.out_dir, 300, name + '_test_')
        self.record_meta_test = tfr.RecordMeta(name + '_test_')

        for ind, f in enumerate(files_list_train):
            values_train = self._get_data(f, mean_diff, mean_rgb)
            for d in values_train:
                for key in self.keys:
                    train_data[key] += [d[key]]
            if len(train_data['image']) > 1000:
                train_size, val_size = self._save_train_val(train_data)
                train_count += train_size
                val_count += val_size
                train_data = {key: [] for key in self.keys}
        if len(train_data['image']) > 0:
            train_size, val_size = self._save_train_val(train_data)
            train_count += train_size
            val_count += val_size

        values_test = self._get_data(test_file, mean_diff, mean_rgb, 'test')
        for d in values_test:
            for key in self.keys:
                test_data[key] += [d[key]]
        self._save(test_data)

        # save the meta information
        count = train_count + val_count + 1
        fi = open(os.path.join(self.out_dir, 'info_' + name + '.txt'), 'w')
        fi.write('Num data points: ' + str(count) + '\n')
        fi.write('Num train: ' + str(train_count) + '\n')
        fi.write('Num val: ' + str(val_count) + '\n')
        fi.write('Num test: ' + str(1) + '\n')
        fi.write('mean depth: ' + str(mean_diff / (count)) + '\n')
        fi.write('mean rgb: ' + str(mean_rgb / (count)) + '\n')
        fi.close()

        self.record_writer_train.close()
        self.record_writer_test.close()
        self.record_writer_val.close()

    def _save_train_val(self, data):
        """
        Save the training and validation data

        Parameters
        ----------
        data : dict of lists
            A dictionary containing the example data

        Returns
        -------
        train_size : int
            Number of training examples saved.
        val_size : int
            Number of validation examples saved.

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

        train_size = int(np.floor(length * 6. / 7.))
        val_size = length - train_size

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
                val_data[key] = np.copy(data[key][train_size:])
            rw = self.record_writer_val
            rm = self.record_meta_val
            tfr.write_tfr(val_data, rw, rm, self.out_dir)
        return train_size, val_size

    def _save(self, data):
        """
        Save the test data

        Parameters
        ----------
        data : dict of lists
            A dictionary containing the example data

        Returns
        -------
        None.

        """
        rw = self.record_writer_test
        rm = self.record_meta_test

        # convert lists to numpy arrays
        for key in self.keys:
            if type(data[key]) == list:
                v = np.array(data[key])
                data[key] = v
            if type(data[key]) == np.ndarray and data[key].dtype == np.float64:
                data[key] = data[key].astype(np.float32)

        tfr.write_tfr(data, rw, rm, self.out_dir)
        return

    def _get_data(self, filename, mean_diff, mean_rgb, mode='train'):
        """
        Generat data for one sequence

        Parameters
        ----------
        filename : str
            name of the files belonging to this sequence.
        mean_diff : np.array
            Mean values of the difference image observations (can be used for
            normalization)
        mean_rgb :  np.array
            Mean values of the rgb image observations (can be used for
            normalization)
        mode : str, optional
            Flag to signal if the sequence is used for training or testing.
            The default is 'train'.

        Returns
        -------
        out : list
            List with all generated examples.

        """
        self.log.info(mode + ': ' + filename)
        out = []
        # load the txt file
        with open(filename + '_image1.txt', 'r') as f:
            tmp = np.loadtxt(f)

        # read in position and orientation data
        xs = tmp[:, 11:12]
        ys = -tmp[:, 3:4]
        thetas = self._wrap_angle(-np.arctan2(-tmp[:, 8:9], tmp[:, 10:11]))
        vs = np.sqrt((ys[1:] - ys[:-1])**2 + (xs[1:] - xs[:-1])**2) / 0.103
        theta_dots = self._wrap_angle(thetas[1:] - thetas[:-1]) / 0.103

        # read the images from the first camera, create rgb and difference
        # observations and mirrored versions of the observations
        ims, ims_m, im_diffs, im_diffs_m = self._read_images(filename,
                                                             'image_2')
        self.log.info('sequence length ' + str(len(ims)))

        # now ims and im_diffs start at t=1, but xs, ys, thetas, vs, and theta
        # dots start at t = 0, so we cut off the first entry
        xs = xs[1:]
        ys = ys[1:]
        thetas = thetas[1:]
        vs = vs[1:]
        theta_dots = theta_dots[1:]

        # we also cut off the last entries for the images, xs, ys and thetas,
        # since we do not have velocities for the last step
        ims = ims[:-1]
        im_diffs = im_diffs[:-1]
        ims_m = ims_m[:-1]
        im_diffs_m = im_diffs_m[:-1]
        xs = xs[:-1]
        ys = ys[:-1]
        thetas = thetas[:-1]

        assert len(ims) == len(xs) and len(ims) == len(vs)

        # positions and velocities are now aligned such that
        # p[i] + v[i] = p[i+1]
        # however, velocities and difference images are not aligned:
        # v[i] = p[i+1] = p[i] but im_diff[i] = im[i] - im[i-1]
        if mode == 'train':
            out += self._get_subsequences(ims, im_diffs, xs, ys, thetas, vs,
                                           theta_dots, 50, 100)
            # do the same again with the mirrored data
            out += self._get_subsequences(ims_m, im_diffs_m, xs, -ys, -thetas,
                                          vs, -theta_dots, 50, 100)
        elif mode == 'test':
            vals = {}
            # use the whole sequence
            vals['image'] = np.array(ims)
            vals['image_diff'] = np.array(im_diffs)
            vals['state'] = np.concatenate([xs, ys, thetas,
                                            vs, theta_dots], axis=1)
            out += [vals]
            # and the whole mirrored sequence
            vals = {}
            vals['image'] = np.array(ims_m)
            vals['image_diff'] = np.array(im_diffs_m)
            vals['state'] = np.concatenate([xs, -ys, -thetas,
                                            vs, -theta_dots], axis=1)
            out += [vals]

        # read the images from the second camera and repeat the process
        ims, ims_m, im_diffs, im_diffs_m = self._read_images(filename,
                                                             'image_3')
        # cut off the last entries for the images
        ims = ims[:-1]
        im_diffs = im_diffs[:-1]
        ims_m = ims_m[:-1]
        im_diffs_m = im_diffs_m[:-1]

        assert len(ims) == len(xs) and len(ims) == len(vs)

        if mode == 'train':
            out += self._get_subsequences(ims, im_diffs, xs, ys, thetas, vs,
                                           theta_dots, 50, 100)
            # do the same again with the mirrored data
            out += self._get_subsequences(ims_m, im_diffs_m, xs, -ys, -thetas,
                                          vs, -theta_dots, 50, 100)
        elif mode == 'test':
            vals = {}
            # use the whole sequence
            vals['image'] = np.array(ims)
            vals['image_diff'] = np.array(im_diffs)
            vals['state'] = np.concatenate([xs, ys, thetas,
                                            vs, theta_dots], axis=1)
            out += [vals]
            # and the whole mirrored sequence
            vals = {}
            vals['image'] = np.array(ims_m)
            vals['image_diff'] = np.array(im_diffs_m)
            vals['state'] = np.concatenate([xs, -ys, -thetas,
                                            vs, -theta_dots], axis=1)
            out += [vals]

        return out

    @staticmethod
    def _read_images(filename, cam_key):
        """
        Reads in the images from the raw kitti data, creates the difference
        images and encodes them as byte strings.
        Also returns mirrired versions of each image for data augmentation

        Parameters
        ----------
        filename : str
            name of the files belonging to this sequence.
        cam_key : str
            Either image_2 or image_3 for images from the first or second
            camera

        Returns
        -------
        ims : list
            The original rgb images
        ims_m : list
            The mirrored rgb images
        im_diffs : list
            The original difference images
        im_diffs_m : list
            The mirrored difference images

        """
        ims = []
        ims_m = []
        im_diffs = []
        im_diffs_m = []
        im_files = [os.path.join(filename, cam_key, fi)
                    for fi in os.listdir(os.path.join(filename, cam_key))]
        im_files = sorted(im_files)
        i = Image.open(im_files[0])
        p = np.asarray(i, 'float32')
        for ind, f in enumerate(im_files[1:]):
            i = Image.open(f)
            i = np.asarray(i, 'float32')
            diff = i - p
            # take care not to cut off the negative part by shifting by the
            # maximum pixel value (this has a minimal loss)
            diff = (diff + 255) / 2
            im = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            im = cv2.imencode('.png', im)[1].tobytes()
            im_m = cv2.cvtColor(np.fliplr(i), cv2.COLOR_RGB2BGR)
            im_m = cv2.imencode('.png', im_m)[1].tobytes()
            d = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)
            d = cv2.imencode('.png', d)[1].tobytes()
            d_m = cv2.cvtColor(np.fliplr(diff), cv2.COLOR_RGB2BGR)
            d_m = cv2.imencode('.png', d_m)[1].tobytes()

            im_diffs += [d]
            im_diffs_m += [d_m]
            ims += [im]
            ims_m += [im_m]
            p = np.copy(i)
        return ims, ims_m, im_diffs, im_diffs_m


    @staticmethod
    def _get_subsequences(ims, im_diffs, xs, ys, thetas, vs, theta_dots, num,
                          sl):
        """
        Extracts num sequences of length sl from each file, all starting
        at different timesteps.

        Parameters
        ----------
        ims : list
            The full sequence of rgb images.
        im_diffs : list
            The full sequence of difference images
        xs : list
            The full sequence of positional x coordinates
        ys : list
            The full sequence of positional y-coordinates
        thetas : list
            The full sequence of headings.
        vs : list
            The full sequence of linear velocities
        theta_dots : list
            The full sequence of angular velocities
        num : int
            Number of subsequences to extract
        sl : int
            Length of the extracted subsequences

        Returns
        -------
        out : list
            List with data for all extracted subsequences.

        """
        inds = np.random.choice(len(ims) - sl, size=num, replace=False)
        inds = list(inds)
        shuffle(inds)
        out = []
        for i in inds:
            vals = {}
            vals['image'] = np.array(ims[i:i+sl])
            vals['image_diff'] = \
                np.array(im_diffs[i:i+sl])
            vals['state'] = \
                np.concatenate([xs[i:i+sl], ys[i:i+sl], thetas[i:i+sl],
                                vs[i:i+sl], theta_dots[i:i+sl]], axis=1)
            out += [vals]
        return out

    @staticmethod
    def _wrap_angle(angle):
        return ((angle - np.pi) % (2 * np.pi)) - np.pi


def main(argv=None):
    parser = argparse.ArgumentParser('create kitti dataset')
    parser.add_argument('--name', dest='name', type=str, default='kitti',
                        help='the name of the created datasets')
    parser.add_argument('--out-dir', dest='out_dir', type=str, required=True,
                        help='where to store results')
    parser.add_argument('--source-dir', dest='source_dir', type=str,
                        required=True,
                        help='where to find the raw kitti dataset.')

    args = parser.parse_args(argv)
    plt.ioff()

    files = [os.path.join(args.source_dir, f)
             for f in os.listdir(args.source_dir)
             if os.path.isdir(os.path.join(args.source_dir, f))]
    files = sorted(files)

    for i in range(len(files)):
        if not os.path.exists(os.path.join(args.out_dir,
                                           'info_' + args.name +
                                           '_' + str(i) + '.txt')):
            test_file = files[i]
            train_files = files[:i] + files[i+1:]
            print('set ', i, ' test file: ', test_file)
            print('train_files: ', train_files)
            c = KittiDataset(args)
            c.create_dataset(train_files,
                             test_file,
                             args.name + '_' + str(i))
    else:
        print('A dataset with name, ', args.name + '_' + str(i),
              'already exists at ', args.out_dir)
    return


if __name__ == "__main__":
    main()
