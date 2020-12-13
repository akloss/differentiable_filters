#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:25:28 2020

@author: alina
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals


import tensorflow as tf
import numpy as np
import os
import yaml

from differentiable_filters import filter_network as filtering


class TestNet():
    def __init__(self, param, logger):
        self.param = param
        self.batch_size = 1
        self.log = logger
        self.debug = param['debug']
        self.param['learning_rate'] = 0

    def load_data(self):
        """
        Prepare the data loading

        Returns
        -------
        success : bool
            If the test dataset has been found.

        """
        self.log.info("loading data")
        files = os.listdir(self.param['data_dir'])

        if self.param['info_file'] is not None:
            info_file = open(self.param['info_file'], 'r')
        else:
            # see if there is an info file
            infos = [os.path.join(self.param['data_dir'], f)
                     for f in files
                     if f.startswith('info_' + self.param['data_name'] +
                                     '.txt') and not f.endswith('~')]

            self.log.debug(infos)
            if len(infos) == 1:
                info_file = open(infos[0], 'r')
            else:
                self.log.error('No info file found for dataset ' +
                               self.param['data_name'])
                return False

        try:
            info_data = yaml.load(info_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            self.log.exception(exc)
            return False

        # gather the datafiles
        if self.param['on_valset']:
            self.test_files = \
                [os.path.join(self.param['data_dir'], f)
                 for f in os.listdir(self.param['data_dir'])
                 if f.startswith(self.param['data_name'] + '_') and
                 '_val_' in f and '.tfrecord' in f]
            test_size = info_data['Num val']
        else:
            self.test_files = \
                [os.path.join(self.param['data_dir'], f)
                 for f in os.listdir(self.param['data_dir'])
                 if f.startswith(self.param['data_name'] + '_') and
                 '_test_' in f and '.tfrecord' in f]
            test_size = info_data['Num test']

        self.log.info('Test size: ' + str(test_size))
        self.param.update(info_data)
        self.param['epoch_size'] = test_size
        return True

    def test(self, mode):
        """
        Run the evaluation

        Parameters
        ----------
        mode : string
            determines which parts of the model are evaluated. Use "filter" for
            the whole model, "pretrain_obs" for the observation
            related functions of the context or "pretrain_proc"
            for the process-related functions of the context.

        Returns
        -------
        success : bool
            whether the testing was completed successfully or not

        """
        if self.param['gpu']:
            # handle selection of the gpu on which tu run the training (as
            # defined by dev_num)
            # hide every other gpu, then set the device number to 0
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.param['dev_num'])
            self.param['dev_num'] = 0

        with tf.Graph().as_default():
            ###############################################################
            # setup the problem context
            ###############################################################
            try:
                # TODO parameterized import
                if self.param['problem'] == 'kitti':
                    from differentiable_filters.contexts import kitti
                    self.context = kitti.Context(self.param, mode)
                elif self.param['problem'] == 'toy':
                    from differentiable_filters.contexts import toy
                    self.context = toy.Context(self.param, mode)
                elif self.param['problem'] == 'pushing':
                    from differentiable_filters.contexts import pushing
                    self.context = pushing.Context(self.param, mode)
                else:
                    self.log.error('Unknown context: ' +
                                   self.param['problem'])
                    return False
            except Exception as ex:
                self.log.exception(ex)
                return False

            ###############################################################
            # setup the filtering
            ###############################################################
            try:
                if mode == 'filter':
                    self.net = filtering.Filter(self.param, self.context)
                elif mode == 'pretrain_obs':
                    self.net = filtering.Filter(self.param, self.context)
                elif mode == 'pretrain_process':
                    self.net = filtering.Filter(self.param, self.context)
                else:
                    self.log.error('unknown training mode ' + mode)
                    return False

                self.net = filtering.Network(self.param, mode)
            except Exception as ex:
                self.log.exception(ex)
                return False

            ###############################################################
            # setup data input management
            ###############################################################
            self.log.info('Setting up data loading')
            try:
                # Define the test dataset based on the filenames of the
                # tfrecords
                test_set = tf.data.TFRecordDataset(self.test_files)
                test_set = self.net.tf_record_map(self.param['data_dir'],
                                                  self.param['data_name'],
                                                  test_set, 'test', mode,
                                                  num_threads=3)

                test_set = test_set.batch(self.batch_size, drop_remainder=True)
                test_set = test_set.prefetch(100)
                iterator = \
                    tf.compat.v1.data.make_initializable_iterator(test_set)

                self.epoch_size = \
                    self.param['epoch_size'] * self.net.context.test_multiplier
                self.epoch_size = self.epoch_size // self.batch_size
                self.log.info('Extracting ' +
                              str(self.net.context.test_multiplier) +
                              ' examples per record from ' +
                              str(self.param['epoch_size']) +
                              ' records')
                self.log.info('Resulting test set size: ' +
                              str(self.epoch_size))

                # update the epoch sizes
                self.net.epoch_size = self.epoch_size
                self.context.epoch_size = self.epoch_size
            except Exception as ex:
                self.log.error('Error setting up data input structure')
                self.log.exception(ex)
                return False
            ###############################################################
            # setup the filtering network
            ###############################################################
            try:
                # define placeholders
                global_step = tf.Variable(0., trainable=False)
                # input data from the dataset
                input_data, label = iterator.get_next()
                train_phase = tf.compat.v1.placeholder(tf.bool, shape=[])
                noise = \
                    tf.compat.v1.placeholder(tf.float32,
                                             shape=[1, self.context.dim_x])

                # set up the prediction, loss and training
                if mode == 'filter':
                    prediction = self.net(list(input_data) +
                                          [global_step, noise], train_phase)
                elif mode == 'pretrain_obs' or 'pretrain_process':
                    prediction = self.net(input_data, train_phase)
                else:
                    self.log.error('unknown training mode ' + mode)
                    return False
                total_loss, metrics, metric_names = \
                    self.net.get_loss(prediction, label, global_step,
                                      train_phase)
            except Exception as ex:
                self.log.error('Error setting up network')
                self.log.exception(ex)
                return False

            additional = self.net.get_fetches(list(input_data) + [noise],
                                              label, prediction)
            self.log.info('Done setting up network')

            ###################################################################
            # Initialize session
            ###################################################################
            if self.param['gpu']:
                gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
                conf = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            else:
                conf = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
            sess = tf.compat.v1.Session(config=conf)

            ###################################################################
            # Restore network
            ###################################################################
            self.log.info("restoring model ...")
            # initialize the whole model from a previous checkpoint
            try:
                if mode == 'filter':
                    self.checkpoint_full = tf.train.Checkpoint(
                        observation_models=self.context.observation_models,
                        observation_noise=self.context.observation_noise_models,
                        process_models=self.context.process_models,
                        process_noise=self.context.process_noise_models,
                        filter_cell=self.net.cell.filter_layers)
                else:
                    self.checkpoint_full = tf.train.Checkpoint(
                        observation_models=self.context.observation_models,
                        observation_noise=self.context.observation_noise_models,
                        process_models=self.context.process_models,
                        process_noise=self.context.process_noise_models)

                init = tf.initializers.global_variables()
                sess.run(init)
            except Exception as ex:
                self.log.error('Error initializing variables')
                self.log.exception(ex)
                return False

            suc = self.restore_weights(sess)
            if suc:
                self.log.info("... done")
            else:
                return False

            summary_op = tf.compat.v1.summary.merge_all()
            summary_writer = \
                tf.compat.v1.summary.FileWriter(self.param['out_dir'],
                                                sess.graph)

            ###################################################################
            # start the testing
            ###################################################################
            with sess.as_default():
                if self.param['add_initial_noise']:
                    # repeat the testing with all noise sequences to get a fair
                    # result
                    for noise_ind, noise_val in enumerate(self.context.noise_list):
                        self.log.info('Testing with initial error ' +
                                      str(noise_ind) + ' of ' +
                                      str(len(self.net.context.noise_list)) +
                                      ': ' + str(noise_val))
                        sess.run(iterator.initializer)
                        overall_step = 0
                        log_dict = {}
                        add_dict = {}
                        step = 0
                        while True:
                            fetches = {'loss': total_loss, 'metrics': metrics,
                                       'summary': summary_op}
                            noise_val = \
                                noise_val.reshape(1, self.net.context.dim_x)
                            feed_dict = {train_phase: False, noise: noise_val}

                            if additional:
                                fetches.update(additional)
                            try:
                                vals = sess.run(fetches, feed_dict)
                            except tf.errors.OutOfRangeError:
                                self.log.info('done')
                                # this marks the end of the dataset
                                break
                            except tf.errors.DataLossError as ex:
                                self.log.exception(ex)
                                break
                            except Exception as ex:
                                self.log.error('Error running testing')
                                self.log.exception(ex)
                                return False, {}

                            if np.isnan(vals['loss']):
                                self.log.error('nan loss')
                                return False, {}

                            # print an overview fairly often.
                            if step % 50 == 0:
                                format_str = ('Testing step %d, ' +
                                              'total loss = %.5f')
                                self.log.info(format_str % (step,
                                                            vals['loss']))
                                summary_writer.add_summary(
                                    vals['summary'], global_step=overall_step)
                                summary_writer.flush()

                            if step == 0:
                                for ind, name in enumerate(metric_names):
                                    v = vals['metrics'][ind]
                                    if name in ['vis', 'r_pred', 'r_het_diag',
                                                'r_het_tri', 'q_pred', 'cont']:
                                        log_dict[name] = v
                                    else:
                                        if len(v.shape) > 1 and v.shape[-1] == 1:
                                            v = np.squeeze(v, axis=-1)
                                        elif len(v.shape) == 0:
                                            v = v.reshape(1)
                                        if len(v.shape) > 1:
                                            log_dict[name] = np.mean(v, axis=1)
                                            log_dict[name + '_steps'] = v
                                        else:
                                            log_dict[name] = v
                                for k in additional.keys():
                                    v = vals[k]
                                    if type(v) == list:
                                        v = np.array(v)
                                    if type(v) == np.ndarray and len(v.shape) > 0:
                                        v = np.squeeze(v)
                                    if len(v.shape) == 0:
                                        v = np.asscalar(v)
                                    add_dict[k] = [v]
                                log_dict['loss'] = [vals['loss']]
                            else:
                                for ind, name in enumerate(metric_names):
                                    v = vals['metrics'][ind]
                                    if name in ['vis', 'r_pred', 'r_het_diag',
                                                'r_het_tri', 'q_pred', 'cont']:
                                        log_dict[name] = \
                                            np.concatenate([log_dict[name], v],
                                                           axis=0)
                                    else:
                                        if len(v.shape) > 1 and v.shape[-1] == 1:
                                            v = np.squeeze(v, axis=-1)
                                        elif len(v.shape) == 0:
                                            v = v.reshape(1)
                                        if len(v.shape) > 1:
                                            vs = np.mean(v, axis=1)
                                            log_dict[name + '_steps'] = \
                                                np.concatenate([log_dict[name + '_steps'], v],
                                                               axis=0)
                                            log_dict[name] = \
                                                np.concatenate([log_dict[name], vs],
                                                               axis=0)
                                        else:
                                            log_dict[name] = \
                                                np.concatenate([log_dict[name], v],
                                                               axis=0)
                                for k in additional.keys():
                                    v = vals[k]
                                    if type(v) == list:
                                        v = np.array(v)
                                    if type(v) == np.ndarray and len(v.shape) > 0:
                                        v = np.squeeze(v)
                                    if len(v.shape) == 0:
                                        v = np.asscalar(v)
                                    add_dict[k] += [v]
                                log_dict['loss'] += [vals['loss']]
                            step += 1
                            overall_step += 1
                        self.net.save_log(log_dict, self.param['out_dir'],
                                          self.param['step'], noise_ind, mode)
                        if noise_ind == 0:
                            self.net.evaluate(log_dict, add_dict,
                                              self.param['out_dir'],
                                              self.param['step'])
                else:
                    sess.run(iterator.initializer)
                    step = 0
                    log_dict = {}
                    add_dict = {}
                    while True:
                        fetches = {'loss': total_loss, 'metrics': metrics,
                                   'sum': summary_op}
                        feed_dict = {train_phase: False,
                                     noise: np.zeros((1,
                                                      self.net.context.dim_x),
                                                     dtype=np.float32)}

                        if additional:
                            fetches.update(additional)
                        try:
                            vals = sess.run(fetches, feed_dict)
                        except tf.errors.OutOfRangeError:
                            self.log.info('done')
                            # this marks the end of the dataset
                            break
                        except tf.errors.DataLossError as ex:
                            self.log.exception(ex)
                            break
                        except Exception as ex:
                            self.log.error('Error running testing')
                            self.log.exception(ex)
                            return False, {}

                        # print an overview fairly often.
                        if step % 50 == 0:
                            format_str = ('Testing step %d, ' +
                                          'total loss = %.5f')
                            self.log.info(format_str % (step, vals['loss']))
                            summary_writer.add_summary(vals['sum'],
                                                       global_step=step)

                        if step == 0:
                            for ind, name in enumerate(metric_names):
                                v = vals['metrics'][ind]
                                if name in ['vis', 'r_pred', 'r_het_diag',
                                            'r_het_tri']:
                                    log_dict[name] = v
                                else:
                                    if len(v.shape) > 1 and v.shape[-1] == 1:
                                        v = np.squeeze(v, axis=-1)
                                    elif len(v.shape) == 0:
                                        v = v.reshape(1)
                                    if len(v.shape) > 1:
                                        log_dict[name] = np.mean(v, axis=1)
                                        log_dict[name + '_steps'] = v
                                    else:
                                        log_dict[name] = v
                            for k in additional.keys():
                                v = vals[k]
                                if type(v) == list:
                                    v = np.array(v)
                                if type(v) == np.ndarray and len(v.shape) > 0:
                                    v = np.squeeze(v)
                                if len(v.shape) == 0:
                                    v = np.asscalar(v)
                                add_dict[k] = [v]
                            log_dict['loss'] = [vals['loss']]
                        else:
                            for ind, name in enumerate(metric_names):
                                v = vals['metrics'][ind]
                                if name in ['vis', 'r_pred', 'r_het_diag',
                                            'r_het_tri']:
                                    log_dict[name] = \
                                        np.concatenate([log_dict[name], v],
                                                       axis=0)
                                else:
                                    if len(v.shape) > 1 and v.shape[-1] == 1:
                                        v = np.squeeze(v, axis=-1)
                                    elif len(v.shape) == 0:
                                        v = v.reshape(1)
                                    if len(v.shape) > 1:
                                        vs = np.mean(v, axis=1)
                                        log_dict[name + '_steps'] = \
                                            np.concatenate([log_dict[name + '_steps'],
                                                            v], axis=0)
                                        log_dict[name] = \
                                            np.concatenate([log_dict[name],
                                                            vs], axis=0)
                                    else:
                                        log_dict[name] = \
                                            np.concatenate([log_dict[name], v],
                                                           axis=0)
                            for k in additional.keys():
                                v = vals[k]
                                if type(v) == list:
                                    v = np.array(v)
                                if type(v) == np.ndarray and len(v.shape) > 0:
                                    v = np.squeeze(v)
                                if len(v.shape) == 0:
                                    v = np.asscalar(v)
                                add_dict[k] += [v]
                            log_dict['loss'] += [vals['loss']]
                        step += 1

                    self.log.info('Finished after ' + str(step) + ' steps')
                    self.net.save_log(log_dict, self.param['out_dir'],
                                      self.param['step'], 0, mode)
                    self.log.info('Log saved')
                    self.net.evaluate(log_dict, add_dict,
                                      self.param['out_dir'],
                                      self.param['step'], mode)
                    self.log.info('Evaluation done.')
        sess.close()

        # set done
        open(os.path.join(self.param['out_dir'], 'DONE'), 'a').close()

        return True, log_dict

    def restore_weights(self, sess):
        wf = self.param['weight_file']['full']
        try:
            # print(tf.train.list_variables(wf))
            stat = self.checkpoint_full.restore(wf)
            stat.run_restore_ops(sess)
        except Exception as ex:
            self.log.error('Error loading weight file ' + wf)
            self.log.exception(ex)
            return False
        return True