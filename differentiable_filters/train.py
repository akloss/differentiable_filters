#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:47:57 2020

@author: alina
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals


import tensorflow as tf

import numpy as np
import os
import time
import csv
import yaml
from collections import OrderedDict
import sys

from differentiable_filters import filter_network as filtering


class TrainNet():
    def __init__(self, param, logger):
        """
        This class sets up a custom training and validation loop for filtering
        networks

        Parameters
        ----------
        param : dict
            ollection of arguments
        logger : logging.logger
            A logger object that handles output during training

        Returns
        -------
        None.

        """
        self.param = param
        self.log = logger

        self.max_epochs = param['max_epochs']
        self.val_epochs = param['val_epochs']
        self.initial_lr = self.param['learning_rate']

        if 'weight_file' in param.keys() and param['weight_file'] is not None \
                and param['weight_file'] != {}:
            self.restore = True
            self.weight_files = param['weight_file']
        else:
            self.restore = False

        self.debug = param['debug']

        # prepare the output directories
        assert(os.path.exists(self.param['out_dir']))
        if self.param['weight_file'] is None:
            if os.path.exists(self.param['out_dir'] + '/train'):
                tf.gfile.DeleteRecursively(self.param['out_dir'] + '/train')
            os.makedirs(self.param['out_dir'] + '/train')
            if os.path.exists(self.param['out_dir'] + '/val'):
                tf.gfile.DeleteRecursively(self.param['out_dir'] + '/val')
            os.makedirs(self.param['out_dir'] + '/val')
        else:
            if not os.path.exists(self.param['out_dir'] + '/train'):
                os.makedirs(self.param['out_dir'] + '/train')
            if not os.path.exists(self.param['out_dir'] + '/val'):
                os.makedirs(self.param['out_dir'] + '/val')

        self.batch_size = param['batch_size']

    def load_data(self):
        self.log.info("loading data with name " + self.param['data_name'])

        files = os.listdir(self.param['data_dir'])

        # see if there is an info file
        infos = [os.path.join(self.param['data_dir'], f) for f in files
                 if f == 'info_' + self.param['data_name'] + '.txt']
        if len(infos) == 1:
            info_file = open(infos[0], 'r')
        else:
            self.log.error('No or multiple info file found for dataset ' +
                           self.param['data_name'])
            return False

        try:
            info_data = yaml.load(info_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            self.log.exception(exc)
            return False

        if 'Num train' in info_data.keys():
            train_size = info_data['Num train']
        else:
            train_size = 0
        val_size = info_data['Num val']

        # gather the datafiles
        self.train_files = \
            [os.path.join(self.param['data_dir'], f) for f in files
             if f.startswith(self.param['data_name'] + '_') and
             '_train_' in f and '.tfrecord' in f]
        self.val_files = \
            [os.path.join(self.param['data_dir'], f) for f in files
             if f.startswith(self.param['data_name'] + '_') and
             '_val_' in f and '.tfrecord' in f]

        self.train_size = train_size
        self.log.info('Train size: ' + str(train_size) + ', val size: ' +
                      str(val_size))

        self.param.update(info_data)
        return True

    def train(self, mode):
        """
        Run the training

        Parameters
        ----------
        mode : string
            determines which parts of the model are trained. Use "filter" for
            the whole model, "pretrain_obs" for pretraining the observation
            related functions of the context in isolation or "pretrain_proc"
            for pretrainign the process-related functions of the context.

        Returns
        -------
        success : bool
            whether the trainign was completed successfully or not

        """
        if self.param['gpu']:
            # handle selection of the gpu on which tu run the training (as
            # defined by dev_num)
            # hide every other gpu, then set the device number to 0
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.param['dev_num'])
            self.param['dev_num'] = 0

        with tf.Graph().as_default():
            # get the optimizer
            if self.param['optimizer'] == 'adam':
                self.optimizer = \
                    tf.keras.optimizers.Adam(self.initial_lr)
            elif self.param['optimizer'] == 'sgd':
                self.optimizer = \
                    tf.keras.optimizers.SGD(self.initial_lr, momentum=0.0)

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
                # Define training and validation datasets based on the
                # filenames of the tfrecords
                train_set = tf.data.TFRecordDataset(self.train_files)
                train_set = \
                    self.net.tf_record_map(self.param['data_dir'],
                                           self.param['data_name'], train_set,
                                           'train', mode, num_threads=2)

                val_set = tf.data.TFRecordDataset(self.val_files)
                val_set = \
                    self.net.tf_record_map(self.param['data_dir'],
                                           self.param['data_name'], val_set,
                                           'val', mode, num_threads=1)

                # get the real size of the trainignset: some tf_record_map
                # functions might extract more than one example per tf record
                self.epoch_size = \
                    self.train_size * self.net.context.train_multiplier
                self.epoch_size = self.epoch_size // self.param['batch_size']

                # This might be an issue with small RAM, use smaller numbers
                # if necessary
                if not self.param['cache']:
                    train_set = train_set.shuffle(100)
                    train_set = train_set.batch(self.param['batch_size'],
                                                drop_remainder=True)
                    val_set = val_set.batch(self.param['batch_size'],
                                            drop_remainder=True)
                    train_set = train_set.prefetch(50)
                    val_set = val_set.prefetch(10)
                else:
                    train_set = train_set.batch(self.param['batch_size'],
                                                drop_remainder=True)
                    val_set = val_set.batch(self.param['batch_size'],
                                            drop_remainder=True)
                    train_set = train_set.cache()
                    val_set = val_set.cache()
                    train_set = train_set.shuffle(self.epoch_size)

                # A feedable iterator is defined by a handle placeholder and
                # its structure, i.e. `output_types` and `output_shapes`
                handle = tf.compat.v1.placeholder(tf.string, shape=[])
                out_shape = tf.compat.v1.data.get_output_shapes(train_set)
                out_type = tf.compat.v1.data.get_output_types(train_set)

                iterator = tf.compat.v1.data.Iterator.from_string_handle(
                    handle, out_type, out_shape)

                training_iterator = \
                    tf.compat.v1.data.make_initializable_iterator(train_set)
                validation_iterator = \
                    tf.compat.v1.data.make_initializable_iterator(val_set)

                self.log.info('Extracting ' +
                              str(self.net.context.train_multiplier) +
                              ' examples per record from ' +
                              str(self.train_size) + ' records')
                self.log.info('Resulting epoch size for batches of size ' +
                              str(self.param['batch_size']) + ': ' +
                              str(self.epoch_size))

                # update the epoch sizes
                self.net.epoch_size = self.epoch_size
                self.context.epoch_size = self.epoch_size
            except Exception as ex:
                self.log.error('Error setting up data input structure')
                self.log.exception(ex)
                return False

            ##################################################################
            # set up the actual traing graph
            ##################################################################
            self.log.info('Setting up network')
            try:
                # define placeholders
                training_step = tf.compat.v1.placeholder(tf.int32, [])
                # input data from the dataset
                input_data, label = iterator.get_next()
                train_phase = tf.compat.v1.placeholder(tf.bool, shape=[])
                noise = \
                    tf.compat.v1.placeholder(tf.float32,
                                             shape=[self.param['batch_size'],
                                                    self.context.dim_x])

                trainable_variables = []
                if mode == 'filter':
                    prediction = self.net(list(input_data) +
                                          [training_step, noise], train_phase)
                    trainable_variables += \
                        self.net.cell.filter_layers.trainable_variables
                elif mode == 'pretrain_obs' or mode == 'pretrain_process':
                    prediction = self.net(input_data, train_phase)
                else:
                    self.log.error('unknown training mode ' + mode)
                    return False
                total_loss, metrics, metric_names = \
                    self.net.get_loss(prediction, label,
                                      training_step, train_phase)

                trainable_variables += \
                    self.context.observation_models.trainable_variables + \
                    self.context.observation_noise_models.trainable_variables + \
                    self.context.process_models.trainable_variables + \
                    self.context.process_noise_models.trainable_variables

                train_op = self.train_operation(total_loss,
                                                trainable_variables)

            except Exception as ex:
                self.log.error('Error setting up network')
                self.log.exception(ex)
                return False

            ###################################################################
            # Initialize session
            ###################################################################
            # Start the training session
            if self.param['gpu']:
                gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
                conf = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            else:
                conf = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

            if self.param['batch_size']:
                from tensorflow.core.protobuf import rewriter_config_pb2
                off = rewriter_config_pb2.RewriterConfig.OFF
                conf.graph_options.rewrite_options.memory_optimization = off

            sess = tf.compat.v1.Session(config=conf)

            # mark that training is currently running
            open(os.path.join(self.param['out_dir'], 'RUNNING'), 'a').close()

            with sess.as_default():
                # create handles to the two datasets
                try:
                    train_handle = sess.run(training_iterator.string_handle())
                    val_handle = sess.run(validation_iterator.string_handle())
                except:
                    self.log.error('Error initializing dataset handles')
                    return False

                ##############################################################
                # Saving initializing and and restoring
                ##############################################################
                checkpoint_path = os.path.join(self.param['out_dir'] +
                                               '/train', 'model.ckpt')
                if mode == 'filter':
                    self.checkpoint_full = tf.train.Checkpoint(
                        optimizer=self.optimizer,
                        observation_models=self.context.observation_models,
                        observation_noise=self.context.observation_noise_models,
                        process_models=self.context.process_models,
                        process_noise=self.context.process_noise_models,
                        filter_cell=self.net.cell.filter_layers)
                else:
                    self.checkpoint_full = tf.train.Checkpoint(
                        optimizer=self.optimizer,
                        observation_models=self.context.observation_models,
                        observation_noise=self.context.observation_noise_models,
                        process_models=self.context.process_models,
                        process_noise=self.context.process_noise_models)

                init = tf.compat.v1.initializers.global_variables()
                sess.run(init)

                # save the initial state to initialize save_counter
                self.checkpoint_full.save(file_prefix=checkpoint_path)

                if self.restore:
                    self.log.info("restoring model ...")
                    suc, re_steps = self.restore_weights(sess)
                    self.log.info("... done")
                    re_step = max(re_steps)

                # save the initial state imediately after variable loading to
                # minimize the risk of loading a checkpoint where the weights
                # have not been restored yet
                self.checkpoint_full.write(file_prefix=checkpoint_path + '-0')

                ###############################################################
                # set up summary writers
                ###############################################################
                summary_op_train = tf.compat.v1.summary.merge_all()
                summary_op_val = tf.compat.v1.summary.merge_all()
                summary_writer_train = \
                    tf.compat.v1.summary.FileWriter(self.param['out_dir'] +
                                                    '/train', sess.graph)
                summary_writer_val = \
                    tf.compat.v1.summary.FileWriter(self.param['out_dir'] +
                                                    '/val', sess.graph)

                ###############################################################
                # Train
                ###############################################################
                self.log.info('Starting training: ' + str(self.max_epochs) +
                              ' epochs')

                step = 0
                start_epoch = 0
                if self.restore:
                    step = re_step
                    start_epoch = step // self.epoch_size
                    self.log.info("starting in epoch " + str(start_epoch))

                # validate once at the start
                val_loss = self.validate(sess, step, summary_op_val,
                                         validation_iterator, metrics,
                                         metric_names, total_loss,
                                         val_handle, handle, train_phase,
                                         prediction, training_step,
                                         label, noise,
                                         summary_writer_val, first=True)

                last_checkpoint = checkpoint_path + '-' + str(step)
                self.log.info("initial checkpoint saved")
                # we keep a dict of checkpoints and losses
                checkpoints = {}
                # enter the initial checkpoint into the dictionary
                checkpoints[last_checkpoint] = val_loss

                for epoch in np.arange(start_epoch, self.max_epochs):
                    sess.run(training_iterator.initializer)

                    fetches = {'train': train_op, 'loss': total_loss,
                               'summary': summary_op_train}

                    while True:
                        # generate random noise if desired
                        if self.param['add_initial_noise']:
                            n_val = np.random.normal(
                                loc=np.zeros((self.context.dim_x)),
                                scale=self.net.initial_covariance,
                                size=(self.param['batch_size'],
                                      self.context.dim_x))
                        else:
                            n_val = np.zeros((self.param['batch_size'],
                                              self.context.dim_x))

                        feed_dict = {handle: train_handle, train_phase: True,
                                     training_step: step, noise: n_val}
                        try:
                            start_time = time.time()
                            vals = sess.run(fetches, feed_dict)
                            duration = time.time() - start_time
                        except tf.errors.OutOfRangeError:
                            # this marks the end of the dataset, i.e. one epoch
                            break
                        except Exception as ex:
                            self.log.error('Error running training')
                            self.log.exception(ex)
                            return False

                        if step % 50 == 0:
                            format_str = ('step %d, loss = %.5f (%.3f sec)')
                            self.log.info(format_str % (step, vals['loss'],
                                                        float(duration)))
                            summary_writer_train.add_summary(vals['summary'],
                                                             global_step=step)
                            summary_writer_train.flush()

                        step += 1

                    if (epoch + 1) % self.val_epochs == 0 or \
                            (epoch + 1) == self.max_epochs:
                        self.log.info("step " + str(step) + ": Evaluating")
                        val_loss = \
                            self.validate(sess, step, summary_op_val,
                                          validation_iterator,
                                          metrics, metric_names, total_loss,
                                          val_handle, handle, train_phase,
                                          prediction, training_step,
                                          label, noise, summary_writer_val,
                                          first=False)

                        # save a checkpoint
                        last_checkpoint = checkpoint_path + '-' + str(step)
                        pre = checkpoint_path + '-' + str(step)
                        self.checkpoint_full.write(file_prefix=pre)
                        self.log.info("step " + str(step) +
                                      ": checkpoint saved")

                        # we always keep the last checkpoitn at least for one
                        # run, but if we have too many checkpoints, we remove
                        # some old ones
                        num_ckpt = len(checkpoints.keys())
                        if num_ckpt >= self.param['num_checkpoints']:
                            # sort the dictionoary by validation losses
                            sort = \
                                OrderedDict(sorted(checkpoints.items(),
                                                   key=lambda t: t[1]))

                            # remove the last item, ie the one with the highest
                            # validation loss
                            path, _ = sort.popitem()
                            if os.path.exists(path + '.index'):
                                ckpt_dir = os.path.dirname(path)
                                ckpt_name = os.path.basename(path)
                                # delete the checkpoint to save space
                                for fname in os.listdir(ckpt_dir):
                                    if fname.startswith(ckpt_name):
                                        os.remove(os.path.join(ckpt_dir,
                                                               fname))
                            del checkpoints[path]
                        checkpoints[last_checkpoint] = val_loss

                    self.log.info('Done ' + str(epoch + 1) + ' epochs of ' +
                                  str(self.max_epochs))

        # we always keep the last checkpoint
        self.val_log_file.close()
        sess.close()

        # remove running marker and set done
        os.remove(os.path.join(self.param['out_dir'], 'RUNNING'))
        open(os.path.join(self.param['out_dir'], 'DONE'), 'a').close()

        return True

    def train_operation(self, loss, varaiables):
        """
        Training operation

        Parameters
        ----------
        loss : tensor
            current loss of the model
        varaiables : list
            list of the variables that should be trained

        Returns
        -------
        tensorflow operation
            the training operation

        """
        if len(varaiables) > 0:
            gradients = self.optimizer.get_gradients(loss, varaiables)
            grads = list(zip(gradients, varaiables))

            clipped_grads = []
            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    # try catching nans
                    grad = tf.where(tf.math.is_finite(grad), grad,
                                    tf.zeros_like(grad))

                    # limit the maximum change to any variable to 0.5
                    grad = tf.clip_by_value(grad, -0.5/self.initial_lr,
                                            0.5/self.initial_lr)

                    clipped_grads += [(grad, var)]
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    # safe plotting
                    var_plot = tf.where(tf.math.is_finite(var), var,
                                        tf.zeros_like(var))
                    tf.summary.histogram(var.op.name, var_plot)

            # process the updates of the batch norm layers
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops = []
            for lay in self.context.update_ops:
                update_ops += lay.updates
            if len(clipped_grads) > 0:
                apply_gradient_op = \
                    self.optimizer.apply_gradients(clipped_grads)

                with tf.control_dependencies([apply_gradient_op] + update_ops):
                    train_parameters_op = tf.no_op(name='train')
            else:
                with tf.control_dependencies(update_ops):
                    train_parameters_op = tf.no_op(name='train')
            return train_parameters_op
        else:
            return tf.no_op(name='train')

    def validate(self, sess, step, summary_op_val, validation_iterator,
                 metrics, metric_names, total_loss,
                 val_handle, handle, train_phase, prediction,
                 training_step, label, noise, summary_writer_val, first):
        """
        Run one pass through the validation set

        Parameters
        ----------
        sess : tf.Session
            the session
        step : int
            the current trainign step
        summary_op_val :
            tensorflow summary operation
        validation_iterator :
            iterator for the validation set
        metrics : list of tensors
            list of tensors that represent metrics and losses
        metric_names : list of strings
            names for the metrics for logging
        total_loss : tensor
            the (training) loss
        val_handle :
            string handle to the validation set
        handle :
            placeholder for the data set handle
        train_phase :
            placeholder for the training phase
        prediction :
            output of the model
        training_step :
            placeholder for the training step
        label :
            label input to the model
        noise :
            placeholder for the initial noise
        summary_writer_val : tf.SummaryWriter
            summary writer for validation
        first : bool
            indicates it this is the first validation run

        Returns
        -------
        float
            the mean validation loss

        """
        loss_dict = {}
        # validate
        if not self.param['add_initial_noise']:
            val_step = 0
            sess.run(validation_iterator.initializer)
            while True:
                fetches = {'pred': prediction, 'metrics': metrics,
                           'label': label, 'loss': total_loss,
                           'summary': summary_op_val}

                feed_dict = {handle: val_handle, train_phase: False,
                             training_step: step,
                             noise: np.zeros((self.param['batch_size'],
                                              self.context.dim_x))}

                try:
                    start_time = time.time()
                    vals = sess.run(fetches, feed_dict)
                    duration = time.time() - start_time
                    loss = vals['loss']
                    ms = vals['metrics']
                except tf.errors.OutOfRangeError:
                    self.log.info('done')
                    # this marks the end of the dataset
                    break
                except tf.errors.InvalidArgumentError as ex:
                    self.log.exception(ex)
                    continue
                except Exception as ex:
                    self.log.error('Error running validation')
                    self.log.exception(ex)
                    return False

                if val_step == 0:
                    loss_dict['loss'] = [loss]
                else:
                    loss_dict['loss'] += [loss]
                for ind, v in enumerate(ms):
                    k = metric_names[ind]
                    if val_step == 0:
                        loss_dict[k] = [v]
                    else:
                        loss_dict[k] += [v]

                if val_step % 50 == 0:
                    format_str = ('validation step %d, ' +
                                  'loss = %.5f (%.3f sec)')
                    self.log.info(format_str % (val_step, loss,
                                                float(duration)))
                    summary_writer_val.add_summary(vals['summary'],
                                                   global_step=step)
                    summary_writer_val.flush()

                val_step += 1
        else:
            val_step = 0
            for noise_ind, noise_val in enumerate(self.context.noise_list):
                sess.run(validation_iterator.initializer)
                self.log.info('Validating with loss ' + str(noise_ind) + ': ' +
                              str(noise_val))
                while True:
                    fetches = {'pred': prediction, 'metrics': metrics,
                               'label': label, 'loss': total_loss,
                               'summary': summary_op_val}

                    feed_dict = {handle: val_handle, train_phase: False,
                                 training_step: step,
                                 noise: np.tile(noise_val[None, :],
                                                [self.net.batch_size, 1])}

                    try:
                        start_time = time.time()
                        vals = sess.run(fetches, feed_dict)
                        duration = time.time() - start_time
                        loss = vals['loss']
                        ms = vals['metrics']
                    except tf.errors.OutOfRangeError:
                        self.log.info('done')
                        # this marks the end of the dataset
                        break
                    except tf.errors.InvalidArgumentError as ex:
                        self.log.exception(ex)
                        sys.exit()
                        return False
                    except Exception as ex:
                        self.log.error('Error running validation')
                        self.log.exception(ex)
                        sys.exit()
                        return False

                    if val_step == 0:
                        loss_dict['loss'] = [loss]
                    else:
                        loss_dict['loss'] += [loss]
                    for ind, v in enumerate(ms):
                        k = metric_names[ind]
                        if val_step == 0:
                            loss_dict[k] = [v]
                        else:
                            loss_dict[k] += [v]

                    if val_step % 50 == 0:
                        format_str = ('validation step %d, ' +
                                      'loss = %.5f (%.3f sec)')
                        self.log.info(format_str % (val_step, loss,
                                                    float(duration)))
                        summary_writer_val.add_summary(vals['summary'],
                                                       global_step=step)
                        summary_writer_val.flush()

                    val_step += 1

        format_str = ('Validation at step %d, average total loss = %.5f')
        self.log.info(format_str % (step, np.mean(loss_dict['loss'])))
        row = {'step': step}
        for k, v in loss_dict.items():
            row.update({k: np.mean(v), k + '_std': np.std(v)})

        if first:
            write_header = True
            # prepare the logging file for the validation data
            if self.restore and os.path.exists(self.param['out_dir'] +
                                               '/val/log.csv'):
                write_header = False
                self.val_log_file = open(self.param['out_dir'] +
                                         '/val/log.csv', 'a')
            else:
                self.val_log_file = open(self.param['out_dir'] +
                                         '/val/log.csv', 'w')
            keys = ['step', 'loss', 'loss_std']
            keys += loss_dict.keys()
            keys += list(map(lambda x: x + '_std', loss_dict.keys()))
            self.val_log = csv.DictWriter(self.val_log_file,
                                          keys)
            if write_header:
                self.val_log.writeheader()

        self.val_log.writerow(row)
        self.val_log_file.flush()
        return np.mean(loss_dict['loss'])

    def restore_weights(self, sess):
        """
        Load the weights of the model from a checkpoint

        Parameters
        ----------
        sess : tf.Session
            the current session

        Returns
        -------
        success : bool
            if the checkpoitn restoration was successful
        re_steps : list of int
            trainign step(s) at which the restored checkpoints were created

        """
        re_steps = []
        for k, wf in self.weight_files.items():
            if k == 'full':
                self.log.debug(k + ': ' + wf)
                try:
                    self.checkpoint_full.restore(wf).run_restore_ops(sess)
                    re_name = os.path.basename(wf)
                    re_steps += [int(re_name[re_name.find('-')+1:])]
                except Exception as ex:
                    self.log.error('Error loading weight file ' + str(wf))
                    self.log.exception(ex)
                    return False, re_steps
            elif k == 'observations':
                # load the model
                self.log.debug(k + ' models: ' + wf)
                try:
                    ckpt_model = tf.train.Checkpoint(
                        observation_models=self.context.observation_models)
                    restore_op = ckpt_model.restore(wf).expect_partial()
                    restore_op.run_restore_ops(sess)
                    re_name = os.path.basename(wf)
                    re_steps += [int(re_name[re_name.find('-')+1:])]
                except Exception as ex:
                    self.log.error('Error loading weight file ' + str(wf))
                    self.log.exception(ex)
                    return False, re_steps

                # if desired, load the noise
                if self.param['use_pretrained_covar']:
                    self.log.debug(k + ' noise: ' + wf)
                    try:
                        ckpt_model = tf.train.Checkpoint(
                            observation_noise=self.context.observation_noise_models)
                        restore_op = ckpt_model.restore(wf).expect_partial()
                        restore_op.run_restore_ops(sess)
                        re_name = os.path.basename(wf)
                        re_steps += [int(re_name[re_name.find('-')+1:])]
                    except Exception as ex:
                        self.log.error('Error loading weight file ' + str(wf))
                        self.log.exception(ex)
                        return False, re_steps
            elif k == 'process':
                self.log.debug(k + ' models: ' + wf)
                # load the model
                try:
                    ckpt_model = tf.train.Checkpoint(
                        process_models=self.context.process_models)
                    restore_op = ckpt_model.restore(wf).expect_partial()
                    restore_op.run_restore_ops(sess)
                    re_name = os.path.basename(wf)
                    re_steps += [int(re_name[re_name.find('-')+1:])]
                except Exception as ex:
                    self.log.error('Error loading weight file ' + str(wf))
                    self.log.exception(ex)
                    return False, re_steps

                # if desired, load the noise
                if self.param['use_pretrained_covar']:
                    self.log.debug(k + ' noise: ' + wf)
                    try:
                        ckpt_model = tf.train.Checkpoint(
                            process_noise=self.context.process_noise_models)
                        restore_op = ckpt_model.restore(wf).expect_partial()
                        restore_op.run_restore_ops(sess)
                        re_name = os.path.basename(wf)
                        re_steps += [int(re_name[re_name.find('-')+1:])]
                    except Exception as ex:
                        self.log.error('Error loading weight file ' + str(wf))
                        self.log.exception(ex)
                        return False, re_steps
        return True, re_steps
