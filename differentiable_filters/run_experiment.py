#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:40:52 2020

@author: alina
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
import logging
import sys
import shutil
import copy

from differentiable_filters import train
from differentiable_filters import test


class RunExperiment():
    def __init__(self, args):
        self.param = args
        self.redo = args.redo_results
        self.base_dir = args.out_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.data_dir = args.data_dir
        self.name =  \
            os.path.join(args.name, args.data_name_train, args.filter,
                         args.loss)

        self.remove_bad = args.remove_bad

        if type(self.param.data_name_test) is str:
            self.param.data_name_test = [self.param.data_name_test]

        if self.param.data_name_test == ['']:
            self.param.data_name_test = []

        #######################################################################
        # setup logging
        #######################################################################
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s: [%(name)s] ' +
                                      '[%(levelname)s] %(message)s')

        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

        # create file handler which logs warnings errors and criticals
        error_file = os.path.join(self.base_dir,
                                  self.name.replace('/', '_') + '_error.log')

        if os.path.exists(error_file):
            os.remove(error_file)

        fh = logging.FileHandler(error_file)
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

        #######################################################################
        # read in the config file
        #######################################################################
        config_path = os.path.join(self.base_dir, 'config.json')
        if os.path.exists(config_path):
            f = open(config_path, 'r')
            self.confs = json.load(f)
            f.close()
        else:
            self.log.error('No config file found at ' + config_path)

    def run_experiments(self):
        self.log.info('##########################################')
        self.log.info('# Starting experiment ' + self.name)
        self.log.info('##########################################')

        # do pretraining or get pretrained weights
        if self.param.pretrain_observations or self.param.pretrain_process:
            success, pretrain_files = self.run_pretrain()
            if not success:
                self.log.info('# Experiment failed')
                self.log.info('##########################################')
                return
        else:
            pretrain_files = {}
            self.log.info('# Not pretraining')

        if not self.param.train_filter:
            self.log.info('# Not training a filter')
            self.log.info('##########################################')
            return

        # run the actual training
        working_dir = os.path.join(self.base_dir, 'experiments', self.name)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        train_param = self._get_train_configuration(self.param.config_name)
        self.log.info('### ... Starting training ')
        self.log.info('###------------------------------------')
        suc = self.run_train(working_dir, train_param, 'model.ckpt-0',
                             weight_files=pretrain_files)
        # if suc:
        self.log.info('### ... Starting Evaluation ')
        self.log.info('###------------------------------------')
        suc = self.run_eval(working_dir, 'model.ckpt-0')
        if suc:
            self.log.info('# Experiment done successfully')
        else:
            self.log.info('# Experiment failed')
        self.log.info('##########################################')

    def run_pretrain(self):
        """
        If the experiment uses pretraining, we either have to run pretraining
        or we can just use the data that is already there
        """
        self.log.info('### Using pretraining ')
        self.log.info('###---------------------------------------')

        pretrain_dir = os.path.join(self.base_dir, 'pretrain',
                                    self.param.data_name_train,
                                    'sc_' + str(self.param.scale))
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)

        weight_files = {}
        success = True
        # get the paths to the weight files
        if self.param.pretrain_observations:
            self.log.info('###### Pretraining Observations ... ')
            working_dir = os.path.join(pretrain_dir, 'observations')
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            obs_path = os.path.join(working_dir, 'pretrain_obs.ckpt-0')
            if os.path.exists(obs_path + '.index'):
                weight_files['observations'] = obs_path
                self.log.info('###### ... already done ')
            else:
                self.log.info('###### ... starting training ')
                self.log.info('######------------------------------------')
                train_param = \
                    self._get_train_configuration('pretrain_observations')
                suc = self.run_train(working_dir, train_param,
                                     'pretrain_obs.ckpt-0', 'pretrain_obs')
                if suc:
                    self.log.info('###### ... Starting Evaluation ')
                    self.log.info('######---------------------------------')
                    suc = self.run_eval(working_dir, 'pretrain_obs.ckpt-0',
                                        'pretrain_obs')
                success = suc and success
                if success:
                    weight_files['observations'] = obs_path
        if self.param.pretrain_process:
            self.log.info('###### Pretraining process model ... ')
            working_dir = os.path.join(pretrain_dir, 'process')
            if not os.path.exists(working_dir):
                os.makedirs(working_dir)
            proc_path = os.path.join(working_dir, 'pretrain_process.ckpt-0')
            if os.path.exists(proc_path + '.index'):
                weight_files['process'] = proc_path
                self.log.info('###### ... already done ')
            else:
                self.log.info('###### ... starting training ')
                self.log.info('######------------------------------------')
                train_param = \
                    self._get_train_configuration('pretrain_process')
                suc = self.run_train(working_dir, train_param,
                                     'pretrain_process.ckpt-0',
                                     'pretrain_process')
                if suc:
                    self.log.info('###### ... Starting Evaluation ')
                    self.log.info('######---------------------------------')
                    suc = self.run_eval(working_dir, 'pretrain_process.ckpt-0',
                                        'pretrain_process')
                success = suc and success
                if success:
                    weight_files['process'] = proc_path

        if success:
            self.log.info('### Pretraining done!')
        else:
            self.log.info('### Pretraining failed!')
        self.log.info('###---------------------------------------')
        return success, weight_files

    def run_train(self, working_dir, train_param, out_name, mode='filter',
                  weight_files={}):
        # run training for all parameter combinations in train_param
        done_training = True
        all_success = True
        net_param = vars(copy.deepcopy(self.param))
        if 'num_samples_train' in net_param.keys():
            net_param['num_samples'] = self.param.num_samples_train

        for param in train_param:
            name = 'lr' + str(param['learning_rate']) + \
                '_batch' + str(param['batch_size']) + '_max' + \
                str(param['max_epochs'])
            config_dir = os.path.join(working_dir, name)
            param.update(net_param)
            param['out_dir'] = config_dir
            param['data_name'] = self.param.data_name_train
            param['sequence_length'] = self.param.sequence_length_train
            param['weight_file'] = weight_files
            if self.param.weight_file is not None:
                param['weight_file'] = {'observations':
                                        self.param.weight_file}
            if not os.path.exists(os.path.join(config_dir, 'DONE')):
                run = True
                if os.path.exists(os.path.join(config_dir, 'RUNNING')):
                    if self.param.continue_train:
                        self.log.info('###### Training was already running' +
                                      ' with this configuration.')
                        weight_file = self._get_last_checkpoint(config_dir)
                        if weight_file[1] > 1:
                            self.log.info('###### Resuming from last ' +
                                          'checkpoint.')
                            param['weight_file'] = {'full': weight_file[0]}
                        else:
                            self.log.info('###### Only first checkpoint ' +
                                          'found. Restarting')
                            # remove everything in the config directory
                            # (will be created again)
                            shutil.rmtree(config_dir)
                    else:
                        self.log.info('###### Training is already running' +
                                      ' with this configuration. Aborting')
                        run = False
                        done_training = False
                        all_success = False
                if run:
                    if not os.path.exists(config_dir):
                        os.mkdir(config_dir)
                    self.log.info('###### Training configuration ' + name)
                    # start training
                    success = self._train(param, config_dir, mode)
                    done_training = done_training or success
                    all_success = all_success and success
                    if success:
                        self.log.info('###### Training done sucessfully')
                    else:
                        self.log.info('###### Training failed')
            else:
                self.log.info('##### ' + name + ' already done.')
            self.log.info('######------------------------------------')

        if done_training:
            if all_success:
                self.log.info('###### All configurations done successfully ')
            else:
                self.log.info('###### Training done, but some configurations' +
                              ' failed ')
            open(os.path.join(working_dir, 'DONE'), 'a').close()
        else:
            self.log.info('###### Training failed ')
        self.log.info('######------------------------------------')
        return done_training

    def run_eval(self, working_dir, out_name, mode='filter'):
        self.log.info('###### Getting the best model ')
        # select the best
        suc, best = self._get_best(working_dir)
        if not suc:
            return False

        if not self.param.no_test:
            self.log.info('######------------------------------------')
            self.log.info('###### Testing ')
            self.log.info('######------------------------------------')
            # make a directory for the evaluation results
            res_dir = os.path.join(working_dir, 'res')
            if self.redo or not os.path.exists(os.path.join(working_dir,
                                                            out_name)):
                # evaluate
                done_validating = self._evaluate(res_dir, best, out_name, mode)
            else:
                self.log.info('##### Testing already done.')
                self.log.info('######------------------------------------')
                done_validating = True
        else:
            self.log.info('##### Not testing')
            self.log.info('######------------------------------------')
            done_validating = True

        return done_validating

    def _get_last_checkpoint(self, config_dir):
        ckpt_files = [os.path.join(config_dir, 'train', f)
                      for f in os.listdir(os.path.join(config_dir, 'train'))
                      if f.endswith('.index')]
        ckpt_files = [x[:-6] for x in ckpt_files]
        ckpt_files = [(x, int(x[x.find('ckpt-')+5:])) for x in ckpt_files]
        ckpt_files = sorted(ckpt_files, key=lambda x: x[1], reverse=True)
        return ckpt_files[0]

    def _get_train_configuration(self, config_name):
        config = self.confs[config_name]
        # config should have learning rate, decay steps, decay factor
        lrs = config['lr']
        batch_size = config['batch_size']
        if 'max_epochs' in config.keys():
            maxs = config['max_epochs']
        else:
            maxs = config['max_steps']
        if type(maxs) != list:
            maxs = [maxs]
        params = []
        for lr in lrs:
            for bs in batch_size:
                for m in maxs:
                    p = {'learning_rate': lr, 'batch_size': bs,
                         'val_epochs': config['val_epochs'],
                         'checkpoint_epochs': config['checkpoint_epochs']}

                    if 'max_epochs' in config.keys():
                        p['max_epochs'] = m
                    else:
                        p['max_steps'] = m

                    params += [p]

        return params

    def _train(self, param, config_dir, mode):
        # write out the traning parameters
        with open(os.path.join(config_dir, 'parameters.json'), 'w') as f:
            json.dump(param, f)
        # setup the network runner
        runner = train.TrainNet(param, self.log)
        success = runner.load_data()
        if success:
            success = success and runner.train(mode)
        return success

    def _get_best(self, working_dir):
        # read in the data from the tensorflow summary files
        config_dirs = [os.path.join(working_dir, f)
                       for f in os.listdir(working_dir) if 'res' not in f]

        min_loss_list = []
        for ind, c in enumerate(config_dirs):
            val_dir = os.path.join(c, 'val')
            if os.path.exists(os.path.join(c, 'DONE')) and not \
                    os.path.exists(os.path.join(c, 'FAILED_RES')):
                log_file = open(os.path.join(val_dir, 'log.csv'), 'r')
                reader = csv.DictReader(log_file)
                min_loss = 1e20
                min_step = 0
                for row in reader:
                    lo = float(row['loss'])
                    st = int(row['step'])
                    # only consider steps for which we have a checkpoint
                    if os.path.exists(os.path.join(c, 'train',
                                                   'model.ckpt-' + str(st) +
                                                   '.index')):
                        if lo < min_loss and lo > 0.:
                            min_loss = lo
                            min_step = st
                min_loss_list += [(c, min_loss, min_step)]

        # sort the list by the minimum validation loss achieved
        min_loss_list = sorted(min_loss_list, key=lambda x: x[1])

        if len(min_loss_list) == 0:
            self.log.error('Did not find any trained model.')
            return False, []

        # return the best model
        models = list(map(lambda x: (x[0], x[2]), min_loss_list))[0]
        self.log.info('Best model:' + str(models[0][0]) +
                      'step: ' + str(models[0][1]))

        if self.remove_bad:
            # delete the files from the others
            for x in min_loss_list:
                if not x[0] in map(lambda x: x[0], models):
                    self.log.debug('delete ' + x[0])
                    shutil.rmtree(x[0])
        return True, models

    def _evaluate(self, res_dir, model, out_name, mode):
        weight_file = {'full': os.path.join(model[0], 'train',
                                            'model.ckpt-' + str(model[1]))}
        step = model[1]
        val_dir = os.path.join(res_dir, os.path.basename(model[0]))
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        self._plot_val_loss(model[0], val_dir)

        args = vars(copy.deepcopy(self.param))
        args['batch_size'] = 1
        args['data_path'] = self.data_dir
        args['sequence_length'] = self.param.sequence_length_test
        if 'num_samples_test' in args.keys():
            args['num_samples'] = self.param.num_samples_test

        if not os.path.exists(os.path.join(val_dir, 'DONE')) or self.redo:
            self.log.info('###### Evaluate ' + os.path.basename(model[0]))
            all_success = True

            # go through the specified test sets
            for test_set in self.param.data_name_test:
                # once for the test split
                out_dir = os.path.join(val_dir, test_set)

                if 'resample_rate_test' in args.keys() and \
                        args['resample_rate_test'] is not None:
                    out_dir += '_re' + str(args['resample_rate_test'])
                if 'initial_covar_test' in args.keys() and \
                        args['initial_covar_test'] is not None:
                    out_dir += '_ic' + \
                        args['initial_covar_test'].replace(' ', '_')

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                self.log.info('######### On ' + test_set)
                if not os.path.exists(os.path.join(out_dir, 'DONE')):
                    args['out_dir'] = out_dir
                    args['data_name'] = test_set
                    args['weight_file'] = weight_file
                    args['step'] = step
                    args['on_testset'] = 0
                    args['on_valset'] = 0

                    runner = test.TestNet(args, self.log)

                    success = runner.load_data()
                    suc, _ = runner.test(mode)
                    success = success and suc
                    all_success = all_success and success
                    if success:
                        open(os.path.join(out_dir, 'DONE'), 'a').close()
                    else:
                        open(os.path.join(model[0], 'FAILED_RES'),
                             'a').close()
                    self.log.info('#########---------------------------------')
                else:
                    self.log.info('######### Already done!')
                    self.log.info('#########---------------------------------')
            if self.param.on_testset:
                # also automatically test on the test-split of the training
                # set
                out_dir = os.path.join(val_dir, self.param.data_name_train)

                if 'resample_rate_test' in args.keys() and \
                        args['resample_rate_test'] is not None:
                    out_dir += '_re' + str(args['resample_rate_test'])
                if 'initial_covar_test' in args.keys() and \
                        args['initial_covar_test'] is not None:
                    out_dir += '_ic' + \
                        args['initial_covar_test'].replace(' ', '_')

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                self.log.info('######### On ' + self.param.data_name_train)
                if not os.path.exists(os.path.join(out_dir, 'DONE')):
                    args['out_dir'] = out_dir
                    args['data_name'] = self.param.data_name_train
                    args['weight_file'] = weight_file
                    args['step'] = step
                    args['on_valset'] = 0

                    runner = test.TestNet(args, self.log)

                    success = runner.load_data()
                    suc, _ = runner.test(mode)
                    success = success and suc
                    all_success = all_success and success
                    if success:
                        open(os.path.join(out_dir, 'DONE'), 'a').close()
                    else:
                        open(os.path.join(model[0], 'FAILED_RES'),
                             'a').close()
                    self.log.info('#########---------------------------------')
                else:
                    self.log.info('######### Already done!')
                    self.log.info('#########---------------------------------')

            if all_success:
                open(os.path.join(val_dir, 'DONE'), 'a').close()
                self.log.info('###### Evaluation done sucessfully')
            else:
                self.log.info('###### Evaluation failed')
            self.log.info('######------------------------------------')
        else:
            self.log.info('###### ' + os.path.basename(model[0]) +
                          ' already done.')
            self.log.info('######------------------------------------')

        if all_success:
            open(os.path.join(res_dir, 'DONE'), 'a').close()
            # copy the best checkpoint files to the top directory
            weight_file = os.path.join(model[0], 'train',
                                       'model.ckpt-' + str(model[1]))
            out_file = os.path.join(os.path.dirname(res_dir), out_name)
            ckpt_dir = os.path.dirname(weight_file)
            ckpt_name = os.path.basename(weight_file)

            for fi in os.listdir(ckpt_dir):
                if fi.startswith(ckpt_name):
                    ending = fi[len(ckpt_name):]
                    shutil.copyfile(os.path.join(ckpt_dir, fi),
                                    out_file + ending)
        return all_success

    def _plot_val_loss(self, model, out_dir):
        val_dir = os.path.join(model, 'val')
        log_file = open(os.path.join(val_dir, 'log.csv'), 'r')

        plot_vals = {}
        plot_vars = {}

        reader = csv.DictReader(log_file)

        for row in reader:
            for key in row.keys():
                if key is None:
                    continue
                if '_std' not in key:
                    if key in plot_vals.keys():
                        plot_vals[key] += [float(row[key])]
                    else:
                        plot_vals[key] = [float(row[key])]
                else:
                    k = key[:key.find('_std')]
                    if key in plot_vals.keys():
                        plot_vars[k] += [float(row[key])]
                    else:
                        plot_vars[k] = [float(row[key])]

        for k, v in plot_vals.items():
            plot_vals[k] = np.array(v)
        for k, v in plot_vars.items():
            plot_vars[k] = np.array(v)

        num_plots = max(1, len(plot_vals.keys()) - 1)
        try:
            fig, ax = plt.subplots(num_plots, figsize=(20, 5 * num_plots))
            # make sure that the total loss comes first
            ax[0].fill_between(plot_vals['step'],
                               plot_vals['loss'] - plot_vars['loss'],
                               plot_vals['loss'] + plot_vars['loss'],
                               color="lightblue")
            ax[0].plot(plot_vals['step'], plot_vals['loss'], color='darkblue')
            ax[0].set_title('validation loss')

            ind = 1
            for k, v in plot_vals.items():
                if k != 'step' and k != 'loss':
                    if k + '_std' in plot_vars.keys():
                        ax[ind].fill_between(plot_vals['step'],
                                             v - plot_vars[k],
                                             v + plot_vars[k],
                                             color="lightblue")
                    ax[ind].plot(plot_vals['step'], v, color='darkblue')
                    ax[ind].set_title(k)
                    ind += 1

            fig.savefig(os.path.join(out_dir, "validation_error"),
                        bbox_inches="tight")
            plt.close()
        except:
            pass
        log_file.close()


def main(argv=None):
    parser = argparse.ArgumentParser('run_experiments')
    parser.add_argument('--name', dest='name', type=str,
                        required=True, help='experiment name')
    parser.add_argument('--data-name-train', dest='data_name_train', type=str,
                        required=True,
                        help='name for training and validation file')
    parser.add_argument('--data-name-test', dest='data_name_test', type=str,
                        default='',
                        nargs='*', help='name for additional testing sets, ' +
                        'this should not include the test split of the ' +
                        'training set (see "on_testset")')
    parser.add_argument('--problem', dest='problem', type=str,
                        default='kitti', choices=['toy', 'kitti', 'pushing'],
                        help='which problem to address')
    parser.add_argument('--filter', dest='filter', type=str,
                        default='ekf', choices=['ekf', 'ukf', 'mcukf', 'pf',
                                                'lstm'],
                        help='which filter class to use')
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        required=True, help='where to find dataset')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        required=True, help='where to store results')

    parser.add_argument('--weight-file', dest='weight_file', type=str,
                        default=None,
                        help='path to a checkpoint file that should be loaded')
    parser.add_argument('--num_checkpoints', dest='num_checkpoints', type=int,
                        default=3, help='how many checkpoints to keep')
    parser.add_argument('--config-name', dest='config_name', type=str,
                        default='default',
                        help='name of the training configuration to be used')
    parser.add_argument('--dev-num', dest='dev_num', type=int, default=0,
                        help='which gpu to use')
    parser.add_argument('--on-testset', dest='on_testset', type=int, default=1,
                        help='if true, we evaluate on the test split of the ' +
                        'training data.')
    parser.add_argument('--continue-train', dest='continue_train', type=int,
                        default=1, choices=[0, 1],
                        help='if true, if a running marker is encountered, ' +
                        'training will be continued from the last checkpoint' +
                        ' instead of being aborted')
    parser.add_argument('--cache', dest='cache', type=int, default=0,
                        choices=[0, 1],
                        help='option to load small datasets into memory')
    parser.add_argument('--debug', dest='debug',
                        type=int, choices=[0, 1], default=0,
                        help='turns debugging on/off ')
    parser.add_argument('--no-test', dest='no_test',
                        type=int, choices=[0, 1], default=0,
                        help='turns off testing')
    parser.add_argument('--remove-bad', dest='remove_bad',
                        type=int, choices=[0, 1], default=0,
                        help='keep models not among the top_k?')
    parser.add_argument('--train-filter', dest='train_filter',
                        type=int, choices=[0, 1], default=1,
                        help='run filter training? (in case we only want to' +
                        'pretrain)')
    parser.add_argument('--redo-results', dest='redo_results',
                        type=int, choices=[0, 1], default=1,
                        help='repeat testing?')
    parser.add_argument('--gpu', dest='gpu', type=int, choices=[0, 1],
                        default=1, help='run on gpu?')

    # filtering parameters
    parser.add_argument('--normalize', dest='normalize', type=str,
                        default='none',
                        help='use batch or layer normalization in the sensor' +
                        'model')
    parser.add_argument('--loss', dest='loss', type=str, default='mixed',
                        help='which type of loss to use for training filters')
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                        default='adam', help='optimization algorithm')
    parser.add_argument('--update-rate', dest='update_rate',
                        type=int, default=1,
                        help='how often a filter gets observations')
    parser.add_argument('--sequence-length-train',
                        dest='sequence_length_train', type=int, default=9,
                        help='length of the training sequence')
    parser.add_argument('--sequence-length-test', dest='sequence_length_test',
                        type=int, default=99,
                        help='length of the testing sequence')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='factor by which to rescale the state')

    parser.add_argument('--learn-q', dest='learn_q', type=int, choices=[0, 1],
                        help='learn the process noise?')
    parser.add_argument('--hetero-q', dest='hetero_q', type=int,
                        choices=[0, 1], default=1,
                        help='learn heteroscedastic process noise?')
    parser.add_argument('--learn-r', dest='learn_r', type=int, choices=[0, 1],
                        help='learn the observation noise?')
    parser.add_argument('--hetero-r', dest='hetero_r', type=int,
                        choices=[0, 1], default=1,
                        help='learn heteroscedastic process noise?')
    parser.add_argument('--diagonal-covar', dest='diagonal_covar', type=int,
                        choices=[0, 1], default=1,
                        help='learn diagonal noise covariance matrices?')

    parser.add_argument('--add-initial-noise', dest='add_initial_noise',
                        type=int, choices=[0, 1], default=0,
                        help='add noise to the initial state estimate?')
    parser.add_argument('--initial-covar', dest='initial_covar', type=str,
                        default=None,
                        help='diagonal elements of the initial covariance')
    parser.add_argument('--q-diag', dest='q_diag', type=str, default=None,
                        help='diagonal elements of the initial q')
    parser.add_argument('--r-diag', dest='r_diag', type=str, default=None,
                        help='diagonal elements of the initial r')

    parser.add_argument('--learn-process', dest='learn_process', type=int,
                        choices=[0, 1], default=1,
                        help='use a learned or an anlytical process model?')
    parser.add_argument('--pretrain-observations',
                        dest='pretrain_observations', type=int, choices=[0, 1],
                        help='pretrain observations?')
    parser.add_argument('--pretrain-process', dest='pretrain_process',
                        type=int, choices=[0, 1],
                        help='pretrain the process model?')
    parser.add_argument('--train-sensor-model', dest='train_sensor_model',
                        type=int, choices=[0, 1], default=1,
                        help='train the sensor model?')
    parser.add_argument('--train-process-model', dest='train_process_model',
                        type=int, choices=[0, 1], default=1,
                        help='train the process model?')
    parser.add_argument('--use-pretrained-covar', dest='use_pretrained_covar',
                        type=int, choices=[0, 1], default=1,
                        help='use the pretrained noise covariances?')
    parser.add_argument('--train-q', dest='train_q',
                        type=int, choices=[0, 1], default=1,
                        help='train the process noise?')
    parser.add_argument('--train-r', dest='train_r',
                        type=int, choices=[0, 1], default=1,
                        help='train the observation noise?')
    # for particle filter and mcukf
    parser.add_argument('--num-samples-train', dest='num_samples_train',
                        type=int, default=100,
                        help='number of samples for pf/mcukf during training')
    parser.add_argument('--num-samples-test', dest='num_samples_test',
                        type=int, default=1000,
                        help='number of samples for pf/mcukf during testing')
    # for ukf
    parser.add_argument('--kappa', dest='kappa', default=0.5,
                        type=float, help='kappa parameter of the ukf')
    parser.add_argument('--scaled', dest='scaled', default=0,
                        choices=[0, 1], type=int,
                        help='use the scaled sigma point variant of the ukf?')
    # for particle filter
    parser.add_argument('--resample-rate', dest='resample_rate',
                        type=int, default=1.,
                        help='resample the particles every x steps')
    parser.add_argument('--resample-rate-test', dest='resample_rate_test',
                        type=int, default=None,
                        help='resample the particles every x steps during ' +
                        'testing')
    parser.add_argument('--alpha', dest='alpha',
                        type=float, default=0.05,
                        help='soft resampling parameter of the pf')
    parser.add_argument('--learned-likelihood', dest='learned_likelihood',
                        type=int, choices=[0, 1], default=0,
                        help='if the likelihood function of the pf is' +
                        ' learned or gaussian')
    parser.add_argument('--mixture-likelihood', dest='mixture_likelihood',
                        type=int, choices=[0, 1], default=1,
                        help='if the likelihood of the ground truth in the ' +
                        'particle filter is based on a guassian mixture ' +
                        'model or a single gaussian approximation (default)')
    parser.add_argument('--mixture-std', dest='mixture_std',
                        type=float, default=5.,
                        help='width of the gmm distributions')
    # for lstms
    parser.add_argument('--num-units', dest='num_units', type=int, default=512,
                        help='number of units in an lstm model')
    parser.add_argument('--lstm-structure', dest='lstm_structure', type=str,
                        default='none', choices=['lstm', 'lstm1'],
                        help='structure of the encoder for the lstm')

    args = parser.parse_args(argv)
    runner = RunExperiment(args)
    runner.run_experiments()


if __name__ == "__main__":
    main()
