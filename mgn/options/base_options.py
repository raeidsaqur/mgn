#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run_train.py
# Author : FirstName LastName
# Email  : anon@cs.anon.edu
# Date   : 02/22/2020
#
# This file is part of MGN
# Distributed under terms of the MIT license.
# https://github.com/anon/mgn

import os
import argparse
import numpy as np
import torch

import logging, time, platform
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

class BaseOptions():
    """Base option class"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--run_dir', default='_scratch/test_run', type=str, help='experiment directory')
        self.parser.add_argument('--dataset', default='clevr', type=str, help='select dataset, options: clevr, clevr-humans')
        # Dataloader
        self.parser.add_argument('--shuffle', default=1, type=int, help='shuffle dataset')
        # use: num_worker = 4 * num_GPU
        self.parser.add_argument('--num_workers', default=1, type=int, help='number of workers for loading data')
        # Run
        self.parser.add_argument('--manual_seed', default=None, type=int, help='manual seed')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        self.parser.add_argument('--visualize', default=0, type=int, help='visualize experiment')
        # Dataset catalog
        # - CLEVR
        self.parser.add_argument('--clevr_train_scene_path', default='../data/raw/CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
                                 type=str, help='path to clevr train scenes')
        self.parser.add_argument('--clevr_val_scene_path', default='../data/raw/CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
                                 type=str, help='path to clevr val scenes')
        self.parser.add_argument('--clevr_train_question_path', default='../data/reason/clevr_h5/clevr_train_questions.h5',
                                 type=str, help='path to clevr train questions')
        # --clevr_val_question_path the path to the questions
        # self.parser.add_argument('--clevr_val_question_path', default='../data/reason/clevr_h5/clevr_val_questions.h5',
        #                          type=str, help='path to clevr val questions')
        #  : updating clevr_val_questions.h5 to mgn prepreocessed folder
        self.parser.add_argument('--clevr_val_question_path', default='../data/reason/clevr_mgn_h5/clevr_val_questions.h5',
                                 type=str, help='path to clevr val questions')
        self.parser.add_argument('--clevr_test_question_path',
                                 default='../data/reason/clevr_mgn_h5/clevr_test_questions.h5',
                                 type=str, help='path to clevr val questions')
        self.parser.add_argument('--clevr_vocab_path', default='../data/reason/clevr_h5/clevr_vocab.json',
                                 type=str, help='path to clevr vocab')
        #   Added
        self.parser.add_argument('--visualize_training_wandb', default=0, type=int,
                                 help='visualize training with wandb')
        self.parser.add_argument('--tags', default="", help="Tags for this run.", type=str)
        self.parser.add_argument('--is_directed_graph', default=1, type=int,
                                    help='If set, will treat Gs,Gt as nx.MultiDiGraph')
        self.parser.add_argument('--wandb_proj_name', default='mgn_vector', type=str,
                                    help='WandB proj to log results')
        #Parsed image scenes: relative go ${PROJ_DIR}/mgn/reason dir
        self.parser.add_argument('--clevr_train_parsed_scene_path',
                                 default='../../data/CLEVR_v1.0/scenes_parsed/train_scenes_parsed.json',
                                 type=str, help='path to clevr train parsed image scenes')
        self.parser.add_argument('--clevr_val_parsed_scene_path',
                                 default='../../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json',
                                 type=str, help='path to clevr val parsed image scenes')
        # Raw Questions Json path: relative to ${PROJ_DIR}/mgn/reason dir
        self.parser.add_argument('--clevr_train_raw_question_path',
                                 default='../../data/CLEVR_v1.0/questions/CLEVR_train_questions.json',
                                 type=str, help='path to clevr train parsed image scenes')
        self.parser.add_argument('--clevr_val_raw_question_path',
                                 default='../../data/CLEVR_v1.0/questions/CLEVR_val_questions.json',
                                 type=str, help='path to clevr val parsed image scenes')

    def get_run_identifier(self, args):
        """
        Allows for a unique run identifier for runs and corresponding checkpoints with the same time-stamp
        :param args: Run arguments
        :return: A identifier suffix. Usage e.g. {project_name}|{checkpoint}-identifier
        """
        ts = time.strftime('%Y%m%d-%H%M%S')
        self.is_test_options = type(self).__name__ == 'TestOptions'
        self.is_train_options = self.__class__.__name__ == 'TrainOptions'

        tag = args.tags
        if len(tag) > 0:
            tag = f"[{tag}]-"
        val_qp = args.clevr_val_question_path
        val_qp = val_qp.split('/')[-1].split('.')[0]
        train_qp = args.clevr_train_question_path
        train_qp = train_qp.split('/')[-1].split('.')[0]

        if self.is_train_options:
            identifier = f"{tag}reason.run_train-pretrain-{train_qp}-{args.num_iters}num_iters.{args.max_train_samples}tr_samples" \
                         f".{args.max_val_samples}val_samples.{args.batch_size}bsz-{platform.node()}-{ts}"

            if args.reinforce:
                identifier = f"{tag}reason.run_train-reinforce-{args.dataset}-{args.num_iters}num_iters.{args.max_train_samples}tr_samples" \
                             f".{args.max_val_samples}val_samples.{args.batch_size}bsz.{args.learning_rate}lr-{ts}"

        elif self.is_test_options:
            guess = "-GUESS-" if args.exec_guess_ans else ""
            # identifier = f"{tag}[{val_qp}]{guess}-reason-run_test-{platform.node()}" \
            #              f".{args.batch_size}bsz-{ts}"

            identifier = f"{tag}[ {val_qp} ]{guess}-[{args.load_checkpoint_path}]-reason.run_test" \
                         f".{args.batch_size}bsz-{ts}"

        else:
            identifier = f"reason-run-{ts}"
        logging.debug("-" * 100)
        logging.debug(f"Run identifier: {identifier}")
        logging.debug("-" * 100)

        #Update self.opt
        self.opt.run_timestamp = ts
        self.opt.run_identifier = identifier

        return identifier, ts

    def get_save_result_path(self, args):
        """
        get the save result path for test runs
        N.b. 99% acc when train question path and val question path are the same
        If result path specified, then that overrides this logic.

        :param args:
        :return:
        """
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = "../data/reason/results"
        dataset = args.dataset
        val_qp = args.clevr_val_question_path
        val_qp = val_qp.split('/')[-1].split('.')[0]
        logging.debug(f"val_qp: {val_qp}")
        ts = self.run_timestamp
        srp = f"{run_dir}/result_pretrained_{dataset}_{val_qp}_{ts}.json"

        logging.debug("-" * 100)
        logging.debug(f"save_result_path: {srp}")
        logging.debug("-" * 100)

        return srp

    def parse(self):
        # Instantiate option
        self.opt = self.parser.parse_args()
        self.run_identifier, self.run_timestamp = self.get_run_identifier(self.opt)

        val_qp = self.opt.clevr_val_question_path
        val_qfn = val_qp.split('/')[-1]
        self.opt.clevr_val_question_filename = val_qfn

        train_qp = self.opt.clevr_train_question_path
        train_qfn = train_qp.split('/')[-1]
        self.opt.clevr_train_question_filename = train_qfn

        if not self.is_train and self.opt.save_result_path is None:
            self.opt.save_result_path = self.get_save_result_path(self.opt)

        # Parse gpu id list
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []

        # Set manual seed
        if self.opt.manual_seed is not None:
            torch.manual_seed(self.opt.manual_seed)
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed(self.opt.manual_seed)

        #if not self.opt.visualize_training_wandb:
            # Print and save options (if wandb visualization is off)
            # if wandb is on, options and results are recorded in test_opt_ts.json file on cloud
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        if not os.path.isdir(self.opt.run_dir):
            os.makedirs(self.opt.run_dir)
        if self.is_train:
            file_path = os.path.join(self.opt.run_dir, f'train_opt_{self.run_timestamp}.txt')
        else:
            file_path = os.path.join(self.opt.run_dir, f'test_opt_{self.run_timestamp}.txt')
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in args.items():
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt