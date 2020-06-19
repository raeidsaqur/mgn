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


import os, sys
_dir = os.getcwd()
if _dir not in sys.path:
    sys.path.insert(0, _dir)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger('imported_module').setLevel(logging.WARNING)

from options.train_options import TrainOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
from trainer import Trainer

opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt)
executor = get_executor(opt)
trainer = Trainer(opt, train_loader, val_loader, model, executor)

trainer.train()
