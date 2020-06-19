#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: run_train_e2e.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-18
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import os
import sys

_dir = os.path.split(os.getcwd())[0]
if _dir not in sys.path:
    sys.path.insert(0, _dir)
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger('imported_module').setLevel(logging.WARNING)

from options.train_options import TrainOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
from trainer import Trainer

import numpy as np
import random
np.random.seed(42)
random.seed(42)
import clevr_parser

graph_parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                       has_spatial=True,
                                       has_matching=True).get_backend(identifier='spacy')
embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')

opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train', graph_parser=graph_parser, embedder=embedder)
val_loader = get_dataloader(opt, 'val', graph_parser=graph_parser, embedder=embedder)

model = Seq2seqParser(opt)
executor = get_executor(opt)
trainer = Trainer(opt, train_loader, val_loader, model, executor)

trainer.train()