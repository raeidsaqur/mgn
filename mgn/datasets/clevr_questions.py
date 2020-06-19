#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: clevr_questions.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-18
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import logging
import os
import os.path as osp
import sys
from itertools import zip_longest

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
from rsmlkit.logging import get_logger, set_default_level

logger = get_logger(__file__)
set_default_level(logging.INFO)

## quick HACK
PROJECT_PATH = '..'
CLEVR_PARSER_PATH = f'{PROJECT_PATH}/vendors/clevr-parser'
print(f"CLEVR_PARSER_PATH={CLEVR_PARSER_PATH}")
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
if CLEVR_PARSER_PATH not in sys.path:
    sys.path.insert(0, CLEVR_PARSER_PATH)

import clevr_parser

from .data import PairData, ClevrData

graph_parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                      has_spatial=True,
                                      has_matching=True).get_backend(identifier='spacy')
embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')


import utils.utils as utils
from utils.mgn_preproc_utils import get_question_file, get_img_scenes

import torch
from torch.utils.data.dataloader import default_collate
# noinspection PyProtectedMember
from torch._six import container_abcs, string_classes, int_classes
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.debug import set_debug_enabled
set_debug_enabled(True)

import traceback


class ModifiedBatch(Batch):
    def __init__(self, **kwargs):
        super(ModifiedBatch, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}
        for key in keys:
            batch[key] = []
        for key in follow_batch:
            batch['{}_batch'.format(key)] = []
        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                # logger.info(f"key={key}")
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size,), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            logger.debug(f"key = {key}")
            if torch.is_tensor(item):
                logger.debug(f"batch[{key}]")
                logger.debug(f"item.shape = {item.shape}")
                elem = data_list[0]     # type(elem) = Data or ClevrData
                dim_ = elem.__cat_dim__(key, item)      # basically, which dim we want to concat
                batch[key] = torch.cat(batch[key], dim=dim_)
                # batch[key] = torch.cat(batch[key],
                #                        dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()


class ClevrQuestionDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, follow_batch=[], **kwargs):
        def collate_wrapper(batch):
            return collate_fn(batch) if collate_fn else collate(batch)

        def collate(batch):
            elem = batch[0]
            elem_type = type(elem)
            is_tensor = isinstance(elem, torch.Tensor)
            is_numpy = elem_type.__module__ == 'numpy' \
                       and elem_type.__name__ != 'str_' \
                        and elem_type.__name__ != 'string_'
            is_regular_elem = (is_tensor or is_numpy) \
                                or isinstance(elem, int_classes) \
                                or isinstance(elem, float)
            if is_regular_elem:
                # Collate question, program, answer, image_idx #
                return default_collate(batch)
            else:
                # Collate graph data #
                if isinstance(elem, PairData):
                    return Batch.from_data_list(batch, follow_batch=['x_s', 'x_t'])
                elif isinstance(elem, Data):
                    try:
                        #return Batch.from_data_list(batch, follow_batch)
                        return ModifiedBatch.from_data_list(batch, follow_batch)
                    except RuntimeError as rte:
                        logger.error(f"{rte}")
                        logger.debug(f"traceback.format_exc(): {traceback.format_exc()}")
                        #logger.debug(f"traceback.print_stack(): {traceback.print_stack()}")
                        return batch
                elif isinstance(elem, string_classes):
                    return batch
                elif isinstance(elem, container_abcs.Mapping):
                    return {key: collate([d[key] for d in batch]) for key in elem}
                elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
                    return type(elem)(*(collate(s) for s in zip(*batch)))
                elif isinstance(elem, container_abcs.Sequence):
                    return [collate(s) for s in zip(*batch)]

            raise TypeError('DataLoader found invalid type: {}'.format(
                type(elem)))

        super(ClevrQuestionDataLoader, self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda batch: collate_wrapper(batch), **kwargs)


class ClevrQuestionDataset(torch.utils.data.Dataset):

    def _init_graph_data(self, graph_data_dir_path=None):
        """
        Used for driving the flow using preprocessed graph data (*.pt) file. Tries to find and return
        a corresponding, complementary {question_h5}.pt file in the graph_data_dir (if path provided),
        o.w. in the same folder as the question_h5_path.
        """
        question_h5_path = self.question_h5_path
        logger.debug(f"Getting graph data from question_h5_path: {question_h5_path}")
        fdir = osp.dirname(question_h5_path)
        fnp = osp.basename(question_h5_path).split('.')[0]
        logger.debug(f"fnp = {fnp}")

        ## Load PairData from {fp}_pairdata.pt ##
        try:
            if graph_data_dir_path is None:
                graph_data_dir_path = fdir      # same as the question_h5 dir
            preprocessed_graph_data_fn = f"{fnp}_directed_pairdata.pt" if self.is_directed_graph \
                                                    else f"{fnp}_pairdata.pt"
            data_fp = f"{graph_data_dir_path}/{preprocessed_graph_data_fn}"
            if not os.path.exists(data_fp):
                logger.info(f"Couln't find preprocessed graph data {preprocessed_graph_data_fn}. "
                             f"Falling back to dynamic processing flow")
                return None
            logger.debug(f"Loading preprocessed pairdata from: {data_fp} ")
            data_file = torch.load(data_fp)
            data_s_list = data_file['data_s_list']
            data_t_list = data_file['data_t_list']
            return tuple([data_s_list, data_t_list])
        except FileNotFoundError as fne:
            logger.error(f"{fnp}_[directed]_pairdata.pt file not found")
            return None

    def __init__(self, opt, split, *args, **kwargs):
        self.max_samples = opt.max_train_samples if split == 'train' \
                                                                else opt.max_val_samples
        self.question_h5_path = opt.clevr_train_question_path if split == 'train' \
                                                                else opt.clevr_val_question_path
        vocab_json = opt.clevr_vocab_path
        self.vocab = utils.load_vocab(vocab_json)
        self.is_directed_graph = opt.is_directed_graph

        #### Init Questions.h5 Data - Invariant same data as in baseline (ques, progs, ans, img_idx) ####
        questions, programs, answers, image_idxs, orig_idxs, question_families = \
            utils.load_data_from_h5(self.question_h5_path)
        self.questions = questions
        self.programs = programs
        self.answers = answers
        self.image_idxs = image_idxs
        self.orig_idxs = orig_idxs
        self.question_families = question_families
        #### Init Graph Data: START ####
        self.graph_data = None
        # Uncomment the below line to activate preprocessed embedding flow
        data_list = self._init_graph_data()     # Load graph_data from preprocessed embeddings
        if data_list:
            logger.info(f"Found preprocessed graph data: self.__init_graph_data(..)")
            data_s_list, data_t_list = data_list
            self.graph_data = list(zip_longest(data_s_list, data_t_list))
        else:
            # Dynamically load graph_data embeddings (skips the preprocessing requirement)
            # N.b Just remove the corresponding *_pairdata.pt file
            logger.debug(f"Preprocessed graph data *_pairdata.pt not found, dynammicall generate g_data")
            logger.info(f"Dynamic Graph Data Gen Flow")
            # raise ValueError if any of the following are None, required for Dynamic Flow
            self.graph_parser = kwargs.get('graph_parser')
            self.embedder = kwargs.get('embedder')
            self.raw_question_path = opt.clevr_train_raw_question_path if split=='train' \
                                                                    else opt.clevr_val_raw_question_path
            self.parsed_img_scene_path = opt.clevr_train_parsed_scene_path if split=='train' \
                                                                    else opt.clevr_val_parsed_scene_path
            logger.debug(f"split: {split}, raw_question_path: {self.raw_question_path}, "
                         f" parsed_img_scene_path: {self.parsed_img_scene_path}")
            try:
                self.raw_questions = get_question_file(self.raw_question_path)
                self.img_scenes = get_img_scenes(self.parsed_img_scene_path)
            except FileNotFoundError as fne:
                logger.error(f"Raw questions.json or parsed image scenes not found: {fne}")
        #### Init Graph Data: END ####

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.questions))
        else:
            return len(self.questions)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('index %d out of range (%d)' % (idx, len(self)))
        question = self.questions[idx]
        image_idx = self.image_idxs[idx]
        program = -1
        answer = -1
        if self.programs is not None:
            program = self.programs[idx]
        if self.answers is not None:
            answer = self.answers[idx]
        orig_idx = self.orig_idxs[idx]
        if self.question_families is not None:
            question_family = self.question_families[idx]

        # ---- Get Graph Data Item -------#
        data_s, data_t = None, None
        if self.graph_data:
            g_data = self.graph_data[idx]
            data_s, data_t = g_data
        else:
            # Dynamically generate graph data item #
            logger.info(f"Dynamic Graph Data Gen for img_idx: {image_idx}")
            def get_question_from_token_seq(q_seq):
                q = []
                for i in q_seq.tolist():
                    q_token = self.vocab['question_idx_to_token'][i]
                    q.append(q_token)
                return ' '.join(q)

            img_scene = list(filter(lambda x: x['image_index'] == image_idx, self.img_scenes))[0]
            s = list(filter(lambda x: x['question_index'] == orig_idx, self.raw_questions))[0]
            assert s['image_index'] == image_idx
            s = s['question']
            Gs, s_doc = graph_parser.parse(s, return_doc=True, is_directed_graph=self.is_directed_graph)
            X_t, ei_t, e_attr_t = embedder.embed_t(image_idx, self.parsed_img_scene_path,
                                                   img_scene=img_scene)
            X_s, ei_s, e_attr_s = embedder.embed_s(s, Gs=Gs, s_doc=s_doc)
            # Using ClevrData allows us a debug extension to Data
            data_s = ClevrData(x=X_s, edge_index=ei_s, edge_attr=e_attr_s)
            data_t = ClevrData(x=X_t, edge_index=ei_t, edge_attr=e_attr_t)

        return question, program, answer, image_idx, (data_s, data_t)
