#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: sample_questions_mgn.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-05-19
# 
# This file is part of MGN
# Distributed under terms of the MIT License

import os, sys, platform
import argparse
import h5py
import numpy as np
np.random.seed(42)
import random
random.seed(42)

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
sys.path.insert(0, ".")

import utils.programs as program_utils
import utils.preprocess as preprocess_utils
from utils.mgn_preproc_utils import get_questions_and_parsed_scenes, \
        save_graphs, save_graph_pairdata, save_graph_edges, save_graph_embeddings, save_graph_docs, save_h5
# from utils.mgn_preproc_utils import *
from datasets import ClevrData

import clevr_parser
import clevr_parser.utils as parser_utils

from rsmlkit.utils.cache import cached_result

import logging
from rsmlkit.logging import get_logger, set_default_level

logger = get_logger(__file__)
set_default_level(logging.INFO)   # logging.INFO

from functools import lru_cache
from typing import *
from collections import defaultdict, OrderedDict

# from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--random_sample', default=0,
                    type=int, help='randomly sample questions')
parser.add_argument('--max_sample', default=2000,
                    type=int, help='size of split, effective only when random is true')
parser.add_argument('--n_questions_per_family', default=5,
                    type=int, help='number of questions per family, effective when random is false')
parser.add_argument('--input_question_h5', default='../data/reason/clevr_h5/clevr_train_questions.h5',
                    type=str, help='path to input question h5 file')
parser.add_argument('--output_dir', default='../data/reason/clevr_mgn_h5',
                    type=str, help='output dir')
parser.add_argument('--output_index', default=None,
                    type=int, help='output repeat id')
# Needed for building Gs
parser.add_argument('--input_questions_json',
                    default='../../data/CLEVR_v1.0/questions/CLEVR_train_questions.json')
# Needed for building Gt
parser.add_argument('--input_parsed_img_scenes_json',
                    default='../../data/CLEVR_v1.0/scenes_parsed/train_scenes_parsed.json',
                    help='The parsed image scenes file for dataset-split')
parser.add_argument('--input_vocab_json', default='../data/reason/clevr_h5/clevr_vocab.json')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--is_debug', action='store_true', default=False, help="Debug flag")
parser.add_argument('--is_directed_graph', default=1, type=int, help='If set, will parse Gs,Gt as nx.MultiDiGraph')

def get_question_fam_to_indices(args) -> Dict:
    """ Returns a dict with 50 q_idx for each q_fam type"""
    h5file = h5py.File(args.input_question_h5, 'r')
    # enc_questions = h5file['questions'][()]
    # image_idxs = h5file['image_idxs'][()]
    q_families = h5file['question_families'][()]
    fam_freq_dist = np.bincount(q_families)
    fam2idx = defaultdict(list)
    for i, fam in enumerate(q_families):
        fam2idx[fam].append(i)
    fam2indices = OrderedDict(sorted(fam2idx.items()))
    #return (fam2indices, fam_freq_dist)
    return fam2indices

def get_output_filename(args) -> str:
    if args.random_sample:
        max_sample = args.max_sample
        if args.output_index is not None:
            filename = 'clevr_train_%d_questions_%02d.h5' % (max_sample, args.output_index)
        else:
            filename = 'clevr_train_%d_questions.h5' % max_sample
        print('| randomly sampling %d questions' % max_sample)
    else:
        max_sample = args.n_questions_per_family * 90
        if args.output_index is not None:
            filename = 'clevr_train_%dquestions_per_family_%02d.h5' \
                       % (args.n_questions_per_family, args.output_index)
        else:
            # Default filename #
            filename = 'clevr_train_%dquestions_per_family.h5' \
                       % args.n_questions_per_family
        print('| drawing questions, %d per family' % args.n_questions_per_family)

    return filename

def main(args):
    """
        Save nx.graph (Gss, Gts,...) and corresponding torch_geometric.data.PairData
        (via clevr_parse embedder api).
    """
    if args.is_debug:
        set_default_level(10)
    is_directed_graph = args.is_directed_graph
    logger.debug(f"Parser flag is_directed_graph = {is_directed_graph}")
    graph_parser = clevr_parser.Parser(backend="spacy", model='en_core_web_sm',
                                       has_spatial=True,
                                       has_matching=True).get_backend(identifier='spacy')
    embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')
    raw_questions, img_scenes = get_questions_and_parsed_scenes(args.input_questions_json,
                                                                args.input_parsed_img_scenes_json )
    logger.info('| importing questions from %s' % args.input_question_h5)
    input_questions = h5py.File(args.input_question_h5, 'r')
    #N = len(input_questions['questions'])

    # Baseline Entities #
    questions, programs, answers, question_families, orig_idxs, img_idxs = [], [], [], [], [], []
    family_count = np.zeros(90)

    # Graphs and Embeddings #
    data_s_list = []                            # List [torch_geometric.data.Data]
    data_t_list = []                            # List [torch_geometric.data.Data]

    filename = get_output_filename(args)
    __all_question_families: np.ndarray = input_questions['question_families'][()]
    __all_enc_questions: np.ndarray = input_questions['questions'][()]
    __all_img_indices: np.ndarray = input_questions['image_idxs'][()]
    logger.debug(f"__all_question_families len {len(__all_question_families)}")

    # Sample N items for each 90 families #
    fam2indices = get_question_fam_to_indices(args)
    M = len(fam2indices.keys())                 # 90
    N = args.n_questions_per_family        # 50
    max_sample = N * M                          # 90 * 50 = 4500
    family_count = np.zeros(M)                  # family_count = Counter()

    # TODO: accumulating values here need to be parallelized, and joined write ex-post
    num_skipped = 0  # Counter for tracking num of samples skipped
    for fam_idx, i_samples in enumerate(fam2indices):
        all_fam_samples = fam2indices[fam_idx]
        logger.debug(f"Question_family {fam_idx} has {len(all_fam_samples)} samples to choose {N} samples")
        N_question_sample_indices = np.random.choice(all_fam_samples, N, replace=False) # N.b seed is fixed
        assert len(N_question_sample_indices) == N
        # TODO: parallelize this iteration loop
        for i in N_question_sample_indices:
            try:
                img_idx = __all_img_indices[i]
                logger.debug(f"\tProcessing Image - {img_idx} from fam_idx {fam_idx}: {i} of {i_samples}")
                img_scene = list(filter(lambda x: x['image_index'] == img_idx, img_scenes))[0]
            except IndexError as ie:
                logger.warning(f"For {img_idx}: {ie}")
                num_skipped += 1
                continue
            try:
                Gt, t_doc = graph_parser.get_doc_from_img_scene(img_scene, is_directed_graph=is_directed_graph)
                X_t, ei_t, e_attr_t = embedder.embed_t(img_idx, args.input_parsed_img_scenes_json)
            except AssertionError as ae:
                logger.warning(f"AssertionError Encountered: {ae}")
                logger.warning(f"[{img_idx}] Excluding images with > 10 objects")
                num_skipped += 1
                continue
            if Gt is None and ("SKIP" in t_doc):
                # If the derendering pipeline failed, then just skip the
                # scene, don't process the labels (and text_scenes) for the image
                logger.warning(f"Got None img_doc at image_index: {img_idx}")
                print(f"Skipping all text_scenes for imgage idx: {img_idx}")
                num_skipped += 1
                continue
            q_idx = input_questions['orig_idxs'][i]
            q_obj = list(filter(lambda x: x['question_index'] == q_idx, raw_questions))[0]
            assert q_obj['image_index'] == img_idx
            s = q_obj['question']
            try:
                Gs, s_doc = graph_parser.parse(s, return_doc=True, is_directed_graph=is_directed_graph)
                X_s, ei_s, e_attr_s = embedder.embed_s(s)
            except ValueError as ve:
                logger.warning(f"ValueError Encountered: {ve}")
                logger.warning(f"Skipping question: {s} for {img_fn}")
                num_skipped += 1
                continue
            if Gs is None and ("SKIP" in s_doc):
                logger.warning("Got None as Gs and 'SKIP' in Gs_embd. (likely plural with CLEVR_OBJS label) ")
                logger.warning(f"SKIPPING processing {s}  at {img_idx}")
                num_skipped += 1
                continue

            data_s = ClevrData(x=X_s, edge_index=ei_s, edge_attr=e_attr_s)
            data_t = ClevrData(x=X_t, edge_index=ei_t, edge_attr=e_attr_t)
            data_s_list.append(data_s)
            data_t_list.append(data_t)

            family_count[fam_idx] += 1
            questions.append(input_questions['questions'][i])
            programs.append(input_questions['programs'][i])
            answers.append(input_questions['answers'][i])
            question_families.append(input_questions['question_families'][i])
            orig_idxs.append(input_questions['orig_idxs'][i])
            img_idxs.append(img_idx)

            logger.info(f"\nCount = {family_count.sum()}\n")

        if family_count.sum() >= max_sample:
            break

    logger.debug(f"Total samples skipped (due to errors/exceptions) = {num_skipped}")
    # ---------------------------------------------------------------------------#
    ## SAVE .H5
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_file = os.path.join(args.output_dir, filename)
    out_dir = args.output_dir
    out_f_prefix = filename.split('.')[0]
    out_fpp = f"{out_dir}/{out_f_prefix}"
    logger.debug(f"out_fpp = {out_fpp}")

    print('sampled question family distribution')
    print(family_count)
    print('| saving output file to %s' % output_file)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('questions', data=np.asarray(questions, dtype=np.int32))
        f.create_dataset('programs', data=np.asarray(programs, dtype=np.int32))
        f.create_dataset('answers', data=np.asarray(answers))
        f.create_dataset('image_idxs', data=np.asarray(img_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        f.create_dataset('question_families', data=np.asarray(question_families))

    ## ------------  SAVE GRAPH DATA ------------ ##
    save_graph_pairdata(out_fpp, data_s_list, data_t_list, is_directed_graph=is_directed_graph)
    logger.info(f"Saved Graph Data in: {out_fpp}_*.[h5|.gpickle|.npz|.pt] ")
    print('| done')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
