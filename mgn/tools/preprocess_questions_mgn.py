#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : preprocess_questions_mgn.py
# Author : FirstName LastName
# Email  : anon@cs.anon.edu
# Date   : 02/22/2020
#
# This file is part of MGN Parser.
# Distributed under terms of the MIT license.

# preprocess clevr questions
# code adopted from https://github.com/facebookresearch/clevr-iep/blob/master/scripts/preprocess_questions.py

import os, sys
import os.path as osp
import argparse
import json
from typing import *

import random

random.seed(42)

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
sys.path.insert(0, ".")

import utils.preprocess as preprocess_utils
import utils.programs as program_utils
import utils.utils as utils
from utils.mgn_preproc_utils import get_questions_and_parsed_scenes, \
    save_graph_pairdata, save_h5
from datasets import ClevrData

import clevr_parser
import clevr_parser.utils as parser_utils

from rsmlkit.logging import get_logger, set_default_level

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--input_parsed_img_scenes_json',
                    default='../../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json',
                    help='The parsed image scenes file for dataset-split')
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)

parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='')
parser.add_argument('--is_debug', action='store_true', default=False, help="IS_DEBUG flag")
parser.add_argument('--max_sample', default=0,
                    type=int, help='max ques. samples to collect from clevr_train. all 669K if 0')
parser.add_argument('--checkpoint_every', default=10000,
                    type=int, help='checkpoint after processing K samples 669K if 0')
parser.add_argument('--is_directed_graph', default=1, type=int, help='If set, will parse Gs,Gt as nx.MultiDiGraph')
parser.add_argument('--parser_lm', default='en_core_web_sm', type=str, help='Model to init spacy LM with ')
parser.add_argument('--model', type=str, default='', help='Name of the embedder model type. '
                                                          'For e.g. mgn.e2e.gt-pos')


def program_to_str(program, mode):
    if mode == 'chain':
        if not program_utils.is_chain(program):
            return None
        return program_utils.list_to_str(program)
    elif mode == 'prefix':
        program_prefix = program_utils.list_to_prefix(program)
        return program_utils.list_to_str(program_prefix)
    elif mode == 'postfix':
        program_postfix = program_utils.list_to_postfix(program)
        return program_utils.list_to_str(program_postfix)
    return None


def _process_vocab(args, questions) -> Dict:
    """If input_vocab_json is provided, then use (or expand) it, o.w. build vocab from train files"""
    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab == 1:
        logger.info('Building vocab')
        if 'answer' in questions[0]:
            answer_token_to_idx = preprocess_utils.build_vocab(
                (q['answer'] for q in questions)
            )
        question_token_to_idx = preprocess_utils.build_vocab(
            (q['question'] for q in questions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )
        all_program_strs = []
        for q in questions:
            if 'program' not in q: continue
            program_str = program_to_str(q['program'], args.mode)
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = preprocess_utils.build_vocab(all_program_strs)
        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
        }

    if args.input_vocab_json != '':
        logger.info('Loading vocab')
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json) as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab['question_token_to_idx']:
                if word not in vocab['question_token_to_idx']:
                    logger.info('Found new word %s' % word)
                    idx = len(vocab['question_token_to_idx'])
                    vocab['question_token_to_idx'][word] = idx
                    num_new_words += 1
            logger.info('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        utils.mkdirs(os.path.dirname(args.output_vocab_json))
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    return vocab

def _get_out_dir_and_file_prefix(args):
    output_h5_file = args.output_h5_file
    out_dir = osp.dirname(output_h5_file)
    out_f_prefix = osp.basename(output_h5_file).split('.')[0]  # clevr_train_questions

    return out_dir, out_f_prefix

def main(args):
    """
    Save nx.graph (Gss, Gts,...) and corresponding torch_geometric.data.PairData
    (via clevr_parse embedder api).
    """
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        logger.info('Must give one of --input_vocab_json or --output_vocab_json')
        return
    graph_parser = clevr_parser.Parser(backend='spacy', model=args.parser_lm,
                                       has_spatial=True,
                                       has_matching=True).get_backend(identifier='spacy')
    embedder = clevr_parser.Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')
    is_directed_graph = args.is_directed_graph  # Parse graphs as nx.MultiDiGraph

    out_dir, out_f_prefix = _get_out_dir_and_file_prefix(args)
    checkpoint_dir = f"{out_dir}/checkpoints"
    utils.mkdirs(checkpoint_dir)

    questions, img_scenes = get_questions_and_parsed_scenes(args.input_questions_json,
                                                            args.input_parsed_img_scenes_json)
    if args.is_debug:
        set_default_level(10)
        questions = questions[:128]  # default BSZ is 64 ensuring enought for batch iter
        logger.debug(f"In DEBUG mode, sampling {len(questions)} questions only..")
    # Process Vocab #
    vocab = _process_vocab(args, questions)

    # Encode all questions and programs
    logger.info('Encoding data')
    questions_encoded, programs_encoded, answers, image_idxs = [], [], [], []
    question_families = []
    orig_idxs = []

    # Graphs and Embeddings #
    data_s_list = []  # List [torch_geometric.data.Data]
    data_t_list = []  # List [torch_geometric.data.Data]
    num_samples = 0  # Counter for keeping track of processed samples
    num_skipped = 0  # Counter for tracking num of samples skipped
    for orig_idx, q in enumerate(questions):
        # First See if Gss, Gts are possible to extract.
        # If not (for e.g., some edges cases like plurality, skip data sample
        img_idx = q['image_index']
        img_fn = q['image_filename']
        logger.debug(f"\tProcessing Image - {img_idx}: {img_fn} ...")
        # q_idx = q['question_index']
        # q_fam_idx = q['question_family_index']
        ## 1: Ensure both Gs,Gt is parseable for this question sample, o.w. skip
        img_scene = list(filter(lambda x: x['image_index'] == img_idx, img_scenes))[0]
        try:
            Gt, t_doc = graph_parser.get_doc_from_img_scene(img_scene, is_directed_graph=is_directed_graph)
            X_t, ei_t, e_attr_t = embedder.embed_t(img_idx, args.input_parsed_img_scenes_json)
        except AssertionError as ae:
            logger.warning(f"AssertionError Encountered: {ae}")
            logger.warning(f"[{img_fn}] Excluding images with > 10 objects")
            num_skipped += 1
            continue
        if Gt is None and ("SKIP" in t_doc):
            # If the derendering pipeline failed, then just skip the
            # scene, don't process the labels (and text_scenes) for the image
            print(f"Got None img_doc at image_index: {img_idx}")
            print(f"Skipping all text_scenes for imgage idx: {img_idx}")
            num_skipped += 1
            continue
        s = q['question']
        orig_idx = q['question_index']
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
            logger.warning(f"SKIPPING processing {s} for {img_fn} and at {img_idx}")
            num_skipped += 1
            continue

        # Using ClevrData allows us a debug extension to Data
        data_s = ClevrData(x=X_s, edge_index=ei_s, edge_attr=e_attr_s)
        data_t = ClevrData(x=X_t, edge_index=ei_t, edge_attr=e_attr_t)
        data_s_list.append(data_s)
        data_t_list.append(data_t)

        question = q['question']
        orig_idxs.append(orig_idx)
        image_idxs.append(img_idx)
        if 'question_family_index' in q:
            question_families.append(q['question_family_index'])
        question_tokens = preprocess_utils.tokenize(question,
                                                    punct_to_keep=[';', ','],
                                                    punct_to_remove=['?', '.'])
        question_encoded = preprocess_utils.encode(question_tokens,
                                                   vocab['question_token_to_idx'],
                                                   allow_unk=args.encode_unk == 1)
        questions_encoded.append(question_encoded)

        has_prog_seq = 'program' in q
        if has_prog_seq:
            program = q['program']
            program_str = program_to_str(program, args.mode)
            program_tokens = preprocess_utils.tokenize(program_str)
            program_encoded = preprocess_utils.encode(program_tokens, vocab['program_token_to_idx'])
            programs_encoded.append(program_encoded)

        if 'answer' in q:
            ans = q['answer']
            answers.append(vocab['answer_token_to_idx'][ans])

        num_samples += 1
        logger.info("-" * 50)
        logger.info(f"Samples processed count = {num_samples}")
        if has_prog_seq:
            logger.info(f"\n[{orig_idx}]: question: {question} \n"
                        f"\tprog_str: {program_str} \n"
                        f"\tanswer: {ans}")
        logger.info("-" * 50)

        # ---- CHECKPOINT ---- #
        if num_samples % args.checkpoint_every == 0:
            logger.info(f"Checkpointing at {num_samples}")
            checkpoint_fn_prefix = f"{out_f_prefix}_{num_samples}"
            _out_dir = f"{checkpoint_dir}/{out_f_prefix}_{num_samples}"
            utils.mkdirs(_out_dir)
            out_fpp = f"{_out_dir}/{checkpoint_fn_prefix}"
            # ------------ Checkpoint .H5 ------------#
            logger.info(f"CHECKPOINT: Saving checkpoint files at directory: {out_fpp}")
            save_h5(f"{out_fpp}.h5", vocab,
                    questions_encoded, image_idxs, orig_idxs, programs_encoded, question_families, answers)
            # ------------ Checkpoint GRAPH DATA ------------#
            save_graph_pairdata(out_fpp, data_s_list, data_t_list, is_directed_graph=is_directed_graph)
            logger.info(f"-------------- CHECKPOINT: COMPLETED --------")

        if (args.max_sample > 0) and (num_samples >= args.max_sample):
            logger.info(f"len(questions_encoded = {len(questions_encoded)}")
            logger.info("args.max_sample reached: Completing ... ")
            break

    logger.debug(f"Total samples skipped = {num_skipped}")
    logger.debug(f"Total samples processed = {num_samples}")
    out_fpp = f"{out_dir}/{out_f_prefix}"
    ## SAVE .H5: Baseline {dataset}_h5.h5 file (q,p,ans,img_idx) as usual
    logger.info(f"Saving baseline (processed) data in: {out_fpp}.h5")
    save_h5(f"{out_fpp}.h5", vocab,
            questions_encoded, image_idxs, orig_idxs, programs_encoded, question_families, answers)
    ## ------------  SAVE GRAPH DATA ------------ ##
    ## N.b. Ensure the len of theses lists are all equals
    save_graph_pairdata(out_fpp, data_s_list, data_t_list, is_directed_graph=is_directed_graph)
    logger.info(f"Saved Graph Data in: {out_fpp}_*.[h5|.gpickle|.npz|.pt] ")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

