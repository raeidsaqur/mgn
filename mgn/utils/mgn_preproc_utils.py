#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import json
import os

import h5py
import networkx as nx
import numpy as np
import torch
from rsmlkit import get_logger, set_default_level

logger = get_logger(__file__)
set_default_level(10)

from rsmlkit.utils.decorators import timeit
# from rsmlkit.utils.cache import cached_result
from typing import *

__all__ = [ 'get_questions_and_parsed_scenes',
            'get_question_file',
            'get_img_scenes',
            'mkdirs',
            'save_graphs' ,
            'save_graph_pairdata',
            'save_graph_edges',
            'save_graph_embeddings',
            'save_graph_docs',
            'save_h5']

@functools.lru_cache()
def get_questions_and_parsed_scenes(q_fp, img_scenes_fp):
    """
    Gets the questions from args.input_questions_json
    and parsed image scenes from args.input_img_scenes_json
    :return:
    """
    text_type = "questions"
    logger.info(f"q_fp = {q_fp}")
    logger.info(f"img_scenes_fp = {img_scenes_fp}")
    is_questions_exist = os.path.exists(q_fp)
    is_scenes_parsed_exist = os.path.exists(img_scenes_fp)
    if not (is_questions_exist and is_scenes_parsed_exist):
        logger.debug(f"is_questions_exist = {is_questions_exist}")
        logger.debug(f"is_scenes_parsed_exist: {is_scenes_parsed_exist}")
        emsg = f"Required text scenes and/or image scenes missing for {q_fp} or {img_scenes_fp}"
        logger.error(emsg)
        raise FileNotFoundError(emsg)

    # Load text (captions | questions) and img scenes
    questions = None; scene_objs = None
    questions = get_question_file(q_fp)
    scene_objs = get_img_scenes(img_scenes_fp)
    assert questions is not None
    assert scene_objs is not None

    return questions, scene_objs

@functools.lru_cache()
def get_question_file(q_fp) -> List:
    with open(q_fp) as f:
        logger.info(f'Loading questions from: {q_fp}...')
        questions = json.load(f)[f"questions"]
    return questions

@functools.lru_cache()
def get_img_scenes(img_scenes_fp):
    with open(img_scenes_fp) as f:
        logger.info(f'Loading parsed img scenes from: {img_scenes_fp}...')
        scene_objs = json.load(f)["scenes"]
    return scene_objs

def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

def save_graphs(fp, Gss, Gts, Gus=None, Gus_matched=None) -> None:
    logger.info(f'| saving nx.Graphs  outputs to {fp}_*.gpickle')
    success = True
    try:
        nx.write_gpickle(Gss, f"{fp}_Gss.gpickle")
    except NotImplementedError as e:
        success = False
        logger.error(f"Error pickling Gss: {e}")

    try:
        nx.write_gpickle(Gts, f"{fp}_Gts.gpickle")
    except NotImplementedError as nie:
        success = False
        logger.error(f"Error pickling Gts: {nie}")
    try:
        if Gus:
            nx.write_gpickle(Gus, f"{fp}_Gus.gpickle")
        if Gus_matched:
            nx.write_gpickle(Gus_matched, f"{fp}_Gus_matched.gpickle")
    except NotImplementedError as nie:
        success = False
        logger.error(f"Error pickling Gus: {nie}")
    if success:
        logger.info(f"SUCCESS: All graphs (Gss, Gts, Gus, Gus_matched) were successfully saved as {fp}_*.gpickle")
    else:
        logger.info(f"ERROR: All graphs (Gss, Gts, Gus, Gus_matched) failed to be saved as {fp}_*.gpickle")

def save_graph_docs(fp, s_docs:List, t_docs:List) -> None:
    logger.info(f"| saving graph spacy ::doc:: outputs to {fp}_*.pickle")
    import pickle
    assert len(s_docs) == len(t_docs)
    docs = s_docs + t_docs
    try:
        with open(f"{fp}_docs.pickle", 'wb') as f:
            pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    except NotImplementedError as nie:
        logger.error(f"Error pickling docs: {nie}")

def save_graph_embeddings(fp, Gs_embds, Gt_embds, Gts_pos=None) -> None:
    ## Save Graph Embeddings (using np.savez as .npz files#
    # Size of Gs, Gt embeddings -> ndarray (# of questions, EMBD_DIM) -> (n, 96)
    logger.info(f'| saving G_embds outputs to {fp}_G_embds.npz')
    if Gts_pos:
        np.savez(f"{fp}_G_embds.npz", Gs_embds=Gs_embds, Gt_embds=Gt_embds, Gts_pos=Gts_pos)
    else:
        np.savez(f"{fp}_G_embds.npz", Gs_embds=Gs_embds, Gt_embds=Gt_embds)

def save_graph_pairdata(fp, data_s_list, data_t_list, is_directed_graph=False) -> None:
    logger.info(f"save_graph_pairdata(..)")
    data_fp = f"{fp}_pairdata.pt"
    if is_directed_graph:
        data_fp = f"{fp}_directed_pairdata.pt"
    logger.info(f"Saving graph pairdata tensors as: {data_fp}")
    cache_kwargs = {'data_s_list': data_s_list,
                    'data_t_list': data_t_list}
    torch.save(cache_kwargs, data_fp)

def save_graph_pairdata2(fp, pairdata_list, data_s_list, data_t_list) -> None:
    logger.info(f"save_graph_pairdata(..)")
    data_fp = f"{fp}_pairdata.pt"
    logger.info(f"Saving graph pairdata tensors as: {data_fp}")
    cache_kwargs = {'pairdata_list': pairdata_list,
                    'data_s_list': data_s_list,
                    'data_t_list': data_t_list}
    torch.save(cache_kwargs, data_fp)

def save_graph_edges(fp, Ess, Ets, Eus_matched) -> None:
    logger.warn(f"save_graph_edges is a redundant call. Use get_edges(G) instead")
    edges_fp = f"{fp}_edges.pt"
    logger.info(f"Saving graph edges tensors as: {edges_fp}")
    edges_kwargs = {'Ess': Ess, 'Ets': Ets, 'Eus_matched': Eus_matched}
    torch.save(edges_kwargs, edges_fp)

def save_h5(fp, vocab, questions_encoded, image_idxs, orig_idxs, programs_encoded,
            question_families, answers, *args, **kwargs) -> None:

    """ fp has .h5 affixed unlike the other helpers
    # h5_kwargs = { 'questions': questions_encoded,
    #              'image_idxs': image_idxs,
    #              'orig_idxs': orig_idxs,
    #              'programs': programs_encoded,
    #              'question_families': question_families,
    #              'answers': answers}
    # #save_h5(args.output_h5_file, vocab, **h5_kwargs)
    """
    ## Save Baseline {dataset}_h5.h5 file (q,p,ans,img_idx) as usual
    logger.info(f"Saving baseline (processed) data in: {fp}")
    mkdirs(os.path.dirname(fp))
    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)

    with h5py.File(fp, 'w') as f:
        f.create_dataset('questions', data=questions_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        if len(programs_encoded) > 0:
            f.create_dataset('programs', data=programs_encoded)
        if len(question_families) > 0:
            f.create_dataset('question_families', data=np.asarray(question_families))
        if len(answers) > 0:
            f.create_dataset('answers', data=np.asarray(answers))