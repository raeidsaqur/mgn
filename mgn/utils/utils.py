import json
import os
import os.path as osp
from functools import lru_cache, wraps
from typing import *

import h5py
import networkx as nx
import numpy as np
import torch
from rsmlkit.collections.frozendict import frozendict
from rsmlkit.logging import get_logger

logger = get_logger('__file__')

def freezeargs(func):
    """
    Transform mutable dictionnary
    Into immutable Useful to be compatible with cache
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped


def mkdirs(paths):
    try:
        if isinstance(paths, list):
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
        else:
            if not os.path.exists(paths):
                os.makedirs(paths)
    except FileExistsError as fe:
        logger.error(fe)

@freezeargs
@lru_cache(maxsize=128)
def invert_dict(d):
  return {v: k for k, v in d.items()}


@lru_cache(maxsize=16)
def load_vocab(path):
    with open(path) as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def load_scenes(scenes_json):
    scenes = []
    if scenes_json is None:
        print("No scenes_json file specified, returning empty scenes")
        return scenes
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            item = {'id': '%d-%d' % (s['image_index'], i)}
            if '3d_coords' in o:
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            table.append(item)
        scenes.append(table)
    return scenes
    

def load_embedding(path):
    return torch.Tensor(np.load(path))

def load_data_from_h5(question_h5_path):
    question_h5 = h5py.File(question_h5_path, 'r')
    questions = torch.LongTensor(np.asarray(question_h5['questions'], dtype=np.int64))
    image_idxs = np.asarray(question_h5['image_idxs'], dtype=np.int64)
    orig_idxs = np.asarray(question_h5['orig_idxs'], dtype=np.int64)
    programs, answers = None, None
    if 'programs' in question_h5:
        programs = torch.LongTensor(np.asarray(question_h5['programs'], dtype=np.int64))
    if 'answers' in question_h5:
        answers = np.asarray(question_h5['answers'], dtype=np.int64)
    if 'question_families' in question_h5:
        question_families = np.asarray(question_h5['question_families'], dtype=np.int64)

    return questions, programs, answers, image_idxs, orig_idxs, question_families

def load_mgn_graph_data(question_h5_path):
    print(f"Getting graph data from question_h5_path: {question_h5_path}")
    fdir = osp.dirname(question_h5_path)
    fnp = osp.basename(question_h5_path).split('.')[0]
    print(f"fnp = {fnp}")
    load_Gs_fn = lambda x: nx.read_gpickle(f"{fdir}/{fnp}_{x}.gpickle")
    graphs = ['Gss', 'Gts', 'Gus', 'Gus_matched']
    loaded_Gs = []
    for gn in graphs:
        loaded_Gs.append(load_Gs_fn(gn))
    print(f"Number of Graphs loaded = {len(loaded_Gs)}")
    assert len(loaded_Gs[0]) == len(loaded_Gs[1])
    assert len(loaded_Gs[1]) == len(loaded_Gs[3])

    # Load G embeddings ( {fnp}_G_embds.npz )
    embds_fp = f"{fdir}/{fnp}_G_embds.npz"
    print(f"Loading Graph embeddings from: {embds_fp} ")
    G_embds = np.load(embds_fp, allow_pickle=True)

    # load_G_embds_fn = lambda x: G_embds[x]
    def load_G_embds_fn(G_embds, x):
        # Convert ndArray to Tensor here
        return G_embds[x]

    loaded_embds = []
    for embd in ['Gs_embds', 'Gt_embds', 'Gts_pos']:
        loaded_embds.append(load_G_embds_fn(G_embds, embd))

    # Load G edges ( {fnp}_edges.pt )
    edges_fp = f"{fdir}/{fnp}_edges.pt"
    print(f"Loading Graph edges from: {edges_fp} ")
    G_edges = torch.load(edges_fp)
    load_G_edges_fn = lambda x: G_edges[x]
    loaded_edges = []
    for e in ['Ess', 'Ets', 'Eus_matched']:
        loaded_edges.append(load_G_edges_fn(e))

    return tuple(loaded_Gs), tuple(loaded_embds), tuple(loaded_edges)

# mgn.reason.run_test
def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type

def get_prog_from_seq(pseq: List, vocab:dict) -> str:
    i2t = vocab.get('program_idx_to_token')
    t2i = vocab.get('program_token_to_idx')
    if not i2t:
        logger.error("Invalid vocab: no program_idx_to_token")
        return "N/A"
    if not t2i:
        logger.error("Invalid vocab: no program_idx_to_token")
        return "N/A"
    _start_idx = t2i.get('<START>')
    if not _start_idx:
        _start_idx = 1
    _end_idx = t2i.get('<END>')
    if not _end_idx:
        _end_idx = 2
    pstr = []
    for pi in pseq:
        if pi == _end_idx: break;
        if pi == _start_idx: continue;
        pstr.append(i2t.get(pi))
    pstr = "->".join(pstr)
    logger.debug(f"program seq: {pstr}")
    return pstr

## Analysis Helper Functions
def get_qtype_distribution_from_fp(fp, template=None):
    print(f"question type distribution in {fp}:")
    try:
        with open(fp) as f:
            all_questions = json.load(f)['questions']
            get_qtype_distribution_from_questions(all_questions)
    except FileNotFoundError as fne:
        print(fne)

def get_qtype_distribution_from_h5(fp, vocab_path):
    print(f"question type distribution in {fp}:")
    x, y, ans, idx, *_ = load_data_from_h5(fp)
    y = y.numpy()
    vocab = load_vocab(vocab_path)
    all_q_types = []
    c = Counter()
    for pg in y:
        pT = vocab['program_idx_to_token'][pg[1]]
        q_type = find_clevr_question_type(pT)
        all_q_types.append(q_type)
        c[q_type] += 1

    return c, all_q_types

def get_qtype_distribution_from_questions(all_questions) -> Tuple[Counter, List]:
    l = len(all_questions)
    # print("Num of questions %d" % l)
    all_programs = list(map(lambda x: x['program'], all_questions))
    all_q_types = []
    c = Counter()
    for pg in all_programs:
        pT = pg[-1]['function']
        q_type = find_clevr_question_type(pT)
        all_q_types.append(q_type)
        c[q_type] += 1

    return c, all_q_types


