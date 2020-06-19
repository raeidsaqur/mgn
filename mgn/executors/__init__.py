from .clevr_executor import ClevrExecutor
from functools import lru_cache

def get_executor(opt, *args, **kwargs):
    print('| creating %s executor' % opt.dataset)
    graph_parser = kwargs.get('graph_parser')
    embedder = kwargs.get('embedder')
    if opt.dataset == 'clevr':
        train_scene_json = opt.clevr_train_scene_path if opt.is_train else None
        val_scene_json = opt.clevr_val_scene_path
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(train_scene_json,
                             val_scene_json,
                             vocab_json,
                             graph_parser=graph_parser,
                             embedder=embedder)
    return executor


def get_executor_orig(opt):
    print('| creating %s executor' % opt.dataset)
    if opt.dataset == 'clevr':
        train_scene_json = opt.clevr_train_scene_path
        val_scene_json = opt.clevr_val_scene_path
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    executor = ClevrExecutor(train_scene_json, val_scene_json, vocab_json)
    return executor