from .data import ClevrData, PairData, PairDataset
from .clevr_questions import ClevrQuestionDataset, ClevrQuestionDataLoader

def get_dataset(opt, split, **kwargs):
    """Get function for dataset class
      For the case of 0-shot generalization, this will suffice as
      we are not loading any CLOSURE questions for training
    """
    assert split in ['train', 'val']
    if opt.dataset == 'clevr':
        graph_parser = kwargs.get('graph_parser')
        embedder = kwargs.get('embedder')
        dataset = ClevrQuestionDataset(opt, split,
                                       graph_parser=graph_parser,
                                       embedder=embedder)
    else:
        raise ValueError('Invalid dataset')
    return dataset

def get_dataloader(opt, split, **kwargs):
    """Get function for dataloader class"""
    graph_parser = kwargs.get('graph_parser')
    embedder = kwargs.get('embedder')
    dataset = get_dataset(opt, split, graph_parser=graph_parser, embedder=embedder)
    shuffle = opt.shuffle if split == 'train' else 0
    loader = ClevrQuestionDataLoader(dataset=dataset, batch_size=opt.batch_size,
                                         shuffle=shuffle, num_workers=opt.num_workers,
                                            follow_batch=[])
    print('| %s %s loader has %d samples' % (opt.dataset, split, len(loader.dataset)))
    return loader


