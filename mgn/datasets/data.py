import re
import torch
import torch_geometric
import random

class ClevrData(torch_geometric.data.Data):
    r"""Data for handling clevr data specific quirks (like edge_attrs)"""
    def __init__(self, src=None, **kwargs):
        super(ClevrData, self).__init__(**kwargs)
        self.src = src      # 'Gs' | 'Gt'

    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return -1 if bool(re.search('(index|face)', key)) else 0

    def __inc__(self, key, value):
        r"""" Increment both edge_index and edge_attr  """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0


class PairData(torch_geometric.data.Data):
    """ For data sample with both Gs, Gt data"""
    def __inc__(self, key, value):
        #if bool(re.search("^edge_[\w]_s$", key)):
        if bool(re.search('index_s', key)):
            return self.x_s.size(0)
        #if bool(re.search("^edge_[\w]*_t$", key)):
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0

class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.

    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample

    def __len__(self):
        return len(self.dataset_s) if self.sample else len(
            self.dataset_s) * len(self.dataset_t)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx // len(self.dataset_t)]
            data_t = self.dataset_t[idx % len(self.dataset_t)]

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            edge_attr_s=data_s.edge_attr,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            edge_attr_t=data_t.edge_attr,
            num_nodes=None,
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)
