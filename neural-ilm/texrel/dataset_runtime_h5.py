"""
handles sampling data from ondisk file at runtime
"""
import argparse
import time
import json
from collections import defaultdict

import numpy as np
import h5py

import torch
from torch import nn
import torch.nn.functional as F

from ulfs.params import Params
from ulfs.utils import die, expand
from ulfs import h5_utils

from texrel.texturizer import Texturizer


class DSRefSlice(object):
    """
    represents eg 'train' slice of some dsref
    """
    def __init__(self, h5_f, slice_name, ds_ref, ds_ref_id):
        """
        h5_f is for a single ds_ref
        so we can just query this, based on the slice_name
        """
        self.h5_f = h5_f
        self.slice_name = slice_name
        self.ds_ref = ds_ref
        self.ds_ref_id = ds_ref_id
        self.values_by_name = {}

        keys = [k.replace(f'{slice_name}_', '') for k in list(h5_f.keys()) if k.startswith(f'{slice_name}_')]
        self.N = h5_f[f'{slice_name}_labels'].shape[0]
        for k in keys:
            v = h5_f[f'{slice_name}_{k}']
            self.values_by_name[k] = v
        # dsrefs_t = torch.full((self.N, ), fill_value=ds_ref_id, dtype=torch.int64)
        self.meta = Params(json.loads(h5_utils.get_value(h5_f, 'meta')))
        print('meta', self.meta)

    @property
    def hypothesis_len(self):
        return self.values_by_name['hypotheses_t'].shape[0]

    # def sample(self, batch_size):
    def batch_from_idxes(self, idxes):
        # idxes = torch.from_numpy(np.random.choice(self.N, batch_size, replace=False))

        idxes = sorted(idxes.tolist())
        _N = len(idxes)
        receiver_shapes_t = torch.from_numpy(self.values_by_name['receiver_shapes'][idxes])
        receiver_colors_t = torch.from_numpy(self.values_by_name['receiver_colors'][idxes])

        hypotheses_t = torch.from_numpy(self.values_by_name['hypotheses_t'][:, idxes])

        input_shapes_t = torch.from_numpy(self.values_by_name['input_shapes'][:, idxes])
        input_colors_t = torch.from_numpy(self.values_by_name['input_colors'][:, idxes])

        dsrefs_t = torch.full((_N, ), fill_value=self.ds_ref_id, dtype=torch.int64)
        # dsrefs_t = torch.from_numpy(self.values_by_name['dsrefs_t'][idxes])
        labels_t = torch.from_numpy(self.values_by_name['labels'][idxes])
        input_labels_t = torch.from_numpy(self.values_by_name['input_labels'][:, idxes])
        # input_examples_t = self.texturizer.forward(
        #     texture_idxes=input_shapes_t, color_idxes=input_colors_t)
        # receiver_examples_t = self.texturizer.forward(
        #     texture_idxes=receiver_shapes_t, color_idxes=receiver_colors_t)

        res = {
            'N': _N,
            # 'input_examples_t': input_examples_t.detach(),
            'hypotheses_t': hypotheses_t.detach(),
            # 'receiver_examples_t': receiver_examples_t.detach(),
            'labels_t': labels_t.detach(),
            'input_shapes_t': input_shapes_t,
            'input_colors_t': input_colors_t,
            'receiver_shapes_t': receiver_shapes_t,
            'receiver_colors_t': receiver_colors_t,
            'input_labels_t': input_labels_t.detach(),
            'dsrefs_t': dsrefs_t.detach(),
            # 'hypotheses': torch.from_numpy(self.values_by_name['hypotheses_t'][:, idxes]).detach()
        }

        return res


class Textured3PlanesDatasetSlice(object):
    """
    just eg train, or just eg test
    """
    def __init__(self, ds_refs, h5_f_by_dsref, slice_name):
        self.ds_refs = ds_refs
        self.h5_f_by_dsref = h5_f_by_dsref
        self.slice_name = slice_name

        dsref_name2id = {name: i for i, name in enumerate(ds_refs)}

        self.ds_ref_slice_by_ds_ref = {}
        self.hypothesis_len = 0
        self.N = 0
        for ds_ref, h5_f in h5_f_by_dsref.items():
            ds_ref_slice = DSRefSlice(
                h5_f=h5_f,
                slice_name=slice_name,
                ds_ref=ds_ref,
                ds_ref_id=dsref_name2id[ds_ref]
            )
            self.ds_ref_slice_by_ds_ref[ds_ref] = ds_ref_slice
            self.hypothesis_len = max(self.hypothesis_len, ds_ref_slice.hypothesis_len)
            self.N += ds_ref_slice.N
        print('slice', slice_name, 'hypothesis_len', self.hypothesis_len, 'N', self.N)

        self.global_start_idx_by_ds_ref = {}
        self.global_end_idx_excl_by_ds_ref = {}
        pos = 0
        for i, ds_ref in enumerate(ds_refs):
            self.global_start_idx_by_ds_ref[ds_ref] = pos
            ds_ref_slice = self.ds_ref_slice_by_ds_ref[ds_ref]
            size = ds_ref_slice.N
            pos += size
            self.global_end_idx_excl_by_ds_ref[ds_ref] = pos

        meta_overall = {}
        for ds_ref, ds_ref_slice in self.ds_ref_slice_by_ds_ref.items():
            _meta = ds_ref_slice.meta
            for k, v in _meta.__dict__.items():
                if k not in ['ref', 'ds_ref', 'hypothesis_generators', 'seed', 'num_distractors', 'vocab_size']:
                    if k in meta_overall:
                        if meta_overall[k] != v:
                            print(f'meta mismatch {k}: {meta_overall[k]} != {v}')
                        assert meta_overall[k] == v
                    meta_overall[k] = v
        meta_overall = Params(meta_overall)
        self.meta = meta_overall
        self.meta.utt_len = self.hypothesis_len
        print('overall meta', self.meta)

    def sample(self, batch_size):
        idxes = torch.from_numpy(np.random.choice(self.N, batch_size, replace=False))
        # batch = {}
        batch = None

        tensor_dim_by_name = {
            'hypotheses_t': 1,
            'input_labels_t': 1,
            'input_shapes_t': 1,
            'input_colors_t': 1,
            'receiver_shapes_t': 0,
            'receiver_colors_t': 0,
            'labels_t': 0,
            'dsrefs_t': 0
        }

        for i, ds_ref in enumerate(self.ds_refs):
            start = self.global_start_idx_by_ds_ref[ds_ref]
            end_excl = self.global_end_idx_excl_by_ds_ref[ds_ref]
            mask = (idxes >= start) * (idxes < end_excl)
            batch_idxes = mask.view(-1).nonzero().view(-1).long()
            idxes_for_child = idxes[batch_idxes] - start
            sub_batch = self.ds_ref_slice_by_ds_ref[ds_ref].batch_from_idxes(idxes_for_child)

            if batch is None:
                batch = {}
                for k, dim in tensor_dim_by_name.items():
                    size = list(sub_batch[k].size())
                    size[dim] = batch_size
                    if k == 'hypotheses_t':
                        size[0] = self.hypothesis_len
                    dtype = sub_batch[k].dtype
                    _t = torch.zeros(*size, dtype=dtype)
                    batch[k] = _t

            for k, dim in tensor_dim_by_name.items():
                if dim == 0:
                    batch[k][batch_idxes] = sub_batch[k]
                elif dim == 1:
                    # print('k', k, 'batch_idxes.size()', batch_idxes.size(), 'k', k, 'sub_batch[k].size()', sub_batch[k].size(),
                    #     'batch[k].size()', batch[k].size())
                    if k == 'hypotheses_t':
                        _sub_len = sub_batch[k].size(0)
                        batch[k][:_sub_len, batch_idxes] = sub_batch[k]
                    else:
                        batch[k][:, batch_idxes] = sub_batch[k]
                else:
                    raise Exception('unhandled dim ' + str(dim))
        return batch


class Textured3PlanesDataset(object):
    def __init__(
            self, ds_filepath_templ, ds_refs, ds_seed, ds_texture_size, ds_background_noise, ds_mean, ds_mean_std
        ):
        print('loading data...')
        filepath_templ = ds_filepath_templ
        refs = ds_refs
        seed = ds_seed
        texture_size = ds_texture_size
        background_noise = ds_background_noise
        self.ds_mean_std = ds_mean_std
        self.background_noise = background_noise

        meta_keys = ['grid_size', 'num_colors', 'num_shapes', 'num_holdout_objects', 'num_pos', 'num_neg', 'vocab_size']
        self.h5_f_by_dsref = {}
        self.h5_wrapper_by_dsref = {}
        for ds_ref in ds_refs:
            print(f'opening {ds_ref} ...', end='', flush=True)
            filepath = ds_filepath_templ.format(ds_ref=ds_ref)
            h5_f = h5py.File(expand(filepath), 'r')
            self.h5_f_by_dsref[ds_ref] = h5_f
            h5_wrapper = h5_utils.H5Wrapper(h5_f)
            self.h5_wrapper_by_dsref[ds_ref] = h5_wrapper

        self.ds_by_slice_name = {}
        for slice_name in ['train', 'test']:
            slice_ds = Textured3PlanesDatasetSlice(
                slice_name=slice_name,
                ds_refs=ds_refs,
                h5_f_by_dsref=self.h5_f_by_dsref
            )
            self.ds_by_slice_name[slice_name] = slice_ds

        self.meta = Params(self.ds_by_slice_name['train'].meta.__dict__.copy())

        self.meta.grid_planes = 3
        self.meta.grid_size *= texture_size
        # self.meta.utt_len = hypothesis_len

        self.training = True
        self.texturizer = Texturizer(
            num_textures=self.meta.num_shapes,
            num_colors=self.meta.num_colors,
            texture_size=ds_texture_size,
            seed=ds_seed,
            background_noise=background_noise,
            background_mean_std=ds_mean_std,
            background_mean=ds_mean
        )
        print(' ... finished loading data')

    def summarize_datas(self):
        for name, datas in self.datas.items():
            print(f'{name}:')
            for k, v in datas.items():
                if isinstance(v, int):
                    print('  ', k, v)
                else:
                    print('  ', k, v.dtype, v.size())

    def summarize_datas_by_dsref(self):
        for dsref, data in self.datas_by_dsref.items():
            print(f'{dsref}:')
            for name, datas in data.items():
                print(f'  {name}:')
                for k, v in datas.items():
                    if isinstance(v, int):
                        print('    ', k, v)
                    else:
                        print('    ', k, v.dtype, v.size())

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    @property
    def N(self):
        if self.training:
            return self.datas['train']['N']
        return self.datas['test']['N']

    # def holdout_iterator(self, batch_size):
    #     class Iterator(object):
    #         def __init__(self, parent):
    #             self.b = 0
    #             self.parent = parent

    #         def __iter__(self):
    #             return self

    #         def __next__(self):
    #             if self.b < num_batches:
    #                 idxes = torch.arange(self.b * batch_size, (self.b + 1) * batch_size, dtype=torch.int64)
    #                 self.b += 1
    #                 return self.parent._batch_from_idxes(set_name='test', idxes=idxes)
    #             else:
    #                 raise StopIteration

    #     data = self.datas['test']
    #     N = data['N']
    #     print('N', N)
    #     num_batches = N // batch_size
    #     print('num_batches', num_batches)
    #     return Iterator(parent=self)

    def sample(self, batch_size, training=None):
        print('creating batch...')
        start_time = time.time()
        if training is None:
            training = self.training
        set_name = 'train' if training else 'test'
        res = self.ds_by_slice_name[set_name].sample(batch_size=batch_size)
        print('fetched data', time.time() - start_time)

        res['input_examples_t'] = self.texturizer.forward(
            texture_idxes=res['input_shapes_t'],
            color_idxes=res['input_colors_t']
        )
        res['receiver_examples_t'] = self.texturizer.forward(
            texture_idxes=res['receiver_shapes_t'],
            color_idxes=res['receiver_colors_t']
        )
        elapsed = time.time() - start_time
        print(f'created batch in {elapsed:.3f} seconds')
        return res

        # data = self.datas[set_name]
        # N = data['N']
        # idxes = torch.from_numpy(np.random.choice(N, batch_size, replace=False))
        # return self._batch_from_idxes(set_name=set_name, idxes=idxes)


def run(ds_ref, ds_filepath, batch_size, ds_texture_size, ds_seed, ds_background_noise, ds_mean_std):
    import matplotlib.pyplot as plt
    ds_filepath = ds_filepath.format(ref=ds_ref)
    dataset = Textured3PlanesDataset(
        ds_filepath=ds_filepath,
        ds_background_noise=ds_background_noise,
        ds_seed=ds_seed,
        ds_texture_size=ds_texture_size,
        ds_mean_std=ds_mean_std
    )
    d = dataset.sample(batch_size=batch_size)
    print('input labels', d['input_labels_t'][:, 0].tolist())
    print('label', d['labels_t'][0].item())
    for m in range(5):
        plt.imshow(d['input_examples_t'][m][0].transpose(-3, -2).transpose(-2, -1).detach().numpy())
        plt.savefig(f'/tmp/inputs_M{m}_N{0}.png')

    plt.imshow(d['receiver_examples_t'][0].transpose(-3, -2).transpose(-2, -1).detach().numpy())
    plt.savefig(f'/tmp/receiver_examples_N{0}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-ref', type=str, required=True)
    parser.add_argument('--ds-filepath', type=str, default='~/data/reftask/{ref}.dat')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--ds-texture-size', type=int, default=2)
    parser.add_argument('--ds-seed', type=int, required=True)
    parser.add_argument('--ds-background-noise', type=float, default=0, help='std of noise (with mean 0.5)')
    parser.add_argument('--ds-mean-std', type=float, default=0)
    args = parser.parse_args()
    run(**args.__dict__)
