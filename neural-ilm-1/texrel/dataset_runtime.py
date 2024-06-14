"""
handles sampling data from ondisk file at runtime
"""
import argparse
import time
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ulfs.params import Params
from ulfs.utils import die, expand

from texrel.texturizer import Texturizer


class Textured3PlanesDataset(object):
    def __init__(
            self, ds_filepath_templ, ds_refs, ds_seed, ds_texture_size, ds_background_noise, ds_mean, ds_mean_std
        ):
        filepath_templ = ds_filepath_templ
        refs = ds_refs
        seed = ds_seed
        texture_size = ds_texture_size
        background_noise = ds_background_noise
        self.ds_mean_std = ds_mean_std

        self.background_noise = background_noise

        meta_keys = ['grid_size', 'num_colors', 'num_shapes', 'num_holdout_objects', 'num_pos', 'num_neg', 'vocab_size']

        self.metas = []
        dsref_name2id = {}
        self.datas_by_dsref = {}  # eg {'ds63': {'train': {'N': ..., 'input_shapes': ..., ...}}}
        self.datas = {}  # {'train': {'N': ..., 'input_shapes': ..., ...}}
        for ds_ref in ds_refs:
            """
            ds_ref, eg dsref64
            """
            dsref_name2id[ds_ref] = len(dsref_name2id)
            print(f'loading {ds_ref} ...', end='', flush=True)
            filepath = ds_filepath_templ.format(ds_ref=ds_ref)
            with open(expand(filepath), 'rb') as f:
                d = torch.load(f)
            print(' done')
            self.metas.append(Params(d['meta']))
            self.datas_by_dsref[ds_ref] = d['data']
            for k1, data in d['data'].items():
                """
                k1, eg 'train', 'holdout'
                """
                _N = data['labels'].size(0)
                # print('_N', _N)
                if k1 not in self.datas:
                    self.datas[k1] = defaultdict(list)
                for k2, v in data.items():
                    """
                    k2, eg 'N', 'input_labels', 'input_shapes', ...
                    """
                    self.datas[k1][k2].append(v)
                dsrefs_t = torch.full((_N, ), fill_value=dsref_name2id[ds_ref], dtype=torch.int64)
                self.datas[k1]['dsrefs_t'].append(dsrefs_t)
        # self.summarize_datas_by_dsref()
        datas_new = {}
        hypothesis_len = 0
        for k1, data in self.datas.items():
            datas_new[k1] = {}
            d = datas_new[k1]
            d['N'] = np.sum(data['N']).item()
            tensor_dim_by_name = {
                'hypotheses_t': 1,
                'input_labels': 1,
                'input_shapes': 1,
                'input_colors': 1,
                'receiver_shapes': 0,
                'receiver_colors': 0,
                'labels': 0,
                'dsrefs_t': 0
            }
            for name, dim in tensor_dim_by_name.items():
                if len(ds_refs) == 1 or name not in 'hypotheses_t':
                    # print(name)
                    # for t in data[name]:
                        # print('    ', t.size(), t.dtype)
                    v = torch.cat(data[name], dim=dim)
                    d[name] = v
                else:
                    # print('hypotheses_t')
                    _max_utt_len = 0
                    _N = 0
                    for t in data[name]:
                        _max_utt_len = max(_max_utt_len, t.size(0))
                        _N += t.size(1)
                        # print('    ', t.size(), t.dtype)
                    # print('_max_utt_len', _max_utt_len, '_N', _N)
                    _fused_shape = list(t.size())
                    _fused_shape[0] = _max_utt_len
                    # print('_fused_shape', _fused_shape)
                    _fused_shape[1] = _N
                    v = torch.zeros(*_fused_shape, dtype=torch.int64)
                    # print('v.size()', v.size())
                    _n = 0
                    for t in data[name]:
                        v[:t.size(0), _n:_n + t.size(1)] = t
                        _n += t.size(1)
                    d[name] = v
                if name == 'hypotheses_t':
                    hypothesis_len = d[name].size(0)
            _sample_idxes = torch.from_numpy(np.random.choice(d['N'], 10, replace=False))
            # print('hyp sample', d['hypotheses_t'][:, _sample_idxes].transpose(0, 1))
            # print('dsrefs_t sample', d['dsrefs_t'][_sample_idxes])

        overall_meta = {}
        print('')
        for meta in self.metas:
            print('meta')
            print(meta)
            for k, v in meta.__dict__.items():
                if k not in ['ref', 'ds_ref', 'hypothesis_generators', 'seed', 'num_distractors', 'vocab_size']:
                    if k in overall_meta:
                        if overall_meta[k] != v:
                            print(f'meta mismatch {k}: {overall_meta[k]} != {v}')
                        assert overall_meta[k] == v
                    overall_meta[k] = v
        overall_meta = Params(overall_meta)
        self.meta = overall_meta
        print('overall meta', self.meta)

        self.datas = datas_new

        # self.summarize_datas()

        self.meta.grid_planes = 3
        self.meta.grid_size *= texture_size
        self.meta.utt_len = hypothesis_len

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

    def _batch_from_idxes(self, set_name, idxes):
        start_time = time.time()
        data = self.datas[set_name]
        receiver_shapes_t = data['receiver_shapes'][idxes].long()
        receiver_colors_t = data['receiver_colors'][idxes].long()

        hypotheses_t = data['hypotheses_t'][:, idxes].long()

        input_shapes_t = data['input_shapes'][:, idxes].long()
        input_colors_t = data['input_colors'][:, idxes].long()

        dsrefs_t = data['dsrefs_t'][idxes].long()
        labels_t = data['labels'][idxes].long()
        input_labels_t = data['input_labels'][:, idxes].long()
        input_examples_t = self.texturizer.forward(
            texture_idxes=input_shapes_t, color_idxes=input_colors_t)
        receiver_examples_t = self.texturizer.forward(
            texture_idxes=receiver_shapes_t, color_idxes=receiver_colors_t)

        res = {
            'N': labels_t.size(0),
            'input_examples_t': input_examples_t.detach(),
            'hypotheses_t': hypotheses_t.detach(),
            'receiver_examples_t': receiver_examples_t.detach(),
            'labels_t': labels_t.detach(),
            'input_labels_t': input_labels_t.detach(),
            'dsrefs_t': dsrefs_t.detach()
        }

        if 'hypotheses_t' in data:
            res['hypotheses_t'] = data['hypotheses_t'][:, idxes].detach()
        # print('batch sample time %.3f' % (time.time() - start_time))
        return res

    def holdout_iterator(self, batch_size):
        class Iterator(object):
            def __init__(self, parent):
                self.b = 0
                self.parent = parent

            def __iter__(self):
                return self

            def __next__(self):
                if self.b < num_batches:
                    idxes = torch.arange(self.b * batch_size, (self.b + 1) * batch_size, dtype=torch.int64)
                    self.b += 1
                    return self.parent._batch_from_idxes(set_name='test', idxes=idxes)
                else:
                    raise StopIteration

        data = self.datas['test']
        N = data['N']
        print('N', N)
        num_batches = N // batch_size
        print('num_batches', num_batches)
        return Iterator(parent=self)

    def sample(self, batch_size, training=None):
        if training is None:
            training = self.training
        set_name = 'train' if training else 'test'
        data = self.datas[set_name]
        N = data['N']
        idxes = torch.from_numpy(np.random.choice(N, batch_size, replace=False))
        return self._batch_from_idxes(set_name=set_name, idxes=idxes)


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
