"""
this is going to provide data from 32x32 images, with attributes
we'll return a tensor of images, and also a tensor of attributes (binary flags)
"""
import torch, mat4py, sys, os, time, datetime, json, csv, argparse
from os.path import expanduser as expand
from collections import defaultdict
import numpy as np
import PIL.Image

def att_vec_to_str(v):
    K = v.size(0)
    return ''.join([str(x) for x in v.tolist()])

class Dataset(object):
    def __init__(self, image_size, images_file, num_neg, att_filepath, num_holdout=128):
        """
        each negative example will be drawn from a different class
        we will consider a class to be each unique combination of attributes

        a key question is: how will we draw the negative samples? uniformly? or
        choosing ones that resemble the target attributes closely? maybe choosing
        uniformly, and just using more negative samples is simpler conceptually,
        and likely to work reasonably (albeit perhaps more slowly).

        another question is, how to split the dataset? just uniformly sample from
        entire dataset? or sample within each attribute value combination? if the latter, how to
        deal with attribute value combinations which only occur once?

        lets just sample uniformly globally for now, for simplicity

        oh wait, for hte holdout, we want unseen combinations in fact?

        lets pick a the att combinations which occur least times ,and use those as holdout set?

        ===========================

        update tue july 16th: we are going to use a different image for the receiver than for the sender, but
        drawn from the same attribute combinnation
        => this also means we need to reconsider how we will handle the holdout attribute combinations
        - we will need at least 128 holdout attribute combinations with >=2 images in

        lets start by partitioning the attributes into those with >=2 images, and those without

        =============================

        - att_t, images_t => all atts, images
        - train_images, train_att_t: all positive training images and atts
        - holdout_images, holdout_att_t: all positive holdout images and atts

        - att_v_by_str: att vector by att string
        - self.idxes_by_att: indexes into att_t, images_t, by att_str (list per att_str value)
        - self.holdout_atts: att_strs for holdout set
        - self.train_atts: att_strs for train set

        - self.holdout_neg_idxes: indexes into images_t and att_t, for holdout negative distractors ([M - 1][N])
        """
        self.image_size = image_size
        self.images_file = images_file
        self.num_neg = num_neg
        self.att_filepath = att_filepath

        print('loading images file...')
        with open(expand(images_file), 'rb') as f:
            self.images_t = torch.load(f)
        print('... done')
        print('images_t.size()', self.images_t.size())

        print('loading attributes file...')
        self.mat = mat4py.loadmat(expand(att_filepath))
        self.att_t = torch.LongTensor(self.mat['attrann']['labels'])
        self.att_t = self.att_t.clamp(min=0)
        print('... loaded')
        print('att_t.size()', self.att_t.size())

        self.N, self.K = self.att_t.size()
        print('N', self.N, 'K', self.K)
        self.idxes_by_att = defaultdict(list)

        self.att_v_by_str = {}
        for n in range(self.N):
            v_str = att_vec_to_str(self.att_t[n])
            self.att_v_by_str[v_str] = self.att_t[n]
            self.idxes_by_att[v_str].append(n)

        self.unique_image_atts = set()
        self.multi_image_atts = set()
        for att, idxes in self.idxes_by_att.items():
            if len(idxes) >= 2:
                self.multi_image_atts.add(att)
            else:
                self.unique_image_atts.add(att)

        # we'll pick 128 of the multi atts as holdout atts, and use those also as distractors
        # count_l = [{'att': att, 'c': len(idxes)} for att, idxes in self.idxes_by_att.items()]
        count_l = [{'att': att, 'c': len(self.idxes_by_att[att])} for att in self.multi_image_atts]
        count_l.sort(key=lambda x: x['c'])
        count = 0
        split_idx = 0
        for n in range(num_holdout):
            count += count_l[split_idx]['c']
            split_idx += 1
        self.holdout_atts = [i['att'] for i in count_l[:split_idx]]
        # for holdout_att in self.holdout_atts:
        #     print(holdout_att, len(self.idxes_by_att[holdout_att]))
        self.train_atts = [i['att'] for i in count_l[split_idx:]]
        print('len(holdout_atts)', len(self.holdout_atts))
        print('len(train_atts)', len(self.train_atts))

        r = np.random.RandomState(seed=123)

        # we are going to store two images from each att in sequence
        # for each example, we will pick an att (indexed to 128)
        # then for the att, use first image as sender, second as receiver
        # and distractors drawn from the rest
        self.N_holdout_atts = len(self.holdout_atts)
        self.holdout_neg_idxes = torch.from_numpy(r.choice(self.N_holdout_atts * 2 - 2, self.N_holdout_atts * self.num_neg, replace=True)).view(self.num_neg, self.N_holdout_atts)
        compare = torch.ones(self.num_neg, self.N_holdout_atts, dtype=torch.int64).cumsum(dim=0) - 1
        self.holdout_neg_idxes[self.holdout_neg_idxes >= compare] += 2
        self.holdout_images = torch.zeros(self.N_holdout_atts * 2, 3, self.image_size, self.image_size)
        self.holdout_att_t = torch.zeros(self.N_holdout_atts * 2, self.K)
        for n in range(self.N_holdout_atts):
            holdout_att_str = self.holdout_atts[n]
            image_idxes = self.idxes_by_att[holdout_att_str]
            assert len(image_idxes) == 2

            self.holdout_images[n * 2] = self.images_t[image_idxes[0]]
            self.holdout_att_t[n * 2] = self.att_v_by_str[holdout_att_str]

            self.holdout_images[n * 2 + 1] = self.images_t[image_idxes[1]]
            self.holdout_att_t[n * 2 + 1] = self.att_v_by_str[holdout_att_str]

        self.N_train_atts = len(self.train_atts)
        # self.train_images = torch.zeros(self.N_train, 3, self.image_size, self.image_size)
        # self.train_att_t = torch.zeros(self.N_train, self.K)
        train_images_l = []
        train_att_l = []
        # train_n = 0
        for train_att_str in self.train_atts:
            _att_v = self.att_v_by_str[train_att_str]
            for idx in self.idxes_by_att[train_att_str]:
                train_images_l.append(self.images_t[idx])
                train_att_l.append(_att_v)
                # train_n += 1
        self.train_images = torch.stack(train_images_l)
        self.train_att_t = torch.stack(train_att_l)
        print('self.train_images.size()', self.train_images.size())
        print('self.train_att_t.size()', self.train_att_t.size())
        self.N_train = len(train_att_l)

    def iter_holdout(self, batch_size):
        N = self.N_holdout_atts
        num_batches = self.N_holdout_atts // batch_size

        for b in range(num_batches):
            b_start = b * batch_size
            b_end = b_start + batch_size
            sender_idxes = torch.arange(b_start * 2, b_end * 2, 2).unsqueeze(0)
            receiver_idxes = torch.arange(b_start * 2, b_end * 2, 2).unsqueeze(0) + 1
            neg_idxes = self.holdout_neg_idxes[:, b_start:b_end]
            idxes = torch.cat([sender_idxes, receiver_idxes, neg_idxes], dim=0)
            image_batch = self.holdout_images[idxes]
            att_batch = self.holdout_att_t[idxes]
            yield image_batch, att_batch

    def sample_batch(self, batch_size, training=True):
        """
        we need to sample self.num_neg + 1 classes
        then from each class, sample one image, then return those
        we also need to provide the ground truth att for the target class
        """
        M = self.num_neg + 2
        atts_b_t = torch.zeros(M, batch_size, self.K)
        images_b_t = torch.zeros(M, batch_size, 3, self.image_size, self.image_size)
        atts = self.train_atts if training else self.holdout_atts
        # atts = self.train_atts
        for n in range(batch_size):
            att_strs = np.random.choice(atts, self.num_neg + 1, replace=False)
            tgt_att = att_strs[0]
            # handle pos images separately from neg images
            for m in range(2, M):
                _att_str = att_strs[m - 1]
                _att = self.att_v_by_str[_att_str]
                atts_b_t[m, n] = _att
                image_idxes = self.idxes_by_att[_att_str]
                images_b_t[m, n] = self.images_t[np.random.choice(image_idxes, replace=False)]
            pos_image_idxes = self.idxes_by_att[att_strs[0]]
            pos_images = self.images_t[np.random.choice(image_idxes, 2, replace=False)]
            images_b_t[:2, n] = pos_images
        return images_b_t, atts_b_t

def run():
    N = 10
    num_neg = 5
    dataset = Dataset(
        images_file='~/data/imagenet/att_synsets/att_images_t.dat',
        num_neg=num_neg,
        att_filepath='~/data/imagenet/attrann.mat',
    )
    batch_images, batch_attributes = dataset.sample_batch(batch_size=N)

    # print('batch_attributes', batch_attributes)
    # print('batch_images[:, :, 10, 10]', batch_images[:, :, 0, 10, 10])

    print('batch_images.size()', batch_images.size())
    print('batch_attributes.size()', batch_attributes.size())

    for n in range(3):
        for m in range(num_neg + 2):
            image = batch_images[m, n]
            image = (image * 255).byte()
            pil_image = PIL.Image.fromarray(image.transpose(0, 1).transpose(1, 2).detach().numpy())
            pil_image.save(f'html/{n}_{m}.png')

    for i, (batch_images, batch_attributes) in enumerate(dataset.iter_holdout(batch_size=4)):
        print('att_batch[:, 0]', batch_attributes[:, 0])
        print('att_batch[:, 1]', batch_attributes[:, 1])

        if i == 0:
            for n in range(4):
                for m in range(num_neg + 2):
                    image = batch_images[m, n]
                    image = (image * 255).byte()
                    pil_image = PIL.Image.fromarray(image.transpose(0, 1).transpose(1, 2).detach().numpy())
                    pil_image.save(f'html/holdout_{n}_{m}.png')

        if i == 1:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(**args.__dict__)
