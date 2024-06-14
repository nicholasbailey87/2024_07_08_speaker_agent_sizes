#!/usr/bin/env python
"""
just write from scratch ...

so, we need:
- an image encoder model. we'll use something like a lenet-5 cnn
- sender network, that takes the encoding as input (?)
- receiver network
    - ... and we will dot product the output of the receiver with the various encoded images
"""
import argparse, time, os, math, json, sys, contextlib, random

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import numpy as np

from ulfs.utils import die
from ulfs.params import Params
from ulfs import name_utils, rl_common, tensor_utils,  utils, nn_modules, metrics
from ulfs.runner_base_v1 import RunnerBase
from ulfs.stats import Stats

from data_code.clevr.three_shapes_dataset import Dataset
from ilm.cnn_models import *

@contextlib.contextmanager
def torch_random_state():
    rnd_tch = torch.get_rng_state()
    rnd_cuda = torch.cuda.get_rng_state()
    np_state = np.random.get_state()
    rand_state = random.getstate()
    yield
    torch.set_rng_state(rnd_tch)
    torch.cuda.set_rng_state(rnd_cuda)
    np.random.set_state(np_state)
    random.setstate(rand_state)

class LangSenderModel(nn.Module):
    """
    generator model
    we'll use a differentiable teacher-forcing model, with fixed utterance length
    """
    def __init__(
            self, opt_name, embedding_size, vocab_size, utt_len, rnn_type, num_layers, input_size,
            dropout
        ):
        self.embedding_size = embedding_size
        self.utt_len = utt_len
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        super().__init__()
        self.h_in = nn.Linear(input_size, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        else:
            RNN = getattr(nn, f'{rnn_type}')
        rnn_params = {
            'input_size': embedding_size,
            'hidden_size': embedding_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        if rnn_type == 'SRU':
            rnn_params['rescale'] = False
            rnn_params['use_tanh'] = True
        self.rnn = RNN(**rnn_params)
        self.h_out = nn.Linear(embedding_size, vocab_size)
        self.drop = nn.Dropout(dropout)

        Opt = getattr(optim, opt_name)
        self.opt = Opt(lr=0.001, params=self.parameters())

    def forward(self, thoughts):
        """
        thoughts ae [N][input_size]
        """
        N, K = thoughts.size()
        embs = self.h_in(thoughts)
        embs = self.drop(embs)
        device = embs.device

        if self.rnn_type in ['SRU']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=device)
            h[0] = embs
        elif self.rnn_type in ['GRU']:
            h = torch.zeros(self.num_layers, N, self.embedding_size, dtype=torch.float32, device=device)
            h[0] = embs
        else:
            raise Exception(f'unrecognized rnn type {self.rnn_type}')

        fake_input = torch.zeros(self.utt_len, N, self.embedding_size, dtype=torch.float32, device=device)
        if self.rnn_type == 'SRU':
            output, state = self.rnn(fake_input, h)
        elif self.rnn_type in ['GRU']:
            output, h = self.rnn(fake_input, h)
        else:
            raise Exception(f'rnn type {self.rnn_type} not recognized')
        utts = self.h_out(output)
        return utts

class LangReceiverModel(nn.Module):
    def __init__(
            self, opt_name, embedding_size, vocab_size, utt_len,
            rnn_type, dropout, num_layers, output_size
        ):
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size

        super().__init__()
        self.embedding = nn_modules.EmbeddingAdapter(vocab_size, embedding_size)
        if rnn_type == 'SRU':
            from sru import SRU
            RNN = SRU
        else:
            RNN = getattr(nn, f'{rnn_type}')
        self.rnn = RNN(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers
        )
        self.h_out = nn.Linear(embedding_size, output_size)
        Opt = getattr(optim, opt_name)
        self.opt = Opt(lr=0.001, params=self.parameters())

    def forward(self, utts, do_predict_correct=False):
        embs = self.embedding(utts)
        seq_len, batch_size, embedding_size = embs.size()
        output, state = self.rnn(embs)
        state = state[-1]
        x = self.h_out(state)
        return x

class SenderPathway(nn.Module):
    def __init__(self, opt_name, cnn, lang_sender, clip_grad):
        super().__init__()
        self.cnn = cnn
        self.lang_sender = lang_sender
        self.clip_grad = clip_grad

        Opt = getattr(optim, opt_name)
        self.params = self.parameters()
        self.opt = Opt(lr=0.001, params=self.params)

    def forward(self, images_t):
        """
        assumes that images_t just contains a batch of single images, not multipel images per example
        """
        image_enc = self.cnn(images_t)
        utt_logits = self.lang_sender(image_enc)
        return utt_logits

    def sup_train_batch(self, images, utts):
        utts_logits = self(images)

        # logits = self.model(meanings)
        _, utts_pred = utts_logits.max(dim=-1)
        correct = utts_pred == utts
        acc = correct.float().mean().item()

        crit = nn_modules.GeneralCrossEntropyLoss()
        loss = crit(utts_logits, utts)
        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.opt.step()
        return loss.item(), acc

class ReceiverPathway(nn.Module):
    def __init__(self, opt_name, cnn, lang_receiver, clip_grad):
        super().__init__()
        self.cnn = cnn
        self.lang_receiver = lang_receiver
        self.clip_grad = clip_grad

        Opt = getattr(optim, opt_name)
        self.params = self.parameters()
        self.opt = Opt(lr=0.001, params=self.params)

    def forward(self, utts, images):
        """
        we assume images are all the receiver images, ie the goal image, and some distractors
        the sender image is not included in images
        """
        lang_enc = self.lang_receiver(utts)  # [N][E]

        d = images.size()
        images_flat_t = tensor_utils.merge_dims(images, 0, 1)  # [M * N][C][H][W]
        images_enc_flat = self.cnn(images_flat_t)   # [M * N][E]
        images_enc = tensor_utils.split_dim(images_enc_flat, 0, d[0], d[1])  # [M][N][E]
        lang_enc_flat_exp = lang_enc.unsqueeze(0).expand_as(images_enc)  # [M][N][E]
        lang_enc_flat_exp = tensor_utils.merge_dims(lang_enc_flat_exp.contiguous(), 0, 1)   # [M * N][E]

        dp_left = lang_enc_flat_exp.unsqueeze(-2)  # [M * N][1][E]
        dp_right = images_enc_flat.unsqueeze(-1)   # [M * N][E][1]
        dp = torch.bmm(dp_left, dp_right)     # [M * N][1][1]
        dp = dp.view(d[0], d[1])   # [M][N]
        dp = dp.transpose(0, 1)   # [N][M]
        return dp

    def sup_train_batch(self, images, utts):
        """
        we assume a batch of single images, no distractors added
        """
        d = images.size()
        with autograd.no_grad():
            images_enc = self.cnn(images)   # [N][E]

        lang_enc = self.lang_receiver(utts)  # [N][E]
        crit = nn.MSELoss()
        loss = crit(lang_enc, images_enc)
        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.opt.step()
        acc = 0  # placeholder
        return loss.item(), acc

class Agent(nn.Module):
    def __init__(self, p, image_size, img_embedding_size):
        super().__init__()
        self.p = p
        self.img_embedding_size = img_embedding_size

        CNN = globals()[p.conv_class]
        # share cnn between sender and receiver for now (can try unsharing later)
        self.cnn = CNN(dropout=p.dropout, num_layers=p.num_conv_layers, image_size=image_size)
        self.lang_sender = LangSenderModel(
            opt_name=p.opt,
            embedding_size=p.embedding_size,
            vocab_size=p.vocab_size,
            utt_len=p.utt_len,
            rnn_type=p.rnn_type,
            num_layers=p.num_layers,
            input_size=self.img_embedding_size,
            dropout=p.dropout
        )
        self.lang_receiver = LangReceiverModel(
            opt_name=p.opt,
            embedding_size=p.embedding_size,
            vocab_size=p.vocab_size,
            utt_len=p.utt_len,
            rnn_type=p.rnn_type,
            dropout=p.dropout,
            num_layers=p.num_layers,
            output_size=self.img_embedding_size
        )
        self.sender_pathway = SenderPathway(opt_name=p.opt, cnn=self.cnn, lang_sender=self.lang_sender, clip_grad=p.clip_grad)
        self.receiver_pathway = ReceiverPathway(opt_name=p.opt, cnn=self.cnn, lang_receiver=self.lang_receiver, clip_grad=p.clip_grad)
        if p.enable_cuda:
            self.lang_sender = self.lang_sender.cuda()
            self.lang_receiver = self.lang_receiver.cuda()
            self.cnn = self.cnn.cuda()
        Opt = getattr(optim, p.opt)
        self.opt_both = Opt(
            lr=0.001,
            params=list(self.lang_sender.parameters()) + list(self.lang_receiver.parameters()) + list(self.cnn.parameters())
        )

    def state_dict(self):
        return {
            'lang_sender_state': self.lang_sender.state_dict(),
            'lang_receiver_state': self.lang_receiver.state_dict(),
            'cnn_state': self.cnn.state_dict(),
            'opt_both_state': self.opt_both.state_dict()
        }

    def load_state_dict(self, statedict):
        self.lang_sender.load_state_dict(statedict['lang_sender_state'])
        self.lang_receiver.load_state_dict(statedict['lang_receiver_state'])
        self.opt_both.load_state_dict(statedict['opt_both_state'])

class SoftmaxLink(object):
    def __init__(self, p):
        pass

    def sample_utterances(self, utt_probs):
        return utt_probs

    def sample_image_choice(self, dp):
        return dp

    def calc_loss(self, image_choice):
        dp = image_choice
        batch_size, _ = dp.size()
        crit = nn.CrossEntropyLoss()
        loss = crit(dp, torch.zeros(batch_size, dtype=torch.int64, device=dp.device))
        loss_v = loss.item()
        return loss, loss_v

class RLLink(object):
    def __init__(self, p):
        self.s_ent = p.s_ent
        self.r_ent = p.r_ent

    def sample_utterances(self, utt_probs, training):
        # print('rllink training', self.training)
        self.s_sender = rl_common.draw_categorical_sample(
            action_probs=utt_probs,
            batch_idxes=None,
            training=training
        )
        utts = self.s_sender.actions.detach()
        return utts

    def sample_image_choice(self, training, dp):
        self.s_recv = rl_common.draw_categorical_sample(
            action_probs=dp,
            batch_idxes=None,
            training=training
        )
        return (self.s_sender.greedy_matches, self.s_recv.greedy_matches), self.s_recv.actions

    def calc_loss(self, image_choice, training):
        # s_recv = image_choice
        s_recv = self.s_recv
        rewards = (s_recv.actions == 0).float()

        # for reporting purposes:
        loss_v = - rewards.mean().item()

        # lets baseline the reward first
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        rewards = rewards - rewards_mean
        if rewards_std > 1e-1:
            rewards = rewards / rewards_std

        if training:
            rl_loss = self.s_sender.calc_loss(rewards) + s_recv.calc_loss(rewards)
            loss_all = rl_loss
            ent_loss = 0
            if self.s_ent is not None and self.s_ent > 0:
                ent_loss -= self.s_sender.entropy * self.s_ent
            if self.r_ent is not None and self.r_ent > 0:
                ent_loss -= s_recv.entropy * self.r_ent
            loss_all += ent_loss
        else:
            # rl_loss = 0
            loss_all = 0

        loss = loss_all
        return loss, loss_v

class RefTaskGame(object):
    def __init__(self, sender_pathway, receiver_pathway, link):
        self.sender_pathway = sender_pathway
        self.receiver_pathway = receiver_pathway
        self.link = link

    def forward(self, images_t, training):
        utt_logits = self.sender_pathway(images_t[0])
        utt_probs = F.softmax(utt_logits, dim=-1)

        utts = self.link.sample_utterances(utt_probs, training=training)

        self.dp = self.receiver_pathway(utts=utts, images=images_t[1:])
        self.dp = F.softmax(self.dp, dim=-1)

        (sender_greedy, receiver_greedy), pred = self.link.sample_image_choice(training=training, dp=self.dp)

        # _, pred = self.dp.max(dim=-1)
        acc = (pred == 0).float().mean().item()
        return (sender_greedy, receiver_greedy), utts.detach(), acc

    def calc_loss(self, training):
        loss, loss_v = self.link.calc_loss(self.dp, training=training)
        # return (sender_greedy, receiver_greedy), loss, loss_v
        return loss, loss_v

def batched_run_nograd(params, model, inputs, batch_size, input_batch_dim, output_batch_dim):
    """
    assumes N is multiple of batch_size
    """
    p = params
    N = inputs.size(input_batch_dim)
    num_batches = (N + batch_size - 1) // batch_size
    outputs = None
    count = 0
    for b in range(num_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, N)
        count += (b_end - b_start)
        input_batch = inputs.narrow(dim=input_batch_dim, start=b_start, length=b_end - b_start)
        if p.enable_cuda:
            input_batch = input_batch.cuda()
        with torch.no_grad():
            output_batch = model(input_batch).detach().cpu()
        if outputs is None:
            out_size_full = list(output_batch.size())
            out_size_full[output_batch_dim] = N
            outputs = torch.zeros(*out_size_full, dtype=output_batch.dtype, device='cpu')
        out_narrow = outputs.narrow(dim=output_batch_dim, start=b_start, length=b_end - b_start)
        out_narrow[:] = output_batch
    assert count == N
    return outputs

class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['teacher'],
            additional_save_keys=[],
            step_key='training_step'
        )

    def setup(self, p):
        if p.seed is not None:
            torch.manual_seed(p.seed)
            np.random.seed(p.seed)
            random.seed(p.seed)
            torch.backends.cudnn.deterministic = True
            print('seeding torch and numpy using ', p.seed)

        self.dataset = Dataset(data_dir=p.data_dir)

        # determine image size from dataset
        images_t, _ = self.dataset.sample_batch(batch_size=2)
        self.image_size = images_t.size(-1)
        print('self.image_size', self.image_size)

        CNN = globals()[p.conv_class]

        # determine output size from cnn:
        _cnn = CNN(dropout=p.dropout, num_layers=p.num_conv_layers, image_size=self.image_size)  # share cnn across everything
        print(_cnn.convnet)
        print('convnet output size', _cnn.output_size)
        if p.enable_cuda:
            _cnn = _cnn.cuda()
        if p.enable_cuda:
            images_t = images_t.cuda()
        with autograd.no_grad():
            enc_images_t = _cnn(images_t[0])
        _, self.img_embedding_size = enc_images_t.size()
        print('self.img_embedding_size', self.img_embedding_size)


        self.teacher = Agent(p=p, img_embedding_size=self.img_embedding_size, image_size=self.image_size)

        Link = globals()[f'{p.link}Link']
        self.link = Link(p=p)

        self.sup_train_N = int(self.dataset.N_train * p.sup_train_frac)
        print('sup_train_N', self.sup_train_N)

    def step(self, p):
        training_step = self.training_step
        step = self.training_step
        render = self.should_render()
        link = self.link

        sup_images = self.dataset.sample_images(self.sup_train_N)
        if p.enable_cuda:
            sup_images = sup_images.cuda()

        print('generating teacher utterances...', end='', flush=True)
        _gen_start = time.time()
        self.teacher.eval()
        # with autograd.no_grad():
        #     sup_images_enc = self.teacher.cnn(sup_images).detach()
        sup_utts_logits = batched_run_nograd(
            params=p,
            model=self.teacher.sender_pathway,
            inputs=sup_images,
            batch_size=p.batch_size,
            input_batch_dim=0,
            output_batch_dim=1
        )
        # self.teacher.train()
        # print('sup_utts_logits.size()', sup_utts_logits.size())
        _, sup_utts = sup_utts_logits.max(dim=-1)
        # print('sup_utts.size()', sup_utts.size())
        print(' done in %.0f seconds' % (time.time() - _gen_start))

        student = Agent(p=p, img_embedding_size=self.img_embedding_size, image_size=self.image_size)

        # and then train each half supervised on this data
        # sender first...
        sup_sender_epochs = 0
        sup_receiver_epochs = 0
        sup_sender_acc = 0
        sup_receiver_acc = 0
        sup_sender_time = 0
        sup_receiver_time = 0
        if (
                (p.sup_acc is not None and p.sup_acc > 0)
                or (p.sup_ksteps is not None and p.sup_ksteps > 0)
            ) and (step > 0 or not p.train_e2e):
            for (agent_str, pathway) in [('send', student.sender_pathway), ('recv', student.receiver_pathway)]:
                # print('sup training on' + agent_str)
                _epoch = 0
                _sup_start = time.time()
                _last_print = time.time()
                sup_stats = Stats([
                    'loss_sum',
                    'acc_sum',
                    'episodes_count',
                ])
                while True:
                    b_idxes = torch.from_numpy(np.random.choice(self.sup_train_N, p.batch_size, replace=False))
                    b_utts = sup_utts[:, b_idxes]
                    b_images = sup_images[b_idxes]
                    if p.enable_cuda:
                        b_utts = b_utts.cuda()
                        b_images = b_images.cuda()
                    b_loss, b_acc = pathway.sup_train_batch(images=b_images, utts=b_utts)
                    sup_stats.episodes_count += 1
                    sup_stats.loss_sum += b_loss
                    sup_stats.acc_sum += b_acc

                    _epoch += 1
                    _done_training = False
                    if p.sup_acc is not None and epoch_acc >= p.sup_acc:
                        # print('done sup training (reason: acc)')
                        _done_training = True
                    if p.sup_ksteps is not None and _epoch >= p.sup_ksteps * 1000:
                        # print('done sup training (reason: steps)')
                        _done_training = True
                    if _done_training or time.time() - _last_print >= 30.0:
                        _elapsed_time = time.time() - _sup_start
                        _loss = sup_stats.loss_sum / sup_stats.episodes_count
                        _acc = sup_stats.acc_sum / sup_stats.episodes_count
                        log_dict = {
                            'record_type': f'sup_{agent_str}',
                            'agent': agent_str,
                            'ilm_epoch': step,
                            'epoch': _epoch,
                            'sps': int(_epoch / _elapsed_time),
                            'sup_time': int(_elapsed_time),
                            'loss': _loss,
                            'acc': _acc,
                        }
                        formatstr = (
                            '{record_type} g={ilm_epoch} e={epoch} '
                            't={sup_time:.0f} '
                            'sps={sps:.0f} '
                            'loss={loss:.3f} '
                            'acc={acc:.3f} '
                        )
                        self.print_and_log(log_dict, formatstr=formatstr)
                        sup_stats.reset()
                        _last_print = time.time()
                    if _done_training:
                        # print('done training for pathway', pathway.__class__.__name__)
                        break
                if pathway == student.sender_pathway:
                    sup_sender_epochs = _epoch
                    sup_sender_acc = _acc
                    sup_sender_time = time.time() - _sup_start
                elif pathway == student.receiver_pathway:
                    sup_receiver_epochs = _epoch
                    sup_receiver_acc = _acc
                    sup_receiver_time = time.time() - _sup_start
                else:
                    raise Exception('invalid pathway value')
            # print('done supervised training')

        e2e_time = 0
        if p.train_e2e:
            # then train end to end for a bit, as decoder-encoder, looking at reconstruction accuracy
            # we'll do this on the same meanings as we got from the teacher? or different ones? or
            # just rnadomly sampled from everything except heldout?
            # maybe train on everything except holdout?
            last_print = time.time()
            _e2e_start = time.time()
            e2e_stats = Stats([
                'episodes_count',
                'e2e_loss_sum',
                'e2e_acc_sum',
                'sender_greedy_sum',
                'receiver_greedy_sum',
            ])
            epoch = 0
            student.train()
            ref_task_game = RefTaskGame(
                sender_pathway=student.sender_pathway, receiver_pathway=student.receiver_pathway, link=link)
            student_params = student.parameters()
            while True:
                images_t, labels = self.dataset.sample_batch(batch_size=p.batch_size)
                if p.enable_cuda:
                    images_t, labels = images_t.cuda(), labels.cuda()

                (sender_greedy, receiver_greedy), _, acc = ref_task_game.forward(images_t, training=True)
                loss, loss_v = ref_task_game.calc_loss(training=True)

                e2e_stats.e2e_loss_sum += loss_v
                e2e_stats.e2e_acc_sum += acc
                e2e_stats.sender_greedy_sum += sender_greedy
                e2e_stats.receiver_greedy_sum += receiver_greedy
                e2e_stats.episodes_count += 1

                student.opt_both.zero_grad()
                loss.backward()
                if p.clip_grad is not None and p.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(student_params, p.clip_grad)
                student.opt_both.step()

                _done_training = False
                if p.e2e_acc is not None and acc >= p.e2e_acc:
                    # print('reached target e2e acc %.3f' % acc, ' => breaking')
                    _done_training = True
                if p.e2e_ksteps is not None and epoch >= p.e2e_ksteps * 1000:
                    # print('reached target e2e step', epoch, ' => breaking')
                    _done_training = True
                save_e2e = p.save_e2e_everyk is not None and p.save_e2e_everyk > 0 and (epoch % (p.save_e2e_everyk * 1000)) == 0
                if time.time() - last_print >= self.render_every_seconds or _done_training or save_e2e:
                    holdout_acc_sum = 0
                    holdout_ep_count = 0
                    holdout_rho_sum = 0
                    holdout_send_greed_sum = 0
                    holdout_recv_greed_sum = 0
                    # self.teacher.lang_sender.eval()
                    student.eval()
                    ho_utts_l = []
                    ho_labels_l = []
                    with torch_random_state():
                        for i, (batch_image, batch_labels) in enumerate(self.dataset.iter_holdout(batch_size=p.batch_size)):
                            if p.enable_cuda:
                                batch_image = batch_image.cuda()
                                batch_labels = batch_labels.cuda()
                            with autograd.no_grad():
                                _, utts, acc = ref_task_game.forward(batch_image, training=False)
                                if utts.dtype == torch.float32:
                                    _, utts = utts.max(dim=-1)
                                holdout_acc_sum += acc
                                holdout_ep_count += 1
                                utts = utts.transpose(0, 1)
                                holdout_rho_sum += metrics.topographic_similarity(utts, batch_labels)
                            ho_utts_l.append(utts)
                            ho_labels_l.append(batch_labels)
                    ho_utts = torch.cat(ho_utts_l)
                    ho_labels = torch.cat(ho_labels_l)
                    if save_e2e:
                        samples_filename = p.utt_samples.format(epoch=epoch)
                        with open(samples_filename, 'wb') as f:
                            torch.save({'samples': {'utts': ho_utts, 'labels': ho_labels}, 'meta': p.__dict__}, f)
                        print('saved samples to ' + samples_filename)
                        model_save = p.model_save.format(epoch=epoch)
                        self.save_to(model_save)
                        print('saved model to ' + model_save)
                    student.train()
                    # self.teacher.lang_sender.train()
                    rho = holdout_rho_sum / holdout_ep_count
                    holdout_acc = holdout_acc_sum / holdout_ep_count
                    acc = e2e_stats.e2e_acc_sum / e2e_stats.episodes_count
                    loss = e2e_stats.e2e_loss_sum / e2e_stats.episodes_count
                    sender_greedy = e2e_stats.sender_greedy_sum / e2e_stats.episodes_count
                    receiver_greedy = e2e_stats.receiver_greedy_sum / e2e_stats.episodes_count

                    _elapsed_time = time.time() - _e2e_start
                    log_dict = {
                        'record_type': 'e2e',
                        'ilm_epoch': step,
                        'epoch': epoch,
                        'sps': int(epoch / _elapsed_time),
                        'e2e_time': int(_elapsed_time),
                        'acc': acc,
                        'holdout_acc': holdout_acc,
                        'rho': rho,
                        'loss': loss,
                        'send_greed': sender_greedy,
                        'recv_greed': receiver_greedy,
                    }
                    formatstr = (
                        '{record_type} '
                        'g={ilm_epoch} '
                        'e={epoch} '
                        't={e2e_time:.0f} '
                        'sps={sps:.0f} '
                        'acc={acc:.3f} '
                        'loss={loss:.3f} '
                        'ho_acc={holdout_acc:.3f} '
                        'rho={rho:.3f} '
                        's_g={send_greed:.3f} '
                        'r_g={recv_greed:.3f} '
                    )
                    self.print_and_log(log_dict, formatstr=formatstr)

                    e2e_stats.reset()
                    last_print = time.time()
                if _done_training:
                    break
                epoch += 1
            e2e_time = time.time() - _e2e_start
            e2e_acc = acc
            e2e_holdout_acc = holdout_acc
            e2e_rho = rho
            e2e_send_greedy = sender_greedy
            e2e_recv_greedy = receiver_greedy

            self.teacher = student

        if True:
            log_dict = {
                'type': 'ilm',
                'sps': int(step / (time.time() - self.start_time)),
                'elapsed_time': time.time() - self.start_time,
                'e2e_time': e2e_time,
                'e2e_acc': e2e_acc,
                'e2e_holdout_acc': e2e_holdout_acc,
                'e2e_rho': e2e_rho,
                'e2e_send_greedy': e2e_send_greedy,
                'e2e_recv_greedy': e2e_recv_greedy,
                'sup_sender_epochs': sup_sender_epochs,
                'sup_receiver_epochs': sup_receiver_epochs,
                'sup_sender_acc': sup_sender_acc,
                'sup_receiver_acc': sup_receiver_acc,
                'sup_sender_time': sup_sender_time,
                'sup_receiver_time': sup_receiver_time,
            }

            formatstr = (
                '{type} '
                'g={training_step} '
                't={elapsed_time:.0f} '
                'sps={sps:.0f}\n'
                '    sup_snd[e={sup_sender_epochs} acc={sup_sender_acc:.3f} t={sup_sender_time:.0f}]\n'
                '    sup_rcv[e={sup_receiver_epochs} acc={sup_receiver_acc:.3f} t={sup_receiver_time:.0f}]\n'
                '    e2e[acc={e2e_acc:.3f} t={e2e_time:.0f} sg={e2e_send_greedy:.3f} rg={e2e_recv_greedy:.3f}]\n'
                'ho_acc={e2e_holdout_acc:.3f} '
                'rho={e2e_rho:.3f} '
            )
            self.print_and_log(log_dict, formatstr=formatstr)
        if p.max_gen is not None and step + 1 >= p.max_gen:
            print('reached max generations', p.max_gen, '=> terminating')
            self.finish = True

if __name__ == '__main__':
    utils.clean_argv()
    runner = Runner()

    runner.add_param('--ds-ref', type=str)
    runner.add_param('-s', type=int, help='num shapes')
    runner.add_param('--opt', type=str, default='RMSprop')
    runner.add_param('--conv-class', type=str, default='CNNALPoolingAll')
    runner.add_param('--num-conv-layers', type=int, default=8)
    runner.add_param('--data-family', type=str, default='objects_gl')
    runner.add_param('--data-dir', type=str, default='~/data/{data_family}/{ds_ref}')
    runner.add_param('--utt-samples', type=str, default='tmp/{ref}_samples_{epoch}.pth')
    runner.add_param('--model-save', type=str, default='tmp/{ref}_model_{epoch}.pth')

    runner.add_param('--seed', type=int)
    runner.add_param('--batch-size', type=int, default=32)
    runner.add_param('--link', type=str, default='RL')
    runner.add_param('--clip-grad', type=float, default=0)
    runner.add_param('--sup-acc', type=float)
    runner.add_param('--e2e-acc', type=float)
    runner.add_param('--ilm', type=str)
    runner.add_param('--max-gen', type=int)
    runner.add_param('-f', type=float, default=0.4, help='supervised train fraction')
    runner.add_param('--no-train-e2e', action='store_true')
    runner.add_param('--save-e2e-everyk', type=int, default=100)

    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--vocab-size', type=int, default=100, help='excludes any terminator')
    runner.add_param('--model', type=str, default='RNN')
    runner.add_param('--rnn-type', type=str, default='GRU')
    runner.add_param('--num-layers', type=int, default=1)
    runner.add_param('--dropout', type=float, default=0.5)
    runner.add_param('--utt-len', type=int, default=6)
    runner.add_param('--nle', type=str, default='2,3', help='negative log10 entropy reg')

    runner.parse_args()
    args = runner.params
    if args.ilm is not None:
        args.e2e_ksteps, args.sup_ksteps = [float(v) for v in args.ilm.split(',')]
    else:
        args.e2e_ksteps, args.sup_ksteps = None, None
    del args.__dict__['ilm']
    args.utt_samples = args.utt_samples.format(ref=args.ref, epoch='{epoch}')
    args.model_save = args.model_save.format(ref=args.ref, epoch='{epoch}')
    args.s_ent, args.r_ent = [math.pow(10, -float(v)) for v in args.nle.split(',')]
    print(f'ent reg {args.s_ent:.1e} {args.r_ent:.1e}')
    del args.__dict__['nle']
    args.shapes = args.s
    del args.__dict__['s']
    args.sup_train_frac = args.f
    del args.__dict__['f']
    if args.shapes is not None:
        args.ds_ref = {
            1: 'dsd39_1sb',
            2: 'dsd37_2s_ho2',
            3: 'dsd38_3s_ho2'
            # 2: 'dsd32_twoshape_123',
            # 3: 'dsd33_threeshapes_123'
        }[args.shapes]
    assert args.ds_ref is not None
    utils.reverse_args(runner.params, 'no_train_e2e', 'train_e2e')
    runner.params.data_dir = runner.params.data_dir.format(**runner.params.__dict__)
    print('runner.params', runner.params)
    runner.setup_base()
    runner.run_base()
