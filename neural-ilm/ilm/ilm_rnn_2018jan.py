"""
do kirby2001, but with an rnn

This file was forked from reprod/ilm/ilm_rnn.py, 2018 jan 26
"""
import argparse
import time
import string
import os
from os import path

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


def Hadamard(one, two):
    if one.size() != two.size():
        print('one.size()', one.size())
        print('two.size()', two.size())
        raise Exception('size mismatch')
    return one * two


class AgentModel(nn.Module):
    """
    the meaning will be two one-hot vectors, concatenated
    """
    def __init__(self, num_meaning_types, meanings_per_type, embedding_size, vocab_size, max_utterance_len):
        super().__init__()
        self.num_meaning_types = num_meaning_types
        self.meanings_per_type = meanings_per_type
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_utterance_len = max_utterance_len

        self.meaning_embeddings = nn.ModuleList()
        for i in range(num_meanings):
            self.meaning_embeddings.append(nn.Embedding(meaning_size, embedding_size))
        # self.rnn = nn.LSTMCell(embedding_size, embedding_size)
        self.rnn = nn.LSTMCell(embedding_size, embedding_size)
        self.enable_cuda = False
        self.torch_constr = torch
        self.e2v = nn.Linear(embedding_size, vocab_size)
        self.v2e = nn.Embedding(vocab_size, embedding_size)

    def cuda(self):
        self.enable_cuda = True
        print('enabled cuda in ', self.__class__)
        self.torch_constr = torch.cuda

    def forward(self, meanings):
        """
        assume that meanings looks like:

        3, 4
        2, 1
        ...

        so need to fluff up, and concatenate

        ... then it will go into rnn state, and be used to generate an utterance


        for now, no stochasticity, and no entropy regularization
        """
        batch_size = meanings.size()[0]
        r = self.torch_constr.Tensor(batch_size, self.embedding_size).zero_()
        for i, l in enumerate(self.meaning_embeddings):
            r = r + l(meanings[:, i])

        state = r
        cell = self.torch_constr.Tensor(batch_size, self.embedding_size).zero_()
        utterances_out = self.torch_constr.Tensor(batch_size, self.max_utterance_len, self.vocab_size).zero_()
        utterances_probs = self.torch_constr.Tensor(batch_size, self.max_utterance_len, self.vocab_size).zero_()
        utterances_argmax = self.torch_constr.LongTensor(batch_size, self.max_utterance_len).zero_()
        last_token = self.torch_constr.LongTensor(batch_size).zero_()
        lens = self.torch_constr.LongTensor(batch_size).fill_(self.max_utterance_len)
        terminated = self.torch_constr.ByteTensor(batch_size).zero_()
        for t in range(self.max_utterance_len):
            token_emb = self.v2e(last_token)
            state, cell = self.rnn(token_emb, (state, cell))
            this_token_logits = self.e2v(state)
            utterances_out[:, t] = this_token_logits
            this_token_probs = F.softmax(this_token_logits)
            utterances_probs[:, t] = this_token_probs
            _, utt_argmax = this_token_probs.max(dim=-1)
            utterances_argmax[:, t] = utt_argmax
            last_token = utt_argmax
            is_terminated_mask = (utt_argmax == 0) & (terminated == 0)
            is_terminated_idxes = is_terminated_mask.view(-1).nonzero().long().view(-1)
            lens[is_terminated_idxes] = t + 1
            terminated[is_terminated_idxes] = 1

        return utterances_out, utterances_probs, utterances_argmax, lens


class Agent(object):
    def __init__(
            self, num_meanings, meaning_size, embedding_size, max_utterance_len, vocab_size,
            target_study_acc):
        self.vocab_size = vocab_size
        self.max_utterance_len = max_utterance_len
        self.target_study_acc = target_study_acc
        self.model = AgentModel(
            num_meanings=num_meanings,
            meaning_size=meaning_size,
            embedding_size=embedding_size,
            max_utterance_len=max_utterance_len,
            vocab_size=vocab_size
        )
        self.torch_constr = torch
        # self.rules = []

    def get_utterances(self, meanings):
        utterances_out, utterances_probs, utterances_argmax, lens = self.model(meanings)
        return utterances_argmax, lens

    def study(self, meanings, utterances, utt_lens):
        batch_size = meanings.size()[0]
        acc = 0
        step = 0
        last_render = time.time()
        while acc < self.target_study_acc:
            pred_out, pred_probs, pred_argmax, pred_lens = self.model(meanings)
            # just directly use teacher forcing loss perhaps?
            # (but taking into account the lengths)
            # (so, using the probs, not the argmax)
            cumsum = self.torch_constr.LongTensor(batch_size, self.max_utterance_len).fill_(1).cumsum(-1) - 1
            # print('cumsum', cumsum)
            expanded_lens = utt_lens.view(-1, 1).expand(batch_size, self.max_utterance_len)
            # print('pred_lens', pred_lens)
            # print('expanded_lens', expanded_lens)
            alive_mask = cumsum < expanded_lens
            total_poss = alive_mask.sum().item()
            match_pred = pred_argmax == utterances
            match_pred = match_pred * alive_mask
            total_right = match_pred.sum().item()
            acc = total_right / total_poss
            # print('acc', acc)

            # print('alive_mask[:5]', alive_mask[:5])
            opt = optim.Adam(lr=0.001, params = self.model.parameters())
            crit = nn.CrossEntropyLoss(reduce=False)
            # loss = crit(pred_out.view(-1, self.vocab_size), utterances.view(-1))
            loss = crit(pred_out.view(-1, self.vocab_size), utterances.view(-1)).view(batch_size, self.max_utterance_len)
            loss = Hadamard(loss, alive_mask.float()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if time.time() - last_render > 3.0:
                print(step, '%.3f' % loss.item(), '%.3f' % acc)
                last_render = time.time()
        print(step, '%.3f' % loss.item(), '%.3f' % acc)
        print('studied in %s steps' % step)


def generate_meanings(batch_size):
    idxes = torch.from_numpy(np.random.choice(5, (batch_size, 2), replace=True))
    # print('idxes', idxes)
    return idxes
    # return dcg.ab(idxes[0].item(), idxes[1].item())


def utt_to_str(utt, utt_len):
    res = ''
    for i in range(utt_len - 1):
        res += string.ascii_lowercase[utt[i] - 1]
    return res


def print_table(params, agent):
    meanings = torch.LongTensor(5 * 5, 2).zero_()
    for a in range(5):
        for b in range(5):
            meanings[b * 5 + a, 0] = a
            meanings[b * 5 + a, 1] = b
    # print('meanings', meanings)
    utterances, utt_lens = agent.get_utterances(meanings)

    for a in range(5):
        line = ''
        for b in range(5):
            utterance = utt_to_str(utterances[b * 5 + a], utt_lens[b * 5 + a].item())
            if utterance is None:
                utterance = '.'
            line += ' ' + utterance.rjust(10)
        print(line)
    print('')


def run_episode(learner, adult, params):
    # meanings = generate_meanings(params=params)
    p = params
    N_meanings_total = int(math.pow(p.meanings_per_type, p.num_meaning_types))
    print('N_meanings_total', N_meanings_total)
    N_train = int(N_meanings_total * p.train_frac)
    N_val = int(N_meanings_total * p.val_frac)
    N_sample = N_train + N_val
    print(N_train, N_val, N_sample)
    meanings_all = torch.from_numpy(np.random.choice(p.meanings_per_type, (N_sample, p.num_meaning_types), replace=False))
    meanings_train = meanings_all[:N_train]
    meanings_val = meanings_all[N_train:]
    # if self.enable_cuda:
    #     meanings = meanings.cuda()

    # print('meaning', meaning)
    utterances = adult.get_utterances(meanings_train)
    # print('meaning', meaning, 'utterance', utterance)
    learner.study(meanings_train, utterances)
    # asdf
    # print_table(adult)
    # print(learner.rules)
    # print_table(learner.rules)
    # if step > 1:
    #     asdfs


def run(params):
    p = params
    agent_params = {k: args.__dict__[k] for k in [
        'num_meaning_types', 'meanings_per_type', 'embedding_size', 'utt_len', 'vocab_size',
        'target_study_acc']}

    learner = Agent(**agent_params)
    adult = Agent(**agent_params)

    episode = 0
    while True:
        print('episode', episode)
        run_episode(learner, adult, params=args)
        # print_table(adult.rules)
        print_table(params=params, learner=learner)
        adult = learner
        learner = Agent(**agent_params)
        # asdf
        episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-study-acc', type=float, default=0.8)
    parser.add_argument('--train-frac', type=float, default=0.5)
    parser.add_argument('--val-frac', type=float, default=0.5)
    parser.add_argument('--meanings', type=str, default='5x10')
    parser.add_argument('--embedding-size', type=int, default=50)
    parser.add_argument('--utt-len', type=int, default=20)
    parser.add_argument('--vocab-size', type=int, default=4)
    args = parser.parse_args()

    args.num_meaning_types, args.meanings_per_type = [int(v) for v in args.meanings.split('x')]
    del args.__dict__['meanings']

    run(params=args)
