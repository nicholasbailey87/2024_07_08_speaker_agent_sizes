import string

import numpy as np

from my_dcg import *
import my_dcg as dcg
import dcg_generalization as dcg_gen


def create_word(max_len):
    _len = np.random.choice(max_len, 1).item() + 1
    idxes = list(np.random.choice(26, _len, replace=True))
    word = ''.join([string.ascii_lowercase[i] for i in idxes])
    # print('word', word)
    return word


class Agent(object):
    def __init__(self):
        self.rules = []

    def get_utterance(self, meaning):
        utterance = dcg.get_output(self.rules, meaning)
        if utterance is None:
            utterance = self.invent_word()
            self.rules.append(
                dcg.Rule(category='S', arg=meaning, out=[utterance])
            )
        # print('utterance', utterance)
        return utterance

    def invent_word(self):
        if len(self.rules) == 0:
            return create_word(max_len=10)
        return create_word(max_len=10)

    def study(self, meaning, utterance):
        pred = dcg.get_output(self.rules, meaning)
        if pred == utterance:
            return   # all good :)
        self.rules.append(dcg.Rule(
            category='S',
            arg=meaning,
            out=[utterance]
        ))
        # print('self.rules', self.rules)
        self.rules, _ = dcg_gen.generalize(self.rules)
        # print('self.rules', self.rules)


def generate_meaning():
    idxes = np.random.choice(5, 2, replace=True)
    return dcg.ab(idxes[0].item(), idxes[1].item())


def print_table(rules):
    for a in range(5):
        line = ''
        for b in range(5):
            utterance = dcg.get_output(rules, dcg.ab(a, b))
            if utterance is None:
                utterance = '.'
            line += ' ' + utterance.rjust(10)
        print(line)
    print('')


def run_episode(learner, adult, num_steps=50):
    for step in range(num_steps):
        meaning = generate_meaning()
        # print('meaning', meaning)
        utterance = adult.get_utterance(meaning)
        # print('meaning', meaning, 'utterance', utterance)
        learner.study(meaning, utterance)
        # print_table(adult)
        # print(learner.rules)
        # print_table(learner.rules)
        # if step > 1:
        #     asdfs


def run():
    learner = Agent()
    adult = Agent()

    episode = 0
    while True:
        print('episode', episode)
        run_episode(learner, adult)
        # print_table(adult.rules)
        print_table(learner.rules)
        adult = learner
        learner = Agent()
        # asdf
        episode += 1


if __name__ == '__main__':
    run()
