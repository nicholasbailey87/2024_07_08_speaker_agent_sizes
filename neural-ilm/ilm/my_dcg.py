"""
simple dcg, just sufficient for kirby 2001


examples of things we need to conceptually handle:

A: a1 --> 'abc'
A: a2 --> 'abc'
B: b1 --> 'abc'
C: a1,b2 --> 'abc'
D: x,b2 -> E:x'abc'

we'll use a(None) and b(None) to represent the variable arguments such as x,y
I'm *fairly* sure that the variables are typed, ie a or b

"""
import argparse
import sys
import os
from os import path


class Meaning(object):
    def matches_arg(self, second):
        if self.__class__ == second.__class__ and \
                self.i == second.i:
            return True
        return False

    def needs_substitution(self):
        return self.i is None

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.i)

    def __eq__(self, second):
        if self.__class__ != second.__class__:
            return False
        return self.i == second.i


class a(Meaning):
    def __init__(self, i=None):
        self.i = i

    def substitute_from(self, non_terminal):
        assert non_terminal.arg.ai is not None
        return a(non_terminal.arg.ai)
        # self.i = non_terminal.arg.ai


class b(Meaning):
    def __init__(self, i=None):
        self.i = i

    def substitute_from(self, non_terminal):
        assert non_terminal.arg.bi is not None
        # self.i = non_terminal.arg.bi
        return b(non_terminal.arg.bi)


class ab(Meaning):
    def __init__(self, a=None, b=None):
        # assert a is not None or b is not None
        self.ai = a
        self.bi = b

    def matches_arg(self, second):
        if self.__class__ != second.__class__:
            return False
        if self.ai == second.ai and self.bi == second.bi:
            return True
        if self.ai is None and self.bi == second.bi:
            return True
        if self.bi is None and self.ai == second.ai:
            return True
        if self.ai is None and self.bi is None:
            return True

        return False

    def __repr__(self):
        return '%s(%s,%s)' % (self.__class__.__name__, self.ai, self.bi)

    def __eq__(self, second):
        if self.__class__ != second.__class__:
            return False
        return self.ai == second.ai and self.bi == second.bi


class Rule(object):
    def __init__(self, category, arg, out):
        self.category = category
        self.arg =  arg
        self.out = out

    def __repr__(self):
        return '%s %s %s' % (self.category, self.arg, self.out)

    def __eq__(self, second):
        return self.category == second.category and \
            self.arg == second.arg and \
            self.out == second.out


def make_rule(category, arg, out):
    return Rule(category=category, arg=arg, out=out)


class NonTerminal(object):
    def __init__(self, category, arg):
        self.category = category
        self.arg = arg

    def __repr__(self):
        return 'NonTerminal(' + self.category +',' + str(self.arg) + ')'

    def __eq__(self, second):
        if not isinstance(second, self.__class__):
            return False
        return self.category == second.category and \
            self.arg == second.arg

    def substitute_from(self, second):
        return NonTerminal(self.category, self.arg.substitute_from(second))


def translate(rules, non_terminal):
    assert isinstance(non_terminal, NonTerminal)
    # print('translate    non_terminal', non_terminal)
    category = non_terminal.category
    meaning = non_terminal.arg
    for rule in rules:
        skip_rule = False
        if rule.category == category and rule.arg.matches_arg(meaning):
            # print('match', rule)
            out_l = []
            out = rule.out
            for item in out:
                if isinstance(item, str):
                    out_l.append(item)
                elif isinstance(item, NonTerminal):
                    if item.arg.needs_substitution():
                        item = item.substitute_from(non_terminal)
                    res = translate(rules, item)
                    if res is None:
                        skip_rule = True
                        break
                        # return None
                    out_l.append(res)
                else:
                    raise Exception('not handled %s' % str(item))
            # return str(rule) + ' => ' + ''.join(out_l)
            if skip_rule:
                continue
            return ''.join(out_l)
    return None
    # raise Exception('no rule found')


def get_output(rules, meaning):
    return translate(rules, NonTerminal(category='S', arg=meaning))


# def parse(rules, meaning, sentence):
#     matched = []
#     for rule in rules:
#         if rule.out
