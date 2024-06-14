import random
import math

import numpy as np

import torch

from ulfs import gru_utils


FACTOR_RELU = 1.43


# class UniformInitializer(object):
#     def __init__(self, lower, upper):
#         self.lower = lower
#         self.upper = upper

#     def apply(self, tensor):
#         tensor.uniform_(self.lower, self.upper)


# class SusilloInitializer(object):
#     def __init__(self, factor=1, input_size=input_size):
#         # self.input_size = input_size
#         self.factor = factor

#     def apply(self, tensor, input_size):
#         rng = math.sqrt(3) / math.sqrt(input_size) * self.factor
#         tensor.uniform_(-rng, rng)


def susillo_init_linear(linear, factor=1):
    input_size = linear.weight.size(1)
    linear.bias.data[:] = 0
    rng = math.sqrt(3) / math.sqrt(input_size) * factor
    linear.weight.data.uniform_(-rng, rng)


def susillo_init_embedding(embedding, factor=1):
    input_size = embedding.weight.size(0)
    rng = math.sqrt(3) / math.sqrt(input_size) * factor
    embedding.weight.data.uniform_(-rng, rng)


def susillo_initialize_gru_reset_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_ir(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hr(gru_cell).data.uniform_(-rng, rng)


def susillo_initialize_gru_update_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_iz(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hz(gru_cell).data.uniform_(-rng, rng)


def susillo_initialize_gru_candidate_weight(gru_cell, factor, num_inputs):
    rng = math.sqrt(3) / math.sqrt(num_inputs) * factor
    gru_utils.get_gru_weight_in(gru_cell).data.uniform_(-rng, rng)
    gru_utils.get_gru_weight_hn(gru_cell).data.uniform_(-rng, rng)


def constant_initialize_gru_reset_bias(gru_cell, value):
    gru_utils.get_gru_bias_ir(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hr(gru_cell).data.fill_(value)


def constant_initialize_gru_update_bias(gru_cell, value):
    gru_utils.get_gru_bias_iz(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hz(gru_cell).data.fill_(value)


def constant_initialize_gru_candidate_bias(gru_cell, value):
    gru_utils.get_gru_bias_in(gru_cell).data.fill_(value)
    gru_utils.get_gru_bias_hn(gru_cell).data.fill_(value)


def init_gru_cell(gru_cell):
    hidden_size = gru_cell.bias_hh.data.size(0) // 3
    input_size = gru_cell.weight_ih.data.size(1)
    print('gru input_size', input_size, 'hidden_size', hidden_size)

    susillo_initialize_gru_reset_weight(gru_cell, factor=1, num_inputs=input_size + hidden_size)
    susillo_initialize_gru_update_weight(gru_cell, factor=1, num_inputs=input_size + hidden_size)
    susillo_initialize_gru_candidate_weight(gru_cell, factor=1, num_inputs=(hidden_size * 3) // 2)

    constant_initialize_gru_reset_bias(gru_cell, value=1)
    constant_initialize_gru_update_bias(gru_cell, value=1)
    constant_initialize_gru_candidate_bias(gru_cell, value=0)


# class SusilloInitializer(object):
#     def __init__(self, factor=1):
#         # self.input_size = input_size
#         self.factor = factor

#     def apply(self, tensor, input_size):
#         rng = math.sqrt(3) / math.sqrt(input_size) * self.factor
#         tensor.uniform_(-rng, rng)


# class ConstantInitializer(object):
#     def __init__(self, constant=0):
#         self.constant = constant

#     def apply(self, tensor, input_size):
#         tensor.fill_(self.constant)


# def initialize_linear(factor, linear, weight_initializer):
#     input_size = linear.weight.data.size(1)
#     initializer.apply(linear=linear, input_size=input_size)


    # def apply_to_linear(self, linear):
    #     in_size = linear.weight.data.size(1)
    #     linear.bias.data[:] = 0
    #     rng = math.sqrt(3) / math.sqrt(input_size) * factor
    #     linear.weight.data.uniform_(-rng, rng)

    # def apply_to_embedding(self, embedding):
    #     in_size = embedding.weight.data.size(0)
    #     rng = math.sqrt(3) / math.sqrt(input_size) * factor
    #     embedding.weight.data.uniform_(-rng, rng)

    # def apply_to_tensor(self, tensor, input_dim=None, input_size=None):
    #     """
    #     Use input_dim to get input_size, or use input_size
    #     one or the other must be given
    #     """
    #     # at least one must be given:
    #     assert input_dim is not None or input_size is not None
    #     # cant both be given (so one must not be given):
    #     assert input_dim is None or input_size is None

    #     if input_size is None:
    #         input_size = tensor.size(input_dim)
    #     rng = math.sqrt(3) / math.sqrt(input_size) * factor
    #     tensor.uniform_(-rng, rng)
