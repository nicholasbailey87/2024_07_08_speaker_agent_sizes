"""
test of drawing relations from relation space
"""
import torch
import numpy as np

from texrel import relations, things


def test_prepositions():
    prep_space = relations.PrepositionSpace()
    for i in range(5):
        print(prep_space.sample())
    print('')

    prep_space = relations.PrepositionSpace(available_preps=['Above', 'RightOf'])
    for i in range(5):
        print(prep_space.sample())
    print('')


def test_relations():
    rel_space = relations.RelationSpace(things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9)))
    for i in range(5):
        print(rel_space.sample())
    print('')

    color_space = things.ColorSpace(num_colors=2)
    shape_space = things.ShapeSpace(num_shapes=2)
    thing_space = things.ThingSpace(color_space=color_space, shape_space=shape_space)
    prep_space = relations.PrepositionSpace(available_preps=['Above', 'RightOf'])
    rel_space = relations.RelationSpace(thing_space=thing_space, prep_space=prep_space)
    for i in range(5):
        print(rel_space.sample())
    print('')


def test_complements():
    print('')
    rel_space = relations.RelationSpace(things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9)))
    rels = []
    for i in range(5):
        r = rel_space.sample()
        rels.append(r)
        print(r)
    print('')

    comps = []
    print('comps:')
    for r in rels:
        comp = r.complement()
        print(comp)
        comps.append(comp)
        assert comp != r
    print('')

    comp2s = []
    print('comp2s:')
    for i, comp in enumerate(comps):
        comp2 = comp.complement()
        print(comp2)
        comp2s.append(comp2)
        assert comp2 != comp
        assert comp2 == rels[i]
    print('')


def test_preps_encode_decode():
    prep_space = relations.PrepositionSpace()
    for i in range(10):
        p = prep_space.sample()
        p_indices = p.as_onehot_indices(prep_space)
        p_onehot = torch.Tensor(p.as_onehot_tensor_size(prep_space)).zero_()
        p_onehot[torch.LongTensor(p_indices)] = 1
        p2 = relations.Preposition.from_onehot_tensor(prep_space, p_onehot)
        print('p', p, 'p2', p2)
        assert p2 == p


def test_relations_encode_decode():
    rel_space = relations.RelationSpace(things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9)))
    for i in range(10):
        r = rel_space.sample()
        r_indices = r.as_onehot_indices(rel_space)
        r_onehot = torch.Tensor(r.as_onehot_tensor_size(rel_space)).zero_()
        r_onehot[torch.LongTensor(r_indices)] = 1
        r2 = relations.Relation.from_onehot_tensor(rel_space, r_onehot)
        print('r', r, 'r2', r2)
        assert r2 == r


def test_relations_encode_decode2():
    rel_space = relations.RelationSpace(things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9)))
    for i in range(10):
        r = rel_space.sample()
        r_onehot = r.encode_onehot(rel_space=rel_space)
        r2 = relations.Relation.from_onehot_tensor(rel_space, r_onehot)
        print('r', r, 'r2', r2)
        assert r2 == r


def test_relations_encode3():
    rel_space = relations.RelationSpace(things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9)))
    for i in range(10):
        r = rel_space.sample()
        indices = r.as_indices(rel_space=rel_space)
        print('indices', indices)
        r2 = relations.Relation.eat_from_indices(rel_space=rel_space, indices=indices[0])[0]
        print('r2.indices', r2.as_indices(rel_space=rel_space))
        print('r', r, 'r2', r2)
        assert r == r2
