"""
Various latent hypotheses, that we can use to generate data from
"""
import numpy as np
import torch

from texrel import things, relations
from texrel.grid import Grid


class Hypothesis(object):
    def create_example(self, label, grid_size, num_distractors):
        """
        label is 1 for positive example, or 0 for negative example
        """
        if label == 1:
            return self.create_positive_example(grid_size=grid_size, num_distractors=num_distractors)
        elif label == 0:
            return self.create_negative_example(grid_size=grid_size, num_distractors=num_distractors)
        else:
            raise Exception('unhandled label', label)


class RelationHypothesis(Hypothesis):
    """
    represents a single relations, and
    generates examples of this relation
    """
    def __init__(self, relation, rel_space, distractor_thing_space):
        self.r = relation
        self.rel_space = rel_space
        self.thing_space = rel_space.thing_space
        self.distractor_thing_space = distractor_thing_space

    def __str__(self):
        return 'relation: ' + str(self.r)

    def __eq__(self, b):
        return self.r == b.r

    def as_seq(self):
        idxes, types = self.r.as_indices(rel_space=self.rel_space)
        seq = torch.LongTensor(idxes)
        return seq, types

    @classmethod
    def from_seq(cls, seq, rel_space, distractor_thing_space):
        return cls(
            relation=relations.Relation.eat_from_indices(indices=seq, rel_space=rel_space)[0],
            rel_space=rel_space,
            distractor_thing_space=distractor_thing_space
    )

    def _create_example(self, r, grid_size, num_distractors):
        # r = self.r
        grid = Grid(size=grid_size)

        grid_size = grid.size
        thing_space = self.thing_space

        first_h_range = [0, grid_size]
        first_w_range = [0, grid_size]
        second_h_range = [0, grid_size]
        second_w_range = [0, grid_size]
        if isinstance(r.prep, relations.RightOf):
            # eg first rightof second
            # means some constraints on the x range of each
            # eg if second is in right-most column, impossible for first to fit somewhere
            first_w_range = [1, grid_size]
        elif isinstance(r.prep, relations.LeftOf):
            first_w_range = [0, grid_size - 1]
        elif isinstance(r.prep, relations.Above):
            # ie first above second
            # assume h is downwards
            first_h_range = [0, grid_size - 1]
        elif isinstance(r.prep, relations.Below):
            first_h_range = [1, grid_size]
        elif r.prep.__class__ in [
                relations.NotAbove, relations.NotBelow,
                relations.NotRightOf, relations.NotLeftOf,
                relations.HorizSame, relations.VertSame]:
            # no constraints in fact; first can go anywhere
            pass    
        else:
            raise Exception('preposition not handled ' + str(r.prep))

        h1 = np.random.randint(*first_h_range)
        w1 = np.random.randint(*first_w_range)

        # now figure out second ranges
        if isinstance(r.prep, relations.RightOf):
            # eg first rightof second
            # means some constraints on the x range of each
            # eg if second is in right-most column, impossible for first to fit somewhere
            second_w_range = [0, w1]
        elif isinstance(r.prep, relations.LeftOf):
            second_w_range = [w1 + 1, grid_size]
        elif isinstance(r.prep, relations.Above):
            # ie first above second
            # assume h is downwards
            second_h_range = [h1 + 1, grid_size]
        elif isinstance(r.prep, relations.Below):
            second_h_range = [0, h1]
        elif isinstance(r.prep, relations.NotRightOf):
            # eg first not rightof second
            second_w_range = [w1, grid_size]
        elif isinstance(r.prep, relations.NotLeftOf):
            second_w_range = [0, w1 + 1]
        elif isinstance(r.prep, relations.NotAbove):
            # ie first not above second
            # assume h is downwards
            second_h_range = [0, h1 + 1]
        elif isinstance(r.prep, relations.NotBelow):
            second_h_range = [h1, grid_size]
        elif isinstance(r.prep, relations.VertSame):
            second_w_range = [w1, w1 + 1]
        elif isinstance(r.prep, relations.HorizSame):
            second_h_range = [h1, h1 + 1]
        else:
            raise Exception('preposition not handled ' + str(r.prep))
        h2 = h1
        w2 = w1
        while h2 == h1 and w2 == w1:
            h2 = np.random.randint(*second_h_range)
            w2 = np.random.randint(*second_w_range)

        o1 = r.left
        o2 = r.right

        grid.add_object([h1, w1], o1)
        grid.add_object([h2, w2], o2)

        distractors = []
        for i in range(num_distractors):
            o = None
            while o is None or o in grid.objects_set:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)

        return grid

    def create_positive_example(self, grid_size, num_distractors):
        return self._create_example(r=self.r, grid_size=grid_size, num_distractors=num_distractors)

    def create_negative_example(self, grid_size, num_distractors):
        return self._create_example(r=self.r.complement(), grid_size=grid_size, num_distractors=num_distractors)


class ColorPairHypothesis(Hypothesis):
    def __init__(self, thing_space, distractor_thing_space):
        self.thing_space = thing_space
        self.distractor_thing_space = distractor_thing_space

    def __str__(self):
        return 'color-pair'

    def __eq__(self, b):
        return self.__class__ == b.__class__

    def as_seq(self):
        seq = torch.zeros(0, dtype=torch.int64)
        return seq, []

    @classmethod
    def from_seq(cls, seq, thing_space, distractor_thing_space):
        return cls()

    def create_positive_example(self, grid_size, num_distractors):
        """
        so a positive example means two objects of the same color
        (but possibly different shapes)
        """
        grid = Grid(size=grid_size)
        thing_space = self.thing_space
        shape_space = thing_space.shape_space
        pass

    def create_negative_example(self, grid_size, num_distractors):
        pass


class SingleColorHypothesis(Hypothesis):
    def __init__(self, color, thing_space, distractor_thing_space):
        self.color = color
        self.thing_space = thing_space
        self.distractor_thing_space = distractor_thing_space

    def __str__(self):
        return 'has-color: ' + str(self.color)

    def __eq__(self, b):
        return self.color == b.color

    def as_seq(self):
        seq = torch.zeros(1, dtype=torch.int64)
        seq[0] = self.color
        return seq, ['C']

    @classmethod
    def from_seq(cls, seq, thing_space, distractor_thing_space):
        assert len(seq.size()) == 1
        assert seq.size(0) == 1
        color_id = seq[0].item()
        # return color_id
        # print('color_id', color_id)
        # color = thing_space.color_space.colors[color_id]()
        return cls(color_id, thing_space=thing_space, distractor_thing_space=distractor_thing_space)

    def create_positive_example(self, grid_size, num_distractors):
        grid = Grid(size=grid_size)
        thing_space = self.thing_space
        shape_space = thing_space.shape_space
        o = None
        if len(thing_space) > 0.4 * len(thing_space.color_space) * len(thing_space.shape_space):
            while o is None or not thing_space.is_valid(o):
                shape = shape_space.sample()
                o = things.ShapeColor(shape=shape, color=self.color)
        else:
            while o is None or o.color != self.color:
                o = thing_space.sample()
                # print('o', o, 'self.color', self.color)
        pos = grid.generate_available_pos()
        grid.add_object(pos=pos, o=o)

        distractors = []
        for i in range(num_distractors):
            o = None
            while o is None or o.color == self.color:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid

    def create_negative_example(self, grid_size, num_distractors):
        """
        in fact everything is just a distractor...
        """
        grid = Grid(size=grid_size)
        distractors = []
        for i in range(num_distractors + 1):
            o = None
            while o is None or o.color == self.color:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid


class SingleShapeHypothesis(Hypothesis):
    def __init__(self, shape, thing_space, distractor_thing_space):
        self.shape = shape
        self.thing_space = thing_space
        self.distractor_thing_space = distractor_thing_space

    def __str__(self):
        return 'has-shape: ' + str(self.shape)

    def __eq__(self, b):
        return self.shape == b.shape

    def as_seq(self):
        seq = torch.zeros(1, dtype=torch.int64)
        seq[0] = self.shape
        return seq, ['S']

    @classmethod
    def from_seq(cls, seq, thing_space, distractor_thing_space):
        assert len(seq.size()) == 1
        assert seq.size(0) == 1
        shape_id = seq[0].item()
        # return shape_id
        # shape = thing_space.shape_space.shapes[shape_id]()
        return cls(shape_id, thing_space=thing_space, distractor_thing_space=distractor_thing_space)

    def create_positive_example(self, grid_size, num_distractors):
        grid = Grid(size=grid_size)
        thing_space = self.thing_space
        color_space = thing_space.color_space
        o = None
        while o is None or not thing_space.is_valid(o):
            color = color_space.sample()
            o = things.ShapeColor(shape=self.shape, color=color)
        pos = grid.generate_available_pos()
        grid.add_object(pos=pos, o=o)

        distractors = []
        for i in range(num_distractors):
            o = None
            while o is None or o.shape == self.shape:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid

    def create_negative_example(self, grid_size, num_distractors):
        """
        in fact everything is just a distractor...
        """
        grid = Grid(size=grid_size)
        distractors = []
        for i in range(num_distractors + 1):
            o = None
            while o is None or o.shape == self.shape:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid

class SingleThingHypothesis(Hypothesis):
    def __init__(self, o, distractor_thing_space):
        self.distractor_thing_space = distractor_thing_space
        # self.thing_space = thing_space
        self.o = o

    def __str__(self):
        return 'has-thing: ' + str(self.o)

    def __eq__(self, b):
        return self.o == b.o

    def as_seq(self):
        seq, types = self.o.as_indices()
        seq = torch.LongTensor(seq)
        return seq, types

    @classmethod
    def from_seq(cls, seq, thing_space, distractor_thing_space):
        o = things.ShapeColor.eat_from_indices(seq)[0]
        return SingleThingHypothesis(o=o, distractor_thing_space=distractor_thing_space)

    def create_positive_example(self, grid_size, num_distractors):
        grid = Grid(size=grid_size)
        # thing_space = self.thing_space
        pos = grid.generate_available_pos()
        grid.add_object(pos=pos, o=self.o)

        distractors = []
        for i in range(num_distractors):
            o = None
            while o is None or o == self.o:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid

    def create_negative_example(self, grid_size, num_distractors):
        """
        in fact everything is just a distractor...
        """
        grid = Grid(size=grid_size)
        distractors = []
        for i in range(num_distractors + 1):
            o = None
            while o is None or o == self.o:
                o = self.distractor_thing_space.sample()
            distractors.append(o)
        for i, o in enumerate(distractors):
            pos = grid.generate_available_pos()
            grid.add_object(pos, o)
        return grid



class HypothesisGenerator(object):
    pass

class RelationHG(HypothesisGenerator):
    def __init__(self, rel_space, distractor_thing_space):
        self.rel_space = rel_space
        self.distractor_thing_space = distractor_thing_space

    def __call__(self):
        """
        generates a single RelationHypothesis
        """
        r = self.rel_space.sample()
        return  RelationHypothesis(
            relation=r,
            rel_space=self.rel_space,
            distractor_thing_space=self.distractor_thing_space
        )

class SingleColorHG(HypothesisGenerator):
    def __init__(self, thing_space, distractor_thing_space):
        self.thing_space = thing_space
        self.color_space = thing_space.color_space
        self.distractor_thing_space = distractor_thing_space

    def __call__(self):
        # color = self.color_space.sample()
        o = self.thing_space.sample()
        color = o.color
        h = SingleColorHypothesis(
            thing_space=self.thing_space,
            distractor_thing_space=self.distractor_thing_space,
            color=color
        )
        return h


class ColorPairHG(HypothesisGenerator):
    """
    the rule is that if there is a pair of colors in the input,
    then there should be a pair in the output (ie: analogy)
    the colors in the input, and the colors in the output dont have to be
    the same
    so a question is: should the color pairs for the sender be identical
    colors to each other, or also different colors?
    perhaps: different colors

    I think this rule/hypothesis only makes sense when combined with
    some other rule/hypothesis, eg ShapePair

    in a sense, this HG only has a single instance, which generates
    pairs of colors, different for each inner example....
    """
    def __init__(self, thing_space, distractor_thing_space):
        self.thing_space = thing_space
        self.color_space = thing_space.color_space
        self.distractor_thing_space = distractor_thing_space

    def __call__(self):
        # color = self.color_space.sample()
        # o = self.thing_space.sample()
        # color = o.color
        h = ColorPairHypothesis(
            thing_space=self.thing_space,
            distractor_thing_space=self.distractor_thing_space
        )
        return h


class SingleShapeHG(HypothesisGenerator):
    def __init__(self, thing_space, distractor_thing_space):
        self.thing_space = thing_space
        self.shape_space = thing_space.shape_space
        self.distractor_thing_space = distractor_thing_space

    def __call__(self):
        o = self.thing_space.sample()
        # shape = self.shape_space.sample()
        shape = o.shape
        h = SingleShapeHypothesis(
            thing_space=self.thing_space,
            distractor_thing_space=self.distractor_thing_space,
            shape=shape
        )
        return h


class SingleThingHG(HypothesisGenerator):
    def __init__(self, thing_space, distractor_thing_space):
        self.thing_space = thing_space
        self.distractor_thing_space = distractor_thing_space

    def __call__(self):
        o = self.thing_space.sample()
        h = SingleThingHypothesis(
            # thing_space=self.thing_space,
            distractor_thing_space=self.distractor_thing_space,
            o=o
        )
        return h
