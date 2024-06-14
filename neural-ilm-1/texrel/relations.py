"""
definition of relations, and factory to create them
"""
import torch
import numpy as np

from texrel import things


# this should be factorized really...
def class_name_to_name(class_name):
    case_transitions = []
    class_name = class_name.replace('_', '-')
    for i, c in enumerate(class_name):
        if c != class_name[i].lower():
            case_transitions.append(i)
    name = ''
    for i, case_transition in enumerate(case_transitions):
        if i > 0:
            name += '-'
        if i == len(case_transitions) - 1:
            segment = class_name[case_transition:]
        else:
            segment = class_name[case_transition:case_transitions[i + 1]]
        name += segment.lower()
    name = name.replace('--', '-')
    return name


class Preposition(object):
    hash_by_class = {}

    def __repr__(self):
        return class_name_to_name(self.__class__.__name__)

    def complement(self):
        complement_class = prep_complements[self.__class__]
        return complement_class()

    def __eq__(self, second):
        return self.__class__ == second.__class__

    def id(self, prep_space):
        return prep_space.id_by_preposition[self.__class__]

    def __hash__(self):
        if self.__class__ in self.hash_by_class:
            return self.hash_by_class[self.__class__]
        self.hash_by_class[self.__class__] = len(self.hash_by_class)
        return self.hash_by_class[self.__class__]

    def as_onehot_indices(self, prep_space):
        return [self.id(prep_space=prep_space)]

    def as_indices(self, prep_space):
        return [self.id(prep_space=prep_space)], ['P']

    def as_onehot_tensor_size(self, prep_space):
        return prep_space.onehot_size

    @classmethod
    def eat_from_indices(cls, prep_space, indices):
        prep = None
        id = indices[0]
        indices = indices[1:]
        prep = prep_space.prepositions[id]()
        return prep, indices

    @classmethod
    def from_onehot_tensor(cls, prep_space, tensor):
        return cls.eat_from_onehot_tensor(prep_space, tensor)[0]

    @classmethod
    def eat_from_onehot_tensor(cls, prep_space, tensor):
        """
        returns tensor with shapcecolor removed from front
        """
        prep = None
        id = tensor.view(-1).nonzero().view(-1)[0]
        this_len = prep_space.onehot_size
        prep = prep_space.prepositions[id]()
        tensor = tensor[this_len:]
        return prep, tensor


class Above(Preposition):
    pass

class Below(Preposition):
    pass

class RightOf(Preposition):
    pass

class LeftOf(Preposition):
    pass

class HorizSame(Preposition):
    pass

class VertSame(Preposition):
    pass

class NotHorizSame(Preposition):
    pass

class NotVertSame(Preposition):
    pass

class NextTo(Preposition):
    pass

class FarFrom(Preposition):
    pass

class SameColorAs(Preposition):
    pass

class SameShapeAs(Preposition):
    pass


_prepositions = [Above, Below, RightOf, LeftOf]
"""
IMPORTANT: not actually complements as of 28th August, more like ... opposites
(context where they are used: negative examples)
I think it's safe to say that r.complement().complement() == r
but it is NOT true that r union r.complement() == all possible arrangements
"""
prep_complements = {
    Above: Below,
    Below: Above,
    RightOf: LeftOf,
    LeftOf: RightOf
}
# print('_prepositions', _prepositions)


class Relation(object):
    def __init__(self, left, prep, right):
        self.left = left
        self.prep = prep
        self.right = right
        self.torch_constr = torch

    def __eq__(self, second):
        return self.left == second.left and \
            self.prep == second.prep and \
            self.right == second.right

    def complement(self):
        # return Relation(
        #     left=self.left,
        #     prep = self.prep.complement(),
        #     right=self.right
        # )
        return Relation(
            left=self.right,
            prep = self.prep,
            right=self.left
        )

    def __repr__(self):
        return \
            str(self.left) + ' ' + \
            str(self.prep) + ' ' + \
            str(self.right)

    def as_onehot_indices(self, rel_space):
        """
        these indices can be used to create a onehot, since later indices
        are offset
        """
        thing_space = rel_space.thing_space
        color_space = thing_space.color_space
        shape_space = thing_space.shape_space
        prep_space = rel_space.prep_space

        left_indices = self.left.as_onehot_indices(thing_space=thing_space)
        prep_indices = self.prep.as_onehot_indices(prep_space=prep_space)
        right_indices = self.right.as_onehot_indices(thing_space=thing_space)

        indices = []
        indices += left_indices
        indices += [i + thing_space.onehot_size for i in prep_indices]
        indices += [i + thing_space.onehot_size + prep_space.onehot_size for i in right_indices]
        return indices

    def as_indices(self, rel_space):
        """
        these indices are not offset (cf as_onehot_indices)
        """
        thing_space = rel_space.thing_space
        prep_space = rel_space.prep_space

        left_indices, thing_types = self.left.as_indices()
        prep_indices, prep_types = self.prep.as_indices(prep_space=prep_space)
        right_indices, _ = self.right.as_indices()
        return left_indices + prep_indices + right_indices, thing_types + prep_types + thing_types

    @classmethod
    def eat_from_indices(self, rel_space, indices):
        thing_space = rel_space.thing_space
        prep_space = rel_space.prep_space

        left, indices = things.ShapeColor.eat_from_indices(indices=indices)
        prep, indices = Preposition.eat_from_indices(prep_space=prep_space, indices=indices)
        right, indices = things.ShapeColor.eat_from_indices(indices=indices)
        r = Relation(left=left, right=right, prep=prep)
        return r, indices

    def as_onehot_tensor_size(self, rel_space):
        return rel_space.onehot_size

    @classmethod
    def from_onehot_tensor(cls, rel_space, tensor):
        return cls.eat_from_onehot_tensor(rel_space=rel_space, tensor=tensor)[0]

    @classmethod
    def eat_from_onehot_tensor(cls, rel_space, tensor):
        """
        returns tensor with shapcecolor removed from front
        """
        relation_t = tensor
        thing_space = rel_space.thing_space
        prep_space = rel_space.prep_space
        assert len(relation_t.size()) == 1

        left, relation_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, relation_t)
        prep, relation_t = Preposition.eat_from_onehot_tensor(prep_space, relation_t)
        right, relation_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, relation_t)
        r = Relation(left=left, right=right, prep=prep)

        return r, relation_t

    def encode_onehot(self, rel_space):
        """
        so, we'll have one-hot for first color, first shape, preposition
        second color, and second shape; and just stack those up

        we only need to consider a single example/relation
        """
        size = rel_space.onehot_size
        res = self.torch_constr.FloatTensor(size).zero_()
        indices = self.as_onehot_indices(rel_space=rel_space)
        res[indices] = 1
        return res


class PrepositionSpace(object):
    def __init__(self, available_preps=None):
        self.prepositions = _prepositions
        if available_preps is not None:
            self.prepositions = []
            for prep in available_preps:
                self.prepositions.append(globals()[prep])
        self.id_by_preposition = {preposition: i for i, preposition in enumerate(self.prepositions)}
        self.num_prepositions = len(self.prepositions)
        self.size = self.num_prepositions
        self.onehot_size = self.num_prepositions
        self.sizes_l = [self.num_prepositions]

    def sample(self):
        id = np.random.randint(0, self.num_prepositions)
        return self.prepositions[id]()


class RelationSpace(object):
    def __init__(self, thing_space, prep_space=PrepositionSpace()):
        self.thing_space = thing_space
        self.prep_space = prep_space
        self.onehot_size = 2 * self.thing_space.onehot_size + self.prep_space.onehot_size
        self.sizes_l = self.thing_space.sizes_l + self.prep_space.sizes_l + self.thing_space.sizes_l

    def sample(self):
        thing1 = self.thing_space.sample()
        thing2 = None
        while thing2 is None or thing2 == thing1:
            thing2 = self.thing_space.sample()
        prep = self.prep_space.sample()
        return Relation(left=thing1, prep=prep, right=thing2)
