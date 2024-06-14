"""
a 'thing' is a physical object. I'd prefer to say 'object', but that gets confused with
python / oop objects
"""
import random

import torch
import numpy as np

from colorama import init as colorama_init, Fore


# def class_name_to_name(class_name):
#     case_transitions = []
#     class_name = class_name.replace('_', '-')
#     for i, c in enumerate(class_name):
#         if c != class_name[i].lower():
#             case_transitions.append(i)
#     name = ''
#     for i, case_transition in enumerate(case_transitions):
#         if i > 0:
#             name += '-'
#         if i == len(case_transitions) - 1:
#             segment = class_name[case_transition:]
#         else:
#             segment = class_name[case_transition:case_transitions[i + 1]]
#         name += segment.lower()
#     name = name.replace('--', '-')
#     return name


# class Color(object):
#     def __repr__(self):
#         return class_name_to_name(self.__class__.__name__)

#     def to_fore_color(self):
#         return getattr(Fore, self.__class__.__name__.upper())

#     def __eq__(self, second):
#         return self.__class__ == second.__class__

#     def id(self, color_space):
#         return color_space.id_by_color[self.__class__]


# class Red(Color):
#     pass

# class Yellow(Color):
#     pass

# class Blue(Color):
#     pass

# class Green(Color):
#     pass

# class Cyan(Color):
#     pass

# class Magenta(Color):
#     pass

# class Black(Color):
#     pass

# class LightRed_Ex(Color):
#     pass

# class LightGreen_Ex(Color):
#     pass

_colors = [
    Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.MAGENTA,
    Fore.BLACK, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX,
    Fore.LIGHTBLACK_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTWHITE_EX, Fore.LIGHTYELLOW_EX, Fore.BLACK
]
# _id_by_color = {color: i for i, color in enumerate(_colors)}
# num_colors = len(colors)


# class Shape(object):
#     def __repr__(self):
#         # return class_name_to_name(self.__class__.__name__)
#         return self.get_as_char()

#     def __eq__(self, second):
#         return self.__class__ == second.__class__

#     def id(self, shape_space):
#         return shape_space.id_by_shape[self.__class__]


# class Cube(Shape):
#     def get_as_char(self):
#         return '#'

# class Sphere(Shape):
#     def get_as_char(self):
#         return 'O'

# class Pyramid(Shape):
#     def get_as_char(self):
#         return '^'

# class Tube(Shape):
#     def get_as_char(self):
#         return 'U'

# class Plane(Shape):
#     def get_as_char(self):
#         return '-'

# class Torus(Shape):
#     def get_as_char(self):
#         return 's'

# class Prism(Shape):
#     def get_as_char(self):
#         return 'V'

# class Pacman(Shape):
#     def get_as_char(self):
#         return 'C'

# class Spiral(Shape):
#     def get_as_char(self):
#         return '$'

# _shapes = [Cube, Sphere, Pyramid, Tube, Plane, Torus, Prism, Pacman, Spiral]
_shapes = '#O^U@XVC$ABCDEFG'
# _id_by_shape = {shape: i for i, shape in enumerate(_shapes)}


class ColorSpace(object):
    def __init__(self, num_colors):
        self.num_colors = num_colors
        self.size = self.num_colors
        self.onehot_size = self.num_colors
        self.sizes_l = [self.num_colors]

    def __len__(self):
        return self.num_colors

    def __getitem__(self, i):
        return i

    def by_index(self, i):
        return i

    def sample(self):
        id = np.random.randint(0, self.num_colors)
        return id


class ShapeSpace(object):
    def __init__(self, num_shapes):
        self.num_shapes = num_shapes
        self.size = self.num_shapes
        self.onehot_size = self.num_shapes
        self.sizes_l = [self.num_shapes]

    def __len__(self):
        return self.num_shapes

    def __getitem__(self, i):
        return i

    def by_index(self, i):
        return i

    def sample(self):
        id = np.random.randint(0, self.num_shapes)
        return id


class ThingSpace(object):
    def __init__(self, color_space, shape_space, available_items=None):
        self.color_space = color_space
        self.shape_space = shape_space
        self.onehot_size = self.color_space.onehot_size + self.shape_space.onehot_size
        self.sizes_l =  self.shape_space.sizes_l + self.color_space.sizes_l
        if available_items is None:
            available_items = []
            for color in range(color_space.size):
                for shape in range(shape_space.size):
                    available_items.append(ShapeColor(shape=shape, color=color))
        self.available_items = available_items
        self.available_items_set = set(self.available_items)

    def __len__(self):
        return len(self.available_items)

    def is_valid(self, o):
        return o in self.available_items_set

    def sample(self):
        item_idx = random.randint(0, len(self.available_items) - 1)
        return self.available_items[item_idx]

    @property
    def num_unique_things(self):
        return len(self.available_items)

    def partition(self, partition_sizes):
        """
        returns new ThingSpaces. Each ThingSpace is for a partition_size'd
        subset of this thingspace. All thingspaces are disjoint
        """
        assert np.sum(partition_sizes).item() == len(self.available_items)

        shuffled_items = random.sample(self.available_items, len(self.available_items))

        new_spaces = []
        pos = 0
        for i, partition_size in enumerate(partition_sizes):
            items = shuffled_items[pos: pos + partition_size]
            pos += partition_size
            new_space = ThingSpace(
                color_space=self.color_space,
                shape_space=self.shape_space,
                available_items=items
            )
            new_spaces.append(new_space)
        return new_spaces


class ShapeColor(object):
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color

    def __repr__(self):
        return _colors[self.color] + _shapes[self.shape] + Fore.RESET

    def __eq__(self, second):
        return self.shape == second.shape and self.color == second.color

    def __hash__(self):
        return self.shape * len(_colors) + self.color

    def as_onehot_indices(self, thing_space):
        return [
            self.shape,
            self.color + thing_space.shape_space.onehot_size
        ]

    def as_indices(self):
        return [
            self.shape,
            self.color
        ], ['S', 'C']

    def as_onehot_tensor_size(self, thing_space):
        return thing_space.shape_space.onehot_size + thing_space.color_space.onehot_size

    @classmethod
    def eat_from_indices(cls, indices):
        shape, color = indices[:2]
        indices = indices[2:]
        return ShapeColor(shape=shape, color=color), indices

    @classmethod
    def from_onehot_tensor(cls, thing_space, tensor):
        return cls.eat_from_onehot_tensor(thing_space, tensor)[0]

    @classmethod
    def eat_from_onehot_tensor(cls, thing_space, tensor):
        """
        returns tensor with shapcecolor removed from front
        """
        color_space = thing_space.color_space
        shape_space = thing_space.shape_space
        def eat(tensor, size):
            id = tensor.view(-1)[:size].nonzero().view(-1)[0]
            tensor = tensor[size:]
            return id, tensor
        shape, tensor = eat(tensor=tensor, size=shape_space.onehot_size)
        color, tensor = eat(tensor=tensor, size=color_space.onehot_size)
        return ShapeColor(shape=shape, color=color), tensor
