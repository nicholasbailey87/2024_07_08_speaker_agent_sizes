import torch

from texrel import things


def test_colors():
    color_space = things.ColorSpace(num_colors=9)
    for i in range(5):
        print(color_space.sample())
    print('')

    color_space = things.ColorSpace(num_colors=2)
    for i in range(5):
        print(color_space.sample())


def test_shapes():
    shape_space = things.ShapeSpace(num_shapes=9)
    for i in range(5):
        print(shape_space.sample())
    print('')

    shape_space = things.ShapeSpace(num_shapes=2)
    for i in range(5):
        print(shape_space.sample())


def test_things():
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    for i in range(5):
        print(thing_space.sample())
    print('')

    thing_space = things.ThingSpace(
        shape_space=things.ShapeSpace(num_shapes=2),
        color_space=things.ColorSpace(num_colors=2)
    )
    for i in range(5):
        print(thing_space.sample())

def test_things_encode_decode():
    thing_space = things.ThingSpace(
        shape_space=things.ShapeSpace(num_shapes=9),
        color_space=things.ColorSpace(num_colors=9)
    )
    for i in range(10):
        o = thing_space.sample()
        o_indices = o.as_onehot_indices(thing_space)
        o_onehot = torch.Tensor(o.as_onehot_tensor_size(thing_space)).zero_()
        o_onehot[torch.LongTensor(o_indices)] = 1
        o2 = things.ShapeColor.from_onehot_tensor(thing_space, o_onehot)
        print('o', o, 'o2', o2)
        assert o2 == o

def test_things_eat():
    thing_space = things.ThingSpace(
        shape_space=things.ShapeSpace(num_shapes=9),
        color_space=things.ColorSpace(num_colors=9)
    )
    o_sample = thing_space.sample()
    thing_onehot_size = o_sample.as_onehot_tensor_size(thing_space)
    print('thing_onehot_size', thing_onehot_size)
    for i in range(10):
        o1 = thing_space.sample()
        o2 = thing_space.sample()
        if o1 == o2:
            continue
        assert o1 != o2

        o_t = torch.Tensor(thing_onehot_size * 2).zero_()

        o1_indices = o1.as_onehot_indices(thing_space)
        o2_indices = o2.as_onehot_indices(thing_space)
        o_t[torch.LongTensor(o1_indices)] = 1
        o_t[torch.LongTensor(o2_indices) + thing_onehot_size] = 1
        o1b, o_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, o_t)
        o2b, o_t = things.ShapeColor.eat_from_onehot_tensor(thing_space, o_t)
        assert o_t.size()[0] == 0

        print('o1', o1, 'o1b', o1b, 'o2', o2, 'o2b', o2b)
        assert o1 == o1b
        assert o2 == o2b
        assert o1 != o2
        assert o1b != o2b

def test_partition():
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    num_items = thing_space.num_unique_things
    print('num_items', num_items)

    partitions = [num_items - 5, 5]
    spaces = thing_space.partition(partitions)
    samples_l = []
    for i, s in enumerate(spaces):
        print('i', i, 'size', s.num_unique_things)
        for k in range(5):
            print(s.sample())
        samples = set()
        for i in range(10000):
            samples.add(s.sample())
        print('len(samples)', len(samples))
        print('samples[:5]', list(samples)[:5])
        samples_l.append(samples)
    for i, s in enumerate(samples_l):
        print('len(samples)', len(s))
        others = set()
        for j, s2 in enumerate(samples_l):
            if i != j:
                others |= s2
        print('len(others)', len(others))
        for o in s:
            if o in others:
                raise Exception("intersection")
