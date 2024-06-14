"""
test ability to generate hypothesis
"""
from texrel import hypothesis, things, relations


def test_single_color():
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    generator = hypothesis.SingleColorHG(thing_space=thing_space, distractor_thing_space=thing_space)
    h = generator()
    print('h', h)

    h_as_seq = h.as_seq()
    print('h.as_seq()', h_as_seq)
    h_prime = hypothesis.SingleColorHypothesis.from_seq(h_as_seq[0], thing_space=thing_space, distractor_thing_space=thing_space)
    print('h_prime', h_prime)
    assert h == h_prime

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(3):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
        print('')
        print('neg examples:')
        for i in range(3):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')


def test_relation():
    # thing_space = things.ThingSpace()
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    rel_space = relations.RelationSpace(thing_space=thing_space)
    thing_space = rel_space.thing_space
    generator = hypothesis.RelationHG(rel_space=rel_space, distractor_thing_space=thing_space)
    r = generator()
    print('r', r)

    r_as_seq = r.as_seq()
    print('r.as_seq()', r_as_seq)
    r_prime = hypothesis.RelationHypothesis.from_seq(seq=r_as_seq[0], rel_space=rel_space, distractor_thing_space=thing_space)
    print('r_prime', r_prime)
    assert r == r_prime

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(3):
            grid = r.create_positive_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
        print('')
        print('neg examples:')
        for i in range(3):
            grid = r.create_negative_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')


def test_single_shape():
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    generator = hypothesis.SingleShapeHG(thing_space=thing_space, distractor_thing_space=thing_space)
    h = generator()
    print('h', h)

    h_as_seq = h.as_seq()
    print('h.as_seq()', h_as_seq)
    h_prime = hypothesis.SingleShapeHypothesis.from_seq(h_as_seq[0], thing_space=thing_space, distractor_thing_space=thing_space)
    print('h_prime', h_prime)
    assert h == h_prime

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(3):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
        print('')
        print('neg examples:')
        for i in range(3):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')


def test_single_thing():
    thing_space = things.ThingSpace(shape_space=things.ShapeSpace(num_shapes=9), color_space=things.ColorSpace(num_colors=9))
    generator = hypothesis.SingleThingHG(thing_space=thing_space, distractor_thing_space=thing_space)
    h = generator()
    print('h', h)

    h_as_seq = h.as_seq()
    print('h.as_seq()', h_as_seq)
    h_prime = hypothesis.SingleThingHypothesis.from_seq(h_as_seq[0], thing_space=thing_space, distractor_thing_space=thing_space)
    print('h_prime', h_prime)
    assert h == h_prime

    for num_distractors in range(3):
        print('')
        print('=================')
        print('distractors=', num_distractors)
        print('pos examples:')
        for i in range(3):
            grid = h.create_positive_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
        print('')
        print('neg examples:')
        for i in range(3):
            grid = h.create_negative_example(num_distractors=num_distractors, grid_size=5)
            print(grid)
            print('')
