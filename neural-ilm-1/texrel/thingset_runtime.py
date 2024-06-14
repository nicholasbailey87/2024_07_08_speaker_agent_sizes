import argparse
import json

import torch

from ulfs.params import Params
from ulfs.utils import expand, die

from texrel import things


class ThingSet(object):
    """
    A 'thingset' comprises multiple ThingSpaces, probably disjoint, eg for train and holdout
    It can be serialized to a file, and restored from a file
    """
    def __init__(self, thing_space_by_name, meta):
        self.thing_space_by_name = thing_space_by_name
        self.meta = meta

    @classmethod
    def from_file(cls, ts_ref):
        ts_filepath = '~/data/reftask/thingset_{ts_ref}.json'.format(ts_ref=ts_ref)
        with open(expand(ts_filepath), 'r') as f:
            d = json.loads(f.read())
        p = Params(d['meta'])
        print('p', p)
        color_space = things.ColorSpace(num_colors=p.num_colors)
        shape_space = things.ShapeSpace(num_shapes=p.num_shapes)

        thing_space_by_name = {}
        for name in ['train', 'holdout']:
            available_items_dicts = d[name]['available_items']
            available_items_objects = []
            for thing_dict in available_items_dicts:
                shape = shape_space[thing_dict['shape_id']]
                color = color_space[thing_dict['color_id']]
                thing = things.ShapeColor(shape=shape, color=color)
                available_items_objects.append(thing)
            thing_space = things.ThingSpace(
                available_items=available_items_objects, color_space=color_space, shape_space=shape_space)
            thing_space_by_name[name] = thing_space
        return cls(thing_space_by_name=thing_space_by_name, meta=p)


def run(ts_ref):
    thing_set = ThingSet.from_file(ts_ref=ts_ref)
    for name, thing_space in thing_set.thing_space_by_name.items():
        print(name)
        for item in thing_space.available_items:
            print('    ', item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts-ref', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
