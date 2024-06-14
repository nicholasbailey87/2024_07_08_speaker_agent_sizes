"""
Generates index_auto.md, from the first line of a block comment at the top of each script,
which line should be prefixed with 'summary:'

I suppose I could use sphinx or something for this, but seems like quite a specialized requirement,
Sphinx might be overkill (for now)

Can also add an option line 'category:', which shouldh ave a string, which is looked up in this file
"""
import argparse
from collections import defaultdict
import json
import os
from os import path
from os.path import join


category_description_by_name = {
}


def get_candidate_dirs_and_files(target_dir):
    files_by_dir = {}
    for root, dir, files in os.walk(target_dir):
        if root.startswith('./'):
            root = root[2:]
        root_comps = set(root.split('/'))
        if root in ['logs', 'model_saves', 'images', 'plots', 'debug_dumps', 'test', 'perf'] or root.startswith('conda'):
            continue
        skip = False
        for c in root_comps:
            if c.startswith('.') or c.startswith('_'):
                skip = True
                break
        if skip:
            continue
        files = [f for f in files if f not in ['__init__.py']]
        # print('root=', root, files[:3])
        # files_by_dir[root] = files
        yield root, files
    # return files_by_dir


def get_summary_category(filepath):
    print('filepath', filepath)
    with open(filepath, 'r') as f:
        lines = f.read().split('\n')
    summary = None
    category = None
    in_block = False
    block_lines = []
    for line in lines:
        if line.strip() == '"""':
            in_block = not in_block
            continue
        if in_block:
            block_lines.append(line)
            if line.startswith('summary:') and summary is None:
                summary = line.replace('summary: ', '').strip()
                if category is not None:
                    break
            if line.startswith('category:') and summary is None:
                summary = line.replace('category: ', '').strip()
                if summary is not None:
                    break
    if summary is None:
        if len(block_lines) > 0:
            summary = block_lines[0].strip()
        else:
            summary = ''
    return summary, category


def run(target_dir, index_filepath):
    summary_by_file_by_category = {}
    for dir, files in get_candidate_dirs_and_files(target_dir):
        print(dir, files[:3])
        for file in files:
            summary, category = get_summary_category(join(target_dir, dir, file))
            if category is None:
                category = f'[{dir}]({dir})'
            else:
                category = f'[{category}]({dir})'
            print('summary', summary)
            print('category', category)
            if category not in summary_by_file_by_category:
                summary_by_file_by_category[category] = {}
            summary_by_file_by_category[category][file] = summary
    # print(json.dumps(summary_by_file_by_category, indent=2))

    s = ''
    s += '\n*This file is auto-generated, by* [index_scripts.py](utils/index_scripts.py)\n\n'
    f_width = 22
    summary_width = 80
    for category in sorted(summary_by_file_by_category.keys()):
        summary_by_file = summary_by_file_by_category[category]
        s += f'## {category}\n'
        s += f'\n'
        s += '| ' + 'file'.ljust(f_width) + ' | ' + 'summary'.ljust(summary_width) + ' |\n'
        s += '|-' + '-' * f_width + '-|-' + summary_width* '-' + '-|\n'
        for file in sorted(summary_by_file.keys()):
            summary = summary_by_file[file]
            s += f'| ' + file.ljust(f_width) + ' | ' + summary.ljust(summary_width) + ' |\n'
        s += '\n'
    with open(index_filepath, 'w') as f:
        f.write(s)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', type=str, default='.')
    parser.add_argument('--index-filepath', type=str, default='index_auto.md')
    args = parser.parse_args()
    run(**args.__dict__)
