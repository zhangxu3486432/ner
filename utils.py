from functools import reduce
from os.path import join

import numpy as np
import torch


def load_data(path="./ResumeNER", dataset=None):
    assert dataset in ['train', 'dev', 'test']

    with open(join(path, f"{dataset}.char.bmes"), 'r') as f:
        lines = f.read().splitlines()
        lines_np = np.array(lines)
        split_index = np.where(lines_np == '')[0]
        data = np.split(lines, split_index)[:-1]
        for i in range(1, len(data)):
            data[i] = data[i][1:]
    word_lists = []
    tag_lists = []

    for line in data:
        word_list = []
        tag_list = []
        for item in line:
            item = item.split()
            word_list.append(item[0])
            tag_list.append(item[1])
        word_lists.append(word_list)
        tag_lists.append(tag_list)

    words = reduce(lambda x, y: x + y, word_lists)
    word_size = len(words)

    word2id = map_(word_lists)
    tag2id = map_(tag_lists)

    tag2id = sorted(tag2id.items(), key=lambda x: x[1])
    tag2id = dict(tag2id)

    word_tag = list(zip(word_lists, tag_lists))
    word_tag.sort(key=lambda item: len(item[0]), reverse=True)
    word_lists, tag_lists = list(zip(*word_tag))

    lengths = [len(l) for l in word_lists]

    return word_lists, tag_lists, lengths, word2id, tag2id, word_size


def map_(lists):
    items = reduce(lambda x, y: x + y, lists)
    items = set(items)
    items = zip(items, range(len(items)))
    items = dict(items)
    return items


def pad(lists, lengths, max_len, map_, PAD, UNK):
    data_size = len(lengths)
    id_lists = torch.ones(data_size, max_len).long() * PAD
    for i, line in enumerate(lists):
        for j, word in enumerate(line):
            id_lists[i][j] = map_.get(word, UNK)
    items = reduce(lambda x, y: x + y, lists)
    ids = [map_.get(i, UNK) for i in items]
    return id_lists, ids
