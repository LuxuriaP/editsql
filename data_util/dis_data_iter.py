import json
import random
import math

import torch
import numpy as np


class DisDataIter(object):
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data = self.read_file(real_data_file)
        fake_data = self.read_file(fake_data_file)
        # self.src_data = real_data[0] + fake_data[0]
        # self.tgt_data = real_data[1] + fake_data[1]
        # self.labels = [random.uniform(0.7, 1.2) for _ in range(len(real_data[1]))] \
        #     + [random.uniform(0., .3) for _ in range(len(fake_data[1]))]
        # self.pairs = list(zip(self.src_data, self.tgt_data, self.labels))
        # self.data_num = len(self.pairs)
        # self.indices = range(self.data_num)
        # self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.src_real, self.tgt_real = real_data
        self.src_fake, self.tgt_fake = fake_data
        self.real_labels = [1 for _ in range(len(real_data[1]))]
        self.fake_labels = [0 for _ in range(len(fake_data[1]))]
        self.real_pairs = list(zip(self.src_real, self.tgt_real, self.real_labels))
        self.fake_pairs = list(zip(self.src_fake, self.tgt_fake, self.fake_labels))
        self.real_num = len(self.real_pairs)
        self.fake_num = len(self.fake_pairs)
        self.data_num = self.real_num + self.fake_num
        self.real_indices = range(self.real_num)
        self.fake_indices = range(self.fake_num)
        self.num_batches = int(math.ceil(float(self.real_num)/self.batch_size)) + \
            int(math.ceil(float(self.fake_num)/self.batch_size))
        self.real_idx = 0
        self.fake_idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.real_idx = 0
        self.fake_idx = 0
        random.shuffle(self.real_pairs)
        random.shuffle(self.fake_pairs)

    def next(self):
        if self.real_idx >= self.real_num and self.fake_idx >= self.fake_num:
            raise StopIteration
        if (random.random() < 0.5 and self.real_idx < self.real_num) \
                or self.fake_idx >= self.fake_num:
            index = self.real_indices[self.real_idx:self.real_idx+self.batch_size]
            pairs = [self.real_pairs[i] for i in index]
            self.real_idx += self.batch_size
        else:
            index = self.fake_indices[self.fake_idx:self.fake_idx+self.batch_size]
            pairs = [self.fake_pairs[i] for i in index]
            self.fake_idx += self.batch_size
        # index = self.indices[self.idx:self.idx+self.batch_size]
        # pairs = [self.pairs[i] for i in index]
        src_data = [p[0] for p in pairs]
        tgt_data = [p[1] for p in pairs]
        label = [p[2] for p in pairs]
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        # self.idx += self.batch_size
        return src_data, tgt_data, label

    @staticmethod
    def read_file(data_file):
        with open(data_file, "r") as f:
            data = json.load(f)

        """
        [
            {
                "src": List[str]
                "tgt": List[str]
            },
            ...
        ]
        """

        src_data, tgt_data = [], []
        for datum in data:
            src_data.append(datum["src"])
            tgt_data.append(datum["tgt"])

        return src_data, tgt_data
