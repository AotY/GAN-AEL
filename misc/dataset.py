#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2019 LeonTao
#
# Distributed under terms of the MIT license.

"""
Load dataset.
"""

import torchtext.data as data
#  import Dataset, Field, BucketIterator

import os
import spacy

#  class Dataset

import io


SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# borrow from https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py


class ConversationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a ConversationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('q', fields[0]), ('r', fields[1])]

        q_path, r_paht = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(q_path, mode='r', encoding='utf-8') as q_file, \
                io.open(r_paht, mode='r', encoding='utf-8') as r_file:
            for q_line, r_line in zip(q_file, r_file):
                try:
                    q_line, r_line = q_line.strip(), r_line.strip()
                    if q_line != '' and r_line != '':
                        examples.append(data.Example.fromlist(
                            [q_line, r_line], fields))
                except UnicodeDecodeError as e:
                    continue

        super(ConversationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """
        Create dataset objects for splits of a ConversationDataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


def build_iterator(config):
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    q_field = data.Field(
        use_vocab=True,
        init_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        fix_length=config.q_max_len,
        included_lengths=True,
        lower=True,
        tokenize=tokenize_en
    )

    r_field = data.Field(
        use_vocab=True,
        init_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        fix_length=config.r_max_len,
        included_lengths=True,
        lower=True,
        tokenize=tokenize_en
    )

    train_data, test_data = ConversationDataset.splits(
        path=config.data_dir, exts=('.enc', '.dec'), fields=(q_field, r_field), test=None)

    q_field.build_vocab(train_data, mid_freq=config.min_count)
    r_field.vocab = q_field.vocab

    train_iterator = data.BucketIterator.splits(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device

    )
    valid_iterator = data.BucketIterator(
        test_data,
        batch_size=config.batch_size,
        device=config.device
    )

    return train_iterator, valid_iterator, q_field, r_field
