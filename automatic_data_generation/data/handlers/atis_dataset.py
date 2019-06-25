#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import random

from pandas import read_csv

from automatic_data_generation.data.base_dataset import BaseDataset


class AtisDataset(BaseDataset):
    """
        Handler for the ATIS dataset
    """

    def __init__(self,
                 dataset_folder,
                 input_type,
                 dataset_size,
                 tokenizer_type,
                 preprocessing_type,
                 max_sequence_length,
                 embedding_type,
                 embedding_dimension,
                 max_vocab_size,
                 output_folder,
                 none_folder,
                 none_idx,
                 none_size):
        super(AtisDataset, self).__init__(dataset_folder,
                                          input_type,
                                          dataset_size,
                                          tokenizer_type,
                                          preprocessing_type,
                                          max_sequence_length,
                                          embedding_type,
                                          embedding_dimension,
                                          max_vocab_size,
                                          output_folder,
                                          none_folder,
                                          none_idx,
                                          none_size)

    @staticmethod
    def get_datafields(text, delex, label, intent):
        skip_header = True
        datafields = [(" ", None), ("utterance", text), (" ", None),
                      ("intent", intent)]
        return skip_header, datafields

    @staticmethod
    def filter_intents(sentences, intents):
        return [row for row in sentences if row[3] in intents]

    @staticmethod
    def add_nones(sentences, none_folder, none_idx, none_size):
        none_path = none_folder / 'train.csv'
        none_sentences = read_csv(none_path)
        random.shuffle(none_sentences)
        for row in none_sentences[:none_size]:
            utterance = row[none_idx]
            new_row = [" ", utterance, " ", "None"]
            sentences.append(new_row)
        return sentences
