#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import random

from automatic_data_generation.data.base_dataset import BaseDataset
from automatic_data_generation.utils.io import read_csv


class PTBDataset(BaseDataset):
    """
        Handler for the Snips dataset
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
        super(PTBDataset, self).__init__(dataset_folder,
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
        skip_header = False
        datafields = [("utterance", text)]
        return skip_header, datafields

    @staticmethod
    def filter_intents(sentences, intents):
        raise TypeError("Intents are not defined for PTB")

    @staticmethod
    def add_nones(sentences, none_folder, none_idx, none_size):
        none_path = none_folder / 'train.csv'
        none_sentences = read_csv(none_path)
        random.shuffle(none_sentences)
        new_rows = [[row[none_idx]] for row in none_sentences[:none_size]]
        return sentences + new_rows
