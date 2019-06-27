#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import random

from automatic_data_generation.data.base_dataset import BaseDataset
from automatic_data_generation.utils.io import read_csv


class YelpDataset(BaseDataset):
    """
        Handler for the Yelp dataset
    """

    def __init__(self,
                 dataset_folder,
                 restrict_to_intent,
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
        super(YelpDataset, self).__init__(dataset_folder,
                                          restrict_to_intent,
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
        datafields = [("", None), ("", None), ("", None),
                      ("intent", intent), ("", None), ("utterance", text),
                      ("", None), ("", None), ("", None)]
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
            new_row = ["", "", "", "None", "", utterance, "", "", ""]
            sentences.append(new_row)
        return sentences
