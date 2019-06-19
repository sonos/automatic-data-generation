#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from automatic_data_generation.data.base_dataset import BaseDataset


class YelpDataset(BaseDataset):
    """
        Handler for the Yelp dataset
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
                 output_folder):
        super(YelpDataset, self).__init__(dataset_folder,
                                          input_type,
                                          dataset_size,
                                          tokenizer_type,
                                          preprocessing_type,
                                          max_sequence_length,
                                          embedding_type,
                                          embedding_dimension,
                                          max_vocab_size,
                                          output_folder)

    @staticmethod
    def get_datafields(text, delex, label, intent):
        skip_header = True
        datafields = [("", None), ("", None), ("", None),
                      ("intent", intent), ("", None), ("utterance", text),
                      ("", None), ("", None), ("", None)]
        return skip_header, datafields
