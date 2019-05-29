#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from automatic_data_generation.data.base_dataset import BaseDataset


class SpamDataset(BaseDataset):
    """
        Handler for the Snips dataset
    """

    def __init__(self,
                 dataset_path,
                 input_type,
                 tokenizer_type,
                 preprocessing_type,
                 max_sequence_length,
                 emb_dim,
                 emb_type,
                 max_vocab_size):
        super(SpamDataset, self).__init__(dataset_path,
                                          input_type,
                                          tokenizer_type,
                                          preprocessing_type,
                                          max_sequence_length,
                                          emb_dim,
                                          emb_type,
                                          max_vocab_size)

    @staticmethod
    def get_datafields(text, delex, intent):
        skip_header = True
        datafields = [("utterance", text), ("intent", intent)]
        return skip_header, datafields
