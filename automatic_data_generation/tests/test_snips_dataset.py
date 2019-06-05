#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import os
import unittest
from pathlib import Path

from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_snips_dataset'

VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', 'the', 'in', 'what', 'weather',
         'be', 'for', 'forecast', 'is', 'will', "'s", '2038', '3', 'area',
         'bonaire', 'by', 'close', 'fall', 'flight', 'fog', 'lesotho',
         'march', 'minutes', 'nv', 'on', 'park', 'recreation', 'sligo',
         'starting', 'state', 'there', 'this', 'three', 'twenty']

DELEX_VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', 'the', 'what',
               '_timerange_', 'in', 'weather', '_country_', 'be', 'for',
               'forecast', 'is', 'will', "'s", '_city_',
               '_condition_description_', '_geographic_poi_',
               '_spatial_relation_', '_state_', 'on', 'starting', 'there']


def create_snips_dataset(dataset_pah, input_type, dataset_size=None):
    return SnipsDataset(
        dataset_folder=dataset_pah,
        input_type=input_type,
        dataset_size=dataset_size,
        tokenizer_type='nltk',
        preprocessing_type=None,
        max_sequence_length=10,
        embedding_type=None,
        embedding_dimension=100,
        max_vocab_size=10000
    )


class TestSnipsDataset(unittest.TestCase):
    def test_should_read_vocab(self):
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='utterance')
        self.assertListEqual(dataset.i2w, VOCAB)

    def test_should_read_delex_vocab(self):
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='delexicalised')
        self.assertListEqual(dataset.i2w, DELEX_VOCAB)

    def test_should_trim_train(self):
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='delexicalised',
                                       dataset_size=3)
        self.assertEqual(dataset.len_train, 3)
        self.assertEqual(dataset.len_valid, 3)


if __name__ == '__main__':
    unittest.main()
