#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import os
import random
import unittest
from pathlib import Path

import numpy as np

from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_snips_dataset'
PTB_DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_ptb_dataset'

VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', 'in', 'a', 'the', 'weather',
         'what', 'at', 'be', 'book', 'for', 'of', 'party', 'restaurant',
         'will', "'s", '2038', '3', '7', '8', 'area', 'bonaire',
         'brasserie', 'by', 'close', 'flight', 'fog', 'ford', 'forecast',
         'heights', 'is', 'jain', 'march', 'minutes', 'nv', 'on', 'park',
         'pub', 'recreation', 'serves', 'sligo', 'state', 'table', 'that',
         'there', 'three', 'twenty', 'type']

DELEX_VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', '_restaurant_type_', 'a',
               'in', 'the', 'weather', 'what', '_city_', '_party_size_number_',
               '_timerange_', 'at', 'be', 'book', 'for', 'of', 'party', 'will',
               "'s", '_condition_description_', '_country_', '_cuisine_',
               '_geographic_poi_', '_spatial_relation_', '_state_',
               'forecast', 'is', 'on', 'serves', 'table', 'that', 'there',
               'type']


def create_snips_dataset(dataset_path, restrict_intents=None,
                         input_type='utterance',
                         dataset_size=None, none_folder=None,
                         none_size=None, none_idx=None, none_intents=None,
                         output_folder=None):
    return SnipsDataset(
        dataset_folder=dataset_path,
        dataset_size=dataset_size,
        restrict_intents=restrict_intents,
        none_folder=none_folder,
        none_size=none_size,
        none_intents=none_intents,
        none_idx=none_idx,
        cosine_threshold=None,
        input_type=input_type,
        tokenizer_type='nltk',
        preprocessing_type='no_preprocessing',
        max_sequence_length=10,
        embedding_type='random',
        embedding_dimension=100,
        max_vocab_size=10000,
        output_folder=output_folder
    )


class TestSnipsDataset(unittest.TestCase):

    def test_should_read_vocab(self):
        random.seed(42)
        np.random.seed(42)
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='utterance')
        self.assertListEqual(dataset.i2w, VOCAB)

    def test_should_read_delex_vocab(self):
        random.seed(42)
        np.random.seed(42)
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='delexicalised')
        self.assertListEqual(dataset.i2w, DELEX_VOCAB)

    def test_should_trim_train(self):
        random.seed(42)
        np.random.seed(42)
        output_folder = DATASET_ROOT / "trimmed_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='delexicalised',
                                       dataset_size=3,
                                       output_folder=output_folder)
        self.assertEqual(dataset.len_train, 3)
        self.assertEqual(dataset.len_valid, 3)
        trimmed_dataset_file = output_folder / "train_3.csv"
        self.assertTrue(trimmed_dataset_file.exists())
        # trim should be stratified
        n_weather = len([item for item in dataset.train.examples if
                         item.intent == 'GetWeather'])
        n_restaurant = len([item for item in dataset.train.examples if
                            item.intent == 'BookRestaurant'])
        self.assertEqual(n_weather, 2)
        self.assertEqual(n_restaurant, 1)

    def test_should_add_none(self):
        random.seed(42)
        np.random.seed(42)
        none_folder = PTB_DATASET_ROOT
        output_folder = DATASET_ROOT / "none_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       input_type='delexicalised',
                                       none_folder=none_folder, none_idx=0,
                                       none_size=4,
                                       output_folder=output_folder)
        self.assertEqual(dataset.len_train, 10)
        self.assertEqual(dataset.len_valid, 9)
        train_dataset_file = output_folder / "train_none_4.csv"
        self.assertTrue(train_dataset_file.exists())
        validated_dataset_file = output_folder / "validate_with_none.csv"
        self.assertTrue(validated_dataset_file.exists())

    def test_should_restrict_intents(self):
        random.seed(42)
        np.random.seed(42)
        output_folder = DATASET_ROOT / "restricted_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       restrict_intents=['GetWeather'],
                                       input_type='utterance',
                                       output_folder=output_folder)
        self.assertEqual(dataset.len_train, 4)
        self.assertEqual(dataset.len_valid, 3)
        self.assertListEqual(dataset.i2int, ['GetWeather'])
        train_dataset_file = output_folder / "train_filtered.csv"
        self.assertTrue(train_dataset_file.exists())
        validated_dataset_file = output_folder / "validate_filtered.csv"
        self.assertTrue(validated_dataset_file.exists())

    def test_should_mix_everything(self):
        random.seed(42)
        np.random.seed(42)
        none_folder = PTB_DATASET_ROOT
        output_folder = DATASET_ROOT / "all"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       restrict_intents=['GetWeather'],
                                       none_folder=none_folder, none_idx=0,
                                       none_size=4,
                                       dataset_size=3,
                                       input_type='utterance',
                                       output_folder=output_folder)
        self.assertEqual(dataset.len_train, 7)
        self.assertEqual(dataset.len_valid, 9)
        self.assertListEqual(dataset.i2int, ['None', 'GetWeather'])
        train_dataset_file = output_folder / "train_3_none_4_filtered.csv"
        self.assertTrue(train_dataset_file.exists())
        validated_dataset_file = output_folder / \
                                 "validate_with_none_filtered.csv"
        self.assertTrue(validated_dataset_file.exists())


if __name__ == '__main__':
    unittest.main()
