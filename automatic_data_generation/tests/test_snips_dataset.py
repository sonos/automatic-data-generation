#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import os
import unittest
from pathlib import Path

from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_snips_dataset'
PTB_DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_ptb_dataset'

VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', 'the', 'in', 'what', 'is',
         'weather', 'a', 'be', 'for', 'forecast', 'this', 'will', "'s",
         '2038', '3', 'area', 'bonaire', 'by', 'close', 'fall', 'flight',
         'fog', 'from', 'intent', 'lesotho', 'march', 'minutes', 'nv', 'on',
         'park', 'random', 'recreation', 'sentence', 'sligo', 'starting',
         'state', 'there', 'three', 'twenty']

DELEX_VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', 'the', 'what',
               '_timerange_', 'in', 'is', 'weather', '_country_', 'a', 'be',
               'for', 'forecast', 'will', "'s", '_city_',
               '_condition_description_', '_geographic_poi_',
               '_spatial_relation_', '_state_', 'from', 'intent', 'on',
               'random', 'sentence', 'starting', 'there', 'this']


def create_snips_dataset(dataset_path, restrict_to_intent=None,
                         input_type='utterance',
                         dataset_size=None, none_folder=None,
                         none_size=None, none_idx=None,
                         output_folder=None):
    return SnipsDataset(
        dataset_folder=dataset_path,
        restrict_to_intent=restrict_to_intent,
        input_type=input_type,
        dataset_size=dataset_size,
        tokenizer_type='nltk',
        preprocessing_type='no_preprocessing',
        max_sequence_length=10,
        embedding_type='random',
        embedding_dimension=100,
        max_vocab_size=10000,
        none_folder=none_folder,
        none_idx=none_idx,
        none_size=none_size,
        output_folder=output_folder
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

    def test_should_add_none(self):
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
        output_folder = DATASET_ROOT / "restricted_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       restrict_to_intent=['GetWeather'],
                                       input_type='utterance',
                                       output_folder=output_folder)
        self.assertEqual(dataset.len_train, 5)
        self.assertEqual(dataset.len_valid, 3)
        self.assertListEqual(dataset.i2int, ['GetWeather'])
        train_dataset_file = output_folder / "train_filtered.csv"
        self.assertTrue(train_dataset_file.exists())
        validated_dataset_file = output_folder / "validate_filtered.csv"
        self.assertTrue(validated_dataset_file.exists())

    def test_should_mix_everything(self):
        none_folder = PTB_DATASET_ROOT
        output_folder = DATASET_ROOT / "all"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_snips_dataset(DATASET_ROOT,
                                       restrict_to_intent=['GetWeather'],
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
