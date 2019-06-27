#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import os
import unittest
from pathlib import Path

from automatic_data_generation.data.handlers.ptb_dataset import PTBDataset

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_ptb_dataset'
SNIPS_DATASET_ROOT = ROOT_PATH / 'resources' / 'mock_snips_dataset'

VOCAB = ['<unk>', '<pad>', '<sos>', '<eos>', '<', '>', 'unk', 'of', 'a', 'n',
         'the', 'it', 'to', 'years', 'asbestos', 'chairman', 'director',
         'group', 'is', 'nonexecutive', 'old', 'once', 'researchers', 'aer',
         'ago', 'among', 'and', 'as', 'banknote', 'berlitz', 'board', 'brief',
         'british', 'calloway', 'cancer', 'caused', 'causing', 'centrust',
         'cigarette', 'cluett', 'conglomerate', 'consolidated', 'deaths',
         'decades', 'dutch', 'enters', 'even', 'exposed', 'exposures', 'fiber',
         'fields', 'filters', 'form', 'former', 'fromstein', 'gitano', 'gold',
         'guterman', 'has', 'high', 'hydro-quebec', 'industrial', 'ipo',
         'join', 'kent', 'kia', 'later', 'make', 'memotec', 'mlx', 'more',
         'mr.', 'n.v.', 'nahb', 'named', 'nov.', 'percentage', 'pierre', 'plc',
         'publishing', 'punts', 'rake', 'regatta', 'reported', 'rubens',
         'rudolph', 'said', 'show', 'sim', 'snack-food', 'ssangyong', 'swapo',
         'symptoms', 'than', 'that', 'this', 'unusually', 'up', 'used',
         'wachter', 'was', 'will', 'with', 'workers']


def create_ptb_dataset(dataset_pah, dataset_size=None,
                       none_folder=None, none_size=None, none_idx=None,
                       output_folder=None):
    return PTBDataset(
        dataset_folder=dataset_pah,
        input_type='utterance',
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


class TestPTBDataset(unittest.TestCase):
    def test_should_read_vocab(self):
        dataset = create_ptb_dataset(DATASET_ROOT)
        self.assertListEqual(dataset.i2w, VOCAB)

    def test_should_trim_train(self):
        output_folder = DATASET_ROOT / "trimmed_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_ptb_dataset(DATASET_ROOT,
                                     dataset_size=3,
                                     output_folder=output_folder)
        self.assertEqual(dataset.len_train, 3)
        self.assertEqual(dataset.len_valid, 4)
        trimmed_dataset_file = output_folder / "train_3.csv"
        self.assertTrue(trimmed_dataset_file.exists())

    def test_should_add_none(self):
        none_folder = SNIPS_DATASET_ROOT
        output_folder = DATASET_ROOT / "none_dataset"
        if not output_folder.exists():
            output_folder.mkdir()
        dataset = create_ptb_dataset(DATASET_ROOT,
                                     none_folder=none_folder, none_idx=0,
                                     none_size=4,
                                     output_folder=output_folder)
        self.assertEqual(dataset.len_train, 10)
        self.assertEqual(dataset.len_valid, 10)
        train_dataset_file = output_folder / "train_none_4.csv"
        self.assertTrue(train_dataset_file.exists())
        validated_dataset_file = output_folder / "validate_with_none.csv"
        self.assertTrue(validated_dataset_file.exists())


if __name__ == '__main__':
    unittest.main()
