#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import tempfile
import unittest
from pathlib import Path

from automatic_data_generation.models.cvae import CVAE


class TestModel(unittest.TestCase):
    def test_should_serialize(self):
        model = CVAE(
            conditional="supervised",
            compute_bow=True,
            vocab_size=300,
            embedding_size=100,
            rnn_type='gru',
            hidden_size_encoder=128,
            hidden_size_decoder=128,
            word_dropout_rate=0,
            embedding_dropout_rate=0,
            z_size=100,
            n_classes=10,
            sos_idx=0,
            eos_idx=0,
            pad_idx=0,
            unk_idx=0,
            max_sequence_length=30,
            num_layers_encoder=1,
            num_layers_decoder=1,
            bidirectional=False,
            temperature=1,
            force_cpu=False
        )

        with tempfile.TemporaryDirectory() as t_dir:
            model_dir = Path(t_dir) / "test_model"
            model.save(model_dir)

            reloaded = CVAE.from_folder(model_dir)
            self.assertIsInstance(reloaded, CVAE)
            self.assertEqual(reloaded.vocab_size, 300)


if __name__ == '__main__':
    unittest.main()
