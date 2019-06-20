#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from pathlib import Path

from automatic_data_generation.utils.io import read_csv, write_csv


def save_augmented_dataset(generated_sentences, n_generated, train_path,
                           output_dir):
    dataset = read_csv(train_path)
    for s, l, d, i in zip(generated_sentences['utterances'],
                          generated_sentences['labellings'],
                          generated_sentences['delexicalised'],
                          generated_sentences['intents']):
        dataset.append([s, l, d, i])
    augmented_path = output_dir / Path(train_path.name.replace(
        '.csv', '_aug_{}.csv'.format(n_generated)
    ))
    write_csv(dataset, augmented_path)