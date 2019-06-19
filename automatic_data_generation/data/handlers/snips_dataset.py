#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import pickle

import torch

from automatic_data_generation.data.base_dataset import BaseDataset
from automatic_data_generation.data.utils import get_groups


class SnipsDataset(BaseDataset):
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
                 output_folder):
        self.skip_header = True
        super(SnipsDataset, self).__init__(dataset_folder,
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
        datafields = [("utterance", text), ("labels", label),
                      ("delexicalised", delex), ("intent", intent)]
        return skip_header, datafields

    def get_slotdic(self):
        slotdic = {}
        encountered_slot_values = {}
        for example in list(self.train):
            utterance, labelling, delexicalised, intent = \
                example.utterance, example.labels, example.delexicalised, \
                example.intent
            groups = get_groups(utterance, labelling)
            for group in groups:
                if 'slot_name' in group.keys():
                    slot_name = group['slot_name']
                    slot_value = group['text']
                    if slot_name not in encountered_slot_values.keys():
                        encountered_slot_values[slot_name] = []
                    if slot_name not in slotdic.keys():
                        slotdic[slot_name] = []
                    if slot_value not in encountered_slot_values[slot_name]:
                        slotdic[slot_name].append(slot_value)
                    encountered_slot_values[slot_name].append(slot_value)
        return slotdic

    def embed_slots(self, averaging='micro',
                    slotdic_path='./data/snips/train_slot_values.pkl'):
        """
        Create embeddings for the slots in the Snips dataset
        """
        if self.input_type == "utterance":
            raise TypeError(
                "Slot embedding only available for delexicalised utterances"
            )

        if averaging is None:
            return

        with open(slotdic_path, 'rb') as f:
            slotdic = pickle.load(f)

        for i, token in enumerate(self.i2w):
            if token.startswith("_") and token.endswith("_"):
                slot = token.lstrip('_').rstrip('_')
                new_vectors = []

                slot_values = slotdic[slot]

                if averaging == 'micro':
                    for slot_value in slot_values:
                        for word in self.tokenize(slot_value):
                            if self.text.vocab.stoi[word] != '<unk>':
                                new_vectors.append(
                                    self.text.vocab.vectors[
                                        self.text.vocab.stoi[word]]
                                )
                    new_vector = torch.mean(torch.stack(new_vectors))

                elif averaging == 'macro':
                    for slot_value in slot_values:
                        tmp = []
                        for word in self.tokenize(slot_value):
                            if self.text.vocab.stoi[word] != '<unk>':
                                tmp.append(
                                    self.text.vocab.vectors[
                                        self.text.vocab.stoi[word]]
                                )
                        new_vectors.append(torch.mean(torch.stack(tmp)))
                    new_vector = torch.mean(torch.stack(new_vectors))

                else:
                    raise ValueError("Unknown averaging strategy")

                self.delex.vocab.vectors[
                    self.delex.vocab.stoi[token]] = new_vector
