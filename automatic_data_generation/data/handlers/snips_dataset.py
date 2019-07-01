#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import pickle
import random

from pathlib import Path
import torch
from nltk import word_tokenize
from automatic_data_generation.data.base_dataset import BaseDataset
from automatic_data_generation.data.utils import get_groups
from automatic_data_generation.utils.constants import NO_SLOT_AVERAGING
from automatic_data_generation.utils.io import read_csv


class SnipsDataset(BaseDataset):
    """
        Handler for the Snips dataset
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
        self.skip_header = True
        super(SnipsDataset, self).__init__(dataset_folder,
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
        datafields = [("utterance", text), ("labels", label),
                      ("delexicalised", delex), ("intent", intent)]
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
            new_row = [utterance, 'O '*len(word_tokenize(utterance)),
                       utterance, 'None']
            sentences.append(new_row)
        return sentences
                                    
    def get_slotdic(self):
        from collections import defaultdict
        slotdic = defaultdict(set)
        for example in list(self.train):
            utterance, labelling, delexicalised, intent = \
                example.utterance, example.labels, example.delexicalised, \
                example.intent
            groups = get_groups(utterance, labelling)
            for group in groups:
                if 'slot_name' in group.keys():
                    slot_name = group['slot_name']
                    slot_value = group['text']
                    slotdic[slot_name].add(slot_value)
        slotdic = {k:sorted(list(v)) for k,v in slotdic.items()} # sort for reproducibility
        self.slotdic = slotdic

        

    def embed_slots(self, averaging, slotdic):
        """
        Create embeddings for the slots in the Snips dataset
        """
        if self.input_type == "utterance":
            raise TypeError(
                "Slot embedding only available for delexicalised utterances"
            )

        if averaging == NO_SLOT_AVERAGING:
            return

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

    
    def update_slotdic(self, new_slotdic):
        
        def merge_dols(dol1, dol2): # Merge dictionaries of lists
            keys = set(dol1).union(dol2)
            no = []
            return dict((k, list(set(dol1.get(k, no) + dol2.get(k, no)))) for k in keys)

        updated_slotdic = merge_dols(self.slotdic, new_slotdic)

        self.slotdic = updated_slotdic
