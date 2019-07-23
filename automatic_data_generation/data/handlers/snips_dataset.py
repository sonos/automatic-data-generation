#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import random
from collections import defaultdict
from pathlib import Path

import torch
from nltk import word_tokenize

from automatic_data_generation.data.base_dataset import BaseDataset
from automatic_data_generation.data.utils import get_groups
from automatic_data_generation.utils.constants import NO_SLOT_EMBEDDING
from automatic_data_generation.utils.io import read_csv


class SnipsDataset(BaseDataset):
    """
        Handler for the Snips dataset
    """

    def __init__(self,
                 dataset_folder,
                 dataset_size,
                 restrict_intents,
                 none_folder,
                 none_size,
                 none_intents,
                 none_idx,
                 infersent_selection,
                 cosine_threshold,
                 input_type,
                 tokenizer_type,
                 preprocessing_type,
                 max_sequence_length,
                 embedding_type,
                 embedding_dimension,
                 max_vocab_size,
                 output_folder):
        self.skip_header = True
        self.slotdic = None
        super(SnipsDataset, self).__init__(dataset_folder,
                                           dataset_size,
                                           restrict_intents,
                                           none_folder,
                                           none_size,
                                           none_intents,
                                           none_idx,
                                           infersent_selection,
                                           cosine_threshold,
                                           input_type,
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

    @staticmethod
    def filter_intents(sentences, intents):
        return [row for row in sentences if row[3] in intents]

    @staticmethod
    def get_intents(sentences):
        return [row[3] for row in sentences]

    def add_nones(self, sentences, none_folder, none_size=None, none_intents=None, pseudolabels=None, none_idx=None):
        none_path = none_folder / 'train.csv'
        none_sentences = read_csv(none_path)

        if none_intents is not None:
            none_sentences = self.filter_intents(none_sentences, none_intents)            

        random.shuffle(none_sentences)
        for row in none_sentences[:none_size]:
            if 'snips' in str(none_folder):
                new_row = row
                if pseudolabels:
                    new_row[3] = pseudolabels[row[3]]
                else :
                    new_row[3] = 'None'
            else:
                utterance = row[none_idx]
                new_row = [utterance, 'O ' * len(word_tokenize(utterance)),
                       utterance, 'None']
            sentences.append(new_row)
        return sentences

    def build_slotdic(self):
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
        slotdic = {k: sorted(list(v)) for k, v in
                   slotdic.items()}  # sort for reproducibility
        self.slotdic = slotdic

    def update(self, folder):
        folder = Path(folder)
        loaded_dict = torch.load(str(folder / "vocab.pth"))
        loaded_i2w = loaded_dict['i2w']
        loaded_i2int = loaded_dict['i2int']
        self.update_vocab(self.vocab, loaded_i2w)
        self.update_vocab(self.intent.vocab, loaded_i2int)
        self.update_vectors()

        if self.input_type == 'delexicalised':
            loaded_slotdic = loaded_dict['slotdic']
            self.update_slotdic(loaded_slotdic)

        self.i2w = self.vocab.itos
        self.w2i = self.vocab.stoi
        self.i2int = self.intent.vocab.itos
        self.int2i = self.intent.vocab.stoi
        self.vectors = self.vocab.vectors

        return len(loaded_i2w)

    def embed_slots(self, slot_embedding, slotdic):
        """
        Create embeddings for the slots in the Snips dataset
        """
        if self.input_type == "utterance":
            raise TypeError(
                "Slot embedding only available for delexicalised utterances"
            )

        if slot_embedding == NO_SLOT_EMBEDDING:
            return

        for i, token in enumerate(self.i2w):
            if token.startswith("_") and token.endswith("_"):
                slot = token.lstrip('_').rstrip('_')
                new_vectors = []

                slot_values = slotdic[slot]
                
                if slot_embedding == "litteral":
                    slot_tokens = slot.split('_')
                    for slot_token in slot_tokens:
                        new_vectors.append(self.text.vocab.vectors[
                        self.text.vocab.stoi[slot_token]])
                    new_vector = torch.mean(torch.stack(new_vectors), dim=0)
                
                elif slot_embedding == 'micro':
                    for slot_value in slot_values:
                        for word in self.tokenize(slot_value):
                            if self.text.vocab.stoi[word] != '<unk>':
                                new_vectors.append(
                                    self.text.vocab.vectors[
                                        self.text.vocab.stoi[word]]
                                )
                    new_vector = torch.mean(torch.stack(new_vectors), dim=0)

                elif slot_embedding == 'macro':
                    for slot_value in slot_values:
                        tmp = []
                        for word in self.tokenize(slot_value):
                            if self.text.vocab.stoi[word] != '<unk>':
                                tmp.append(
                                    self.text.vocab.vectors[
                                        self.text.vocab.stoi[word]]
                                )
                        new_vectors.append(torch.mean(torch.stack(tmp)))
                    new_vector = torch.mean(torch.stack(new_vectors), dim=0)

                else:
                    raise ValueError("Unknown averaging strategy: {}".format(slot_embedding))

                self.delex.vocab.vectors[
                    self.delex.vocab.stoi[token]] = new_vector

    def update_slotdic(self, new_slotdic):
        def merge_dols(dol1, dol2):  # merge dictionaries of lists
            keys = set(dol1).union(dol2)
            no = []
            return dict(
                (k, list(set(dol1.get(k, no) + dol2.get(k, no))))
                for k in keys
            )

        updated_slotdic = merge_dols(self.slotdic, new_slotdic)
        self.slotdic = updated_slotdic
