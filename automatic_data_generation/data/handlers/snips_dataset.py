#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from automatic_data_generation.data.base_dataset import BaseDataset


class SnipsDataset(BaseDataset):
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

        super(SnipsDataset, self).__init__(dataset_path,
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
        datafields = [("utterance", text), ("labels", None),
                      ("delexicalised", delex), ("intent", intent)]
        return skip_header, datafields

    def embed_slots(self, averaging='micro',
                    slotdic_path='./data/snips/train_slot_values.pkl'):
        """
        Create embeddings for the slots in the Snips dataset
        """
        if self.input_type == "utterance":
            raise TypeError(
                "Slot embedding only available for delexicalized utterances"
            )

        if averaging == 'none':
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
                            if self.w2i[word] != '<unk>':
                                new_vectors.append(
                                    self.text.vocab.vectors[self.w2i[word]]
                                )
                    new_vector = torch.mean(torch.stack(new_vectors))

                elif averaging == 'macro':
                    for slot_value in slot_values:
                        tmp = []
                        for word in self.tokenize(slot_value):
                            if self.w2i[word] != '<unk>':
                                tmp.append(
                                    self.text.vocab.vectors[self.w2i[word]]
                                )
                        new_vectors.append(torch.mean(torch.stack(tmp)))
                    new_vector = torch.mean(torch.stack(new_vectors))

                else:
                    raise ValueError("Unknwon averaging strategy")

                self.delex.vocab.vectors[
                    self.delex.vocab.stoi[token]] = new_vector
