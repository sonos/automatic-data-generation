import os
from abc import ABCMeta, abstractmethod

import torch
import torchtext
from torchtext.data import BucketIterator

from automatic_data_generation.data.utils import get_fields, make_tokenizer


class BaseDataset(object):
    """
        Abstract class setting the API for using the package with a custom data
        set. Inherit from this class to implement training with a new data set.
        See 'handlers' for examples with open Snips data set.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 dataset_path,
                 input_type,
                 tokenizer_type,
                 preprocessing_type,
                 max_sequence_length,
                 emb_dim,
                 emb_type,
                 max_vocab_size):
        self.input_type = input_type
        self.tokenize = make_tokenizer(tokenizer_type, preprocessing_type)
        text, delex, intent = get_fields(self.tokenize, max_sequence_length)
        skip_header, datafields = self.get_datafields(text, delex, intent)

        train_path, valid_path = self.get_dataset_paths(dataset_path)

        train, valid = torchtext.data.TabularDataset.splits(
            path='.',  # the root directory where the data lies
            train=train_path,
            validation=valid_path,
            format='csv',
            skip_header=skip_header,
            # if your csv header has a header, make sure to pass this to
            # ensure it doesn't get proceesed as data!
            fields=datafields
        )

        if emb_type == 'glove':
            emb_vectors = "glove.6B.{}d".format(emb_dim)
            text.build_vocab(train, max_size=max_vocab_size,
                             vectors=emb_vectors)
            delex.build_vocab(train, max_size=max_vocab_size,
                              vectors=emb_vectors)
        elif emb_type is None:
            text.build_vocab(train, max_size=max_vocab_size)
            delex.build_vocab(train, max_size=max_vocab_size)
            text.vocab.vectors = torch.randn(len(text.vocab.itos), emb_dim)
            delex.vocab.vectors = torch.randn(len(delex.vocab.itos), emb_dim)
        else:
            raise NotImplementedError

        intent.build_vocab(train)

        self.vocab = text.vocab if input_type == 'utterance' else delex.vocab
        self.i2w = self.vocab.itos
        self.w2i = self.vocab.stoi
        self.i2int = intent.vocab.itos
        self.int2i = intent.vocab.stoi

        self.train = train
        self.valid = valid
        self.text = text
        self.delex = delex
        self.intent = intent

    @staticmethod
    @abstractmethod
    def get_datafields(text, delex, intent):
        """
        Get metadata relating to sample with index `item`.
        Args:
            text (torchtext.data.Field): field for the text entries
            delex (torchtext.data.Field): field for the delexicalized entries
            intent (torchtext.data.Field): field for the intent labels

        Returns:
            skip_header (bool): whether or not skip the csv header
                datafields list(tuple(str, torchtext.data.Field)): the fields
                should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that
                will be ignored.
        """
        raise NotImplementedError

    @staticmethod
    def get_dataset_paths(dataset_path):
        return os.path.join(dataset_path, 'train.csv'), os.path.join(
            dataset_path, 'validate.csv')

    @property
    def len_train(self):
        return len(self.train)

    @property
    def len_valid(self):
        return len(self.valid)

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def n_classes(self):
        return len(self.i2int)

    def get_iterators(self, batch_size=64):
        # make iterator for splits
        train_iter, valid_iter = BucketIterator.splits(
            (self.train, self.valid),
            # we pass in the datasets we want the iterator to draw data from
            batch_sizes=(batch_size, batch_size),
            device='cpu',
            # if you want to use the GPU, specify the GPU number here
            sort_key=lambda x: len(x.utterance),
            # the BucketIterator needs to be told what function it should
            # use to group the data.
            sort_within_batch=False,
            repeat=False,
            # we pass repeat=False because we want to wrap this Iterator layer.
        )
        return train_iter, valid_iter

    def embed_unks(self, init="randn", num_special_toks=2):
        sweep_range = len(self.vocab)
        running_norm = 0.
        num_non_zero = 0
        total_words = 0
        for i in range(num_special_toks, sweep_range):
            if len(self.vocab.vectors[i].nonzero()) == 0:
                # std = 0.05 is based on the norm of average GloVE 100-dim
                # word vectors
                if init == "randn":
                    torch.nn.init.normal_(self.vocab.vectors[i], mean=0,
                                          std=0.05)
            else:
                num_non_zero += 1
                running_norm += torch.norm(self.vocab.vectors[i])
            total_words += 1
        print(
            "average GloVE norm is {}, number of known words are {}, "
            "total number of words are {}".format(
                running_norm / num_non_zero,
                num_non_zero,
                total_words
            )
        )
