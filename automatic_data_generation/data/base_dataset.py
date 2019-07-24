import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torchtext
from sklearn.model_selection import StratifiedShuffleSplit
from torchtext.data import BucketIterator

from automatic_data_generation.data.utils.utils import (get_fields,
                                                        make_tokenizer)
from automatic_data_generation.utils.io import (read_csv, write_csv)
from automatic_data_generation.utils.constants import NO_INFERSENT_SELECTION


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class BaseDataset(object):
    """
        Abstract class setting the API for using the package with a custom data
        set. Inherit from this class to implement training with a new data set.
        See 'handlers' for examples.
    """
    __metaclass__ = ABCMeta

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

        self.input_type = input_type
        self.tokenize = make_tokenizer(tokenizer_type, preprocessing_type)
        self.embedding_dimension = embedding_dimension
        self.embedding_type = embedding_type

        text, delex, label, intent = get_fields(self.tokenize,
                                                max_sequence_length)
        skip_header, datafields = self.get_datafields(text, delex, label,
                                                      intent)

        train_path, valid_path = self.build_data_files(
            dataset_folder, dataset_size, restrict_intents,
            none_folder, none_size, none_intents, none_idx,
            infersent_selection, cosine_threshold,
            output_folder, skip_header)
        self.original_train_path = dataset_folder / 'train.csv'
        self.train_path = train_path

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

        if embedding_type == 'glove':
            emb_vectors = "glove.6B.{}d".format(embedding_dimension)
            text.build_vocab(train, max_size=max_vocab_size,
                             vectors=emb_vectors)
            delex.build_vocab(train, max_size=max_vocab_size,
                              vectors=emb_vectors)
        elif embedding_type == 'random':
            text.build_vocab(train, max_size=max_vocab_size)
            delex.build_vocab(train, max_size=max_vocab_size)
            text.vocab.vectors = torch.randn(len(text.vocab.itos),
                                             embedding_dimension)
            delex.vocab.vectors = torch.randn(len(delex.vocab.itos),
                                              embedding_dimension)
        else:
            raise TypeError("Unknown embedding type")

        label.build_vocab(train)
        intent.build_vocab(train)

        self.vocab = text.vocab if input_type == 'utterance' else delex.vocab
        self.vectors = self.vocab.vectors
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
    def get_datafields(text, delex, label, intent):
        """
        Get metadata relating to sample with index `item`.
        Args:
            text (torchtext.data.Field): field for the text entries
            delex (torchtext.data.Field): field for the delexicalised entries
            label (torchtext.data.Field): field for the slot label entries
            intent (torchtext.data.Field): field for the intent labels

        Returns:
            skip_header (bool): whether or not skip the csv header
            datafields list(tuple(str, torchtext.data.Field)): the fields
                should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that
                will be ignored.
        """
        raise NotImplementedError

    @abstractmethod
    def add_nones(self, sentences, none_folder, none_size=None,
                  none_intents=None, none_idx=None):
        """
        Get metadata relating to sample with index `item`.
        Args:
            sentences (list(list(str)): sentences to augment with None data
            none_folder (Path): path to the folder with None data
            none_size (int): number of None sentences to add
            none_intents (list(str)): restriction on intents of the None
                sentences
            none_idx (int): column index for data in None file


        Returns:
            augmented_sentences (list(list(str)): list of sentences
                augmented with None data
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def filter_intents(sentences, intents):
        """
        Get metadata relating to sample with index `item`.
        Args:
            sentences (list(list(str)): sentences be filtered
            intents (list(str): list of intents to keep

        Returns:
            augmented_sentences (list(list(str)): list of sentences
                augmented with None data
        """
        raise NotImplementedError

    def update_slotdic(self, new_slotdic):
        """
        Update the slot dictionnary of the dataset.
        """
        raise NotImplementedError

    def build_data_files(self, dataset_folder,  dataset_size=None, restrict_intents=None,
                         none_folder=None, none_size=None, none_intents=None, none_idx=None,
                         infersent_selection="no_infersent_selection", cosine_threshold=None,
                         output_folder=None, skip_header=True):

        original_train_path = dataset_folder / 'train.csv'
        original_test_path = dataset_folder / 'validate.csv'

        new_train = read_csv(original_train_path)
        new_test = read_csv(original_test_path)

        if skip_header:
            header_train = new_train[0]
            header_test = new_test[0]
            new_train = new_train[1:]
            new_test = new_test[1:]

        # filter intents
        filter_prefix = ''
        if restrict_intents is not None:
            filter_prefix = '_filtered'
            new_train = self.filter_intents(new_train, restrict_intents)
            new_test = self.filter_intents(new_test, restrict_intents)

        # trim_dataset
        trim_prefix = ''
        if dataset_size is not None:
            trim_prefix = '_{}'.format(dataset_size)
            original_dataset_size = len(new_train)
            keep_fraction = dataset_size / original_dataset_size
            intents = self.get_intents(new_train)
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=1 - keep_fraction)
            keep_indices = list(sss.split(intents, intents))[0][0]
            new_train = [new_train[i] for i in keep_indices]
            # new_train = random.sample(new_train, dataset_size)

        # add nones
        train_none_prefix = ''
        test_none_prefix = ''
        if none_size is not None:
            train_none_prefix = '_none_{}'.format(none_size)
            test_none_prefix = '_with_none'
            pseudolabels = None
            if infersent_selection != NO_INFERSENT_SELECTION:
                assert(none_intents is None)
                none_intents, pseudolabels = self.select_none_intents(dataset_folder, restrict_intents, none_folder, cosine_threshold)
                if infersent_selection == 'unsupervised':
                    pseudolabels = None # ignore pseudolabels
            new_train = self.add_nones(new_train, none_folder, none_size=none_size, none_intents=none_intents, pseudolabels=pseudolabels, none_idx=none_idx)
            new_test = self.add_nones(new_test, none_folder, none_size=200, none_intents=none_intents, none_idx=none_idx)

        if output_folder is not None:
            new_train_path = output_folder / 'train{}{}{}.csv'.format(
                trim_prefix, train_none_prefix, filter_prefix)
            new_test_path = output_folder / 'validate{}{}.csv'.format(
                test_none_prefix, filter_prefix)
        else:
            new_train_path = dataset_folder / 'train{}{}{}.csv'.format(
                trim_prefix, train_none_prefix, filter_prefix)
            new_test_path = dataset_folder / 'validate{}{}.csv'.format(
                test_none_prefix, filter_prefix)

        if skip_header:
            new_train = [header_train] + new_train
            new_test = [header_test] + new_test

        write_csv(new_test, new_test_path)
        write_csv(new_train, new_train_path)

        return new_train_path, new_test_path

    def select_none_intents(self, dataset_folder, restrict_intents,
                            none_folder, cosine_threshold):
        """
        Select none intents which embeddings are close to original intents
        """
        selected_none_intents = []
        pseudolabels = {}
        def cosine(u, v):
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        intent_vectors = self.load_intent_vectors(dataset_folder) # this is cheating a bit !
        none_vectors = self.load_intent_vectors(none_folder)
        for none_intent, none_vector in none_vectors.items():
            for intent, intent_vector in intent_vectors.items():
                if restrict_intents is not None and intent not in restrict_intents:
                    continue
                if cosine(none_vector, intent_vector) > cosine_threshold:
                    print('none intent {} is close to {}'.format(none_intent, intent))
                    selected_none_intents.append(none_intent)
                    pseudolabels[none_intent] = intent
                    break
        return selected_none_intents, pseudolabels

    
    @property
    def len_train(self):
        return len(self.train)

    @property
    def len_valid(self):
        return len(self.valid)

    @property
    def vocab_size(self):
        return len(self.i2w)

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
        sweep_range = self.vocab_size
        running_norm = 0.
        num_non_zero = 0
        total_words = 0
        for i in range(num_special_toks, sweep_range):
            if len(self.vectors[i].nonzero()) == 0:
                # std = 0.05 is based on the norm of average GloVE 100-dim
                # word vectors
                if init == "randn":
                    torch.nn.init.normal_(self.vectors[i], mean=0,
                                          std=0.05)
            else:
                num_non_zero += 1
                running_norm += torch.norm(self.vectors[i])
            total_words += 1
        print(
            "average GloVE norm is {}, number of known words are {}, "
            "total number of words are {}"
                .format(running_norm / num_non_zero, num_non_zero, total_words)
        )

    def load_intent_vectors(self, dataset_folder):
        intent_vectors_path = dataset_folder / 'vectors.pkl'
        return torch.load(intent_vectors_path)

    def save(self, folder):
        # TODO slotdic is not defined except for Snips dataset
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir()
        vocab_dict = {'i2w': self.i2w, 'i2int': self.i2int}
        if self.input_type == 'delexicalised':
            vocab_dict['slotdic'] = self.slotdic
        torch.save(vocab_dict, folder / "vocab.pth")

    def update(self, folder):
        # TODO slotdic is not defined except for Snips dataset
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

    @staticmethod
    def update_vocab(field_vocab, loaded_itos):
        # order matters: the indices of the old dataset should be conserved
        field_vocab.itos = loaded_itos + [w for w in field_vocab.itos if
                                          w not in loaded_itos]
        field_vocab.stoi.update({w: i for (i, w) in enumerate(loaded_itos)})

    def update_vectors(self):
        if self.embedding_type == 'glove':
            emb_vectors = "glove.6B.{}d".format(self.embedding_dimension)
            self.vocab.load_vectors(vectors=emb_vectors)
        elif self.embedding_type == 'random':
            self.vocab.vectors = torch.randn(len(self.vocab.itos),
                                             self.embedding_dimension)
