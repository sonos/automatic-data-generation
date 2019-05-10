from nltk import word_tokenize
import torchtext
from torchtext.data import Iterator, BucketIterator
import pickle
import torch
import spacy

class Datasets():
    
    def __init__(self, train_path='train.csv', valid_path='validate.csv', emb_dim=100, tokenizer='split'):
    
        if tokenizer == 'spacy':
            from spacy.symbols import ORTH
            my_tok = spacy.load('en')
            my_tok.tokenizer.add_sp
            def tokenize(x):
                return [tok.text for tok in my_tok.tokenizer(x)]
        elif tokenizer=='nltk':
            tokenize = word_tokenize
        elif tokenizer=='split':
            tokenize = lambda s : s.split(" ")

        self.tokenize = tokenize

        TEXT   = torchtext.data.Field(lower=True, tokenize=self.tokenize, sequential=True, batch_first=False)
        DELEX  = torchtext.data.Field(lower=True, tokenize=self.tokenize, sequential=True, batch_first=False)
        INTENT = torchtext.data.Field(sequential=False, batch_first=True, unk_token=None)

        if 'snips' in train_path:
            datafields = [("utterance", TEXT), ("labels", None), ("delexicalised", DELEX), ("intent", INTENT)]
        elif 'atis' in train_path:
            datafields = [(" ", None), ("utterance", TEXT), (" ", None), ("intent",  INTENT)]
        elif 'sentiment' in train_path:
            datafields = [("intent", INTENT), ("", None), ("", None), ("", None), ("", None), ("utterance", TEXT)]
        elif 'spam' in train_path:
            datafields = [("utterance", TEXT), ("intent", INTENT)]

            
        train, valid = torchtext.data.TabularDataset.splits(
                       path='.', # the root directory where the data lies
                       train=train_path, validation=valid_path,
                       format='csv',
                       skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                       fields=datafields,)

        TEXT.build_vocab(train,  vectors="glove.6B.{}d".format(emb_dim))
        DELEX.build_vocab(train, vectors="glove.6B.{}d".format(emb_dim))
        INTENT.build_vocab(train)
        
        self.train = train
        self.valid = valid
        self.TEXT = TEXT
        self.DELEX = DELEX
        self.INTENT = INTENT
    
    def embed_slots(self, averaging='micro', slotdic_path='./data/train_slot_values.pkl'):
        '''
        Create embeddings for the slots
        '''

        if averaging ==  'none':
            return

        with open(slotdic_path, 'rb') as f:
            slotdic = pickle.load(f)

        for i, token in enumerate(self.DELEX.vocab.itos):
            if token.startswith("_") and token.endswith("_"):
                slot = token.lstrip('_').rstrip('_')
                new_vectors = []

                slot_values = slotdic[slot]

                if averaging == 'micro':
                    for slot_value in slot_values:
                        for word in self.tokenize(slot_value):
                            if self.TEXT.vocab.stoi[word] != '<unk>' :
                                new_vectors.append(self.TEXT.vocab.vectors[self.TEXT.vocab.stoi[word]])
                    new_vector = torch.mean(torch.stack(new_vectors))   

                elif averaging == 'macro':
                    for slot_value in slot_values:
                        tmp = []
                        for word in self.tokenize(slot_value):
                            if self.TEXT.vocab.stoi[word] != '<unk>' :
                                tmp.append(self.TEXT.vocab.vectors[self.TEXT.vocab.stoi[word]])
                        new_vectors.append(torch.mean(torch.stack(tmp)))
                    new_vector = torch.mean(torch.stack(new_vectors))

                self.DELEX.vocab.vectors[self.DELEX.vocab.stoi[token]] = new_vector

    def get_iterators(self, batch_size=64):
    
        # make iterator for splits
        train_iter, valid_iter = BucketIterator.splits(
            (self.train, self.valid), # we pass in the datasets we want the iterator to draw data from
            batch_sizes=(batch_size, batch_size),
            device='cpu', # if you want to use the GPU, specify the GPU number here
            sort_key=lambda x: len(x.utterance), # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=False,
            repeat=False, # we pass repeat=False because we want to wrap this Iterator layer.
        )

        return train_iter, valid_iter
