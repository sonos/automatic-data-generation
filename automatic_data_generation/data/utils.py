import random

import torchtext

from automatic_data_generation.utils.constants import NO_PREPROCESSING


NONE_COLUMN_MAPPING = {
    'penn-tree-bank': 0,
    'yelp': 5,
    'shakespeare': 5,
    'subtitles': 0
}


def make_tokenizer(tokenizer_type, preprocessing_type):
    if tokenizer_type == 'spacy':
        import spacy
        my_tok = spacy.load('en')

        def tokenize(x):
            return [tok.lemma_ for tok in my_tok.tokenizer(x)]

    elif tokenizer_type == 'nltk':
        from nltk import word_tokenize
        from nltk.stem import WordNetLemmatizer, PorterStemmer

        def tokenize(x):
            if preprocessing_type == 'stem':
                stemmer = PorterStemmer()
                return [stemmer.stem(tok) for tok in word_tokenize(x)]
            elif preprocessing_type == 'lemmatize':
                lemmatizer = WordNetLemmatizer()
                return [lemmatizer.lemmatize(tok) for tok in
                        word_tokenize(x)]
            elif preprocessing_type == NO_PREPROCESSING:
                return word_tokenize(x)

    elif tokenizer_type == 'split':

        def tokenize(x):
            return x.split(" ")

    else:
        raise ValueError("Unknown tokenizer")

    return tokenize


def get_fields(tokenize, max_sequence_length):
    text = torchtext.data.Field(lower=True, tokenize=tokenize,
                                sequential=True, batch_first=True,
                                include_lengths=True,
                                fix_length=max_sequence_length,
                                init_token='<sos>', eos_token='<eos>')
    delex = torchtext.data.Field(lower=True, tokenize=tokenize,
                                 sequential=True, batch_first=True,
                                 include_lengths=True,
                                 fix_length=max_sequence_length,
                                 init_token='<sos>', eos_token='<eos>')
    label = torchtext.data.Field(lower=False, tokenize=tokenize,
                                 sequential=True, batch_first=True,
                                 include_lengths=True,
                                 fix_length=max_sequence_length,
                                 init_token='<sos>', eos_token='<eos>')
    intent = torchtext.data.Field(sequential=False, batch_first=True,
                                  unk_token=None, pad_token=None)
    return text, delex, label, intent


def idx2word(idx, i2w, eos_idx):
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == eos_idx:
                # sent_str[i] += "<eos>"
                break
            else:
                sent_str[i] += i2w[word_id] + " "
        sent_str[i] = sent_str[i].strip()

    return sent_str


def word2idx(sentences, w2i):
    idx = [[] for i in range(len(sentences))]
    for i, sent in enumerate(sentences):
        for token in sent:
            idx[i].append(w2i[token])

    return idx


def surface_realisation(idx, i2w, eos_idx, slotdic):
    utterances = [str() for i in range(len(idx))]
    labellings = [str() for i in range(len(idx))]

    for isent, sent in enumerate(idx):

        for word_id in sent:
            word = i2w[word_id]
            if word_id == eos_idx:
                break
            if word.startswith('_') and word.endswith('_'):
                slot = word.lstrip('_').rstrip('_')
                slot_values = slotdic[slot]
                slot_choice = random.choice(slot_values)
                utterances[isent] += slot_choice + " "
                for i, word in enumerate(slot_choice.split(' ')):
                    if word == '':
                        continue
                    if i == 0:
                        labellings[isent] += ('B-' + slot + ' ')
                    else:
                        labellings[isent] += ('I-' + slot + ' ')
            else:
                labellings[isent] += 'O '
                utterances[isent] += word + " "

    return labellings, utterances


def get_groups(words, labels):
    """
    Extracts text, slot name and slot value from a tokenized sentence and
    its BIO labelling
    """

    prev_label = None
    groups = []

    zipped = zip(words, labels)
    for i, (word, label) in enumerate(zipped):
        if label.startswith('B-'):  # start slot group
            if i != 0:
                groups.append(group)  # dump previous group
            slot = label.lstrip('B-')
            group = {'text': (word + ' '), 'entity': slot, 'slot_name': slot}
        elif (label == 'O' and prev_label != 'O'):  # start context group
            if i != 0:
                groups.append(group)  # dump previous group
            group = {'text': (word + ' ')}
        else:
            group['text'] += (word + ' ')
        prev_label = label

    groups.append(group)  # dump previous group

    return groups
