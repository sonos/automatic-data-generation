import torchtext


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
            elif preprocessing_type == 'none':
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
    intent = torchtext.data.Field(sequential=False, batch_first=True,
                                  unk_token=None)
    return text, delex, intent


def get_datafields(dataset_type, text, delex, intent):
    skip_header = True
    if dataset_type == 'snips':
        datafields = [("utterance", text), ("labels", None),
                      ("delexicalised", delex), ("intent", intent)]
    elif dataset_type == 'atis':
        datafields = [(" ", None), ("utterance", text), (" ", None),
                      ("intent", intent)]
    elif dataset_type == 'sentiment':
        datafields = [("intent", intent), ("", None), ("", None),
                      ("", None), ("", None), ("utterance", text)]
    elif dataset_type == 'yelp':
        datafields = [("", None), ("", None), ("", None),
                      ("intent", intent), ("", None), ("utterance", text),
                      ("", None), ("", None), ("", None)]
    elif dataset_type == 'spam':
        datafields = [("utterance", text), ("intent", intent)]
    elif dataset_type == 'ptb':
        datafields = [("utterance", text)]
        skip_header = False
    else:
        raise TypeError("Unknown dataset type")
    return datafields, skip_header
