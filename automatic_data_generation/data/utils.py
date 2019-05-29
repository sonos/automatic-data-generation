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
            elif preprocessing_type is None:
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


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()] * len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:
            if word_id == pad_idx:
                sent_str[i] += "<pad>"
                break
            sent_str[i] += i2w[word_id] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str
