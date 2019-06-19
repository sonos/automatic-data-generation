import os
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle

def my_remove(list, elt):
    list.remove(elt)
    return list

def calc_bleu(sentences, intents, datasets, type='utterance'):

    bleu_scores = {'quality':{}, 'diversity':{}, 'original_diversity':{}}

    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    cc = SmoothingFunction()
    references = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in list(datasets.valid): # VALIDATION SET
        if example.intent in i2int:
            references[example.intent].append(getattr(example, type))
    for i, example in enumerate(sentences):
        candidates[intents[i]].append(datasets.tokenize(example))

    for intent in i2int:

        # QUALITY
        bleu_scores['quality'][intent] = np.mean(
            [sentence_bleu(references[intent], candidate, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=cc.method1) for
             candidate in candidates[intent]])

        # DIVERSITY
        bleu_scores['diversity'][intent] = np.mean(
            [1-sentence_bleu(my_remove(candidates[intent],candidate), candidate, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=cc.method1)
             for candidate in candidates[intent]])

        # ORIGINAL DIVERSITY
        bleu_scores['original_diversity'][intent] = np.mean(
            [1-sentence_bleu(my_remove(references[intent],reference), reference, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=cc.method1)
             for reference in references[intent]])

    bleu_scores['quality']['avg'] = np.mean([bleu_score for bleu_score in bleu_scores['quality'].values()])
    bleu_scores['diversity']['avg'] = np.mean([bleu_score for bleu_score in bleu_scores['diversity'].values()])
    bleu_scores['original_diversity']['avg'] = np.mean([bleu_score for bleu_score in bleu_scores['original_diversity'].values()])

    return bleu_scores


def calc_originality_and_transfer(sentences, intents, datasets, type='utterance'):

    originality = {}
    transfer = {}

    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    references = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in datasets.train: # TRAINING SET
        references[example.intent].append(getattr(example, type))
    for i, example in enumerate(sentences): 
        candidates[intents[i]].append(datasets.tokenize(example))

    # ORIGINALITY
    original_sentences = []
    for intent in i2int:
        original = [candidate for candidate in candidates[intent] if candidate not in references[intent]]
        original_sentences += original
        originality[intent] = float(len(original) / len(candidates[intent]))
    originality['avg'] = np.mean([x for x in originality.values()])

    # TRANSFER
    from sklearn.feature_extraction.text import CountVectorizer
    ref_vocabs  = {intent: CountVectorizer().fit(list(map(' '.join, references[intent]))).vocabulary_ for intent in i2int}
    cand_vocabs = {intent: CountVectorizer().fit(list(map(' '.join, candidates[intent]))).vocabulary_ for intent in i2int}

    for intent in i2int:
        transferred = [token for token in cand_vocabs[intent] if token not in ref_vocabs[intent]]
        transfer[intent] = len(transferred)/len(cand_vocabs[intent])
        print('Transferred to intent {} : '.format(intent), transferred)
    transfer['avg'] = np.mean([x for x in transfer.values()])

    return originality, transfer, original_sentences


def calc_entropy(logp):
    normalization = 1./logp.numel()
    entropy = - normalization * torch.sum(logp.view(-1))
    # perplexity = torch.exp(entropy).item()
    return entropy.item()

def calc_diversity(sentences, datasets):
    tokens = np.concatenate([datasets.tokenize(sentence) for sentence in sentences])
    diversity = len(set(tokens)) / float(len(tokens))
    return diversity

def intent_classification(sentences, intents, train_path, type='utterance'):
    '''This only works for snips dataset for now...'''

    # if not os.path.exists('clf.pkl'):
    print('Training intent classifier')
    intent_classifier = train_intent_classifier(train_path, type)

    # else:
    # with open('clf.pkl', 'rb') as f:
    #     intent_classifier = pickle.load(f)

    preds = intent_classifier.predict(sentences)

    accuracy = float(sum([pred==intent for pred, intent in zip(preds,intents) if intent != 'None'])
                     / len([intent for intent in intents if intent != 'None']))    

    return accuracy

def train_intent_classifier(train_path, type='utterance'):

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    
    data = pd.read_csv(train_path)
    X = getattr(data, type)
    y = data.intent
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    model = Pipeline([
                    ('vect',CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', SGDClassifier())
    ])
    model.fit(X, y)

    class IntentClassifier():
        def __init__(self, model, label_encoder):
            self.label_encoder = label_encoder
            self.classifier = model

        def predict(self,sentences):
            labels = model.predict(sentences)
            intents = list(label_encoder.inverse_transform(labels))
            return intents

    intent_classifier = IntentClassifier(model, label_encoder)

    # with open('clf.pkl', 'wb') as f:
    #         pickle.dump(intent_classifier, f)

    return intent_classifier
