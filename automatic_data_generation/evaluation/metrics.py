import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import (sentence_bleu, SmoothingFunction)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def my_remove(list, elt):
    list.remove(elt)
    return list


def compute_generation_metrics(dataset, sentences, intents, logp,
                               input_type='utterance', compute_entropy=True):
    i2int = dataset.i2int
    references_train = {intent: [] for intent in i2int}
    references_valid = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in list(dataset.valid):
        if example.intent in i2int:
            references_valid[example.intent].append(
                getattr(example, input_type))
    for example in list(dataset.train):
        if example.intent in i2int:
            references_train[example.intent].append(
                getattr(example, input_type))
    for i, example in enumerate(sentences):
        candidates[intents[i]].append(dataset.tokenize(example))

    del references_train['None']
    del references_valid['None']
    del candidates['None']

    accuracies = intent_classification(
        candidates,
        train_path=dataset.original_train_path,
        input_type=input_type
    )  # only keep well-conditioned candidates

    bleu_scores = calc_bleu(candidates, references_valid, input_type)
    originality, transfer = calc_originality_and_transfer(candidates,
                                                          references_train,
                                                          input_type)
    diversity = calc_diversity(dataset, sentences)
    if input_type == 'utterance' and compute_entropy:
        entropy = calc_entropy(logp)
    else:
        entropy = None

    return {
        'bleu_scores': bleu_scores,
        'originality': originality,
        'transfer': transfer,
        'diversity': diversity,
        'intent_accuracy': accuracies,
        'entropy': entropy
    }


def calc_bleu(candidates, references, type='utterance'):
    bleu_scores = {'quality': {}, 'diversity': {}, 'original_diversity': {}}
    cc = SmoothingFunction()

    for intent in candidates.keys():
        # try:
        # QUALITY
        bleu_scores['quality'][intent] = np.mean(
            [sentence_bleu(
                references[intent],
                candidate,
                weights=[0.25, 0.25, 0.25, 0.25],
                smoothing_function=cc.method1
            ) for candidate in candidates[intent]]
        )

        # DIVERSITY
        bleu_scores['diversity'][intent] = np.mean(
            [1 - sentence_bleu(
                my_remove(candidates[intent], candidate),
                candidate,
                weights=[0.25, 0.25, 0.25, 0.25],
                smoothing_function=cc.method1
            ) for candidate in candidates[intent]]
        )

        # ORIGINAL DIVERSITY
        bleu_scores['original_diversity'][intent] = np.mean(
            [1 - sentence_bleu(
                my_remove(references[intent], reference),
                reference,
                weights=[0.25, 0.25, 0.25, 0.25],
                smoothing_function=cc.method1
            ) for reference in references[intent]]
        )
        # except:
        #     print("Failed for intent %s" % intent)

    bleu_scores['quality']['avg'] = np.mean(
        [bleu_score for bleu_score in bleu_scores['quality'].values()])
    bleu_scores['diversity']['avg'] = np.mean(
        [bleu_score for bleu_score in bleu_scores['diversity'].values()])
    bleu_scores['original_diversity']['avg'] = np.mean(
        [bleu_score for bleu_score in
         bleu_scores['original_diversity'].values()])

    return bleu_scores


def calc_originality_and_transfer(candidates, references, type='utterance'):
    originality = {}
    transfer = {'metric': {}, 'tokens': {}}

    # ORIGINALITY
    original_sentences = []
    for intent in candidates.keys():
        original = [candidate for candidate in candidates[intent] if
                    candidate not in references[intent]]
        original_sentences += original
        originality[intent] = float(len(original) / len(candidates[intent]))
    originality['avg'] = np.mean([x for x in originality.values()])

    # TRANSFER
    ref_vocabs = {intent: CountVectorizer().fit(
        list(map(' '.join, references[intent]))).vocabulary_ for intent in
                  candidates.keys()}
    cand_vocabs = {intent: CountVectorizer().fit(
        list(map(' '.join, candidates[intent]))).vocabulary_ for intent in
                   candidates.keys()}

    for intent in candidates.keys():
        transferred = [token for token in cand_vocabs[intent] if
                       token not in ref_vocabs[intent]]
        transfer['tokens'][intent] = transferred
        transfer['metric'][intent] = len(transferred) / len(
            cand_vocabs[intent])
    transfer['metric']['avg'] = np.mean(
        [x for x in transfer['metric'].values()])

    return originality, transfer


def calc_entropy(logp):
    normalization = 1. / logp.numel()
    entropy = - normalization * torch.sum(logp.view(-1))
    # perplexity = torch.exp(entropy).item()
    return entropy.item()


def calc_diversity(dataset, sentences):
    tokens = np.concatenate(
        [dataset.tokenize(sentence) for sentence in sentences])
    return len(set(tokens)) / float(len(tokens))


def intent_classification(candidates, train_path, input_type):
    """
        This only works for snips dataset for now...
    """
    accuracies = {}
    intent_classifier = train_intent_classifier(train_path, input_type)
    for intent, tokenized_sentences in candidates.items():
        sentences = [' '.join(sentence) for sentence in tokenized_sentences]
        preds = intent_classifier.predict(sentences)
        corrects = [pred == intent for pred in preds]
        candidates[intent] = [cand for i, cand in enumerate(candidates[intent])
                              if corrects[i]]
        accuracy = sum(corrects) / len(preds)
        accuracies[intent] = accuracy
    accuracies['avg'] = np.mean([x for x in accuracies.values()])
    return accuracies


def train_intent_classifier(train_path, input_type):
    data = pd.read_csv(train_path)
    X = getattr(data, input_type)
    y = data.intent
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', SGDClassifier())
    ])
    model.fit(X, y)

    class IntentClassifier():
        def __init__(self, model, label_encoder):
            self.label_encoder = label_encoder
            self.classifier = model

        def predict(self, sentences):
            labels = model.predict(sentences)
            intents = list(label_encoder.inverse_transform(labels))
            return intents

    intent_classifier = IntentClassifier(model, label_encoder)

    return intent_classifier
