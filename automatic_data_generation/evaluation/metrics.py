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


def compute_generation_metrics(dataset, generated_utterances,
                               generated_intents, logp,
                               input_type='utterance', compute_entropy=True):
    bleu_scores = calc_bleu(dataset, generated_utterances,
                            generated_intents, input_type)
    originality, transfer = calc_originality_and_transfer(dataset, generated_utterances,
                            generated_intents, input_type)
    diversity = calc_diversity(dataset, generated_utterances)
    intent_accuracy = intent_classification(
        generated_utterances,
        generated_intents,
        train_path=dataset.original_train_path,
        input_type=input_type
    )
    if input_type == 'utterance' and compute_entropy:
        entropy = calc_entropy(logp)
    else:
        entropy = None

    return {
        'bleu_scores': bleu_scores,
        'originality': originality,
        'transfer' : transfer,
        'diversity': diversity,
        'intent_accuracy': intent_accuracy,
        'entropy': entropy
    }


def calc_bleu(dataset, sentences, intents, type='utterance'):

    bleu_scores = {'quality':{}, 'diversity':{}, 'original_diversity':{}}

    i2int = dataset.intent.vocab.itos
    int2i = dataset.intent.vocab.stoi
    cc = SmoothingFunction()
    references = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in list(dataset.valid): # VALIDATION SET
        if example.intent in i2int:
            references[example.intent].append(getattr(example, type))
    for i, example in enumerate(sentences):
        candidates[intents[i]].append(dataset.tokenize(example))

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

def calc_originality_and_transfer(dataset, sentences, intents, type='utterance'):

    originality = {}
    transfer = {}

    i2int = dataset.intent.vocab.itos
    int2i = dataset.intent.vocab.stoi
    references = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in dataset.train: # TRAINING SET
        references[example.intent].append(getattr(example, type))
    for i, example in enumerate(sentences): 
        candidates[intents[i]].append(dataset.tokenize(example))

    print(candidates)
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


def intent_classification(sentences, intents, train_path, input_type):
    """
        This only works for snips dataset for now...
    """

    # if not os.path.exists('clf.pkl'):
    print('Training intent classifier')
    intent_classifier = train_intent_classifier(train_path, input_type)

    # else:
    # with open('clf.pkl', 'rb') as f:
    #     intent_classifier = pickle.load(f)

    preds = intent_classifier.predict(sentences)

    accuracy = float(sum([pred==intent for pred, intent in zip(preds,intents) if intent != 'None'])
                     / len([intent for intent in intents if intent != 'None'])) 

    return accuracy


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

    # with open('clf.pkl', 'wb') as f:
    #         pickle.dump(intent_classifier, f)

    return intent_classifier
