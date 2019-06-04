import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from automatic_data_generation.models.intent_classification import RNN_classifier

def my_remove(list, elt):
    list.remove(elt)
    return list

def calc_bleu(sentences, intents, datasets):

    bleu_scores = {'quality':{}, 'diversity':{}}

    i2int = datasets.INTENT.vocab.itos
    int2i = datasets.INTENT.vocab.stoi
    cc = SmoothingFunction()
    references = {intent: [] for intent in i2int}
    candidates = {intent: [] for intent in i2int}

    for example in datasets.train: # REFERENCES
        references[example.intent].append(example.utterance)
    for i, example in enumerate(sentences): # CANDIDATES
        candidates[i2int[intents[i]]].append(datasets.tokenize(example))

    for intent in i2int:

        # QUALITY
        bleu_scores['quality'][intent] = np.mean(
            [sentence_bleu(references[intent], candidate, weights=[0.5, 0.5, 0, 0], smoothing_function=cc.method1) for
             candidate in candidates[intent]])

        #DIVERSITY
        bleu_scores['diversity'][intent] = np.mean(
            [1-sentence_bleu(my_remove(candidates[intent],candidate), candidate, weights=[0.5, 0.5, 0, 0], smoothing_function=cc.method1)
             for candidate in candidates[intent]])

    bleu_scores['quality']['avg'] = np.mean([bleu_score for bleu_score in bleu_scores['quality'].values()])
    bleu_scores['diversity']['avg'] = np.mean([bleu_score for bleu_score in bleu_scores['diversity'].values()])

    return bleu_scores



def calc_entropy(logp):
    normalization = 1./logp.numel()
    entropy = - normalization * torch.sum(logp.view(-1))
    # perplexity = torch.exp(entropy).item()
    return entropy.item()

def calc_diversity(sentences, datasets):
    tokens = np.concatenate([datasets.tokenize(sentence) for sentence in sentences])
    diversity = len(set(tokens)) / float(len(tokens))
    return diversity

def intent_classification(sentences, intents, datasets):
    state_dict = torch.load('intent_classifier.pyT')

    model = RNN_classifier()
    model.load_state_dict(state_dict)
    model.eval()  # turn on evaluation mode
    input = datasets.tokenize(sentences)
    preds = model()
    preds = F.log_softmax(preds, dim=1)
    loss = loss_func(preds, intents)

    pred_labels = preds.data.max(1)[1].long()
    val_corrects = pred_labels.eq(y.data).cpu().sum()
    accuracy = val_corrects/samples.size(0)

    return accuracy.item()
