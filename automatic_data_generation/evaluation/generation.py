#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from pathlib import Path

from automatic_data_generation.data.utils import (idx2word, surface_realisation,
                                                  word2idx)
from automatic_data_generation.utils.io import (read_csv, write_csv)


def generate_vae_sentences(model, n_to_generate, input_type, i2int, i2w,
                           eos_idx, slotdic, verbose=False):
    generated_sentences = {}

    model.eval()
    samples, z, y_onehot, logp = model.inference(n=n_to_generate)
    samples = samples.cpu().numpy()

    generated_sentences['samples'] = samples

    if model.conditional is not None:
        preds = y_onehot.data.max(1)[1].cpu().numpy()
        intents = [i2int[pred] for pred in preds]
        generated_sentences['intents'] = intents

    if input_type == 'delexicalised':
        delexicalised = idx2word(samples, i2w=i2w, eos_idx=eos_idx)
        labellings, utterances = surface_realisation(samples, i2w=i2w,
                                                     eos_idx=eos_idx,
                                                     slotdic=slotdic)
        generated_sentences['labellings'] = labellings
        generated_sentences['delexicalised'] = delexicalised
        generated_sentences['utterances'] = utterances
    else:
        utterances = idx2word(samples, i2w=i2w, eos_idx=eos_idx)
        generated_sentences['utterances'] = utterances

    if verbose:
        print('----------GENERATED----------')
        for i in range(n_to_generate):
            if model.conditional is not None:
                print('Intents   : ', intents[i])
            if input_type == 'delexicalised':
                print('Delexicalised : ', delexicalised[i])
            print('Utterances : ', utterances[i] + '\n')

    return generated_sentences, logp


def generate_slot_expansion_sentences(delexicalised, intents, n_to_generate,
                                      w2i, i2w, eos_idx, slotdic):
    # TODO: randomize the sentences and intents before generation
    slot_expansion_sentences = {}
    slot_expansion_delexicalised = list(delexicalised)[
                                   :n_to_generate]
    slot_expansion_intents = list(intents)[
                             :n_to_generate]
    slot_expansion_samples = word2idx(slot_expansion_delexicalised,
                                      w2i=w2i)
    slot_expansion_labellings, slot_expansion_utterances = surface_realisation(
        slot_expansion_samples, i2w=i2w, eos_idx=eos_idx,
        slotdic=slotdic)
    slot_expansion_sentences['samples'] = slot_expansion_samples
    slot_expansion_sentences['intents'] = slot_expansion_intents
    slot_expansion_sentences['labellings'] = slot_expansion_labellings
    slot_expansion_sentences['delexicalised'] = slot_expansion_delexicalised
    slot_expansion_sentences['utterances'] = slot_expansion_utterances

    return slot_expansion_sentences


def save_augmented_dataset(generated_sentences, n_generated, train_path,
                           output_dir):
    dataset = read_csv(train_path)
    for s, l, d, i in zip(generated_sentences['utterances'],
                          generated_sentences['labellings'],
                          generated_sentences['delexicalised'],
                          generated_sentences['intents']):
        dataset.append([s, l, d, i])
    augmented_path = output_dir / Path(train_path.name.replace(
        '.csv', '_aug_{}.csv'.format(n_generated)
    ))
    write_csv(dataset, augmented_path)
