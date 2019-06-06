import torch
import numpy as np
import random
import pickle
import csv

force_cpu = False

def to_device(x, volatile=False):
    if torch.cuda.is_available() and not force_cpu :
        x = x.cuda()

        return x

def create_augmented_dataset(args, raw_path, generated):
    augmented_path = raw_path.replace('.csv', '_aug{}.csv'.format(args.datasize, args.n_generated))
    print('Dumping augmented dataset at %s' %augmented_path)
    from shutil import copyfile            
    copyfile(raw_path, augmented_path)
    augmented_csv = open(augmented_path, 'a')
    augmented_writer = csv.writer(augmented_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for s, l, d, i in zip(generated['utterances'], generated['labellings'], generated['delexicalised'], generated['intents']):
        augmented_writer.writerow([s, l, d, i])
    return augmented_path

def idx2word(idx, i2w, eos_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:
            if word_id == eos_idx:
                #sent_str[i] += "<eos>"
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
                    if i==0:
                        labellings[isent] += ('B-'+slot+' ')
                    else:
                        labellings[isent] += ('I-'+slot+' ')
            else:
                labellings[isent] += 'O '
                utterances[isent] += word + " "

    return labellings, utterances


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T


def get_groups(words, labels):
    '''Extracts text, slot name and slot value from a tokenized sentence and its BIO labelling'''
    # import ipdb
    # ipdb.set_trace()
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
