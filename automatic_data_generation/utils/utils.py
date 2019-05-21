import torch
import numpy as np
import random
import pickle


def to_device(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:
            if word_id == pad_idx:
                sent_str[i] += "<pad>"                
                break
            # if word_id in [sos_idx, eos_idx]:
            #     continue
            sent_str[i] += i2w[word_id] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str


def surface_realisation(idx, i2w, pad_idx,
                        slotdic_path='./data/snips/train_slot_values.pkl'):

    with open(slotdic_path, 'rb') as f:
        slotdic = pickle.load(f)

    utterances = [str() for i in range(len(idx))]
    labellings = [str() for i in range(len(idx))]

    for isent, sent in enumerate(idx):

        for word_id in sent:
            word = i2w[word_id]
            if word_id == pad_idx:
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
