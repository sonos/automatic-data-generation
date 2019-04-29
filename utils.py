import torch
import numpy as np

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
            sent_str[i] += i2w[word_id] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str
