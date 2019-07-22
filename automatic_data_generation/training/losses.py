"""
Loss functions for VAE trainings, note that the loss should not be
normalized by the batch size as it is done in the Trainer
"""

import numpy as np
import torch


def annealing_fn(annealing_strategy, step, k, x, m):
    if annealing_strategy == 'logistic':
        return m * float(1 / (1 + np.exp(-k * (step - x))))
    elif annealing_strategy == 'linear':
        return m * min(1, step / x)


def compute_recon_loss(pad_idx, vocab_size, length, logp, target):
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, vocab_size)
    # Negative Log Likelihood
    nll_recon = torch.nn.NLLLoss(reduction='sum', ignore_index=pad_idx)
    return nll_recon(logp, target)


def compute_kl_loss(logv, mean, annealing_strategy, step, k, x, m):
    kl_weight = annealing_fn(annealing_strategy, step, k, x, m)
    kl_losses = -0.5 * torch.sum((1 + logv - mean.pow(2) - logv.exp()), dim=0)
    return kl_weight, kl_losses


def compute_bow_loss(batch_size, bow, target):
    if bow is not None:
        bow.view(batch_size, -1)
        return - torch.einsum('iik->', bow[:, target])
    else:
        return torch.Tensor([0])


def compute_label_loss(logc, target, annealing_strategy, step, k, x, m, none_idx, alpha):
    # Negative Log Likelihood
    preds = logc.max(1)[1]
    none_mask = preds==none_idx
    label_weight = annealing_fn(annealing_strategy, step, k, x, m)
    nll_label = torch.nn.NLLLoss(reduction='none') #, ignore_index=none_idx)
    label_loss = nll_label(logc, target)
    label_loss[none_mask] *= alpha
    label_loss = torch.sum(label_loss)
    return label_weight, label_loss
