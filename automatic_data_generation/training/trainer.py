#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import logging

import torch
from tqdm import tqdm

from automatic_data_generation.training.losses import compute_bow_loss, \
    compute_label_loss, compute_recon_loss, compute_kl_loss
from automatic_data_generation.utils.utils import to_device

LOGGER = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
                 dataset,
                 model,
                 optimizer,
                 batch_size=64,
                 annealing_strategy='logistic',
                 kl_anneal_time=0.005,
                 kl_anneal_rate=1000,
                 kl_anneal_target=1.,
                 label_anneal_time=0.01,
                 label_anneal_rate=100,
                 label_anneal_target=1.,
                 add_bow_loss=False,
                 print_loss_every=50,
                 record_loss_every=5,
                 force_cpu=False):

        self.force_cpu = force_cpu
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every

        self.annealing_strategy = annealing_strategy
        self.kl_anneal_time = kl_anneal_time
        self.kl_anneal_rate = kl_anneal_rate
        self.kl_anneal_target = kl_anneal_target
        self.label_anneal_time = label_anneal_time
        self.label_anneal_rate = label_anneal_rate
        self.label_anneal_target = label_anneal_target
        self.add_bow_loss = add_bow_loss

        self.epoch = -1
        self.step = 0
        self.latent_rep = {i: [] for i in range(self.model.n_classes)}

    def run(self, n_epochs, dev_step_every_n_epochs=1):
        train_iter, val_iter = self.dataset.get_iterators(
            batch_size=self.batch_size)

        for idx in range(n_epochs):
            self.epoch += 1
            is_last_epoch = self.epoch == n_epochs
            train_loss = self.do_one_sweep(train_iter, is_last_epoch, "train")
            LOGGER.info('Training loss after epoch %d: %f', self.epoch,
                        train_loss)

            if (idx + 1) % dev_step_every_n_epochs == 0:
                train_loss = self.do_one_sweep(val_iter, is_last_epoch, "dev")
                LOGGER.info('Dev loss after epoch %d: %f', self.epoch,
                            train_loss)

    def do_one_sweep(self, iter, is_last_epoch, train_or_dev):
        # TODO book keeping
        if train_or_dev not in ['train', 'dev']:
            raise TypeError("train_or_dev should be either train or dev")

        if train_or_dev == "train":
            self.model.train()
        else:
            self.model.eval()

        sweep_loss = 0
        for iteration, batch in enumerate(tqdm(iter)):
            if len(batch) < self.batch_size:
                continue
            self.step += 1
            batch = batch.to(self.device)
            if train_or_dev == "train":
                self.optimizer.zero_grad()

            # compute loss
            x, lengths = getattr(batch, self.dataset.input_type)
            input = x[:, :-1]  # remove <eos>
            target = x[:, 1:]  # remove <sos>
            lengths -= 1  # account for the removal
            input, target = to_device(input, self.force_cpu), to_device(target)

            if self.model.conditional:
                y = batch.intent.squeeze()
                y = to_device(y)
                sorted_lengths, sorted_idx = torch.sort(lengths,
                                                        descending=True)
                y = y[sorted_idx]

            logp, mean, logv, logc, z, bow = self.model(input, lengths)

            if train_or_dev == "train":
                if is_last_epoch and self.model.conditional:
                    for i, intent in enumerate(y):
                        self.latent_rep[int(intent)].append(
                            z[i].cpu().detach().numpy()
                        )

            loss = self.compute_loss(logp, bow, target, lengths, mean, logv,
                                     logc, y)

            sweep_loss += loss

            self.summary_writer.add_scalar(
                train_or_dev+'/loss', loss, self.step)

            if train_or_dev == "train":
                loss.backward()
                self.optimizer.step()

            return sweep_loss / (iteration + 1)

    def compute_loss(self, logp, bow, target, length, mean, logv, logc, y):
        # TODO book keeping
        batch_size, seqlen, vocab_size = logp.size()
        target = target.view(batch_size, -1)

        recon_loss = compute_recon_loss(
            self.dataset.pad_idx, vocab_size, length, logp, target)

        kl_loss, kl_weight = compute_kl_loss(
            logv, mean, self.annealing_strategy, self.step,
            self.kl_anneal_time, self.kl_anneal_rate, self.kl_anneal_target)

        loss = (recon_loss + kl_weight * kl_loss)

        if self.add_bow_loss:
            loss += compute_bow_loss(batch_size, bow, target)

        if self.model.conditional == 'supervised':
            label_loss, label_weight = compute_label_loss(
                logc, y, self.annealing_strategy, self.step,
                self.label_anneal_time, self.label_anneal_rate,
                self.label_anneal_target)
            loss += label_weight * label_loss
        elif self.model.conditional == 'unsupervised':
            entropy = torch.sum(
                torch.exp(logc) * torch.log(
                    self.model.n_classes * torch.exp(logc)
                )
            )
            loss += entropy

        return loss / batch_size
