#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from automatic_data_generation.training.losses import (compute_bow_loss,
                                                       compute_label_loss,
                                                       compute_recon_loss,
                                                       compute_kl_loss)
from automatic_data_generation.utils.utils import to_device

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Trainer(object):
    def __init__(
            self, dataset, model, optimizer, batch_size=64,
            annealing_strategy='logistic',
            kl_anneal_rate=0.01, kl_anneal_time=100, kl_anneal_target=1.,
            label_anneal_rate=0.01, label_anneal_time=100,
            label_anneal_target=1.,
            add_bow_loss=False, force_cpu=False,
            run_dir=None, alpha=1.
    ):

        self.force_cpu = force_cpu
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.i2w = dataset.i2w
        self.i2int = dataset.i2int
        self.w2i = dataset.w2i
        self.int2i = dataset.int2i

        self.batch_size = batch_size

        self.annealing_strategy = annealing_strategy
        self.kl_anneal_time = kl_anneal_time
        self.kl_anneal_rate = kl_anneal_rate
        self.kl_anneal_target = kl_anneal_target
        self.label_anneal_time = label_anneal_time
        self.label_anneal_rate = label_anneal_rate
        self.label_anneal_target = label_anneal_target
        self.add_bow_loss = add_bow_loss
        self.alpha = alpha

        self.epoch = 0
        self.step = 0
        self.latent_rep = {intent: [] for intent in self.i2int}

        self.run_logs = {
            'train': {
                'recon_loss': [],
                'kl_losses': [[] for _ in range(self.model.z_size)],
                'conditioning_accuracy': [],
                'total_loss': [],
                'classifications': {
                    real_intent: {pred_intent: 0 for pred_intent in self.i2int}
                    for real_intent in self.i2int
                },
                'transfer': {
                    real_intent: torch.zeros(model.cat_size) for
                    real_intent in self.i2int}
            },
            'dev': {
                'recon_loss': [],
                'kl_loss': [],
                'conditioning_accuracy': [],
                'total_loss': [],
                'classifications': {
                    real_intent: {pred_intent: 0 for pred_intent in self.i2int}
                    for real_intent in self.i2int
                },
                'transfer': {
                    real_intent: torch.zeros(model.cat_size) for
                    real_intent in self.i2int
                }
            }
        }
        self.summary_writer = SummaryWriter(log_dir=run_dir)

    def run(self, n_epochs, dev_step_every_n_epochs=1):
        train_iter, val_iter = self.dataset.get_iterators(
            batch_size=self.batch_size)

        for idx in range(n_epochs):
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            self.epoch += 1
            is_last_epoch = self.epoch == n_epochs - 1
            train_loss, train_recon_loss, train_kl_loss, train_acc = self.do_one_sweep(
                train_iter, is_last_epoch, "train")

            # logging
            LOGGER.info('Training loss after epoch %d: %f', self.epoch,
                        train_loss.item())
            LOGGER.info('Training reconstruction loss after epoch %d: %f',
                        self.epoch, train_recon_loss.item())
            LOGGER.info('Training KL loss after epoch %d: %f',
                        self.epoch, train_kl_loss.item())
            LOGGER.info('Training accuracy after epoch %d: %f',
                        self.epoch, train_acc)
            self.summary_writer.add_scalar(
                'train/total-loss', train_loss.cpu().detach().numpy().item(),
                self.epoch)
            self.run_logs['train']['total_loss'].append(
                train_loss.cpu().detach().numpy().item())

            if (idx + 1) % dev_step_every_n_epochs == 0:
                dev_loss, dev_recon_loss, dev_kl_loss, dev_acc = self.do_one_sweep(
                    val_iter, is_last_epoch, "dev")

                # logging
                LOGGER.info('Dev loss after epoch %d: %f', self.epoch,
                            dev_loss)
                LOGGER.info('Dev recon loss after epoch %d: %f',
                            self.epoch, dev_recon_loss)
                LOGGER.info('Dev KL loss after epoch %d: %f',
                            self.epoch, dev_kl_loss)
                LOGGER.info('Dev acc after epoch %d: %f',
                            self.epoch, dev_acc)
                # summaries
                self.summary_writer.add_scalar(
                    'dev/recon-loss',
                    dev_recon_loss.cpu().detach().numpy().item(), self.epoch)
                self.run_logs['dev']['recon_loss'].append(
                    dev_recon_loss.cpu().detach().numpy().item())
                self.summary_writer.add_scalar(
                    'dev/kl-loss', dev_kl_loss.detach().cpu().detach().numpy(),
                    self.epoch)
                self.run_logs['dev']['kl_loss'].append(
                    dev_kl_loss.cpu().detach().numpy().item())
                self.summary_writer.add_scalar(
                    'dev/total-loss', dev_loss.detach().cpu().detach().numpy(),
                    self.epoch)
                self.run_logs['dev']['total_loss'].append(
                    dev_loss.cpu().detach().numpy().item())

    def do_one_sweep(self, iter, is_last_epoch, train_or_dev):
        if train_or_dev not in ['train', 'dev']:
            raise TypeError("train_or_dev should be either train or dev")

        if train_or_dev == "train":
            self.model.train()
        else:
            self.model.eval()

        sweep_loss = 0
        sweep_recon_loss = 0
        sweep_kl_loss = 0
        sweep_accuracy = 0
        n_batches = 0
        for iteration, batch in enumerate(tqdm(iter)):
            # if len(batch) < self.batch_size and :
            #     continue
            if train_or_dev == "train":
                self.step += 1
                self.optimizer.zero_grad()

            # forward pass
            x, lengths = getattr(batch, self.dataset.input_type)
            input = x[:, :-1]  # remove <eos>
            target = x[:, 1:]  # remove <sos>
            lengths -= 1  # account for the removal
            input, target = to_device(input, self.force_cpu), to_device(
                target, self.force_cpu)

            y = None
            if self.model.conditional is not None:
                y = batch.intent.squeeze()
                y = to_device(y, self.force_cpu)
                sorted_lengths, sorted_idx = torch.sort(lengths,
                                                        descending=True)
                y = y[sorted_idx]

            logp, mean, logv, logc, z, bow = self.model(input, lengths)

            if is_last_epoch:
                _, reversed_idx = torch.sort(sorted_idx)
                y = y[reversed_idx]
                logc = logc[reversed_idx]
                real_labels = [self.i2int[label] for label in y]
                pred_labels = [
                    self.i2int[label] if label < len(self.i2int) else 'None'
                    for label in logc.max(1)[1]]
                for real_label, pred_label in zip(real_labels, pred_labels):
                    self.run_logs[train_or_dev]['classifications'][real_label][
                        pred_label] += 1
                for real_label in real_labels:
                    self.run_logs[train_or_dev]['transfer'][
                        real_label] += logc.sum(dim=0).cpu().detach()

                # save latent representation
                if train_or_dev == "train" and self.model.conditional:
                    for i, intent in enumerate(y):
                        self.latent_rep[self.i2int[intent]].append(
                            z[i].cpu().detach().numpy()
                        )

            # loss calculation
            loss, recon_loss, kl_loss, accuracy = self.compute_loss(
                logp, bow, target, lengths, mean, logv, logc, y, train_or_dev)

            sweep_loss += loss
            sweep_recon_loss += recon_loss
            sweep_kl_loss += kl_loss
            sweep_accuracy += accuracy

            n_batches += 1
            if train_or_dev == "train":
                loss.backward()
                self.optimizer.step()

        if is_last_epoch:
            for intent1 in self.i2int:
                n_sentences = sum(
                    self.run_logs[train_or_dev]['classifications'][
                        intent1].values())
                self.run_logs[train_or_dev]['transfer'][intent1] /= n_sentences
                for intent2 in self.i2int:
                    self.run_logs[train_or_dev]['classifications'][intent1][
                        intent2] /= n_sentences

        return sweep_loss / n_batches, sweep_recon_loss / n_batches, \
               sweep_kl_loss / n_batches, sweep_accuracy / n_batches

    def compute_loss(self, logp, bow, target, length, mean, logv, logc, y,
                     train_or_dev):
        batch_size, seqlen, vocab_size = logp.size()
        target = target.view(batch_size, -1)

        # reconstruction loss
        recon_loss = compute_recon_loss(
            self.dataset.pad_idx, vocab_size, length, logp, target)

        # kl loss
        kl_weight, kl_losses = compute_kl_loss(
            logv, mean, self.annealing_strategy, self.step,
            self.kl_anneal_rate, self.kl_anneal_time, self.kl_anneal_target)
        kl_loss = torch.sum(kl_losses)

        total_loss = (recon_loss + kl_weight * kl_loss)

        # bow loss
        if self.add_bow_loss:
            total_loss += compute_bow_loss(batch_size, bow, target)

        # labels loss
        if self.model.conditional == 'supervised':
            if 'None' in self.i2int:
                none_idx = self.int2i['None']
            else:
                none_idx = -100
            label_loss, label_weight = compute_label_loss(
                logc, y, self.annealing_strategy, self.step,
                self.label_anneal_time, self.label_anneal_rate,
                self.label_anneal_target, none_idx, self.alpha)
            total_loss += label_weight * label_loss
        elif self.model.conditional == 'unsupervised':
            entropy = torch.sum(
                torch.exp(logc) * torch.log(
                    self.model.n_classes * torch.exp(logc)
                )
            )
            total_loss += entropy

        # summaries
        if train_or_dev == "train":
            self.summary_writer.add_scalar(
                train_or_dev + '/recon-loss',
                recon_loss.detach().cpu().detach().numpy() / batch_size,
                self.step)
            self.run_logs[train_or_dev]['recon_loss'].append(
                recon_loss.cpu().detach().numpy() / batch_size
            )
            for i in range(self.model.z_size):
                self.summary_writer.add_scalars(
                    train_or_dev + '/kl-losses',
                    {str(i): kl_losses[i].cpu().detach().numpy().item() /
                             batch_size},
                    self.step
                )
                self.run_logs[train_or_dev]['kl_losses'][i].append(
                    kl_losses[i].cpu().detach().numpy().item() / batch_size
                )
        n_correct = 0
        if self.model.conditional is not None:
            mask = y != self.int2i['None']  # ignore nones
            pred_labels = logc[mask].data.max(1)[1].long()
            true_labels = y[mask].data
            n_correct = pred_labels.eq(true_labels).cpu().sum().float().item()
        self.summary_writer.add_scalar(
            train_or_dev + '/conditioning-accuracy',
            n_correct / len(true_labels),
            self.step)
        self.run_logs[train_or_dev]['conditioning_accuracy'].append(
            n_correct / len(true_labels)
        )

        return total_loss / batch_size, recon_loss / batch_size, \
               kl_loss / batch_size, n_correct / len(true_labels)
