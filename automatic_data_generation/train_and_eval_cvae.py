#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch

from automatic_data_generation.data.handlers.atis_dataset import AtisDataset
from automatic_data_generation.data.handlers.ptb_dataset import PTBDataset
from automatic_data_generation.data.handlers.snips_dataset import SnipsDataset
from automatic_data_generation.data.handlers.spam_dataset import SpamDataset
from automatic_data_generation.data.handlers.yelp_dataset import YelpDataset
from automatic_data_generation.evaluation.generation import \
    generate_vae_sentences, save_augmented_dataset, \
    generate_slot_expansion_sentences
from automatic_data_generation.evaluation.metrics import \
    compute_generation_metrics
from automatic_data_generation.models.cvae import CVAE
from automatic_data_generation.training.trainer import Trainer
from automatic_data_generation.utils.utils import to_device

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def train_and_eval_cvae(data_folder,
                        output_folder,
                        dataset_type,
                        input_type,
                        dataset_size,
                        tokenizer_type,
                        preprocessing_type,
                        max_sequence_length,
                        embedding_type,
                        embedding_dimension,
                        max_vocab_size,
                        slot_averaging,
                        conditioning,
                        bow_loss,
                        rnn_type,
                        hidden_size,
                        word_dropout_rate,
                        embedding_dropout_rate,
                        latent_size,
                        num_layers,
                        bidirectional,
                        temperature,
                        n_epochs,
                        optimizer_type,
                        learning_rate,
                        batch_size,
                        annealing_strategy,
                        kl_anneal_time,
                        kl_anneal_rate,
                        kl_anneal_target,
                        label_anneal_time,
                        label_anneal_rate,
                        label_anneal_target,
                        n_generated,
                        force_cpu):
    # output folder
    output_dir = Path(output_folder)
    if not output_dir.exists():
        output_dir.mkdir()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = output_dir / current_time
    run_dir.mkdir()

    # data handling
    data_folder = Path(data_folder)
    dataset_folder = data_folder / dataset_type
    slotdic = None
    if dataset_type == "snips":
        dataset = SnipsDataset(
            dataset_folder=dataset_folder,
            input_type=input_type,
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
        if input_type == "delexicalised":
            dataset.embed_slots(slot_averaging)
            slotdic = dataset.get_slotdic()
    elif dataset_type == "atis":
        dataset = AtisDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "spam":
        dataset = SpamDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "yelp":
        dataset = YelpDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    elif dataset_type == "penn-tree-bank":
        dataset = PTBDataset(
            dataset_folder=dataset_folder,
            input_type="utterance",
            dataset_size=dataset_size,
            tokenizer_type=tokenizer_type,
            preprocessing_type=preprocessing_type,
            max_sequence_length=max_sequence_length,
            embedding_type=embedding_type,
            embedding_dimension=embedding_dimension,
            max_vocab_size=max_vocab_size,
            output_folder=run_dir
        )
    else:
        raise TypeError("Unknown dataset type")

    dataset.embed_unks(num_special_toks=4)

    # training
    model = CVAE(
        conditional=conditioning,
        compute_bow=bow_loss,
        vocab_size=dataset.vocab_size,
        embedding_size=embedding_dimension,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        word_dropout_rate=word_dropout_rate,
        embedding_dropout_rate=embedding_dropout_rate,
        z_size=latent_size,
        n_classes=dataset.n_classes,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=max_sequence_length,
        num_layers=num_layers,
        bidirectional=bidirectional,
        temperature=temperature,
        force_cpu=force_cpu
    )

    model = to_device(model, force_cpu)
    optimizer = getattr(torch.optim, optimizer_type)(
        model.parameters(),
        lr=learning_rate
    )

    trainer = Trainer(
        dataset,
        model,
        optimizer,
        batch_size=batch_size,
        annealing_strategy=annealing_strategy,
        kl_anneal_time=kl_anneal_time,
        kl_anneal_rate=kl_anneal_rate,
        kl_anneal_target=kl_anneal_target,
        label_anneal_time=label_anneal_time,
        label_anneal_rate=label_anneal_rate,
        label_anneal_target=label_anneal_target,
        add_bow_loss=bow_loss,
        force_cpu=force_cpu,
        run_dir=run_dir / "tensorboard"
    )

    trainer.run(n_epochs,
                dev_step_every_n_epochs=1)

    model.save(run_dir / "model")

    # evaluation
    run_dict = dict()

    # generate queries
    generated_sentences, logp = generate_vae_sentences(
        model=model,
        n_to_generate=n_generated,
        input_type=input_type,
        i2int=dataset.i2int,
        i2w=dataset.i2w,
        eos_idx=dataset.eos_idx,
        slotdic=slotdic,
        verbose=True
    )
    run_dict['metrics'] = compute_generation_metrics(
        dataset,
        generated_sentences['utterances'],
        generated_sentences['intents'],
        logp
    )

    if input_type == "delexicalised":
        run_dict['delexicalised_metrics'] = compute_generation_metrics(
            dataset,
            generated_sentences['delexicalised'],
            generated_sentences['intents'],
            logp,
            input_type='delexicalised'
        )
        slot_expansion_sentences = generate_slot_expansion_sentences(
            delexicalised=dataset.train.delexicalised,
            intents=dataset.train.intent,
            n_to_generate=n_generated,
            w2i=dataset.w2i,
            i2w=dataset.i2w,
            eos_idx=dataset.eos_idx,
            slotdic=slotdic
        )
        run_dict['slot_expansion_metrics'] = compute_generation_metrics(
            dataset,
            slot_expansion_sentences['utterances'],
            slot_expansion_sentences['intents'],
            logp,
            input_type='utterance',
            compute_entropy=False
        )

    save_augmented_dataset(generated_sentences, n_generated,
                           dataset.train_path, run_dir)

    run_dict['logs'] = trainer.run_logs
    run_dict['latent_rep'] = trainer.latent_rep
    run_dict['i2w'] = dataset.i2w
    run_dict['w2i'] = dataset.w2i
    run_dict['vectors'] = {
        'before': dataset.vocab.vectors,
        'after': model.embedding.weight.data
    }
    run_path = str(run_dir / "run.pkl")
    torch.save(run_dict, run_path)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data-folder', type=str, default='data')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--dataset-type', type=str, default='snips',
                        choices=['snips', 'atis', 'sentiment', 'spam', 'yelp',
                                 'penn-tree-bank'])
    parser.add_argument('-it', '--input_type', type=str,
                        default='delexicalised',
                        choices=['delexicalised', 'utterance'])
    parser.add_argument('--dataset-size', type=int, default=None)
    parser.add_argument('--tokenizer-type', type=str, default='nltk',
                        choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--preprocessing-type', type=str, default=None,
                        choices=['stem', 'lemmatize'])
    parser.add_argument('-msl', '--max_sequence_length', type=int, default=60)
    #
    parser.add_argument('--embedding-type', type=str, default=None,
                        choices=['glove'])
    parser.add_argument('--embedding-dimension', type=int, default=100)
    parser.add_argument('-mvs', '--max-vocab-size', type=int, default=10000)
    parser.add_argument('--slot-averaging', type=str, default=None,
                        choices=['micro', 'macro'])
    # model
    parser.add_argument('--conditioning', type=str, default=None,
                        choices=['supervised', 'unsupervised', None])
    parser.add_argument('--bow-loss', type=bool, default=False)
    parser.add_argument('-rnn', '--rnn-type', type=str, default='gru',
                        choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden-size', type=int, default=256)
    parser.add_argument('-wd', '--word-dropout-rate', type=float, default=0.)
    parser.add_argument('-ed', '--embedding-dropout-rate', type=float,
                        default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=8)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-t', '--temperature', type=float, default=1)
    parser.add_argument('-bi', '--bidirectional', type=bool, default=False)

    # training
    parser.add_argument('-ep', '--n-epochs', type=int, default=5)
    parser.add_argument('-opt', '--optimizer-type', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-as', '--annealing-strategy', type=str,
                        default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k1', '--kl-anneal-time', type=float, default=0.01,
                        help='anneal time for KL weight')
    parser.add_argument('-x1', '--kl-anneal-rate', type=int, default=300,
                        help='anneal rate for KL weight')
    parser.add_argument('-m1', '--kl-anneal-target', type=float, default=1.,
                        help='final value for KL weight')
    parser.add_argument('-k2', '--label-anneal-time', type=float, default=0.01,
                        help='anneal time for label weight')
    parser.add_argument('-x2', '--label-anneal-rate', type=int, default=100,
                        help='anneal rate for label weight')
    parser.add_argument('-m2', '--label-anneal-target', type=float, default=1.,
                        help='final value for label weight')
    parser.add_argument('--force-cpu', type=bool, default=False)

    # evaluation
    parser.add_argument('-ng', '--n-generated', type=int, default=500)

    args = parser.parse_args()

    train_and_eval_cvae(**vars(args))


if __name__ == "__main__":
    main()
