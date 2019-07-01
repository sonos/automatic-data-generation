#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import random

from automatic_data_generation.data.utils import NONE_COLUMN_MAPPING
from automatic_data_generation.evaluation.generation import \
    generate_vae_sentences
from automatic_data_generation.evaluation.metrics import \
    compute_generation_metrics
from automatic_data_generation.evaluation.utils import save_augmented_dataset
from automatic_data_generation.models.cvae import CVAE
from automatic_data_generation.training.trainer import Trainer
from automatic_data_generation.utils.constants import (NO_CONDITIONING,
                                                       NO_SLOT_AVERAGING,
                                                       NO_PREPROCESSING)
from automatic_data_generation.utils.utils import create_dataset
from automatic_data_generation.utils.utils import to_device

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def train_and_eval_cvae(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # output folder
    output_dir = Path(args.output_folder)
    if not output_dir.exists():
        output_dir.mkdir()
    if args.pickle is not None:
        run_dir = Path('.')
    else:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        run_dir = output_dir / current_time
        run_dir.mkdir()

    # data handling
    data_folder = Path(args.data_folder)
    dataset_folder = data_folder / args.dataset_type
    none_folder = data_folder / args.none_type
    none_idx = NONE_COLUMN_MAPPING[args.none_type]

    dataset = create_dataset(
        args.dataset_type, dataset_folder, args.restrict_to_intent,
        args.input_type, args.dataset_size, args.tokenizer_type,
        args.preprocessing_type, args.max_sequence_length,
        args.embedding_type, args.embedding_dimension, args.max_vocab_size,
        args.slot_averaging, run_dir, none_folder, none_idx, args.none_size
    )
    if args.load_folder:
        original_vocab_size = dataset.vocab_size
        dataset.update(args.load_folder)
        LOGGER.info('Loaded vocab from %s' % args.load_folder)

    print(dataset.slotdic)
        
    # training
    if args.conditioning == NO_CONDITIONING:
        args.conditioning = None

    if not args.load_folder:
        model = CVAE(
            conditional=args.conditioning,
            compute_bow=args.bow_loss,
            vocab_size=dataset.vocab_size,
            embedding_size=args.embedding_dimension,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout_rate=args.word_dropout_rate,
            embedding_dropout_rate=args.embedding_dropout_rate,
            z_size=args.latent_size,
            n_classes=dataset.n_classes,
            cat_size=dataset.n_classes if args.cat_size is None else args.cat_size, 
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            max_sequence_length=args.max_sequence_length,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            temperature=args.temperature,
            force_cpu=args.force_cpu
        )
    else:
        model = CVAE.from_folder(args.load_folder)
        LOGGER.info('Loaded model from %s' % args.load_folder)
        model.n_classes = dataset.n_classes        
        print(model.n_classes)
        model.update_embedding(dataset.vectors)
        model.update_outputs2vocab(original_vocab_size, dataset.vocab_size)

    model = to_device(model, args.force_cpu)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = getattr(torch.optim, args.optimizer_type)(
        model.parameters(),
        lr=args.learning_rate
    )
    
    trainer = Trainer(
        dataset,
        model,
        optimizer,
        batch_size=args.batch_size,
        annealing_strategy=args.annealing_strategy,
        kl_anneal_rate=args.kl_anneal_rate,
        kl_anneal_time=args.kl_anneal_time,
        kl_anneal_target=args.kl_anneal_target,
        label_anneal_rate=args.label_anneal_rate,
        label_anneal_time=args.label_anneal_time,
        label_anneal_target=args.label_anneal_target,
        add_bow_loss=args.bow_loss,
        force_cpu=args.force_cpu,
        run_dir=run_dir / "tensorboard"
    )

    trainer.run(args.n_epochs, dev_step_every_n_epochs=1)

    if args.pickle is not None:
        model_path = run_dir / "{}_load".format(args.pickle)
    else:
        model_path = run_dir / "load"
    dataset.save(model_path)
    model.save(model_path)

    # evaluation
    run_dict = dict()

    # generate queries
    generated_sentences, logp = generate_vae_sentences(
        model=model,
        n_to_generate=args.n_generated,
        input_type=args.input_type,
        i2int=dataset.i2int,
        i2w=dataset.i2w,
        eos_idx=dataset.eos_idx,
        slotdic=dataset.slotdic,
        seed=args.seed,
        verbose=True
    )
    run_dict['generated'] = generated_sentences
    run_dict['metrics'] = compute_generation_metrics(
        dataset,
        generated_sentences['utterances'],
        generated_sentences['intents'],
        logp
    )
    LOGGER.debug(*[{k: v} for (k, v) in run_dict['metrics'].items()], sep='\n')

    if args.input_type == "delexicalised":
        run_dict['delexicalised_metrics'] = compute_generation_metrics(
            dataset,
            generated_sentences['delexicalised'],
            generated_sentences['intents'],
            logp,
            input_type='delexicalised'
        )
        LOGGER.debug(*[{k: v} for (k, v) in run_dict[
            'delexicalised_metrics'].items()], sep='\n')

    save_augmented_dataset(generated_sentences, args.n_generated,
                           dataset.train_path, run_dir)

    run_dict['args'] = vars(args)
    run_dict['logs'] = trainer.run_logs
    run_dict['latent_rep'] = trainer.latent_rep
    run_dict['i2w'] = dataset.i2w
    run_dict['w2i'] = dataset.w2i
    run_dict['vectors'] = {
        'before': dataset.vocab.vectors,
        'after': model.embedding.weight.data
    }

    if args.pickle is not None:
        run_dict_path = run_dir / "{}.pkl".format(args.pickle)
    else:
        run_dict_path = run_dir / "run.pkl"
    torch.save(run_dict, str(run_dict_path))


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-folder', type=str, default='data')
    parser.add_argument('--output-folder', type=str, default='output')
    parser.add_argument('--load-folder', type=str, default=None)
    parser.add_argument('--pickle', type=str, default=None,
                        help='for grid search experiments only')
    parser.add_argument('--dataset-type', type=str, default='snips',
                        choices=['snips', 'atis', 'sentiment', 'spam', 'yelp',
                                 'penn-tree-bank'])
    parser.add_argument('--none-type', type=str, default='penn-tree-bank',
                        choices=['penn-tree-bank', 'shakespeare',
                                 'subtitles', 'yelp'])
    parser.add_argument('-it', '--input_type', type=str,
                        default='delexicalised',
                        choices=['delexicalised', 'utterance'])
    parser.add_argument('--dataset-size', type=int, default=None)
    parser.add_argument('--none-size', type=int, default=None)
    parser.add_argument('--restrict-to-intent', nargs='+', type=str,
                        default=None)

    # data representation
    parser.add_argument('--tokenizer-type', type=str, default='nltk',
                        choices=['split', 'nltk', 'spacy'])
    parser.add_argument('--preprocessing-type', type=str,
                        default=NO_PREPROCESSING,
                        choices=['stem', 'lemmatize', NO_PREPROCESSING])
    parser.add_argument('-msl', '--max_sequence_length', type=int, default=20)
    parser.add_argument('--embedding-type', type=str, default='glove',
                        choices=['glove', 'random'])
    parser.add_argument('--embedding-dimension', type=int, default=100)
    parser.add_argument('--freeze_embeddings', type=bool, default=False)
    parser.add_argument('-mvs', '--max-vocab-size', type=int, default=10000)
    parser.add_argument('--slot-averaging', type=str,
                        default=NO_SLOT_AVERAGING,
                        choices=['micro', 'macro', NO_SLOT_AVERAGING])

    # model
    parser.add_argument('--conditioning', type=str, default='supervised',
                        choices=['supervised', 'unsupervised',
                                 NO_CONDITIONING])
    parser.add_argument('--bow-loss', action='store_true')
    parser.add_argument('-rnn', '--rnn-type', type=str, default='gru',
                        choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('-hs', '--hidden-size', type=int, default=256)
    parser.add_argument('-wd', '--word-dropout-rate', type=float, default=0.)
    parser.add_argument('-ed', '--embedding-dropout-rate', type=float,
                        default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=8)
    parser.add_argument('-cs', '--cat_size', type=int, default=None)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-t', '--temperature', type=float, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    # training
    parser.add_argument('-ep', '--n-epochs', type=int, default=5)
    parser.add_argument('-opt', '--optimizer-type', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-as', '--annealing-strategy', type=str,
                        default='logistic', choices=['logistic', 'linear'])
    parser.add_argument('-k1', '--kl-anneal-time', type=float, default=300,
                        help='anneal time for KL weight')
    parser.add_argument('-x1', '--kl-anneal-rate', type=int, default=0.01,
                        help='anneal rate for KL weight')
    parser.add_argument('-m1', '--kl-anneal-target', type=float, default=1.,
                        help='final value for KL weight')
    parser.add_argument('-k2', '--label-anneal-time', type=float, default=100,
                        help='anneal time for label weight')
    parser.add_argument('-x2', '--label-anneal-rate', type=int, default=0.01,
                        help='anneal rate for label weight')
    parser.add_argument('-m2', '--label-anneal-target', type=float, default=1.,
                        help='final value for label weight')
    parser.add_argument('--force-cpu', type=bool, default=False)

    # evaluation
    parser.add_argument('-ng', '--n-generated', type=int, default=500)

    args = parser.parse_args()

    train_and_eval_cvae(args)


if __name__ == "__main__":
    main()
