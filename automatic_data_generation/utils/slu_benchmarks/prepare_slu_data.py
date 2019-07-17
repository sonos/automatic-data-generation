#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import copy
import json
import logging
from pathlib import Path

import click as click
from sklearn.model_selection import StratifiedShuffleSplit

from automatic_data_generation.utils.conversion import extract_intents_entities
from automatic_data_generation.utils.io import dump_json, load_json, read_csv

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                           '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

INTENTS = {
    "AddToPlaylist": {},
    "BookRestaurant": {
        'timerange': 'snips/datetime',
        'party_size_number': 'snips/number',
    },
    "RateBook": {
        'rating_value': 'snips/number',
        'best_rating': 'snips/number'
    },
    "GetWeather": {
        'timerange': 'snips/datetime',
    },
    "SearchCreativeWork": {},
    "PlayMusic": {},
    "SearchScreeningEvent": {
        'timerange': 'snips/datetime',
    },
}

ENTITY_MAPPING = {
    'timerange': 'snips/datetime',
    'party_size_number': 'snips/number',
    'rating_value': 'snips/number',
    'best_rating': 'snips/number',
}

PUNCTUATIONS = [',', '.', ';', '?', '!', '\"']

ROOT_PATH = Path('../nlu-benchmark/2017-06-custom-intent-engines/')


@click.group()
def main():
    pass


@main.command('get_utterance_list')
@click.option('--output_folder', required=True, type=str)
def get_utterance_list(output_folder):
    sentence_list = []
    for intent in INTENTS:
        path = ROOT_PATH / intent / 'validate_{}.json'.format(intent)
        dataset = load_json(path)
        to_add = []
        for query in dataset[intent]:
            text = ''.join([chunk['text'] for chunk in query['data']])
            for p in PUNCTUATIONS:
                text = text.replace(p, '')
            to_add.append(text.replace)

        sentence_list.extend(to_add)

    output_dir = Path(output_folder)
    if not output_dir.exists():
        output_dir.mkdir()

    dump_json(sentence_list, output_dir / 'sentence_list.json')


@main.command('download_audio_files')
@click.option('--path_to_campaign_file', required=True, type=str)
@click.option('--output_folder', required=True, type=str)
@click.option('--nb_positive_votes', type=int, default=0)
def download_audio_files(path_to_campaign_file, output_folder,
                         nb_positive_votes):
    import boto3
    client = boto3.client('s3')
    path_to_file = Path(path_to_campaign_file)
    campaign_file = load_json(path_to_file, encoding='utf-16be')
    output_folder = Path(output_folder)

    for idx, item in enumerate(campaign_file):
        if item['nb_positive_votes'] >= nb_positive_votes:
            filename = 'wav_{}'.format(idx)
            s3_path = item["link_to_file"].replace(
                'https://s3.amazonaws.com/snips/', '')
            local_path = output_folder / filename
            item['path_file'] = filename
            client.download_file('snips', s3_path, str(local_path))

    metadata_path = output_folder / 'metadata.json'
    dump_json(campaign_file, metadata_path)


def make_dataset_dict(intents, entities, language='en'):
    return {
        'language': language,
        'intents': intents,
        'entities': entities
    }


def stratified_trim(data, data_size):
    original_data_size = len(data)
    keep_fraction = data_size / original_data_size
    intents_list = [row[3] for row in data]
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=1 - keep_fraction,
                                 random_state=42)
    keep_indices = list(sss.split(intents_list, intents_list))[0][0]
    return [data[i] for i in keep_indices]


def augment_dataset(train_data, train_entities, augmentation_data,
                    ref_data, augmentation_ratio=0.5):
    aug_size = (len(train_data) - 1) * augmentation_ratio  # skip header

    # augmented dataset
    from_aug = stratified_trim(augmentation_data, aug_size)
    train_aug = train_data + from_aug
    aug_intents, aug_entities = extract_intents_entities(
        train_aug, ENTITY_MAPPING)
    aug_train_entities = merge_entity_dict(train_entities, aug_entities)
    augmented_train_dataset = make_dataset_dict(aug_intents,
                                                aug_train_entities)

    # reference dataset
    # TODO: make sure we are ok adding new entities from reference data
    #  compared to CVAE-generated utterances which have same entities
    unseen_ref_data = [item for item in ref_data if item not in train_data]
    from_ref = stratified_trim(unseen_ref_data, aug_size)
    train_ref = train_data + from_ref
    ref_intents, ref_entities = extract_intents_entities(
        train_ref, ENTITY_MAPPING)
    ref_train_entities = merge_entity_dict(train_entities, ref_entities)
    ref_train_dataset = make_dataset_dict(ref_intents, ref_train_entities)

    return augmented_train_dataset, ref_train_dataset


def merge_entity_dict(lh_dict, rh_dict):
    new_dict = copy.deepcopy(lh_dict)
    for k, v in rh_dict.items():
        if k in new_dict.keys() and v['entity_type'] == "custom":
            new_dict[k]['data'] += v['data']
            new_dict[k]['data'] = list(
                {v['value']: v for v in new_dict[k]['data']}.values()
            )
        else:
            new_dict[k] = v
    return new_dict


def compute_dataset_size(dataset):
    lengths = [len(dataset['intents'][intent]['utterances']) for intent in
               dataset['intents']]
    return sum(lengths)


def compute_dataset_size_per_intent(dataset):
    return {intent: len(dataset['intents'][intent]['utterances']) for
            intent in dataset['intents']}


def process_and_dump_augmentation(current_path, train_data, train_entities,
                                  augmentation_data, ref_data,
                                  augmentation_ratio,
                                  train_size):
    augmented_train_dataset, ref_train_dataset = augment_dataset(
        train_data=train_data,
        train_entities=train_entities,
        augmentation_data=augmentation_data,
        ref_data=ref_data,
        augmentation_ratio=augmentation_ratio
    )

    target_train_size = int(train_size * (1 + augmentation_ratio))
    dump_json(augmented_train_dataset,
              current_path / 'train_{}_aug_{}.json'.format(
                  train_size, target_train_size))
    dump_json(ref_train_dataset,
              current_path / 'train_{}_ref_{}.json'.format(
                  train_size, target_train_size))

    LOGGER.info("target size for augmentation: %s" % target_train_size)
    LOGGER.info(
        "real size for reference: %s" % compute_dataset_size(
            ref_train_dataset))
    LOGGER.info(json.dumps(compute_dataset_size_per_intent(ref_train_dataset)))
    LOGGER.info(
        "real size for augmentation: %s" % compute_dataset_size(
            augmented_train_dataset))
    LOGGER.info(
        json.dumps(compute_dataset_size_per_intent(augmented_train_dataset)))


@main.command('create_datasets')
@click.option('--path_to_data_folders', required=True, type=str)
def create_datasets(path_to_data_folders):
    data_root_folder = Path(path_to_data_folders)
    ref_train_path = data_root_folder / 'train.csv'
    ref_train_dataset = read_csv(ref_train_path)
    for t in data_root_folder.iterdir():  # iterate over train size
        if not t.is_dir():  # ignore reference train
            continue
        if 'runs' not in str(t):
            continue
        try:
            train_size = int(str(t).split('_')[-1])
        except:
            import ipdb; ipdb.set_trace()
        LOGGER.info("Processing datasets of size: %s" % train_size)
        for s in t.iterdir():  # iterate over seeds
            if not s.is_dir():  # ignore pkl
                continue
            LOGGER.info("Processing seed: %s" % str(s.name))
            # validate
            val_csv_data = read_csv(s / 'validate.csv')
            val_intents, val_entities = extract_intents_entities(
                val_csv_data, ENTITY_MAPPING)
            val_dataset = make_dataset_dict(val_intents, val_entities)
            dump_json(val_dataset, s / 'validate.json'.format(train_size))

            # train
            tr_csv_data = read_csv(s / 'train_{}.csv'.format(train_size))
            train_intents, train_entities = extract_intents_entities(
                tr_csv_data, ENTITY_MAPPING)
            enriched_train_entities = merge_entity_dict(train_entities,
                                                        val_entities)
            train_dataset = make_dataset_dict(train_intents,
                                              enriched_train_entities)
            dump_json(train_dataset, s / 'train_{}.json'.format(train_size))

            # augmented
            csv_data = read_csv(s / 'train_{}_aug_2000.csv'.format(train_size))
            augmented_utterances = csv_data[-2000:]

            process_and_dump_augmentation(
                current_path=s,
                train_data=tr_csv_data,
                train_entities=enriched_train_entities,
                augmentation_data=augmented_utterances,
                ref_data=ref_train_dataset,
                augmentation_ratio=0.5,
                train_size=train_size
            )

            process_and_dump_augmentation(
                current_path=s,
                train_data=tr_csv_data,
                train_entities=enriched_train_entities,
                augmentation_data=augmented_utterances,
                ref_data=ref_train_dataset,
                augmentation_ratio=1,
                train_size=train_size
            )


if __name__ == '__main__':
    main()
