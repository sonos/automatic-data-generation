import operator
from pathlib import Path

import click

from automatic_data_generation.utils.conversion import extract_intents_entities
from automatic_data_generation.utils.io import read_csv, load_json


@click.group()
def main():
    pass


def extract_ref_values(ref_data):
    return [' '.join(item[0].split(' ')[1:]).rstrip().lower() for item in
            ref_data]


def extract_data_values(entity_name, train_entities, test_entities):
    return [
               item['value'].rstrip().lower()
               for item in train_entities[entity_name]['data']
           ] + [
               item['value'].rstrip().lower()
               for item in test_entities[entity_name]['data']
           ]


@main.command('check_music_coverage')
@click.option('--data-path', required=True, type=str)
@click.option('--album-path', required=True, type=str)
@click.option('--artist-path', required=True, type=str)
@click.option('--track-path', required=True, type=str)
def check_music_entity_coverage(data_path, album_path, artist_path,
                                track_path):
    data_path = Path(data_path)
    train_csv_path = data_path / 'train.csv'
    test_csv_path = data_path / 'validate.csv'
    train_data = read_csv(train_csv_path)
    test_data = read_csv(test_csv_path)

    _, train_entities = extract_intents_entities(
        train_data, entity_mapping=None)
    _, test_entities = extract_intents_entities(
        test_data, entity_mapping=None)

    album = extract_data_values('album', train_entities, test_entities)
    artist = extract_data_values('artist', train_entities, test_entities)
    track = extract_data_values('track', train_entities, test_entities)

    ref_album = extract_ref_values(read_csv(Path(album_path)))
    ref_artist = extract_ref_values(read_csv(Path(artist_path)))
    ref_track = extract_ref_values(read_csv(Path(track_path)))

    print(len([val for val in album if val in ref_album]) / len(album))
    print(len([val for val in artist if val in ref_artist]) / len(artist))
    print(len([val for val in track if val in ref_track]) / len(track))


def get_patterns_counts_from_dataset(dataset):
    pattern_counts = dict()
    for intent, intent_data in dataset['intents'].items():
        patterns = []
        for query in intent_data['utterances']:
            patterns.append(
                ''.join([chunk['text'] if 'entity' not in chunk
                         else '_' + chunk['slot_name'] + '_'
                         for chunk in query['data']])
            )
        pattern_counts[intent] = {item: patterns.count(item) for item in
                                  patterns}
    return pattern_counts


def get_utterances_as_string(dataset):
    utterances = dict()
    for intent, intent_data in dataset['intents'].items():
        utterances[intent] = []
        for query in intent_data['utterances']:
            utterances[intent].append(
                ''.join([chunk['text'] for chunk in query['data']])
            )
    return utterances


def get_datasets_difference(small_dataset, big_dataset):
    new_utterances = {'intents': {}}
    for intent, intent_data in small_dataset['intents'].items():
        new_utterances['intents'][intent] = {}
        new_utterances['intents'][intent]['utterances'] = \
            [utt for utt in intent_data['utterances'] if utt not in
             big_dataset['intents'][intent]['utterances']]
    return new_utterances


@main.command('analyse_patterns')
@click.option('--train-path', required=True, type=str)
@click.option('--augmented-path', required=True, type=str)
@click.option('--ref-path', required=True, type=str)
@click.option('--validate-path', required=True, type=str)
def analyse_patterns(train_path, augmented_path, ref_path, validate_path):
    # train
    train_path = Path(train_path)
    train_dataset = load_json(train_path)
    train_patterns = get_patterns_counts_from_dataset(train_dataset)

    # augmentation CVAE
    augmented_path = Path(augmented_path)
    augmented_dataset = load_json(augmented_path)
    augmented_utterances = get_datasets_difference(augmented_dataset,
                                                   train_dataset)
    augmented_patterns = get_patterns_counts_from_dataset(augmented_utterances)
    print(sum([len(augmented_utterances['intents'][intent]['utterances']) for
               intent in augmented_utterances['intents']]))

    augmented_sentences = get_utterances_as_string(augmented_utterances)
    for intent, sentences in augmented_sentences.items():
        print(intent)
        for sentence in sentences: print(sentence)

    # augmentation ref
    ref_path = Path(ref_path)
    ref_dataset = load_json(ref_path)
    ref_utterances = get_datasets_difference(ref_dataset, train_dataset)
    ref_patterns = get_patterns_counts_from_dataset(ref_utterances)
    print(sum([len(ref_utterances['intents'][intent]) for intent in ref_utterances['intents']]))

    # test
    test_path = Path(validate_path)
    test_dataset = load_json(test_path)
    test_patterns = get_patterns_counts_from_dataset(test_dataset)

    # for intent, intent_patterns in test_patterns.items():
    #     print('{:<15}\n'.format(intent))
    #     print('{:<15} {:<15} {:<15} {}\n'.format('# in test', '# generated', '# in ref', 'sentence'))
    #     for sentence, count in sorted(intent_patterns.items(),
    #                                   key=operator.itemgetter(1)):
    #         aug_count = augmented_patterns[intent][sentence] \
    #             if sentence in augmented_patterns[intent] else 0
    #         ref_count = ref_patterns[intent][sentence] \
    #             if sentence in ref_patterns[intent] else 0
    #         print('{:<15} {:<15} {:<15} {}'.format(count, aug_count,
    #                                                ref_count, sentence))

    # for intent, intent_patterns in train_patterns.items():
    #     print('{:<15}\n'.format(intent))
    #     print('{:<15} {:<15} {:<15} {}\n'.format('# in train', '# generated',
    #                                              '# in ref', 'sentence'))
    #     for sentence, count in sorted(intent_patterns.items(),
    #                                   key=operator.itemgetter(1)):
    #         aug_count = augmented_patterns[intent][sentence] \
    #             if sentence in augmented_patterns[intent] else 0
    #         ref_count = ref_patterns[intent][sentence] \
    #             if sentence in ref_patterns[intent] else 0
    #         print('{:<15} {:<15} {:<15} {}'.format(count, aug_count,
    #                                                ref_count, sentence))


    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    main()
