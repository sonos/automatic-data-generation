import argparse
import csv
import os
import pickle
from pathlib import Path

from nltk import word_tokenize

from automatic_data_generation.data.utils.utils import get_groups_v2
from automatic_data_generation.utils.io import (load_json, write_csv,
                                                read_csv, dump_json)

remove_punctuation = True


def iso2utf(datadir, outdir):
    for split in ['train', 'validate']:
        csvname = os.path.join(datadir, split)
        output_csv = open(csvname + '_converted.csv', 'w', encoding='utf-8')
        csv_writer = csv.writer(output_csv, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        with open(csvname + '.csv', 'r', encoding='ISO-8859-1') as input_csv:
            reader = csv.reader(input_csv)
            for irow, row in enumerate(reader):
                csv_writer.writerow(row)


def json2csv(datadir, outdir, samples_per_intent):
    print('Starting json2csv conversion...')
    punctuation = [',', '.', ';', '?', '!', '\"']
    data_folder = Path(datadir)
    out_folder = Path(outdir)

    for split in ['train', 'validate']:
        data_dict = {}
        for intent_dir in data_folder.iterdir():
            if not intent_dir.is_dir():
                continue
            intent = intent_dir.stem
            suffix = '{}_{}{}.json'.format(
                split, intent, '_full' if split == 'train' else ''
            )
            data_dict[intent] = load_json(intent_dir / suffix,
                                          encoding='latin1')[intent]

        slotdic = {}
        csv_data = [['utterance', 'labels', 'delexicalised', 'intent']]
        for intent, data in data_dict.items():
            for isent, sentence in enumerate(data):
                if isent >= samples_per_intent:
                    break
                utterance = ''
                labelling = ''
                delexicalised = ''

                for group in sentence['data']:
                    words = group['text']
                    try:
                        words = words.encode('latin-1').decode('utf8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        if 'entity' not in group.keys():
                            print("skipping because of bad encoding:{}".format(
                                words))
                            continue
                        else:
                            words = words.encode('utf8').decode('utf8')

                    if remove_punctuation:
                        for p in punctuation:
                            words = words.replace(p, '')
                    words = words.replace('\n', '')  # trailing new lines are
                    # misread by csv writer
                    utterance += words

                    if 'entity' in group.keys():  # this group is a slot
                        slot = group['entity'].lower()
                        if remove_punctuation:
                            for p in punctuation:
                                slot = slot.replace(p, '')

                        delexicalised += '_' + slot + '_'
                        for i, word in enumerate(word_tokenize(words)):
                            # if word == '':
                            #     continue
                            if i == 0:
                                word = 'B-' + slot + ' '
                            else:
                                word = 'I-' + slot + ' '
                            labelling += word

                        if slot not in slotdic.keys():
                            slotdic[slot] = [words]
                        else:
                            if words not in slotdic[slot]:
                                slotdic[slot].append(words)

                    else:  # this group is just context
                        delexicalised += words
                        labelling += 'O ' * len(word_tokenize(words))

                csv_data.append([utterance, labelling, delexicalised, intent])

        output_file = out_folder / '{}.csv'.format(split)
        write_csv(csv_data, output_file)

        output_pickle_file = out_folder / '{}_slot_values.pkl'.format(split)
        with open(output_pickle_file, 'wb') as f:
            print(slotdic.keys())
            pickle.dump(slotdic, f)
            print('Dumped slot dictionnary')
    print('Example : ')
    print('Original utterance : ', utterance)
    print('Labelled : ', labelling)
    print('Delexicalised : ', delexicalised)

    print('Successfully converted json2csv !')


def new_json2csv(datadir, outdir):
    data_folder = Path(datadir)
    out_folder = Path(outdir)
    data = load_json(data_folder / 'dataset.json', encoding='latin1')
    remove_punctuation = True
    punctuation = [',', '.', ';', '?', '!', '\"']

    val_fraction = 0.2

    for split in ['train', 'validate']:

        csv_data = [['utterance', 'labels', 'delexicalised', 'intent']]

        for intent in data['intents'].keys():

            num_val_sentences = int(
                val_fraction * len(data['intents'][intent]['utterances']))
            print(split, intent, num_val_sentences)
            if split == 'validate':
                sentences = data['intents'][intent]['utterances'][
                            :num_val_sentences]
            else:
                sentences = data['intents'][intent]['utterances'][
                            num_val_sentences:]
            print(len(sentences))

            for sentence in sentences:

                utterance = ''
                labelling = ''
                delexicalised = ''

                for group in sentence['data']:
                    words = group['text']
                    try:
                        words = words.encode('latin-1').decode('utf8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        if 'entity' not in group.keys():
                            print("skipping because of bad encoding:{}".format(
                                words))
                            continue
                        else:
                            words = words.encode('utf8').decode('utf8')
                    if remove_punctuation:
                        for p in punctuation:
                            words = words.replace(p, '')
                    words = words.replace('\n', '')  # trailing new lines are
                    # misread by csv writer
                    utterance += words

                    if 'slot_name' in group.keys():  # this group is a slot
                        slot = group['slot_name'].lower()
                        if remove_punctuation:
                            for p in punctuation:
                                slot = slot.replace(p, '')

                        delexicalised += '_' + slot + '_'
                        for i, word in enumerate(word_tokenize(words)):
                            if i == 0:
                                word = 'B-' + slot + ' '
                            else:
                                word = 'I-' + slot + ' '
                            labelling += word

                    else:  # this group is just context
                        delexicalised += words
                        labelling += 'O ' * len(word_tokenize(words))

                csv_data.append([utterance, labelling, delexicalised, intent])

        output_file = out_folder / '{}.csv'.format(split)
        write_csv(csv_data, output_file)


def extract_intents_entities(data, entity_mapping=None):
    intents = {}
    entities = {}

    encountered_entity_values = {}
    for irow, row in enumerate(data):
        if irow == 0:  # ignore header
            continue

        utterance, labelling, delexicalised, intent = row
        if intent not in intents.keys():
            intents[intent] = {'utterances': []}

        groups = get_groups_v2(word_tokenize(utterance), word_tokenize(
            labelling), entity_mapping)
        intents[intent]['utterances'].append({'data': groups})
        # entities
        for group in groups:
            if 'entity' in group.keys():
                entity = group['entity']
                entity_value = group['text']
                entity_type = "builtin" if "snips/" in entity else "custom"

                if entity not in entities.keys():
                    if entity_type == "builtin":
                        entities[entity] = {
                            "entity_type": entity_type,
                        }
                    else:
                        entities[entity] = {
                            "data": [],
                            "use_synonyms": True,
                            "entity_type": entity_type,
                            "automatically_extensible": False,
                            "matching_strictness": 1.0
                        }

                if entity_type == "custom":
                    if entity not in encountered_entity_values.keys():
                        encountered_entity_values[entity] = []
                    if entity_value not in encountered_entity_values[entity]:
                        entities[entity]['data'].append(
                            {
                                'value': entity_value,
                                'synonyms': []
                            }
                        )
                    encountered_entity_values[entity].append(entity_value)

    return intents, entities


def csv2json(csv_path_in, output_dir):
    print('Starting csv2json conversion...')
    jsondic = {'language': 'en'}
    csv_data = read_csv(Path(csv_path_in))
    intents, entities = extract_intents_entities(csv_data)

    jsondic['intents'] = intents
    jsondic['entities'] = entities

    filename = str(Path(csv_path_in).stem) + '.json'
    path_out = Path(output_dir) / filename
    dump_json(jsondic, path_out)

    print('Successfully converted csv2json !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/snips_original')
    parser.add_argument('--outdir', type=str, default='data/snips')
    parser.add_argument('-spi', '--samples_per_intent', type=int,
                        default=10000)
    parser.add_argument('--augmented', type=int, default=1)
    parser.add_argument('--convert_to', type=str, default='csv')
    parser.add_argument('--remove_punctuation', type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        print('saving data to', args.outdir)
        os.mkdir(args.outdir)

    if args.convert_to == 'csv':
        json2csv(args.datadir, args.outdir, args.samples_per_intent)
    if args.convert_to == 'json':
        csv2json(args.datadir, args.outdir)
    if args.convert_to == 'utf':
        iso2utf(args.datadir, args.outdir)
