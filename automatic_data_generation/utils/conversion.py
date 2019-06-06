import argparse
import csv
import json
import os
import pickle
from automatic_data_generation.utils.utils import get_groups

from nltk import word_tokenize

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
    punctuation = [',', '.', ':', ';', '?', '!']

    for split in ['train', 'validate']:

        datadic = {}
        for intent in os.listdir(datadir):
            with open(
                    os.path.join(
                        datadir,
                        intent,
                        '{}_{}{}.json'.format(split, intent, '_full' if
                            split == 'train' else '')
                    ),
                    encoding='ISO-8859-1'
            ) as json_file:
                datadic[intent] = json.load(json_file)[intent]

        slotdic = {}

        # intentfile = open('{}/{}_intents.txt'       .format(outdir, split), 'w')
        # utterfile  = open('{}/{}_utterances.txt'    .format(outdir, split), 'w')
        # labelfile  = open('{}/{}_labels.txt'        .format(outdir, split), 'w')
        # delexfile  = open('{}/{}_delexicalised.txt' .format(outdir, split), 'w')
        csvfile = open('{}/{}.csv'.format(outdir, split), 'w')
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['utterance', 'labels', 'delexicalised', 'intent'])

        for intent, data in datadic.items():
            for isent, sentence in enumerate(data):
                if isent >= samples_per_intent:
                    break
                utterance = ''
                labelling = ''
                # delexicalised = ''
                delexicalised = ''

                for group in sentence['data']:
                    words = group['text']
                    if remove_punctuation:
                        for p in punctuation:
                            words = words.replace(p, ' ')
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

                # intentfile.write(intent+'\n')
                # utterfile.write(utterance+'\n')
                # labelfile.write(labelling+'\n')
                # delexfile.write(delexicalised+'\n')
                csv_writer.writerow(
                    [utterance, labelling, delexicalised, intent])

        with open('{}/{}_slot_values.pkl'.format(outdir, split), 'wb') as f:
            print(slotdic.keys())
            pickle.dump(slotdic, f)
            print('Dumped slot dictionnary')

    print('Example : ')
    print('Original utterance : ', utterance)
    print('Labelled : ', labelling)
    print('Delexicalised : ', delexicalised)

    print('Successfully converted json2csv !')


def csv2json(csv_path):
    print('Starting csv2json conversion...')

    jsondic = {'language': 'en'}
    intents = {}
    entities = {}

    encountered_slot_values = {}

    csv_file = open(csv_path, 'r')
    reader = csv.reader(csv_file)

    for irow, row in enumerate(reader):

        if irow == 0:  # ignore header
            continue

        utterance, labelling, delexicalised, intent = row
        if intent not in intents.keys():
            intents[intent] = {'utterances': []}
        groups = get_groups(word_tokenize(utterance), word_tokenize(labelling))
        intents[intent]['utterances'].append({'data': groups})
        for group in groups:
            if 'slot_name' in group.keys():
                slot_name = group['slot_name']
                slot_value = group['text']
                if slot_name not in encountered_slot_values.keys():
                    encountered_slot_values[slot_name] = []

                if slot_name not in entities.keys():
                    entities[slot_name] = {"data": [],
                                           "use_synonyms": True,
                                           "automatically_extensible": True,
                                           "matching_strictness": 1.0
                                           }
                if slot_value not in encountered_slot_values[slot_name]:
                    entities[slot_name]['data'].append(
                        {'value': slot_value,
                         'synonyms': []})
                if slot_value == 'added ':
                    print(groups, utterance, labelling)

                encountered_slot_values[slot_name].append(slot_value)

    jsondic['intents'] = intents
    jsondic['entities'] = entities

    json_path = csv_path.replace('.csv', '.json')
    with open(json_path, 'w') as jsonfile:
        json.dump(jsondic, jsonfile)

    print('Successfully converted csv2json !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/snips_original')
    parser.add_argument('--outdir', type=str, default='data/snips')
    parser.add_argument('-spi', '--samples_per_intent', type=int, default=10000)
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
        csv2json(args.datadir, args.outdir, args.augmented)
    if args.convert_to == 'utf':
        iso2utf(args.datadir, args.outdir)
