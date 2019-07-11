#! /usr/bin/env python
# encoding: utf-8

"""
    Interface to the intent data
"""

from pathlib import Path
import json
import csv
from nltk import word_tokenize

INTENT_FOLDER = "intent"
ENTITY_FOLDER = "entity"

class IntentNotFoundError(Exception):
    pass
class EntityNotFoundError(Exception):
    pass

def load_json(path, encoding='utf-8'):
    """
        Load the content of filename
    """
    with path.open('r', encoding=encoding) as _file:
        return json.load(_file)


def dump_json(json_data, path, encoding="utf-8", indent=2, sort_keys=True):
    json_string = json.dumps(json_data, indent=indent, sort_keys=sort_keys,
                             ensure_ascii=False)
    with path.open("w", encoding=encoding) as f:
        f.write(json_string)

def read_csv(filename, newline='', encoding='utf-8'):
    with filename.open(newline=newline, encoding=encoding) as f:
        return list(csv.reader(f))

def write_csv(data, filename):
    with filename.open(mode='w') as _file:
        writer = csv.writer(_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

def loaded_required(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.loaded:
            raise NotLoadedError("%s must be loaded" % self.unit_name)
        return func(self, *args, **kwargs)

    return func_wrapper

class Entity(object):
    def __init__(self, data_dir, entity_id):
        self.entity_id = entity_id
        self.dir = Path(data_dir)
        self.config = None
        self.entity_data = None
        self.loaded = False

    def load(self):
        config_filename = self.entity_id + ".json"
        config_path = self.dir / ENTITY_FOLDER / config_filename
        try:
            config = load_json(config_path.absolute())
        except IOError as e:
            raise EntityNotFoundError(
                "Missing entity {}".format(e.filename)
            )

        self.entity_data = {
            'entity_name': config['name'],
            'automatically_extensible': config['automaticallyExtensible'],
            'data': config['data'],
            'use_synonyms': config['useSynonyms']
        }

        self.loaded = True

class Intent(object):
    # TODO: maybe the load method should be moved to init
    def __init__(self, data_dir, intent_id):
        self.intent_id = intent_id
        self.dir = Path(data_dir)
        self.config = None
        self.intent_data = None
        self.slots_data = None
        self.loaded = False
        self._utterances = None
        self._stats = None
        self._slots = None

    def load(self):
        config_filename = self.intent_id + ".json"
        config_path = self.dir / INTENT_FOLDER / config_filename
        try:
            self.config = load_json(config_path.absolute())
        except IOError as e:
            raise IntentNotFoundError(
                "Missing intent {}".format(e.filename)
            )

        self.intent_data = {
            'user_id': self.config['config']['userId'],
            'name': self.config['config']['displayName'],
            'utterances': self.config['customIntentData']['utterances'],
            'copied_from': self.config.get('copiedFrom'),
            'statistics': self.config['statistics'],
            'is_private': self.config['config']['private']
        }
        self.loaded = True

    @loaded_required
    def is_of_language(self, language):
        return self.config['customIntentData']['language'] == language

    @loaded_required
    def is_of_user(self, user_id):
        return self.config['config']['userId'] == user_id and not \
            self.config.get('copiedFrom')

    @property
    @loaded_required
    def user_id(self):
        return self.config['config']['userId']

    @property
    @loaded_required
    def slots(self):
        if self.slots_data is None:
            slots = dict()
            for slot in self.config['config']['slots']:
                entity_type = slot['entityId']
                slot_id = slot['id']
                if "snips/" not in entity_type:
                    entity = Entity(str(self.dir), entity_type)
                    entity.load()
                    slots[slot_id] = entity.entity_data
                    slots[slot_id]['entity_type'] = entity_type
                    slots[slot_id]['slot_name'] = slot['name']
                else:
                    slots[slot_id] = {
                        'slot_name': slot['name'],
                        'entity_type': entity_type
                    }
            self.slots_data = slots
        return self.slots_data

    @slots.setter
    @loaded_required
    def slots(self, value):
        self.slots = value

    @property
    @loaded_required
    def utterances(self):
        if self._utterances is None:
            self._utterances = [
                ''.join(chunk['text'] for chunk in utterance_data['data']) for
                utterance_data in self.intent_data['utterances']
            ]
        return self._utterances

    @property
    @loaded_required
    def stats(self):
        if self._stats is None:
            n_default = 0
            n_builtin = 0
            n_custom = 0
            for slot_id in self.slots_data:
                if self.slots_data[slot_id]['entity_type'] == 'snips/default':
                    n_default += 1
                elif 'snips/' in self.slots_data[slot_id]['entity_type']:
                    n_builtin += 1
                else:
                    n_custom += 1
            self._stats = {
                'len_utterances': [
                    len(''.join([chunk['text'] for chunk in query['data']]))
                    for
                    query in self.intent_data['utterances']],
                'n_utterances': len(self.intent_data['utterances']),
                'n_slots': len(self.slots_data),
                'quality_score': self.intent_data['statistics'][
                    'qualityScore'],
                'is_private': self.intent_data['is_private'],
                'n_builtin': n_builtin,
                'n_custom': n_custom,
                'n_default': n_default,
                'delta_n_utterances': self.intent_data['statistics'][
                                          'utterancesCount'] - len(
                    self.intent_data['utterances'])
            }
        return self._stats

def sentences2csv(csv_data, sentences, intent):

    punctuation = [',', '.', ';', '?', '!', '\"']
    remove_punctuation = True

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
                    # print("skipping because of bad encoding:{}".format(
                    #     words))
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
    return


if __name__=='__main__':
    import time
    data_dir = Path('/Users/stephane/Dropbox/Work/Codes/snips/dump_snips_db')
    intent_dir = data_dir / 'intent'
    intent_files = intent_dir.glob('*')
    intent_ids = [intent_file.stem for intent_file in intent_files]

    intent_names = []
    csv_data = [['utterance', 'labels', 'delexicalised', 'intent']]

    for i, intent_id in enumerate(intent_ids):
        #if i>100:
        #    break

        try:
            
            intent = Intent(data_dir=data_dir, intent_id=intent_id)
            intent.load()
            n_utterances = len(intent.intent_data['utterances'])
            intent_name = intent.intent_data['name']
            
            if intent_name not in intent_names:
                intent_names.append(intent_name)
            else:
                continue
            
            if intent.is_of_language('en') and n_utterances>100:
                intent.slots
                print(intent_name, n_utterances)
                sentences = intent.intent_data['utterances']

                sentences2csv(csv_data, sentences, intent_name)

        except EntityNotFoundError:
            print(intent_name, 'ERROR')
            pass
                
    write_csv(csv_data, Path('data.csv'))




        
    
