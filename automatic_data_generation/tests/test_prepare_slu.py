#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import unittest

from automatic_data_generation.utils.slu_benchmarks.prepare_slu_data import (
    merge_entity_dict, augment_dataset, compute_dataset_size)


class TestPrepareSLU(unittest.TestCase):
    def test_should_merge_entity_dict(self):
        lhs_dict = {
            'entity1': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'val1', 'synonyms': []},
                    {'value': 'val2', 'synonyms': []},
                ]
            },
            'entity2': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'val12', 'synonyms': []},
                ]
            },
            'snips/entity1': {
                'entity_type': 'builtin'
            }
        }

        rhs_dict = {
            'entity1': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'val1', 'synonyms': []},
                    {'value': 'val3', 'synonyms': []},
                ]
            },
            'snips/entity1': {
                'entity_type': 'builtin'
            },
            'snips/entity2': {
                'entity_type': 'builtin'
            }
        }

        target_dict = {
            'entity1': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'val1', 'synonyms': []},
                    {'value': 'val2', 'synonyms': []},
                    {'value': 'val3', 'synonyms': []},
                ]
            },
            'entity2': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'val12', 'synonyms': []},
                ]
            },
            'snips/entity1': {
                'entity_type': 'builtin'
            },
            'snips/entity2': {
                'entity_type': 'builtin'
            }
        }

        merged_dict = merge_entity_dict(lhs_dict, rhs_dict)
        self.assertDictEqual(merged_dict, target_dict)

    def test_should_augment_dataset(self):
        train_data = [
            ['utterance', 'labels', 'delexicalised', 'intent'],
            [
                'What will the weather be in Paris tomorrow',
                'O O O O O O B-city B-timerange',
                'What will the weather be in _city_ _timerange_',
                'GetWeather'
            ],
        ]

        train_entities = {
            'snips/datetime': {
                'entity_type': 'builtin'
            },
            'city': {
                'entity_type': 'custom',
                'data': [
                    {'value': 'Paris', 'synonyms': []},
                    {'value': 'London', 'synonyms': []},
                ]
            }
        }

        augmentation_data = [
            [
                'How sunny is it in Dublin',
                'O O O O O B-city',
                'How sunny is it in _city_',
                'GetWeather'
            ],
            [
                'How rainy is it in Lille',
                'O O O O O B-city',
                'How rainy is it in _city_',
                'GetWeather'
            ],
            [
                'Find me a pizzeria',
                'O O O B-restaurant_type',
                'Find me a _restaurant_type_',
                'BookRestaurant'
            ],
            [
                'I want a boulangerie',
                'O O O B-restaurant_type',
                'I want a _restaurant_type_',
                'BookRestaurant'
            ],
        ]

        ref_data = [
            [
                'What will the weather be in New York',
                'O O O O O O B-city I-city',
                'What will the weather be in _city_',
                'GetWeather'
            ],
            [
                'How nice is the weather in Singapore',
                'O O O O O O B-city',
                'How nice is the weather in _city_',
                'GetWeather'
            ]
        ]
        ref_data *= 2
        ref_data += [[
            'What will the weather be in Paris tomorrow',
            'O O O O O O B-city B-timerange',
            'What will the weather be in _city_ _timerange_',
            'GetWeather'
        ]] * 10

        augmentation_ratio = 2

        augmented_train_dataset, ref_train_dataset = augment_dataset(
            train_data, train_entities, augmentation_data, ref_data,
            augmentation_ratio)

        # augmented train
        nb_weather, nb_restaurant, nb_city, total_size = \
            extract_info_from_dataset(augmented_train_dataset)
        self.assertEqual(nb_weather, 2)
        self.assertEqual(nb_restaurant, 1)
        self.assertEqual(nb_city, 3)
        self.assertEqual(total_size, 3)

        # reference train
        nb_weather, nb_restaurant, nb_city, total_size = \
            extract_info_from_dataset(ref_train_dataset)
        self.assertEqual(nb_weather, 3)
        self.assertEqual(nb_restaurant, 0)
        self.assertEqual(nb_city, 4)
        self.assertEqual(total_size, 3)


def extract_info_from_dataset(dataset):
    nb_weather = len(dataset['intents']['GetWeather']['utterances']) if \
        'GetWeather' in dataset['intents'] else 0
    nb_restaurant = len(dataset['intents']['BookRestaurant']['utterances']) \
        if 'BookRestaurant' in dataset['intents'] else 0
    nb_city = len(dataset['entities']['city']['data'])
    total_size = compute_dataset_size(dataset)
    return nb_weather, nb_restaurant, nb_city, total_size
