#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

import csv
import json


def load_json(filename, encoding='utf-8'):
    """
        Load the content of filename

        Args:
            filename (pathlib.Path): path of json file to open
            encoding (str): encoding of file to open (default: 'utf-8')

        returns:
            loaded json
    """
    with filename.open(mode='r', encoding=encoding) as _file:
        return json.load(_file)


def dump_json(data, filename, ensure_ascii=False):
    """
      Dump data to filename with utf-8 encoding
      Args:
          data (list or dict): data to dump
          filename (pathlib.Path): path of json file to dump to
          ensure_ascii (bool): whether to ensure that the dumped json file uses
            only ascii characters (default: False)
    """
    with filename.open(mode='w', encoding='utf-8') as _file:
        json.dump(data, _file, indent=2, ensure_ascii=ensure_ascii)


def read_csv(filename, newline='', encoding='utf-8'):
    with filename.open(newline=newline, encoding=encoding) as f:
        return list(csv.reader(f))


def write_csv(data, filename):
    with filename.open(mode='w') as _file:
        writer = csv.writer(_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)
