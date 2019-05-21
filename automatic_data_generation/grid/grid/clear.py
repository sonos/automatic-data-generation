# pylint: disable=C
import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    args = parser.parse_args()


    for path in glob.glob("{}/*.pkl".format(args.log_dir)):
        with open(path, 'rb') as f:
            torch.load(f)
            try:
                torch.load(f, map_location='cpu')
            except EOFError:
                print("remove {}".format(path))
                os.remove(path)


if __name__ == '__main__':
    main()
