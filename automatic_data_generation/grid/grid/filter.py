# pylint: disable=C, eval-used
import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("filter", type=str)
    args = parser.parse_args()

    f = eval(args.filter)

    for path in glob.glob("{}/*.pkl".format(args.log_dir)):
        if not f(torch.load(path)):
            print("remove {}".format(path))
            os.remove(path)


if __name__ == '__main__':
    main()
