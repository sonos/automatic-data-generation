# pylint: disable=C
import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    args = parser.parse_args()

    if os.path.isfile("{}/info".format(args.log_dir)):
        with open("{}/info".format(args.log_dir), 'rb') as f:
            while True:
                try:
                    info = torch.load(f)
                    print(info['args'])
                    print(info['params'])
                    # print("    {}".format(info['git']['log']))
                    # print("    {}".format(info['git']['status']))
                except EOFError:
                    break


    # runs = [
    #     {
    #         key: value
    #         for key, value in r.items()
    #         if key != 'pickle'
    #     }
    #     for r in [torch.load(path) for path in glob.glob("{}/*.pkl".format(args.log_dir))]
    # ]
    # for key in {key for r in runs for key in r.keys()}:

    #     values = {r[key] if key in r else None for r in runs}

    #     print("{}: {}".format(key, values))

    runs = [r for r in [torch.load(path) for path in glob.glob("{}/*.pkl".format(args.log_dir))]]
    for run in runs:
        print(run['args'])
        print(run['metrics'])


if __name__ == '__main__':
    main()
