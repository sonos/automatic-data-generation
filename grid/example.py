# pylint: disable=C,R,E1101
import argparse
import os
import time

import torch


def execute(args):
    for i in range(3):
        time.sleep(0.5)
        print("computation {} / 3".format(i + 1), flush=True)  # need flush=True

    return args.a / args.b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--a", type=int, required=True)
    parser.add_argument("--b", type=int, required=True)

    args = parser.parse_args()

    torch.save(args, args.pickle)
    try:
        results = execute(args)

        with open(args.pickle, 'wb') as f:
            torch.save(args, f)
            torch.save(results, f)
    except:
        os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
