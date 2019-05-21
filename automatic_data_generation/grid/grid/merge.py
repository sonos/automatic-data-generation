# pylint: disable=C
import argparse
import glob
import os
import random
from itertools import count

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir_src", type=str)
    parser.add_argument("log_dir_dst", type=str)
    args = parser.parse_args()

    def tup(args):
        d = [(key, value) for key, value in args.__dict__.items() if key != 'pickle']
        return tuple(sorted(d))

    done_dst = {
        tup(torch.load(f))
        for f in glob.glob("{}/*.pkl".format(args.log_dir_dst))
    }

    for path_src in glob.glob("{}/*.pkl".format(args.log_dir_src)):
        args_src = tup(torch.load(path_src))

        if args_src in done_dst:
            print("{} ok".format(path_src))
            continue

        for i in count(random.randint(0, 50000)):
            name = "{:05d}.pkl".format(i)
            path_dst = os.path.join(args.log_dir_dst, name)
            if not os.path.isfile(path_dst):
                break

        print("[{}] {} -> {}".format(
            " ".join("{}={}".format(key, value) for key, value in args_src),
            path_src, path_dst))
        os.rename(path_src, path_dst)

        done_dst.add(args_src)


if __name__ == '__main__':
    main()
