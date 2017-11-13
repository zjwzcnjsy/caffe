import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, type=str)
    parser.add_argument('--outfile', required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    assert os.path.isfile(args.infile)
    with open(args.infile, mode='r') as fin:
        with open(args.outfile, mode='w') as fout:
            for line in fin:
                if line.strip():
                    items = line.strip().split()
                    fout.write('{} {}\n'.format(items[2], items[1]))