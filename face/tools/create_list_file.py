# -*- coding: utf-8 -*-
import os
import random
import argparse

def get_samples_from_paths(image_path):
    ret = {}
    paths = os.listdir(image_path)
    for path in paths:
        ret[os.path.join(image_path, path)] = os.listdir(os.path.join(image_path, path))
    return ret

def split_sample(samples, args):
    train = []
    test = []
    label_count = 0
    for path in samples:
        filenames = samples[path]
        if args.split:
            random.shuffle(filenames)
            choosed = filenames[:args.num_test]
            for t in choosed:
                test.append((os.path.join(path, t), label_count))
            for t in filenames[args.num_test:]:
                train.append((os.path.join(path, t), label_count))
        else:
            for t in filenames:
                train.append((os.path.join(path, t), label_count))
        label_count += 1
    return train, test

def set2listfile(file_set, listfilename):
    with open(listfilename, mode='w') as f:
        for s in file_set:
            f.write('{} {}\n'.format(*s))

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--split', action='store_true', default=False)
    parser.add_argument('--num_test', default=1, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parser()
    
    dirs = args.dirs.strip().split(',')
    samples = {}
    
    for dir in dirs:
        tmp_samples = get_samples_from_paths(dir)
        samples.update(tmp_samples)
    train, test = split_sample(samples, args)
    if args.split:
        set2listfile(train, os.path.join(args.out_dir, '{}_train.list'.format(args.prefix)))
        set2listfile(test, os.path.join(args.out_dir, '{}_val.list'.format(args.prefix)))
    else:
        set2listfile(train, os.path.join(args.out_dir, '{}_all.list'.format(args.prefix)))
