#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
import multiprocessing

IMAGE_EXTS = ('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP')


def load_image(filename):
    '''load image'''
    img = cv2.imdecode(np.fromfile(filename,dtype=np.uint8), -1)
    return img

def save_image(filename, image):
    '''save image'''
    ret, code = cv2.imencode('.jpg', image)
    code.tofile(filename)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True, help='input image path')
    parser.add_argument('--o', required=True, help='output path')
    parser.add_argument('--output_side_length', default=256, type=int, help='size of shorter side resize to')

    parser.add_argument('--multiprocessing', default=False, action='store_true', help='whether to use multi-process pool')
    parser.add_argument('--pool_size', default=12, type=int, help='pool size')

    parser.add_argument('--v', default=False, action='store_true', help='Print some extra info')
    return parser.parse_args()


def process_path(ipath, opath, files, output_side_length):
    for file in files:
        if '.' in file:
            ext = file[file.rindex('.')+1:]
            if ext in IMAGE_EXTS:
                image_filename = os.path.join(ipath, file)
                if not os.path.exists(opath):
                    os.makedirs(opath)
                output_file = os.path.join(opath, file)
                image = load_image(image_filename)
                assert image is not None, "can't load image [{}]".format(image_filename)
                shape = image.shape
                if len(shape) == 3:
                    height, width, depth = shape
                elif len(shape) == 2:
                    height, width = shape
                else:
                    raise ValueError("can't process image [{}]".format(image_filename))
                new_height = output_side_length
                new_width = output_side_length
                if height > width:
                    new_height = int(output_side_length * height / width)
                else:
                    new_width = int(output_side_length * width / height)
                resized_img = cv2.resize(image, (new_width, new_height))
                save_image(output_file, resized_img)


def process_file(ipath, opath, file, output_side_length):
    if '.' in file:
        ext = file[file.rindex('.')+1:]
        if ext in IMAGE_EXTS:
            image_filename = os.path.join(ipath, file)
            if not os.path.exists(opath):
                os.makedirs(opath)
            output_file = os.path.join(opath, file)
            image = load_image(image_filename)
            assert image is not None, "can't load image [{}]".format(image_filename)
            shape = image.shape
            if len(shape) == 3:
                height, width, depth = shape
            elif len(shape) == 2:
                height, width = shape
            else:
                raise ValueError("can't process image [{}]".format(image_filename))
            new_height = output_side_length
            new_width = output_side_length
            if height > width:
                new_height = int(output_side_length * height / width)
            else:
                new_width = int(output_side_length * width / height)
            resized_img = cv2.resize(image, (new_width, new_height))
            save_image(output_file, resized_img)


def main(args_params):
    if args_params.multiprocessing:
        pool = multiprocessing.Pool(processes=args_params.pool_size)
    for root, dirs, files in os.walk(args_params.i):
        if args_params.v:
            print(root)
        for file in files:
            if args_params.multiprocessing:
                pool.apply_async(process_file, args=(root, root.replace(args_params.i, args_params.o), file, args_params.output_side_length))
            else:
                process_file(root, root.replace(args_params.i, args_params.o), file, args_params.output_side_length)
        for dir in dirs:
            ipath = os.path.join(root, dir)
            if args_params.v:
                print(ipath)
            files = os.listdir(ipath)
            for file in files:
                if args_params.multiprocessing:
                    pool.apply_async(process_file, args=(ipath, ipath.replace(args_params.i, args_params.o), file, args_params.output_side_length))
                else:
                    process_file(ipath, ipath.replace(args_params.i, args_params.o), file, args_params.output_side_length)
    if args_params.multiprocessing:
        pool.close()
        pool.join()
        

if __name__ == '__main__':
    args = parse()
    main(args)