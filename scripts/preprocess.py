import glob
import h5py
import numpy as np
import multiprocessing
import argparse
import os
import shutil
from skimage.io import imread
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out_dir', type=str, help='dir to write hf5 files to')
parser.add_argument('-f', '--format', type=str, choices=['hdf5', 'npz', 'np'],
                    default='hdf5', help='Output format')

def resize_to(w, h):
    def _resize_frame(img):
        pass

    return _resize_frame

PREPROCESS = [
    # resize_to(256, 256),
]

def preprocess_frame(frame):
    for preprocessor in PREPROCESS:
        frame = preprocessor(frame)

    return frame


# def process_dir(in_dir, out_file):
def process_dir(params):
    # pool.imap is annoying in that it won't **kwarg
    in_dir = params['in_dir']
    out_file = params['out_file']
    format = params['format']

    frame_files = glob.glob(in_dir + '/*.jpg')
    # print(in_dir)
    # print(len(frame_files))

    def _do_frame(frame_file):
        img = imread(frame_file)
        img = preprocess_frame(img)
        return img

    frames = list(map(_do_frame, frame_files))
    frames = np.stack(frames)

    if format == 'hdf5':
        with h5py.File(out_file + '.hf5') as out:
            out.create_dataset('img', data=frames, dtype=np.uint8)
    elif format == 'npz':
        np.savez_compressed(out_file, img=frames)


    else: pass


    # return frames
    return frames.shape


if __name__ == '__main__':
    args = parser.parse_args()

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir)

    def _argify(in_dir):
        basename = in_dir.split('/')[-1]
        params = {
            'in_dir': in_dir,
            'out_file': '{}/{}'.format(args.out_dir, basename),
            'format': args.format,
        }
        return params

    all_args = list(map(_argify, glob.glob(args.in_dir + '/*')))

    # for a in all_args:
    #     process_dir(a)

    pool = multiprocessing.Pool(4)

    for result in tqdm(pool.imap_unordered(process_dir, all_args), total=len(all_args)):
        # print('.', end='')
        pass
    # pool.map(process_dir, all_args)
    pool.terminate()
    pool.join()
    # print('')
