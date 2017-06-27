#!/usr/bin/env python

# Quick one-off script to benchmark the use of all.hf5,
# which contains all videos and frames

from datetime import datetime
import glob
import h5py
import numpy as np
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_file', type=str, help='input frames directory, containing subfolders')
parser.add_argument('--parallel', type=int, default=1)

def random_sample_generator(
        vids_group,
        # target_group,
        vid_names,
        rng
        ):

    while True:
        vid_name = rng.choice(vid_names)
        vid = vids_group[vid_name]
        n_frames = vid.shape[0]
        frame_i = rng.randint(n_frames)

        frame = vids_group[vid_name][frame_i]
        # target = target[vid_name][frame_i]

        # yield frame, target
        yield frame



def random_batch_generator(
        batch_size,
        **kwargs
        ):

    sample_gen = random_sample_generator(**kwargs)

    while True:
        yield [next(sample_gen) for _ in range(batch_size)]


def parallel_batch_generator(
        batch_size,
        n,
        **kwargs
        ):

    sample_gen = random_sample_generator(**kwargs)

    _gen = lambda _: next(sample_gen)

    # Silly Ruben you cant pickle a generator it has state
    # TODO? maybe
    while True:
        pool = multiprocessing.Pool(n)
        return list(pool.imap_unordered(_gen, range(batch_size)))


if __name__ == '__main__':
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    num_trials = 10
    batch_size = 100

    with h5py.File(args.in_file, 'r') as f:
        vid_names = list(f['vids'])

        if args.parallel == 1:
            gen = random_batch_generator(
                    batch_size,
                    vids_group = f['vids'],
                    # target_group =
                    vid_names=vid_names,
                    rng=rng
                    )
        elif args.parallel >= 1:
            gen = parallel_batch_generator(
                    batch_size,
                    args.parallel,
                    vids_group = f['vids'],
                    # target_group =
                    vid_names=vid_names,
                    rng=rng
                    )

        start = datetime.now()

        for i in range(num_trials):
            batch = next(gen)
            # print(batch)

        end = datetime.now()

        print((end-start)/num_trials)
        print(end-start)
