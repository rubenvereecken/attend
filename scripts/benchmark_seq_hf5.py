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

def seq_batch_generator(
        # vids_group,
        # target_group,
        # vid_names,
        dset,
        batch_size
        # rng
        ):

    n_samples = dset.shape[0]
    offset = 0

    while True:
        current_batch_size = min(batch_size, n_samples - offset)
        frames = dset[offset:offset+current_batch_size]

        # Loop around
        if current_batch_size < batch_size:
            offset = 0
            extra_batch_size = batch_size - current_batch_size
            extra_frames = dset[offset:extra_batch_size]
            offset += extra_batch_size
            frames = np.stack(frames, extra_frames)
        else:
            offset += batch_size

        yield frames


# def parallel_batch_generator(
#         batch_size,
#         n,
#         **kwargs
#         ):

#     sample_gen = random_sample_generator(**kwargs)

#     _gen = lambda _: next(sample_gen)

#     # Silly Ruben you cant pickle a generator it has state
#     # TODO? maybe
#     while True:
#         pool = multiprocessing.Pool(n)
#         return list(pool.imap_unordered(_gen, range(batch_size)))


if __name__ == '__main__':
    args = parser.parse_args()

    # rng = np.random.RandomState(0)
    num_trials = 10
    batch_size = 100

    with h5py.File(args.in_file, 'r') as f:
        # vid_names = list(f['vids'])

        if args.parallel == 1:
            gen = seq_batch_generator(
                    f['img'],
                    batch_size
                    # rng=rng
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
