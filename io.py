import math

def train_and_test_from_hdf5(
        file_name,
        rng,
        train_size # <= 1.
        ):

    f = h5py.File(file_name, 'r')
    vids = f['annot']

    n_vids = len(vids)

    n_train_vids = math.ceil(n_vids * train_size)
    n_test_vids = n_vids - n_train_vids


def random_sample_generator(
        vids_group,
        target_group,
        vid_names,
        rng
        ):

    vid_name = np.choice(vid_names)
    n_frames = vid.shape[0]
    frame_i = rng.randint(n_frames)

    frame = vids_group[vid_name][frame_i]
    target = target[vid_name][frame_i]

    yield frame, target


def random_batch_generator(
        batch_size,
        *args
        ):

    yield [random_sample_generator(*args) for _ in range(batch_size)]
