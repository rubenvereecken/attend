import h5py

def find_leaf_group_paths(h):
    '''
    Assumes a level with datasets does not hold groups
    '''
    paths = set([])

    def _walk(h):
        for _, child in h.items():
            # print(child)
            if isinstance(child, h5py.Dataset):
                paths.add(h.name)
                break
            _walk(child)

    _walk(h)
    return list(paths)
