import EmoData as ED
import h5py
import glob
from collections import defaultdict
from numpy.random import random, seed, shuffle
import numpy as np
np.random.seed(4)


if 1:
    files = sorted(list(glob.glob('/homes/rw2614/data/similarity_256_256/fera2015/*')))

    TRAIN = [
        'F001',
        'F003',
        'F005',
        'F007',
        'F009',
        'k011',
        'F013',
        'F015',
        'F017',
        'F019',
        'F021',
        'F023',
        'M001',
        'M003',
        'M005',
        'M007',
        'M009',
        'M011',
        'M013',
        'M015',
        'M017'
    ]
    TEST = [
        'F002',
        'F004',
        'F006',
        'F008',
        'F010',
        'F012',
        'F014',
        'F016',
        'F018',
        'F020',
        'F022',
        'M002',
        'M004',
        'M006',
        'M008',
        'M010',
        'M012',
        'M014',
        'M016',
        'M018',
    ]
    sub_list = np.unique(TEST+TRAIN)

    tr_list, te_list = [],[]
    tr_sub , te_sub  = [],[]
    for seq_path in files:
        seq_id = seq_path.split('/')[-1][:4]
        if seq_id in TRAIN:
            sub_id = np.argwhere(seq_id==sub_list)[0,0]
            tr_sub.append(sub_id)
            tr_list.append(seq_path)
        if seq_id in TEST:
            sub_id = np.argwhere(seq_id==sub_list)[0,0]
            te_sub.append(sub_id)
            te_list.append(seq_path)

    print(tr_sub[:2])
    print(tr_list[:2])

    ED.transformer.merge_h5(tr_list, '/homes/rw2614/data/similarity_256_256/fera2015_tr.h5', shuffle=True, subject_ids=tr_sub)
    ED.transformer.merge_h5(te_list, '/homes/rw2614/data/similarity_256_256/fera2015_te.h5', shuffle=True, subject_ids=te_sub)

if 0:
    files = sorted(list(glob.glob('/homes/rw2614/data/similarity_256_256/disfa/*.h5')))
    sub_ids = np.arange(len(files))
    tr_list, te_list = files[:18], files[18:]
    tr_subs, te_subs = sub_ids[:18], sub_ids[18:]
    tr_subs = tr_subs.tolist()
    te_subs = te_subs.tolist()

    ED.transformer.merge_h5(tr_list, '/homes/rw2614/data/similarity_256_256/disfa_tr.h5', shuffle=True, subject_ids=tr_subs)
    ED.transformer.merge_h5(te_list, '/homes/rw2614/data/similarity_256_256/disfa_te.h5', shuffle=True, subject_ids=te_subs)

if 0:
    subs = defaultdict(list)
    files = sorted(list(glob.glob('./similarity_256_256/pain/*.h5')))
    for f in files:
        sub_id = f.split('_')[-2]
        subs[sub_id].append(f)

    subjects = np.array(sorted(list(subs.keys())))
    idx = np.arange(len(subjects))
    np.random.shuffle(idx)
    subjects=subjects[idx]

    tr = subjects[:17]
    te = subjects[17:]

    tr_list = []
    te_list = []
    tr_subs = []
    te_subs = []
    for i, sub in enumerate(subs):
        if sub in tr:
            tr_list.extend(subs[sub])
            tr_subs.extend([i]*len(subs[sub]))
        if sub in te:
            te_list.extend(subs[sub])
            te_subs.extend([i]*len(subs[sub]))

    ED.transformer.merge_h5(tr_list, '/homes/rw2614/data/similarity_256_256/pain_tr.h5', shuffle=True, subject_ids=tr_subs)
    ED.transformer.merge_h5(te_list, '/homes/rw2614/data/similarity_256_256/pain_te.h5', shuffle=True, subject_ids=te_subs)

if 1:
    files = sorted(list(glob.glob('/homes/rw2614/data/similarity_256_256/imdb_wiki/*.h5')))
    files = np.array(files)
    sub_ids = np.array([int(i[-6:-3]) for i in files])
    tr_list, te_list = files[sub_ids<=70], files[sub_ids>70]
    tr_subs, te_subs = sub_ids[sub_ids<=70], sub_ids[sub_ids>70]
    tr_subs = tr_subs.tolist()
    te_subs = te_subs.tolist()
    print(len(tr_subs))
    print(len(te_subs))

    ED.transformer.merge_h5(tr_list, '/homes/rw2614/data/similarity_256_256/imdb_wiki_tr.h5', shuffle=True, subject_ids=tr_subs)
    ED.transformer.merge_h5(te_list, '/homes/rw2614/data/similarity_256_256/imdb_wiki_te.h5', shuffle=True, subject_ids=te_subs)
