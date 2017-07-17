import h5py
import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

class DATA:
    _mean_shape = None

    @classmethod
    def load(cls):
        cls._mean_shape = np.load(dir_path + '/mean_face.npy')

    @classmethod
    def mean_shape(cls):
        if cls._mean_shape is None:
            cls.load()
        return cls._mean_shape


def warp_to_mean_shape(img, pts, mean_shape=None):
    if mean_shape is None:
        mean_shape = DATA.mean_shape()

    pts_rescaled = rescale(pts).astype(np.float32)
    M = cv2.estimateRigidTransform(pts_rescaled, mean_shape, True)
    shape_2d = lambda x:tuple(reversed(list(x.shape[:2])))
    # shape_2d(img)
    img_transformed = cv2.warpAffine(img, M, shape_2d(img))
    pts_transformed = np.matmul(M[:2,:2], pts.T).T + M[:,2]

    return img_transformed, pts_transformed


def extract_face(img, pts, face_ratio=.8):
    """ Extract a square image containing the face, covering `face_ratio` """
    bbox = bounding_box(pts)
    bbox = pad_to_rectangle(bbox)
    bbox_lengths = box_lengths(bbox)
    assert bbox_lengths[0] == bbox_lengths[1]
    bbox_size = np.apply_along_axis(np.diff, 0, bbox)[0,0]
    #face_scale = face_res / bbox_size
    #final_padding = final_res - face_res
    final_padding = bbox_size * (1 - face_ratio)
    # Calculate padding such that (face + padding) * scale == final_resolution
    padding = np.round(np.round(final_padding) / 2)
    bbox_padded = pad_box(bbox, padding)
    # At this point the bounding box is ready to be applied to the image for a proper cropping
    bbox_padded = np.round(bbox_padded).astype(np.int64) # Dunno if needed
    # padded_lengths = box_lengths(bbox_padded)
    # assert padded_lengths[0] == padded_lengths[1]
    img_cropped = crop_to(img, bbox_padded)

    return img_cropped


def preprocess_and_extract_face(img, pts):
    img_transformed, pts_transformed = warp_to_mean_shape(img, pts)
    img_cropped = extract_face(img_transformed, pts_transformed)
    # assert img_cropped.shape[0] == img_cropped.shape[1], paint_and_save(img_cropped, pts_transformed)
    img_final = resize(img_cropped)
    return img_final


def bounding_box(pts):
    bbox_min = np.ceil(np.array([np.min(pts[:,0]), np.min(pts[:,1])]))
    bbox_max = np.floor(np.array([np.max(pts[:,0]), np.max(pts[:,1])]))
    return np.array([bbox_min, bbox_max])


def rescale(pts):
    """ Rescales points to [0,1] while keeping ratio between x and y """
    bbox_min, bbox_max = bounding_box(pts)
    # There'll be a bigger diff on one axis
    max_diff = np.max(bbox_max-bbox_min)
    pts = pts - np.repeat([bbox_min], pts.shape[0], axis=0)
    pts = pts / max_diff
    return pts


def box_lengths(box):
    return np.apply_along_axis(np.diff, 0, box)[0]


def pad_to_rectangle(box):
    box = np.copy(box)
    lengths = np.apply_along_axis(np.diff, 0, box)[0]
    short, long = np.min(lengths), np.max(lengths)
    diff = long - short

    # 0 means x gets padded
    axis = 0 if short == lengths[0] else 1
    pad = np.zeros((2,2))
    pad[0][axis] = np.floor(diff/2)
    pad[1][axis] = np.ceil(diff/2)
    return pad_box(box, pad)


def pad_box(box, pad):
    box = np.copy(box).astype(np.float64)
    if np.ndim(pad) == 0:
        pad = np.array([pad, pad])
    if np.ndim(pad) == 1:
        pad = np.repeat([pad], 2, axis=0)

    box[0] -= pad[0]
    box[1] += pad[1]

    return box


def crop_to(img, box):
    # First take the y slice, then x slice
    min_x = box[0][1] if box[0][1] >= 0 else 0
    min_y = box[0][0] if box[0][0] >= 0 else 0
    max_x = box[1][1] if box[1][1] < img.shape[1] else img.shape[1] # switch right?
    max_y = box[1][0] if box[1][0] < img.shape[0] else img.shape[0]
    return img[min_x:max_x, min_y:max_y]


def resize(img, res=224):
    # Ehh x and y flipped again
    resized = cv2.resize(img,None,fx=res/img.shape[1], fy=res/img.shape[0], interpolation = cv2.INTER_CUBIC)
    if not (resized.shape[0] == res and resized.shape[1] == res):
        print(img.shape)
        print(resized.shape)
        raise Exception('Resize failed')

    return resized


def paint_and_save(img, pts, filename='tmp.png'):
    path = '/home/ruben/tmp'
    import matplotlib.pyplot as plt
    print(img.shape)

    plt.imshow(img)
    plt.scatter(pts[:,0], pts[:,1], marker='.')
    plt.savefig(path + '/' + filename)

