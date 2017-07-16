import h5py
import numpy as np
import cv2


class DATA:
    _mean_shape = None

    @classmethod
    def load(cls):
        cls._mean_shape = np.load('./mean_face.npy')

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
    img_transformed = cv2.warpAffine(img, M, (224,224))
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
    img_cropped = crop_to(img, bbox_padded)

    return img_cropped


def preprocess_and_extract_face(img, pts):
    img_transformed, pts_transformed = warp_to_mean_shape(img, pts)
    img_cropped = extract_face(img_transformed, pts_transformed)
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
    return img[box[0][1]:box[1][1], box[0][0]:box[1][0]]


def resize(img, res=224):
    img = cv2.resize(img,None,fx=res/img.shape[0], fy=res/img.shape[1], interpolation = cv2.INTER_CUBIC)
    assert img.shape[0] == res and img.shape[1] == res
    return img
