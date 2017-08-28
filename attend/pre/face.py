import h5py
import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

class DATA:
    _mean_shape = None
    facial_points = {   'inner' : np.arange(17,68),
            'outer' : np.arange(0,27),
            'stable': np.array([36,39,42,45,33]),
            'Stable': np.array([19, 22, 25, 28, 10, 11, 12, 13, 14, 15, 16, 17, 18])+17,
            'all'   : np.arange(68)}

    @classmethod
    def load(cls):
        cls._mean_shape = np.load(dir_path + '/mean_face.npy')

    @classmethod
    def mean_shape(cls):
        if cls._mean_shape is None:
            cls.load()
        return cls._mean_shape

    @classmethod
    def facial_points_idxs(cls, key='stable'):
        return cls.facial_points.get(key)


class BoundingBox:
    def __init__(self, bbox_min, bbox_max):
        self.min = bbox_min
        self.max = bbox_max
        self._min = self.min
        self._max = self.max


    def __repr__(self):
        return 'BBox {} - {}'.format(tuple(self.min), tuple(self.max))

    def as_array(self):
        return np.array([self.min, self.max])

    def as_anchor_and_lengths(self):
        """
        Returns:
            xy: lower left
            width
            height
        """
        size = self.max - self.min
        return self.min, size[0], size[1]

    def astype(self, t):
        return BoundingBox(*self.as_array().astype(t))

    @property
    def lengths(self):
        return np.apply_along_axis(np.diff, 0, self.as_array())[0]

    @property
    def max_length(self):
        return self.lengths



def warp_to_mean_shape(pts, img=None, idxs=None, mean_shape=None):
    if idxs is None:
        idxs = DATA.facial_points_idxs()
    if mean_shape is None:
        mean_shape = DATA.mean_shape()

    # TODO needed?
    pts = pts.copy()

    # Only use a few points to warp face with, not all of them
    # pts = pts[idxs]
    mean_shape = mean_shape[idxs]

    pts_rescaled = rescale(pts).astype(np.float32)
    pts_rescaled = pts_rescaled[idxs]
    # pts = pts[idxs]
    M = cv2.estimateRigidTransform(pts_rescaled, mean_shape, True)
    A = M[:2,:2]
    b = M[:,2]
    pts_transformed = np.matmul(A, pts.T).T + b
    safety_padding = 50

    if not img is None:
        height, width = img.shape[:2]
        # If some points fall off, resize source so the final image will fit
        max_x, max_y = np.ceil(np.max(pts_transformed, axis=0)).astype(int)
        min_x, min_y = np.floor(np.min(pts_transformed, axis=0)).astype(int)
        padding_x = (abs(min(min_x - safety_padding, 0)),
                     max(max_x + safety_padding, width) - width)
        padding_y = (abs(min(min_y - safety_padding, 0)),
                     max(max_y + safety_padding, height) - height)
        padding = [padding_y, padding_x, (0,0)]

        # TODO change mode
        img_padded = np.pad(img, padding, 'symmetric')

        height, width = img_padded.shape[:2]

        # Pad the points so they are on the same place wrt the face
        bottom_padding = np.array([padding_x[0], padding_y[0]])
        pts += bottom_padding

        img_transformed = cv2.warpAffine(img_padded, M, (width, height))
        pts_transformed = np.matmul(A, pts.T).T + b

    if img is None:
        return pts_transformed
    else:
        return pts_transformed, img_transformed


def extract_face(img, pts, face_ratio=.8):
    """ Extract a square image containing the face, covering `face_ratio` """
    bbox = bounding_box(pts, True)
    bbox = pad_to_rectangle(bbox)
    assert bbox.lengths[0] == bbox.lengths[1], '{} != {}'.format(bbox.lengths[0], bbox_lengths[1])
    # bbox_size = np.apply_along_axis(np.diff, 0, bbox)[0,0]
    # TODO why lengths[0]?
    bbox_size = bbox.lengths[0]
    final_padding = bbox_size * (1 - face_ratio)
    # Calculate padding such that (face + padding) * scale == final_resolution
    padding = np.round(np.round(final_padding) / 2)
    final_bbox = pad_box(bbox, padding)
    # At this point the bounding box is ready to be applied to the image for a proper cropping
    # assert padded_lengths[0] == padded_lengths[1]
    img_cropped = crop_to(img, final_bbox)

    return img_cropped


def preprocess_and_extract_face(img, pts):
    pts_transformed, img_transformed = warp_to_mean_shape(pts, img)
    img_cropped = extract_face(img_transformed, pts_transformed)
    # assert img_cropped.shape[0] == img_cropped.shape[1], paint_and_save(img_cropped, pts_transformed)
    if 0 in img_cropped.shape:
        raise Exception('Cropped img with a 0 dimension')
    img_final = resize(img_cropped)
    # paint_and_save(img_transformed, pts_transformed, 'test.png')
    return img_final


def bounding_box(pts, round=False):
    bbox_min = np.array(np.min(pts, axis=0))
    bbox_max = np.array(np.max(pts, axis=0))
    if round:
        bbox_min, bbox_max = np.ceil(bbox_min), np.floor(bbox_max)

    return BoundingBox(bbox_min, bbox_max)


def rescale(pts):
    """ Rescales points to [0,1] while keeping ratio between x and y """
    bbox = bounding_box(pts)
    # There'll be a bigger diff on one axis
    max_diff = np.max(bbox.lengths)
    num_pts = pts.shape[0]
    pts = pts - np.repeat([bbox.min], num_pts, axis=0)
    pts = pts / max_diff
    return pts


def pad_to_rectangle(box):
    box = np.copy(box.as_array())
    lengths = np.apply_along_axis(np.diff, 0, box)[0]
    short, long = np.min(lengths), np.max(lengths)
    diff = long - short

    # 0 means x gets padded
    axis = 0 if short == lengths[0] else 1
    pad = np.zeros((2,2))
    pad[0][axis] = np.floor(diff/2)
    pad[1][axis] = np.ceil(diff/2)
    padded = pad_box(BoundingBox(*box), pad)
    return padded


def pad_box(box, pad):
    box = np.copy(box.as_array()).astype(np.float64)
    if np.ndim(pad) == 0:
        pad = np.array([pad, pad])
    if np.ndim(pad) == 1:
        pad = np.repeat([pad], 2, axis=0)

    box[0] -= pad[0]
    box[1] += pad[1]

    return BoundingBox(*box)


def crop_to(img, box):
    # img is of shape height x width
    height, width, _ = img.shape
    box = box.astype(np.int64)
    # First take the y slice, then x slice
    min_x = max(box.min[0], 0)
    min_y = max(box.min[1], 0)
    max_x = min(box.max[0], width)
    max_y = min(box.max[1], height)
    final = img[min_y:max_y, min_x:max_x]
    return final



def resize(img, res=224):
    # if img.shape[0] == 0 or img.shape[1] == 0:
    #     import pdb
    #     pdb.set_trace()
    # Ehh x and y flipped again
    resized = cv2.resize(img,None,fx=res/img.shape[1], fy=res/img.shape[0], interpolation = cv2.INTER_CUBIC)
    if not (resized.shape[0] == res and resized.shape[1] == res):
        print(img.shape)
        print(resized.shape)
        raise Exception('Resize failed')

    return resized


def paint_and_save(img, pts, path='/home/ruben/tmp', filename='tmp.png'):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.imshow(img)
    plt.scatter(pts[:,0], pts[:,1], marker='.', s=10, color='red')
    plt.savefig(path + '/' + filename)


def mean_and_stdev_normalize(pts):
    p = np.copy(pts)
    p -= np.mean(p, axis=0)
    # This is the stdec across x/y because I don't want to warp the image
    r = np.std(p)
    p = p / r
    return p
