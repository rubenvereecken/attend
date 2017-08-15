import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="attend",
    version="0.0.1",
    author="Ruben Vereecken",
    author_email="r.vereecken@imperial.ac.uk",
    description=(""),
    license = "BSD",
    keywords = "",
    url = "",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    data_files = [
        # ('./faceKit/data/', ['./faceKit/data/faceKit_ResNet50.h5','./faceKit/data/mean_shape.h5', './faceKit/data/shape_predictor_68_face_landmarks.dat']),
        ],
    packages=find_packages(),
    entry_points= {
        'console_scripts': [
            'confer-to-images = attend.pre.confer_to_images:main',
            'images-to-features = attend.pre.images_to_features:main',
            'confer-annot = attend.pre.confer_annot:main',
            ]
        }
)

