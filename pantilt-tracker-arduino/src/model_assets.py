import bz2
import os
import shutil
import urllib.request

from config import (
    FACE_CASCADE_PATH,
    DLIB_LANDMARK_MODEL_PATH,
    DLIB_LANDMARK_MODEL_URL,
    OPENCV_DNN_PROTOTXT_PATH,
    OPENCV_DNN_MODEL_PATH,
)


CASCADE_URL = (
    'https://raw.githubusercontent.com/opencv/opencv/master/'
    'data/haarcascades/haarcascade_frontalface_default.xml'
)
OPENCV_DNN_PROTOTXT_URL = (
    'https://raw.githubusercontent.com/opencv/opencv/master/'
    'samples/dnn/face_detector/deploy.prototxt'
)
OPENCV_DNN_MODEL_URL = (
    'https://raw.githubusercontent.com/opencv/opencv_3rdparty/'
    'dnn_samples_face_detector_20170830/'
    'res10_300x300_ssd_iter_140000.caffemodel'
)


def _download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def ensure_cascade_file():
    if os.path.exists(FACE_CASCADE_PATH):
        return
    print('Downloading Haar cascade...')
    _download_file(CASCADE_URL, FACE_CASCADE_PATH)
    print('Haar cascade ready.')


def ensure_dlib_landmark_model():
    if os.path.exists(DLIB_LANDMARK_MODEL_PATH):
        return

    compressed_path = DLIB_LANDMARK_MODEL_PATH + '.bz2'
    print('Downloading dlib landmark model (this can take a while)...')
    _download_file(DLIB_LANDMARK_MODEL_URL, compressed_path)
    print('Decompressing landmark model...')
    with bz2.BZ2File(compressed_path, 'rb') as src, open(DLIB_LANDMARK_MODEL_PATH, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    os.remove(compressed_path)
    print('Dlib landmark model ready.')


def ensure_opencv_dnn_face_model():
    if not os.path.exists(OPENCV_DNN_PROTOTXT_PATH):
        print('Downloading OpenCV DNN deploy.prototxt...')
        _download_file(OPENCV_DNN_PROTOTXT_URL, OPENCV_DNN_PROTOTXT_PATH)
    if not os.path.exists(OPENCV_DNN_MODEL_PATH):
        print('Downloading OpenCV DNN Caffe face model...')
        _download_file(OPENCV_DNN_MODEL_URL, OPENCV_DNN_MODEL_PATH)
    print('OpenCV DNN face model ready.')


def ensure_runtime_models():
    # Keep multiple detectors available for runtime switching and benchmarking.
    ensure_cascade_file()
    ensure_dlib_landmark_model()
    ensure_opencv_dnn_face_model()
