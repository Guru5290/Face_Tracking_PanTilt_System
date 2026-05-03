import argparse
import csv
import os

import cv2
import dlib

from config import (
    FACE_CASCADE_PATH,
    FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
    FACE_MIN_SIZE,
    DLIB_HOG_UPSAMPLE,
    OPENCV_DNN_PROTOTXT_PATH,
    OPENCV_DNN_MODEL_PATH,
    OPENCV_DNN_CONFIDENCE_THRESHOLD,
    OPENCV_DNN_TARGET,
)
from model_assets import ensure_runtime_models


def _configure_dnn(net):
    if OPENCV_DNN_TARGET == 'cuda':
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detect_opencv_dnn(frame_bgr, dnn_net):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    out = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < OPENCV_DNN_CONFIDENCE_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        out.append((int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))))
    return out


def image_files_in_dir(dataset_dir):
    return [
        name for name in sorted(os.listdir(dataset_dir))
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]


def detect_haar(gray, cascade):
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_dlib(gray, detector):
    rects = detector(gray, DLIB_HOG_UPSAMPLE)
    boxes = []
    for rect in rects:
        x = max(rect.left(), 0)
        y = max(rect.top(), 0)
        w = max(rect.right() - x, 1)
        h = max(rect.bottom() - y, 1)
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def prelabel_dataset(dataset_dir, output_csv, detector_backend):
    ensure_runtime_models()
    files = image_files_in_dir(dataset_dir)
    if not files:
        raise RuntimeError('No image files found in dataset directory.')

    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector_backend == 'haar' and cascade.empty():
        raise RuntimeError(f'Haar cascade missing or invalid: {FACE_CASCADE_PATH}')
    dlib_detector = dlib.get_frontal_face_detector()
    dnn_net = None
    if detector_backend == 'opencv_dnn':
        dnn_net = cv2.dnn.readNetFromCaffe(OPENCV_DNN_PROTOTXT_PATH, OPENCV_DNN_MODEL_PATH)
        _configure_dnn(dnn_net)

    rows = []
    for filename in files:
        image_path = os.path.join(dataset_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if detector_backend == 'haar':
            boxes = detect_haar(gray, cascade)
        elif detector_backend == 'opencv_dnn':
            boxes = detect_opencv_dnn(img, dnn_net)
        else:
            boxes = detect_dlib(gray, dlib_detector)

        for x, y, w, h in boxes:
            rows.append({
                'filename': filename,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            })

    with open(output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['filename', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Prelabels written: {output_csv}')
    print(f'Total box rows: {len(rows)}')
    print(f'Detector used: {detector_backend}')


def parse_args():
    parser = argparse.ArgumentParser(description='Auto-generate initial face box labels CSV.')
    parser.add_argument('--dataset-dir', required=True, help='Directory with evaluation images')
    parser.add_argument('--output-csv', default='labels.csv', help='Output labels CSV path')
    parser.add_argument(
        '--detector',
        default='opencv_dnn',
        choices=['dlib', 'haar', 'opencv_dnn'],
        help='Detector used for seed boxes before manual review (opencv_dnn recommended for PiCam/Jetson)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prelabel_dataset(
        dataset_dir=args.dataset_dir,
        output_csv=args.output_csv,
        detector_backend=args.detector,
    )
