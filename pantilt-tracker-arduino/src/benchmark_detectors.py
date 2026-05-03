import argparse
import csv
import os
import time
from collections import defaultdict

import cv2
import dlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _clean_csv_row(row):
    """Strip BOM/spaces from keys (Excel UTF-8 often adds \\ufeff to first header)."""
    return {k.lstrip('\ufeff').strip(): v for k, v in row.items()}


def load_ground_truth(csv_path):
    gt = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        cleaned_headers = {f.lstrip('\ufeff').strip() for f in fieldnames}
        required = {'filename', 'x', 'y', 'w', 'h'}
        if not required.issubset(cleaned_headers):
            raise ValueError(
                'Ground truth CSV must include columns: filename,x,y,w,h '
                f'(got {fieldnames})'
            )
        for row in reader:
            row = _clean_csv_row(row)
            fname = (row.get('filename') or '').strip()
            if not fname:
                continue
            gt[fname].append(
                (
                    int(float(row['x'])),
                    int(float(row['y'])),
                    int(float(row['w'])),
                    int(float(row['h'])),
                )
            )
    return gt


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def match_counts(preds, gts, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            current = iou(pred, gt)
            if current > best_iou:
                best_iou = current
                best_idx = idx
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


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
    out = []
    for rect in rects:
        x = max(rect.left(), 0)
        y = max(rect.top(), 0)
        w = max(rect.right() - x, 1)
        h = max(rect.bottom() - y, 1)
        out.append((int(x), int(y), int(w), int(h)))
    return out


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


def metrics_from_counts(tp, fp, fn, elapsed_ms, frame_count):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_latency_ms = elapsed_ms / max(frame_count, 1)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_latency_ms': avg_latency_ms,
        'fps': fps,
        'frames': frame_count,
    }


def write_metrics_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'detector', 'frames', 'tp', 'fp', 'fn',
                'precision', 'recall', 'f1', 'avg_latency_ms', 'fps'
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_metric_plot(output_path, rows):
    detectors = [r['detector'] for r in rows]
    precision = [r['precision'] for r in rows]
    recall = [r['recall'] for r in rows]
    f1 = [r['f1'] for r in rows]
    latency = [r['avg_latency_ms'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(detectors))

    axes[0].bar([i - 0.25 for i in x], precision, width=0.25, label='Precision')
    axes[0].bar(x, recall, width=0.25, label='Recall')
    axes[0].bar([i + 0.25 for i in x], f1, width=0.25, label='F1')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(detectors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Detection Quality')
    axes[0].legend()

    bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    axes[1].bar(
        detectors,
        latency,
        color=bar_colors[: len(detectors)],
    )
    axes[1].set_title('Average Inference Latency (ms)')
    axes[1].set_ylabel('ms/frame')

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_qualitative_examples(image_files, gt_map, out_dir, cascade, dlib_detector, dnn_net):
    os.makedirs(out_dir, exist_ok=True)
    for image_path in image_files[:5]:
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = os.path.basename(image_path)
        gts = gt_map.get(base, [])
        haar_boxes = detect_haar(gray, cascade)
        dlib_boxes = detect_dlib(gray, dlib_detector)

        left = img.copy()
        center = img.copy()
        right = img.copy()
        for x, y, w, h in gts:
            cv2.rectangle(left, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(center, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(right, (x, y), (x + w, y + h), (255, 255, 255), 2)
        for x, y, w, h in haar_boxes:
            cv2.rectangle(left, (x, y), (x + w, y + h), (0, 255, 255), 2)
        for x, y, w, h in dlib_boxes:
            cv2.rectangle(center, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dnn_boxes = detect_opencv_dnn(img, dnn_net)
        for x, y, w, h in dnn_boxes:
            cv2.rectangle(right, (x, y), (x + w, y + h), (255, 120, 0), 2)

        cv2.putText(left, 'Haar (yellow) + GT (white)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(center, 'Dlib HOG+SVM (green) + GT (white)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(right, 'OpenCV DNN (blue) + GT (white)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        h0, w0 = left.shape[:2]
        if center.shape[:2] != (h0, w0) or right.shape[:2] != (h0, w0):
            continue
        combined = cv2.hconcat([left, center, right])
        cv2.imwrite(os.path.join(out_dir, base), combined)


def configure_opencv_dnn_net(net):
    """Match runtime tracker: CUDA when available (Jetson), else CPU."""
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


def evaluate(dataset_dir, labels_csv, output_dir, iou_threshold=0.5):
    ensure_runtime_models()
    os.makedirs(output_dir, exist_ok=True)

    gt = load_ground_truth(labels_csv)
    image_files = [
        os.path.join(dataset_dir, name)
        for name in sorted(os.listdir(dataset_dir))
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    if not image_files:
        raise RuntimeError('No images found in dataset directory.')

    labeled_names = set(gt.keys())
    missing_gt = [
        os.path.basename(p) for p in image_files
        if os.path.basename(p) not in labeled_names
    ]
    if missing_gt:
        print(
            f'Warning: {len(missing_gt)} image(s) have no rows in labels CSV — '
            'they are scored as "zero faces" ground truth. Add rows or remove those files.'
        )

    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            f'Haar cascade failed to load (missing or invalid XML): {FACE_CASCADE_PATH}'
        )
    dlib_detector = dlib.get_frontal_face_detector()

    dnn_net = cv2.dnn.readNetFromCaffe(OPENCV_DNN_PROTOTXT_PATH, OPENCV_DNN_MODEL_PATH)
    configure_opencv_dnn_net(dnn_net)

    def run_detector(label, detector_fn):
        tp = fp = fn = 0
        total_elapsed_ms = 0.0
        processed = 0
        for image_path in image_files:
            img = cv2.imread(image_path)
            if img is None:
                print(f'Warning: could not read image, skipping: {image_path}')
                continue
            processed += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start = time.perf_counter()
            preds = detector_fn(img, gray)
            total_elapsed_ms += (time.perf_counter() - start) * 1000.0
            gts = gt.get(os.path.basename(image_path), [])
            c_tp, c_fp, c_fn = match_counts(preds, gts, iou_threshold=iou_threshold)
            tp += c_tp
            fp += c_fp
            fn += c_fn
        if processed == 0:
            raise RuntimeError('No images could be read from dataset directory.')
        row = metrics_from_counts(tp, fp, fn, total_elapsed_ms, processed)
        row['detector'] = label
        return row

    haar_row = run_detector('haar', lambda _img, gray: detect_haar(gray, cascade))
    dlib_row = run_detector('dlib_hog_svm', lambda _img, gray: detect_dlib(gray, dlib_detector))
    dnn_row = run_detector('opencv_dnn', lambda img, _gray: detect_opencv_dnn(img, dnn_net))
    rows = [haar_row, dlib_row, dnn_row]

    metrics_csv_path = os.path.join(output_dir, 'detector_metrics.csv')
    plot_path = os.path.join(output_dir, 'detector_comparison.png')
    examples_dir = os.path.join(output_dir, 'qualitative_examples')

    write_metrics_csv(metrics_csv_path, rows)
    save_metric_plot(plot_path, rows)
    save_qualitative_examples(image_files, gt, examples_dir, cascade, dlib_detector, dnn_net)

    print(f'Metrics CSV: {metrics_csv_path}')
    print(f'Comparison chart: {plot_path}')
    print(f'Qualitative overlays: {examples_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Haar, dlib HOG+SVM, and OpenCV DNN face detectors on labeled images.'
    )
    parser.add_argument('--dataset-dir', required=True, help='Directory with evaluation images')
    parser.add_argument(
        '--labels-csv',
        required=True,
        help='CSV with columns: filename,x,y,w,h (one row per face)'
    )
    parser.add_argument(
        '--output-dir',
        default='report_artifacts',
        help='Where metrics and plots are saved'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching predictions to ground truth (default 0.5)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.dataset_dir, args.labels_csv, args.output_dir, iou_threshold=args.iou_threshold)
