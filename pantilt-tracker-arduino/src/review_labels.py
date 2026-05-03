import argparse
import csv
import os
from collections import defaultdict

import cv2


def load_labels(csv_path):
    labels = defaultdict(list)
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[row['filename']].append(
                [
                    int(float(row['x'])),
                    int(float(row['y'])),
                    int(float(row['w'])),
                    int(float(row['h'])),
                ]
            )
    return labels


def save_labels(csv_path, labels, ordered_filenames):
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['filename', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        for filename in ordered_filenames:
            for x, y, w, h in labels.get(filename, []):
                writer.writerow({
                    'filename': filename,
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h),
                })


def image_files_in_dir(dataset_dir):
    return [
        name for name in sorted(os.listdir(dataset_dir))
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]


def draw_overlay(img, filename, idx, total, boxes):
    canvas = img.copy()
    for box_idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 255), 2)
        cv2.putText(canvas, str(box_idx), (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(canvas, f'{idx + 1}/{total}: {filename}', (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, 'n/p: next/prev  a:add ROI  x:delete last  c:clear  s:save  q:quit',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
    return canvas


def clamp_box(box, width, height):
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return [int(x), int(y), int(w), int(h)]


def review_labels(dataset_dir, labels_csv):
    filenames = image_files_in_dir(dataset_dir)
    if not filenames:
        raise RuntimeError('No image files found in dataset directory.')

    labels = load_labels(labels_csv)
    idx = 0
    dirty = False

    window_name = 'Label Reviewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        filename = filenames[idx]
        image_path = os.path.join(dataset_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            idx = (idx + 1) % len(filenames)
            continue

        h, w = img.shape[:2]
        labels[filename] = [clamp_box(box, w, h) for box in labels.get(filename, [])]
        canvas = draw_overlay(img, filename, idx, len(filenames), labels[filename])
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('n'), 83):  # n or right arrow
            idx = min(len(filenames) - 1, idx + 1)
        elif key in (ord('p'), 81):  # p or left arrow
            idx = max(0, idx - 1)
        elif key == ord('a'):
            roi = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                labels[filename].append(clamp_box([x, y, rw, rh], w, h))
                dirty = True
        elif key == ord('x'):
            if labels[filename]:
                labels[filename].pop()
                dirty = True
        elif key == ord('c'):
            labels[filename] = []
            dirty = True
        elif key == ord('s'):
            save_labels(labels_csv, labels, filenames)
            print(f'Saved: {labels_csv}')
            dirty = False
        elif key == ord('q'):
            if dirty:
                save_labels(labels_csv, labels, filenames)
                print(f'Auto-saved before exit: {labels_csv}')
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Review and edit face box labels quickly.')
    parser.add_argument('--dataset-dir', required=True, help='Directory with evaluation images')
    parser.add_argument('--labels-csv', required=True, help='CSV label file to edit')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    review_labels(dataset_dir=args.dataset_dir, labels_csv=args.labels_csv)
