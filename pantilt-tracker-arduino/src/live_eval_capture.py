"""
Save evaluation frames during live face tracking (one camera process).

Writes JPEGs + tracking_manifest.csv under eval_sessions/<timestamp>/ for offline
benchmarking against Haar / dlib / OpenCV DNN on the same real tracking scenes.
"""
import csv
import os
import time

import cv2


class LiveEvalSession:
    def __init__(self, base_dir='eval_sessions', every_n_frames=15, max_frames=400):
        every_n_frames = max(1, int(every_n_frames))
        max_frames = max(1, int(max_frames))
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.abspath(os.path.join(base_dir, ts))
        os.makedirs(self.session_dir, exist_ok=True)
        self.every_n = every_n_frames
        self.max_frames = max_frames
        self.saved_count = 0
        self.loop_counter = 0

        self.manifest_path = os.path.join(self.session_dir, 'tracking_manifest.csv')
        self._mf = open(self.manifest_path, 'w', newline='', encoding='utf-8')
        self._mw = csv.writer(self._mf)
        self._mw.writerow([
            'filename', 'unix_ts', 'loop_idx', 'pan_deg', 'tilt_deg',
            'tracking_detector', 'face_locked', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        ])
        self._mf.flush()

        print(f'Live evaluation session: saving up to {max_frames} frames every {every_n_frames} loops → {self.session_dir}')

    def maybe_save(
        self,
        frame_bgr_clean,
        pan_deg,
        tilt_deg,
        tracking_detector,
        face_locked,
        bbox_xywh,
    ):
        """
        frame_bgr_clean: untouched camera frame (no overlays).
        bbox_xywh: largest tracked face (x,y,w,h) or None when unlocked.
        """
        self.loop_counter += 1
        if self.saved_count >= self.max_frames:
            return False
        if self.loop_counter % self.every_n != 0:
            return False

        fname = f'track_{self.saved_count:05d}.jpg'
        path = os.path.join(self.session_dir, fname)
        try:
            if not cv2.imwrite(path, frame_bgr_clean):
                print(f'[eval] imwrite failed: {path}')
                return False
        except Exception as exc:
            print(f'[eval] save error: {exc}')
            return False

        if bbox_xywh is not None:
            bx, by, bw, bh = bbox_xywh
        else:
            bx = by = bw = bh = -1

        self._mw.writerow([
            fname,
            f'{time.time():.3f}',
            self.loop_counter,
            f'{float(pan_deg):.2f}',
            f'{float(tilt_deg):.2f}',
            tracking_detector,
            1 if face_locked else 0,
            int(bx), int(by), int(bw), int(bh),
        ])
        self._mf.flush()
        self.saved_count += 1
        print(f'[eval] {fname} ({self.saved_count}/{self.max_frames})')
        return True

    def close(self):
        try:
            self._mf.close()
        except Exception:
            pass
        print(f'Live evaluation session closed: {self.saved_count} frames in {self.session_dir}')
