import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


_latest_frame = None
_frame_lock = threading.Lock()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame
                if frame is not None:
                    ok, jpeg = cv2.imencode('.jpg', frame)
                    if ok:
                        data = jpeg.tobytes()
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(data)
                        self.wfile.write(b'\r\n')
                time.sleep(0.033)
        except Exception:
            pass


def start_stream_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f'Stream available at http://<jetson-ip>:{port}/')
    return server


def set_latest_frame(frame):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()
