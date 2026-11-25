import json
import threading
import time
from pathlib import Path
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO

CAPTURE_SIZE = (2304, 1296)
INFER_SIZE = (1280, 720)
MODEL_PATH = "yolo11n.pt"
TOTAL_SPOTS = 10
STATE_FILE = Path("counter_state.json")

LINE_START = (0.22, 0.87)
LINE_END = (0.88, 0.66)
VEHICLE_CLASSES = {2, 3, 5, 7}
CONF_THRESHOLD = 0.4

app = Flask(__name__)
latest_jpeg: bytes | None = None
jpeg_lock = threading.Lock()


def update_stream_frame(bgr_frame) -> None:
    global latest_jpeg
    preview = cv2.resize(bgr_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if ok:
        with jpeg_lock:
            latest_jpeg = buf.tobytes()


def mjpeg_generator():
    while True:
        with jpeg_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.05)


@app.route("/")
def index():
    return "<html><body><img src='/video_feed' style='max-width:100%;'></body></html>"


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_web_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    ).start()


def load_occupied_count() -> int:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return max(0, min(TOTAL_SPOTS, int(data.get("occupied", 0))))
        except Exception:
            pass
    return 0


def save_occupied_count(occupied: int) -> None:
    STATE_FILE.write_text(
        json.dumps({"occupied": occupied, "updated_at": time.time()}, indent=2),
        encoding="utf-8",
    )


def signed_side_of_line(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def segments_intersect(p1, p2, q1, q2) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def init_camera() -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"},
        controls={"FrameRate": 15},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    return picam2


def main() -> None:
    occupied = load_occupied_count()
    start_web_server(host="0.0.0.0", port=5000)
    model = YOLO(MODEL_PATH)
    picam2 = init_camera()

    w, h = INFER_SIZE
    p1 = (int(LINE_START[0] * w), int(LINE_START[1] * h))
    p2 = (int(LINE_END[0] * w), int(LINE_END[1] * h))

    track_history: dict[int, tuple[float, float]] = {}
    counted_ids: set[int] = set()

    try:
        while True:
            frame_rgb = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(frame_bgr, INFER_SIZE, interpolation=cv2.INTER_LINEAR)

            results = model.track(
                resized,
                persist=True,
                imgsz=640,
                conf=CONF_THRESHOLD,
                classes=list(VEHICLE_CLASSES),
                verbose=False,
            )

            cv2.line(resized, p1, p2, (0, 0, 255), 3)
            current_ids: set[int] = set()

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = box
                    cx = float((x1 + x2) / 2.0)
                    cy = float(y2)
                    current_ids.add(track_id)
                    curr_pt = (cx, cy)

                    if track_id in track_history:
                        prev_pt = track_history[track_id]
                        if track_id not in counted_ids and segments_intersect(prev_pt, curr_pt, p1, p2):
                            prev_side = signed_side_of_line(prev_pt[0], prev_pt[1], p1[0], p1[1], p2[0], p2[1])
                            curr_side = signed_side_of_line(curr_pt[0], curr_pt[1], p1[0], p1[1], p2[0], p2[1])

                            if prev_side < 0 and curr_side >= 0:
                                occupied = min(TOTAL_SPOTS, occupied + 1)
                                counted_ids.add(track_id)
                                save_occupied_count(occupied)
                            elif prev_side > 0 and curr_side <= 0:
                                occupied = max(0, occupied - 1)
                                counted_ids.add(track_id)
                                save_occupied_count(occupied)

                    track_history[track_id] = curr_pt

                    cv2.rectangle(resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(resized, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                    cv2.putText(
                        resized,
                        f"ID {track_id} ({conf:.2f})",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

            for old_id in list(track_history.keys()):
                if old_id not in current_ids:
                    del track_history[old_id]

            free_spots = max(0, TOTAL_SPOTS - occupied)
            cv2.putText(
                resized,
                f"Frei: {free_spots} | Besetzt: {occupied}/{TOTAL_SPOTS}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            update_stream_frame(resized)
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()