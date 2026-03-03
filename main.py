import cv2
import time
from detectors.base import Detection

# Importa qui i detector che vuoi attivi
from detectors.pedestrians_hog import Detector as PedDetector
from detectors.lanes_hough import Detector as LaneDetector
from detectors.trafficlight_color import Detector as TLDetector

def draw_detection(frame, det: Detection):
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
    label = f"{det.label} {det.conf:.2f}"
    y_text = det.y1 - 8 if det.y1 - 8 > 18 else det.y1 + 18
    cv2.putText(frame, label, (det.x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    video_path = "assets/dashcam.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire {video_path}")

    detectors = [
        PedDetector(),
        LaneDetector(),
        TLDetector(),
    ]

    frame_idx = 0
    last_t = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        all_dets = []
        for d in detectors:
            try:
                dets = d.detect(frame, frame_idx)
                if dets:
                    all_dets.extend(dets)
            except Exception as e:
                print(f"[WARN] detector {getattr(d,'name','?')} crashed: {e}")

        for det in all_dets:
            draw_detection(frame, det)

        # FPS overlay
        now = time.time()
        dt = now - last_t
        last_t = now
        fps = (1.0 / dt) if dt > 0 else 0.0
        fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Dashcam - Combined detectors (press q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
