import cv2
import time
import os
from detectors.base import Detection

# Importa qui i detector che vuoi attivi
from detectors.pedestrians_hog import Detector as PedDetector
from detectors.lanes_hough import Detector as LaneDetector
from detectors.trafficlight_color import Detector as TLDetector


def _has_gui():
    """Controlla se è disponibile un display grafico."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def draw_detection(frame, det: Detection):
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
    label = f"{det.label} {det.conf:.2f}"
    y_text = det.y1 - 8 if det.y1 - 8 > 18 else det.y1 + 18
    cv2.putText(frame, label, (det.x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    video_path = "assets/Dashcam.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire {video_path}")

    detectors = [
        PedDetector(),
        LaneDetector(),
        TLDetector(),
    ]

    gui = _has_gui()

    # Se non c'è display, scrivi il video di output su file
    writer = None
    output_path = "output.mp4"
    if not gui:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_out, (w, h))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Modalità headless — il video verrà salvato in {output_path}")
        print(f"[INFO] Frame totali: {total}")

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

        if gui:
            cv2.imshow("Dashcam - Combined detectors (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            writer.write(frame)
            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx} — dets: {len(all_dets)} — FPS: {fps_smooth:.1f}")

        frame_idx += 1

    cap.release()
    if gui:
        cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"\n[DONE] Video salvato in {output_path} ({frame_idx} frame)")

if __name__ == "__main__":
    main()
