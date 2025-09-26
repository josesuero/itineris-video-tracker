import cv2
from capture import VideoCapture
from detector import Detector
from tracker import Tracker
from jersey_ocr import JerseyOCR
from logger import JsonLogger

VIDEO_PATH = "videos/sample1.mp4"

def main():
    cap = VideoCapture(VIDEO_PATH)
    detector = Detector()
    tracker = Tracker()
    ocr = JerseyOCR()
    logger = JsonLogger("output/metadata.jsonl")

    frame_id = 0

    while True:
        frame = cap.read()
        if frame is None:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        frame_tracks = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            tid = t.track_id
            jersey = ocr.read_number(frame, (x1, y1, x2, y2))

            # Save for logging
            frame_tracks.append({
                "id": tid,
                "bbox": (x1, y1, x2, y2),
                "jersey": jersey
            })

            # Draw for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{tid}"
            if jersey:
                label += f" #{jersey}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log JSON
        logger.log(frame_id, frame_tracks)

        cv2.imshow("Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
