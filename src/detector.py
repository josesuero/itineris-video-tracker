from ultralytics import YOLO
from utils_device import get_device

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.35):
        self.device = get_device()
        print(f"[Detector] Using device: {self.device}")
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == 0:  # person only
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(([x1, y1, x2, y2], conf, cls_id))
        return detections
