import json
import time

class JsonLogger:
    def __init__(self, output_file="output/metadata.jsonl"):
        self.output_file = output_file
        self.start_time = time.time()
        # Ensure file is empty at start
        open(self.output_file, "w").close()

    def log(self, frame_id, tracks):
        timestamp = time.time() - self.start_time
        frame_data = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "tracks": []
        }

        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            entry = {
                "track_id": track["id"],
                "bbox": [x1, y1, x2, y2],
                "jersey": track.get("jersey", None)
            }
            frame_data["tracks"].append(entry)

        with open(self.output_file, "a") as f:
            f.write(json.dumps(frame_data) + "\n")
