from deep_sort_realtime.deepsort_tracker import DeepSort
from utils_device import get_device

class Tracker:
    def __init__(self):
        device = get_device()
        use_gpu = device == "cuda"  # DeepSORT embedder only supports CUDA
        print(f"[Tracker] DeepSORT embedder on GPU: {use_gpu}")

        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=use_gpu,
            half=use_gpu
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
