import numpy as np
import cv2

class Homography:
    def __init__(self, src_points, dst_points):
        self.matrix, _ = cv2.findHomography(np.array(src_points), np.array(dst_points))

    def to_field_coords(self, point):
        px = np.array([[point]], dtype="float32")
        dst = cv2.perspectiveTransform(px, self.matrix)
        return dst[0][0].tolist()
