import cv2
import pytesseract

class JerseyOCR:
    def __init__(self):
        self.config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"

    def read_number(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        text = pytesseract.image_to_string(gray, config=self.config)
        text = text.strip()
        return text if text.isdigit() else None
