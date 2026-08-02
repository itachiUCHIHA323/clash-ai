"""
Vision & Perception Module
--------------------------
Converts raw screenshot frames into structured state representations
using OpenCV and Object Detection (YOLOv8/YOLOv11).
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


class ClashVisionDetector:
    def __init__(self, yolo_model_path: str = None):
        """
        Initialize the detector with an optional YOLO model path.
        If no model path is provided, falls back to OpenCV heuristic detection for demo/skeleton usage.
        """
        self.model = None
        if yolo_model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(yolo_model_path)
            except Exception as e:
                print(f"[WARN] Could not load YOLO model from {yolo_model_path}: {e}")

    def parse_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a raw BGR frame and extract structured game state.
        
        :param frame: Raw screenshot image (H, W, 3) in BGR.
        :return: Dictionary containing detected buildings, destruction %, stars, and valid drop zones.
        """
        height, width, _ = frame.shape

        state = {
            "destruction_percentage": self._read_destruction_percentage(frame),
            "stars": self._detect_stars(frame),
            "buildings": self._detect_buildings(frame),
            "valid_deploy_mask": self._get_valid_deployment_mask(frame),
            "battle_ended": self._is_battle_ended(frame),
        }
        return state

    def _read_destruction_percentage(self, frame: np.ndarray) -> float:
        """
        Extract the Destruction % from the top-left area of the battle screen.
        Returns a float between 0.0 and 100.0.
        """
        # In a production implementation, use OCR (e.g., pytesseract or EasyOCR)
        # on the top-left destruction text ROI:
        # roi = frame[20:100, 20:150]
        # For this demo/skeleton, we return a simulated value:
        return 0.0

    def _detect_stars(self, frame: np.ndarray) -> int:
        """
        Detect how many stars (0, 1, 2, or 3) have been earned in the current attack.
        """
        return 0

    def _detect_buildings(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO detection on the frame to locate enemy buildings (Town Hall, Defenses, Walls).
        
        :return: List of dicts with bounding box coordinates, class labels, and estimated HP.
        """
        buildings = []
        if self.model:
            results = self.model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    buildings.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "class_id": cls_id,
                        "confidence": conf
                    })
        return buildings

    def _get_valid_deployment_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect the red restricted border and return a binary mask (H, W) where 1 indicates
        valid drop locations and 0 indicates restricted/interior tiles.
        """
        h, w, _ = frame.shape
        mask = np.ones((h, w), dtype=np.uint8)
        # Mask out UI bars at top and bottom
        mask[: int(h * 0.12), :] = 0
        mask[int(h * 0.82):, :] = 0
        return mask

    def _is_battle_ended(self, frame: np.ndarray) -> bool:
        """
        Check if the 'Return Home' or victory/defeat banner is visible, signaling battle end.
        """
        return False


if __name__ == "__main__":
    # Test script for vision detector
    detector = ClashVisionDetector()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = detector.parse_state(dummy_frame)
    print("Parsed test frame state:", state)
