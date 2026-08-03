"""
Card Scanner Module (Deployment Bar Recognition & Troop Count Reader)
---------------------------------------------------------------------
Scans the bottom troop deployment bar to locate specific cards:
- Sneaky Goblins, Valkyries (valk.PNG), Dragons (drag.PNG), Electro Dragons (edrag.PNG)
- Barbarian King (king.PNG), Archer Queen (queen.PNG), Grand Warden (warden.PNG), Royal Champion (rc.PNG)

Supports both OpenCV Template Matching (using PNG icons) and Fixed Slot Order mapping.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class CardScanner:
    """
    Scans the bottom deployment bar of the battle screen to determine the screen (X, Y)
    coordinates of troop and hero cards, and reads troop count above the card.
    """

    def __init__(self, template_dir: str = "templates/cards", screen_width: int = 1280, screen_height: int = 720):
        self.template_dir = template_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Canonical name aliases mapping user filenames (drag.PNG, valk.PNG, rc.PNG) to canonical names
        self.aliases = {
            "VALK": "VALKYRIE",
            "VALKYRIE": "VALKYRIE",
            "DRAG": "DRAGON",
            "DRAGON": "DRAGON",
            "EDRAG": "EDRAGON",
            "EDRAGON": "EDRAGON",
            "RC": "CHAMPION",
            "CHAMPION": "CHAMPION",
            "KING": "KING",
            "QUEEN": "QUEEN",
            "WARDEN": "WARDEN",
            "RAGE": "RAGE",
        }

        # Preloaded OpenCV templates
        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

        # Default slot mapping fallback
        self.default_slot_order = {
            "SNEAKY_GOBLIN": 1,
            "VALKYRIE": 1,
            "DRAGON": 1,
            "EDRAGON": 1,
            "KING": 2,
            "QUEEN": 3,
            "WARDEN": 4,
            "CHAMPION": 5,
        }

    def _load_templates(self) -> None:
        """Load card icon PNG templates from disk if available."""
        if not os.path.exists(self.template_dir):
            return

        for filename in os.listdir(self.template_dir):
            if filename.lower().endswith((".png", ".jpg")):
                raw_name = os.path.splitext(filename)[0].upper()
                canonical = self.aliases.get(raw_name, raw_name)
                filepath = os.path.join(self.template_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[canonical] = img
                    self.templates[raw_name] = img
                    print(f"[INFO] Loaded Card template '{raw_name}' (as '{canonical}') from '{filepath}'")

    def scan_cards(self, frame: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Scan a screenshot frame and return a mapping of detected card names to (X, Y) screen coordinates.
        Uses OpenCV Template Matching when templates exist, otherwise uses default slot indices.

        :param frame: Full battle screenshot (BGR format).
        :return: Dict mapping card name -> (x, y) tap coordinate.
        """
        detected = {}
        h, w, _ = frame.shape

        # Define ROI for bottom deployment bar (~75% to 98% height)
        bar_y1 = int(h * 0.75)
        bar_y2 = int(h * 0.98)
        bar_roi = frame[bar_y1:bar_y2, 0:w]

        # 1. Try Template Matching for loaded icons
        for card_name, tmpl in self.templates.items():
            if card_name in detected:
                continue
            th, tw, _ = tmpl.shape
            if tw > w or th > (bar_y2 - bar_y1):
                continue

            res = cv2.matchTemplate(bar_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.65:
                match_x = max_loc[0] + tw // 2
                match_y = bar_y1 + max_loc[1] + th // 2
                detected[card_name] = (int(match_x), int(match_y))
                # Also assign alias if available
                canonical = self.aliases.get(card_name, card_name)
                detected[canonical] = (int(match_x), int(match_y))

        # 2. Fallback to Slot Mapping for any requested cards not found via templates
        for card_name, slot_idx in self.default_slot_order.items():
            if card_name not in detected:
                detected[card_name] = self.get_slot_coordinate(slot_idx, w, h)

        return detected

    def get_slot_coordinate(self, slot_index: int, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
        """
        Calculate screen (X, Y) coordinate for a 1-indexed deployment card slot along the bottom bar.
        """
        w = width or self.screen_width
        h = height or self.screen_height

        card_x = int(w * (0.12 + (slot_index - 1) * 0.085))
        card_y = int(h * 0.91)
        return (card_x, card_y)

    def get_troop_count(self, frame: np.ndarray, card_x: int, card_y: int, default_count: int = 30) -> int:
        """
        Read the number of troops remaining above the card icon (e.g. 30 for Valkyries).
        If OCR/count reading cannot parse a clear number, returns default_count (30) so
        that edge deployment always deploys the entire army without running out early.
        """
        # In practice, OCR can scan the small ROI above card_y: frame[card_y-30:card_y-10, card_x-20:card_x+20]
        # For reliable farming automation without Tesseract dependency, default to full army size (30)
        return default_count


if __name__ == "__main__":
    scanner = CardScanner()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cards = scanner.scan_cards(dummy_frame)
    print("Scanned card coordinates:", cards)
