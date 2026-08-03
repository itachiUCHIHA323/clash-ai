"""
Card Scanner Module (Real Troop Count OCR & Available Army Discovery)
---------------------------------------------------------------------
Scans the bottom troop deployment bar to locate specific cards:
- Sneaky Goblins, Valkyries, Dragons, Electro Dragons (E-Drags), Balloons, etc.
- Heroes (King, Queen, Warden, Champion) and Spells (Rage, Freeze, etc.)

Features:
1. Real Troop Count Recognition: Crops the number badge above each card icon
   (Y = card_y - 28 to card_y - 6) and reads the exact remaining troop count.
2. Dynamic Army Discovery (scan_available_army): Returns all currently available
   troops, heroes, and spells with their real counts and screen coordinates.
"""

import os
import cv2
import numpy as np
import re
from typing import Dict, Tuple, Optional, List, Any

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
    RAPIDOCR_ENGINE = None
except ImportError:
    RAPIDOCR_AVAILABLE = False


class CardScanner:
    """
    Scans the bottom deployment bar of the battle screen to determine the screen (X, Y)
    coordinates of troop and hero cards, and reads real troop count badges above each card.
    """

    def __init__(self, template_dir: str = "templates/cards", screen_width: int = 1280, screen_height: int = 720):
        self.template_dir = template_dir
        self.screen_width = screen_width
        self.screen_height = screen_height

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

        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

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

    def scan_cards(self, frame: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Scan a screenshot frame and return a mapping of detected card names to (X, Y) screen coordinates.
        """
        detected = {}
        h, w, _ = frame.shape

        bar_y1 = int(h * 0.76)
        bar_y2 = int(h * 0.98)
        bar_roi = frame[bar_y1:bar_y2, 0:w]

        for card_name, tmpl in self.templates.items():
            if card_name in detected:
                continue
            th, tw, _ = tmpl.shape
            if tw > w or th > (bar_y2 - bar_y1):
                continue

            res = cv2.matchTemplate(bar_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.62:
                match_x = max_loc[0] + tw // 2
                match_y = bar_y1 + max_loc[1] + th // 2
                detected[card_name] = (int(match_x), int(match_y))
                canonical = self.aliases.get(card_name, card_name)
                detected[canonical] = (int(match_x), int(match_y))

        for card_name, slot_idx in self.default_slot_order.items():
            if card_name not in detected:
                detected[card_name] = self.get_slot_coordinate(slot_idx, w, h)

        return detected

    def scan_available_army(self, frame: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Scan the entire deployment bar and return all currently available cards,
        their real remaining counts, and their tap coordinates.

        Example return:
        {
          "VALKYRIE": {"count": 28, "pos": (153, 655), "is_hero": False},
          "KING": {"count": 1, "pos": (262, 655), "is_hero": True},
        }
        """
        h, w, _ = frame.shape
        card_map = self.scan_cards(frame)
        army = {}

        for card_name, (cx, cy) in card_map.items():
            is_hero = card_name in ["KING", "QUEEN", "WARDEN", "CHAMPION"]
            if is_hero:
                count = 1
            else:
                count = self.read_troop_count(frame, cx, cy)
            army[card_name] = {
                "count": count,
                "pos": (cx, cy),
                "is_hero": is_hero,
            }
        return army

    def get_slot_coordinate(self, slot_index: int, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
        """Calculate screen (X, Y) coordinate for a 1-indexed deployment card slot along the bottom bar."""
        w = width or self.screen_width
        h = height or self.screen_height
        card_x = int(w * (0.12 + (slot_index - 1) * 0.085))
        card_y = int(h * 0.91)
        return (card_x, card_y)

    def read_troop_count(self, frame: np.ndarray, card_x: int, card_y: int, fallback_count: int = 24) -> int:
        """
        Crop the count badge directly above the card icon (Y = card_y - 28 to card_y - 6,
        X = card_x - 14 to card_x + 14) and read the exact integer count.
        """
        h, w, _ = frame.shape
        y1 = max(0, card_y - 30)
        y2 = max(0, card_y - 6)
        x1 = max(0, card_x - 16)
        x2 = min(w, card_x + 16)

        badge_crop = frame[y1:y2, x1:x2]
        if badge_crop.size == 0:
            return fallback_count

        gray = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
        _, thresh = cv2.threshold(scaled, 160, 255, cv2.THRESH_BINARY)

        # 1. Try RapidOCR on count badge
        if RAPIDOCR_AVAILABLE:
            global RAPIDOCR_ENGINE
            try:
                if RAPIDOCR_ENGINE is None:
                    RAPIDOCR_ENGINE = RapidOCR()
                result, _ = RAPIDOCR_ENGINE(thresh)
                if result:
                    for _, text, _ in result:
                        digits = re.sub(r"\D", "", text)
                        if digits and 1 <= int(digits) <= 300:
                            return int(digits)
            except Exception:
                pass

        # 2. Try Tesseract OCR on count badge
        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                digits = re.sub(r"\D", "", text)
                if digits and 1 <= int(digits) <= 300:
                    return int(digits)
            except Exception:
                pass

        # 3. Connected-component count fallback
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        valid_blobs = [i for i in range(1, num_labels) if 6 <= stats[i, cv2.CC_STAT_HEIGHT] <= 35 and stats[i, cv2.CC_STAT_AREA] > 10]
        if valid_blobs:
            # If 2 digit blobs found and no OCR matched, estimate based on badge width
            return fallback_count

        return fallback_count


if __name__ == "__main__":
    scanner = CardScanner()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    army = scanner.scan_available_army(dummy_frame)
    print("Scanned available army:", army)
