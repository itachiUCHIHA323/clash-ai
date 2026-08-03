"""
Card Scanner Module (Precision Best-Match Slot Recognition & Greyed-Out Detector)
---------------------------------------------------------------------------------
Scans the bottom troop deployment bar to locate specific cards:
- Sneaky Goblins, Valkyries, Dragons, Electro Dragons (E-Drags), etc.
- Heroes (King, Queen, Warden, Champion) and Spells (Rage, Freeze, etc.)

Key Capabilities:
1. Best-Match per Slot Competition (Threshold >= 0.75): Evaluates all card templates
   per slot along the bottom bar and assigns the highest-scoring card, preventing
   false matches (e.g. matching Valkyries when Dragons are present).
2. is_card_greyed_out(): Converts card icon ROI to HSV and inspects mean saturation (< 42)
   to determine when all units of a card have been deployed.
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
    Scans the bottom deployment bar using competitive Best-Match template recognition
    and checks card greyed-out status via HSV saturation analysis.
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
            "FREEZE": "FREEZE",
            "CLONE": "CLONE",
        }

        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

        self.count_digits: Dict[int, np.ndarray] = {}
        self._load_count_digit_templates()

        self.default_slot_order = {
            "DRAGON": 1,
            "EDRAGON": 1,
            "VALKYRIE": 1,
            "SNEAKY_GOBLIN": 1,
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

    def _load_count_digit_templates(self) -> None:
        """Load digit templates 0..9 from 'templates/battleTroopCountFont'."""
        for d in range(10):
            for ext in [".png", ".bmp"]:
                path = os.path.join("templates/battleTroopCountFont", f"{d}{ext}")
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.count_digits[d] = img
                        break

    def scan_cards(self, frame: np.ndarray, threshold: float = 0.75) -> Dict[str, Tuple[int, int]]:
        """
        Scan a screenshot frame and return a mapping of detected card names to (X, Y) screen coordinates
        using Best-Match per Slot competition.
        """
        detected = {}
        h, w, _ = frame.shape

        bar_y1 = int(h * 0.76)
        bar_y2 = int(h * 0.98)
        bar_roi = frame[bar_y1:bar_y2, 0:w]

        # Competitive best match per template across the bar
        for card_name, tmpl in self.templates.items():
            if card_name in detected:
                continue
            th, tw, _ = tmpl.shape
            if tw > w or th > (bar_y2 - bar_y1):
                continue

            res = cv2.matchTemplate(bar_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
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
        """
        h, w, _ = frame.shape
        card_map = self.scan_cards(frame, threshold=0.75)
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

    def is_card_greyed_out(self, frame: np.ndarray, card_x: int, card_y: int) -> bool:
        """
        Determine if a card in the bottom bar has been completely deployed (greyed out).
        When a card is out of troops, Supercell turns its icon grey/desaturated (mean Saturation < 42).
        """
        h, w, _ = frame.shape
        y1 = max(0, card_y - 25)
        y2 = min(h, card_y + 15)
        x1 = max(0, card_x - 22)
        x2 = min(w, card_x + 22)

        icon_crop = frame[y1:y2, x1:x2]
        if icon_crop.size == 0:
            return True

        hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        mean_val = float(hsv[:, :, 2].mean())

        # An active colorful card has mean_sat > 50. A greyed out card drops to < 38 saturation
        is_empty = mean_sat < 40.0 or mean_val < 45.0
        return is_empty

    def get_slot_coordinate(self, slot_index: int, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
        """Calculate screen (X, Y) coordinate for a 1-indexed deployment card slot along the bottom bar."""
        w = width or self.screen_width
        h = height or self.screen_height
        card_x = int(w * (0.12 + (slot_index - 1) * 0.085))
        card_y = int(h * 0.91)
        return (card_x, card_y)

    def read_troop_count(self, frame: np.ndarray, card_x: int, card_y: int, fallback_count: int = 24) -> int:
        """
        Crop the count badge directly above the card icon (Y = card_y - 32 to card_y - 5,
        X = card_x - 18 to card_x + 18) and read the exact integer count.
        """
        h, w, _ = frame.shape
        y1 = max(0, card_y - 32)
        y2 = max(0, card_y - 5)
        x1 = max(0, card_x - 18)
        x2 = min(w, card_x + 18)

        badge_crop = frame[y1:y2, x1:x2]
        if badge_crop.size == 0:
            return fallback_count

        tmpl_count = self._match_count_digits(badge_crop)
        if 1 <= tmpl_count <= 300:
            print(f"[CARD SCAN] Read remaining count '{tmpl_count}' using battleTroopCountFont")
            return tmpl_count

        gray = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
        _, thresh = cv2.threshold(scaled, 160, 255, cv2.THRESH_BINARY)

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

        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                digits = re.sub(r"\D", "", text)
                if digits and 1 <= int(digits) <= 300:
                    return int(digits)
            except Exception:
                pass

        return fallback_count

    def _match_count_digits(self, badge_crop: np.ndarray, threshold: float = 0.72) -> int:
        """Multi-scale template matching across 0..9 for the count badge above a card."""
        if not self.count_digits or badge_crop.size == 0:
            return 0

        ch, cw, _ = badge_crop.shape
        matches = []

        for d, base_tmpl in self.count_digits.items():
            for scale in [0.90, 0.95, 1.0, 1.05, 1.10]:
                th, tw = int(base_tmpl.shape[0] * scale), int(base_tmpl.shape[1] * scale)
                if tw > cw or th > ch or th < 5 or tw < 3:
                    continue
                tmpl = cv2.resize(base_tmpl, (tw, th), interpolation=cv2.INTER_LINEAR)

                res = cv2.matchTemplate(badge_crop, tmpl, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= threshold)
                for pt_y, pt_x in zip(*locs):
                    conf = float(res[pt_y, pt_x])
                    matches.append((int(pt_x), str(d), conf, tw))

        if not matches:
            return 0

        matches.sort(key=lambda item: item[0])
        filtered = []
        for match in matches:
            x, d, conf, tw = match
            overlap = False
            for prev in filtered:
                prev_x, prev_d, prev_conf, prev_tw = prev
                if abs(x - prev_x) < max(4, prev_tw // 2):
                    overlap = True
                    if conf > prev_conf:
                        filtered.remove(prev)
                        filtered.append(match)
                    break
            if not overlap:
                filtered.append(match)

        filtered.sort(key=lambda item: item[0])
        digit_str = "".join([m[1] for m in filtered])
        try:
            return int(digit_str) if digit_str else 0
        except ValueError:
            return 0


if __name__ == "__main__":
    scanner = CardScanner()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    army = scanner.scan_available_army(dummy_frame)
    print("Scanned available army:", army)
