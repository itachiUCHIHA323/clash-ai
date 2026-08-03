"""
Loot OCR & Template Reader (Upper-Left Corner Gold, Elixir & Dark Elixir Scanner)
---------------------------------------------------------------------------------
Scans the enemy village scout screen in the UPPER-LEFT corner using gold.PNG,
elixir.PNG, and dark_exlixir.PNG icon templates. Slices the exact numbers immediately
to the RIGHT of each icon, and uses Rapid OCR (Tesseract / EasyOCR) to parse available loot.
"""

import os
import shutil
import cv2
import numpy as np
import re
from typing import Dict, Tuple, Optional, List

# Try importing Tesseract OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER = None
except ImportError:
    EASYOCR_AVAILABLE = False


class LootReader:
    """
    Parses Available Gold, Elixir, and Dark Elixir from the UPPER-LEFT corner
    of battle scout screenshots by matching gold.PNG / elixir.PNG templates.
    """

    def __init__(self, min_gold: int = 800000, min_elixir: int = 800000, template_dir: str = "templates/ui"):
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.template_dir = template_dir

        # Auto-discover Tesseract OCR executable on Windows
        self._setup_tesseract_windows()

        # Load icon templates for Gold, Elixir, and Dark Elixir
        self.icon_templates: Dict[str, np.ndarray] = {}
        self._load_loot_icons()

    def _setup_tesseract_windows(self) -> None:
        """Auto-discover Tesseract executable on Windows if not in PATH."""
        if not PYTESSERACT_AVAILABLE:
            return
        if shutil.which("tesseract"):
            return

        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for p in windows_paths:
            if os.path.exists(p):
                print(f"[INFO] Auto-detected Tesseract OCR executable at: '{p}'")
                pytesseract.pytesseract.tesseract_cmd = p
                break

    def _load_loot_icons(self) -> None:
        """Load gold.PNG, elixir.PNG, and dark_exlixir.PNG icon templates."""
        targets = {
            "gold": ["gold.png", "gold.PNG"],
            "elixir": ["elixir.png", "elixir.PNG"],
            "dark_elixir": ["dark_exlixir.png", "dark_exlixir.PNG", "dark_elixir.png", "dark_elixir.PNG"],
        }
        for resource, filenames in targets.items():
            for fname in filenames:
                path = os.path.join(self.template_dir, fname)
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.icon_templates[resource] = img
                        print(f"[INFO] Loaded Loot Icon template '{resource}' from '{path}'")
                        break

    def read_loot(self, frame: np.ndarray, save_debug_roi: bool = True) -> Dict[str, int]:
        """
        Extract Available Gold, Elixir, and Dark Elixir amounts from the UPPER-LEFT corner of a scout frame.

        :param frame: Screenshot image (BGR format).
        :param save_debug_roi: Whether to save debug_loot_gold_roi.png and debug_loot_elixir_roi.png.
        :return: Dict with 'gold', 'elixir', and 'dark_elixir' integers.
        """
        h, w, _ = frame.shape

        # Search ONLY in the upper-left corner (Y = 15 to 260 px, X = 10 to 360 px at 1280x720)
        y1, y2 = int(h * 0.02), int(h * 0.36)
        x1, x2 = int(w * 0.01), int(w * 0.28)
        upper_left_roi = frame[y1:y2, x1:x2]

        gold_val = self._parse_resource_from_icon(
            upper_left_roi, "gold", default_y=(55, 90), default_x=(55, 230), debug_name="debug_loot_gold_roi.png" if save_debug_roi else None
        )
        elixir_val = self._parse_resource_from_icon(
            upper_left_roi, "elixir", default_y=(90, 125), default_x=(55, 230), debug_name="debug_loot_elixir_roi.png" if save_debug_roi else None
        )
        dark_val = self._parse_resource_from_icon(
            upper_left_roi, "dark_elixir", default_y=(125, 160), default_x=(55, 200), debug_name=None
        )

        return {
            "gold": gold_val,
            "elixir": elixir_val,
            "dark_elixir": dark_val,
        }

    def _parse_resource_from_icon(
        self,
        upper_left_roi: np.ndarray,
        resource_name: str,
        default_y: Tuple[int, int],
        default_x: Tuple[int, int],
        debug_name: Optional[str] = None,
    ) -> int:
        """
        Locate resource icon in upper_left_roi and slice the digits immediately to its right.
        """
        roi_h, roi_w, _ = upper_left_roi.shape
        digit_crop = None

        # 1. Dynamic Template Matching (find icon and slice digits to the right)
        if resource_name in self.icon_templates:
            tmpl = self.icon_templates[resource_name]
            th, tw, _ = tmpl.shape
            if tw <= roi_w and th <= roi_h:
                res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.65:
                    ix, iy = max_loc
                    # Slicing box immediately to the RIGHT of the icon
                    crop_x1 = min(roi_w, ix + tw + 3)
                    crop_x2 = min(roi_w, crop_x1 + 170)
                    crop_y1 = max(0, iy - 4)
                    crop_y2 = min(roi_h, iy + th + 6)
                    digit_crop = upper_left_roi[crop_y1:crop_y2, crop_x1:crop_x2]

        # 2. Fallback to default relative bounding box if icon not matched
        if digit_crop is None or digit_crop.size == 0:
            dy1, dy2 = default_y
            dx1, dx2 = default_x
            digit_crop = upper_left_roi[dy1:dy2, dx1:dx2]

        if digit_crop is None or digit_crop.size == 0:
            return 0

        if debug_name:
            try:
                cv2.imwrite(debug_name, digit_crop)
            except Exception:
                pass

        return self._run_rapid_ocr(digit_crop, resource_name)

    def _run_rapid_ocr(self, crop: np.ndarray, resource_name: str) -> int:
        """
        Run rapid OCR (Tesseract -> EasyOCR) on the sliced digit ROI image.
        """
        # Convert to grayscale and threshold for high-contrast white/magenta digits
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(scaled, 175, 255, cv2.THRESH_BINARY)

        # 1. Try Tesseract OCR
        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                digits = re.sub(r"\D", "", text)
                if digits:
                    val = int(digits)
                    print(f"[LOOT OCR] {resource_name.upper()}: {val:,}")
                    return val
            except Exception as e:
                print(f"[DEBUG] Tesseract OCR failed on {resource_name}: {e}")

        # 2. Try EasyOCR
        if EASYOCR_AVAILABLE:
            global EASYOCR_READER
            try:
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                results = EASYOCR_READER.readtext(thresh, allowlist="0123456789")
                for _, text, conf in results:
                    digits = re.sub(r"\D", "", text)
                    if digits:
                        val = int(digits)
                        print(f"[LOOT OCR - EasyOCR] {resource_name.upper()}: {val:,}")
                        return val
            except Exception as e:
                print(f"[DEBUG] EasyOCR failed on {resource_name}: {e}")

        print(
            f"[LOOT OCR] Could not read {resource_name.upper()} digits. Ensure Tesseract OCR is installed:\n"
            "  -> Download for Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return 0

    def is_loot_sufficient(self, loot_dict: Dict[str, int], force_attack: bool = False) -> bool:
        """
        Return True if both Gold and Elixir are >= the minimum thresholds (default: 800,000).
        """
        if force_attack:
            print("[LOOT CHECK] --force-attack enabled: Bypassing loot threshold verification.")
            return True

        gold = loot_dict.get("gold", 0)
        elixir = loot_dict.get("elixir", 0)
        sufficient = gold >= self.min_gold and elixir >= self.min_elixir
        print(
            f"[LOOT CHECK] Gold: {gold:,} (Min: {self.min_gold:,}) | "
            f"Elixir: {elixir:,} (Min: {self.min_elixir:,}) -> Attack Suitable: {sufficient}"
        )
        return sufficient


if __name__ == "__main__":
    reader = LootReader(min_gold=800000, min_elixir=800000)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    loot = reader.read_loot(dummy)
    print("Test Loot Reader output:", loot)
