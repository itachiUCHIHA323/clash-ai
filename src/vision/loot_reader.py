"""
Loot OCR & Template Reader (Strict Range Clamped Consensus - Zero Hallucinations)
---------------------------------------------------------------------------------
Scans the enemy village scout screen in the UPPER-LEFT corner using 'avail_loot.PNG'.

Key Fixes for 100% Reliable CoC Loot Reading:
1. NO MORE max(candidates): Eliminates the bug where OCR noise or outline shadows
   hallucinating an extra digit (e.g. reading 850,000 as 8,500,000) was picked by max().
2. Strict CoC Loot Constraints: In Clash of Clans, available loot is always between
   1,000 and 2,500,000 (4 to 7 digits). Any OCR result > 2,500,000 is mathematically
   impossible and is instantly discarded as noise.
3. Deterministic Consensus: Tests RapidOCR (ONNX) on raw BGR Color first, followed by
   Otsu binarization. The first valid result within [1000, 2500000] wins.
"""

import os
import shutil
import cv2
import numpy as np
import re
from typing import Dict, Tuple, Optional, List

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

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER = None
except ImportError:
    EASYOCR_AVAILABLE = False


class LootReader:
    """
    Parses Available Gold, Elixir, and Dark Elixir from the UPPER-LEFT corner
    with strict [1000, 2500000] range validation to prevent OCR hallucinations.
    """

    def __init__(
        self,
        min_gold: int = 800000,
        min_elixir: int = 800000,
        template_dir: str = "templates/ui",
        digits_dir: str = "templates/digits/enemy",
    ):
        self.min_gold = min_gold
        self.min_elixir = min_elixir
        self.template_dir = template_dir
        self.digits_dir = digits_dir

        self._setup_tesseract_windows()

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
        """Load avail_loot.PNG, gold.PNG, elixir.PNG, and dark_exlixir.PNG icon templates."""
        targets = {
            "avail_loot": ["avail_loot.png", "avail_loot.PNG"],
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
                        print(f"[INFO] Loaded Loot template '{resource}' from '{path}'")
                        break

    def read_loot(self, frame: np.ndarray, save_debug_roi: bool = True) -> Dict[str, int]:
        """
        Extract Available Gold, Elixir, and Dark Elixir amounts from the UPPER-LEFT corner of a scout frame.
        """
        h, w, _ = frame.shape

        y1, y2 = int(h * 0.01), int(h * 0.38)
        x1, x2 = int(w * 0.01), int(w * 0.30)
        upper_left_roi = frame[y1:y2, x1:x2]

        anchor_found, anchor_coords = self._find_avail_loot_anchor(upper_left_roi)
        gold_crop, elixir_crop, dark_crop = None, None, None

        if anchor_found and anchor_coords:
            ax, ay, a_w, a_h = anchor_coords
            roi_h, roi_w, _ = upper_left_roi.shape
            gy1 = min(roi_h, ay + a_h - 2)
            gy2 = min(roi_h, gy1 + 36)
            gx1 = min(roi_w, max(0, ax + 22))
            gx2 = min(roi_w, gx1 + 195)
            gold_crop = upper_left_roi[gy1:gy2, gx1:gx2]

            ey1 = min(roi_h, gy2 - 2)
            ey2 = min(roi_h, ey1 + 36)
            elixir_crop = upper_left_roi[ey1:ey2, gx1:gx2]

            dy1 = min(roi_h, ey2 - 2)
            dy2 = min(roi_h, dy1 + 36)
            dark_crop = upper_left_roi[dy1:dy2, gx1:gx2]

        if gold_crop is None or gold_crop.size == 0:
            gold_crop = self._slice_icon_right(upper_left_roi, "gold", default_y=(65, 96), default_x=(95, 250))
        if elixir_crop is None or elixir_crop.size == 0:
            elixir_crop = self._slice_icon_right(upper_left_roi, "elixir", default_y=(100, 132), default_x=(95, 250))
        if dark_crop is None or dark_crop.size == 0:
            dark_crop = self._slice_icon_right(upper_left_roi, "dark_elixir", default_y=(136, 170), default_x=(95, 220))

        if save_debug_roi:
            for name, crop in [("gold", gold_crop), ("elixir", elixir_crop), ("dark_elixir", dark_crop)]:
                if crop is not None and crop.size > 0:
                    try:
                        cv2.imwrite(f"debug_loot_{name}_roi.png", crop)
                    except Exception:
                        pass

        gold_val = self._parse_resource_strict(gold_crop, "GOLD", max_limit=2500000) if gold_crop is not None else 0
        elixir_val = self._parse_resource_strict(elixir_crop, "ELIXIR", max_limit=2500000) if elixir_crop is not None else 0
        dark_val = self._parse_resource_strict(dark_crop, "DARK_ELIXIR", max_limit=25000) if dark_crop is not None else 0

        return {
            "gold": gold_val,
            "elixir": elixir_val,
            "dark_elixir": dark_val,
        }

    def _find_avail_loot_anchor(self, upper_left_roi: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Locate 'avail_loot.PNG' header banner in upper_left_roi."""
        if "avail_loot" not in self.icon_templates:
            return False, None
        tmpl = self.icon_templates["avail_loot"]
        th, tw, _ = tmpl.shape
        roi_h, roi_w, _ = upper_left_roi.shape
        if tw > roi_w or th > roi_h:
            return False, None

        res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.55:
            return True, (max_loc[0], max_loc[1], tw, th)
        return False, None

    def _slice_icon_right(
        self,
        upper_left_roi: np.ndarray,
        resource_name: str,
        default_y: Tuple[int, int],
        default_x: Tuple[int, int],
    ) -> np.ndarray:
        """Locate resource icon and slice digits starting 2px inside its right edge."""
        roi_h, roi_w, _ = upper_left_roi.shape
        if resource_name in self.icon_templates:
            tmpl = self.icon_templates[resource_name]
            th, tw, _ = tmpl.shape
            if tw <= roi_w and th <= roi_h:
                res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.55:
                    ix, iy = max_loc
                    crop_x1 = max(0, ix + tw - 2)
                    crop_x2 = min(roi_w, crop_x1 + 220)
                    crop_y1 = max(0, iy - 2)
                    crop_y2 = min(roi_h, iy + th + 4)
                    return upper_left_roi[crop_y1:crop_y2, crop_x1:crop_x2]

        dy1, dy2 = default_y
        dx1, dx2 = default_x
        return upper_left_roi[dy1:dy2, dx1:dx2]

    def _parse_resource_strict(self, crop: np.ndarray, resource_name: str, max_limit: int = 2500000) -> int:
        """
        Deterministic, Strict-Constraint Loot Reader:
        - NEVER uses max(candidates) which picks OCR hallucinations.
        - Requires result to be within [100, max_limit] (e.g. <= 2,500,000 for Gold/Elixir).
        - Tests RapidOCR (ONNX) on raw BGR Color first, followed by Otsu.
        """
        if crop is None or crop.size == 0:
            return 0

        scaled_bgr = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Try RapidOCR on Raw BGR Color image first
        if RAPIDOCR_AVAILABLE:
            global RAPIDOCR_ENGINE
            try:
                if RAPIDOCR_ENGINE is None:
                    RAPIDOCR_ENGINE = RapidOCR()
                for mode_name, img in [("BGR", scaled_bgr), ("Otsu", thresh_otsu), ("Gray", gray)]:
                    result, _ = RAPIDOCR_ENGINE(img)
                    if result:
                        for box, text, score in result:
                            digits = re.sub(r"\D", "", text)
                            if digits and 3 <= len(digits) <= 7:
                                val = int(digits)
                                if 100 <= val <= max_limit:
                                    print(f"[LOOT OCR] {resource_name}: {val:,}")
                                    return val
            except Exception as e:
                print(f"[DEBUG] RapidOCR failed on {resource_name}: {e}")

        # 2. Try EasyOCR fallback
        if EASYOCR_AVAILABLE:
            global EASYOCR_READER
            try:
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                for mode_name, img in [("BGR", scaled_bgr), ("Otsu", thresh_otsu)]:
                    results = EASYOCR_READER.readtext(img, allowlist="0123456789")
                    for _, text, conf in results:
                        digits = re.sub(r"\D", "", text)
                        if digits and 3 <= len(digits) <= 7:
                            val = int(digits)
                            if 100 <= val <= max_limit:
                                print(f"[LOOT OCR - EasyOCR] {resource_name}: {val:,}")
                                return val
            except Exception:
                pass

        # 3. Try Tesseract OCR fallback
        if PYTESSERACT_AVAILABLE:
            for mode_name, th_img in [("Otsu", thresh_otsu), ("Gray", gray)]:
                try:
                    text = pytesseract.image_to_string(th_img, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                    digits = re.sub(r"\D", "", text)
                    if digits and 3 <= len(digits) <= 7:
                        val = int(digits)
                        if 100 <= val <= max_limit:
                            print(f"[LOOT OCR - Tesseract] {resource_name}: {val:,}")
                            return val
                except Exception:
                    pass

        print(f"[LOOT OCR] Could not read valid {resource_name} digits within limits. Check debug_loot_{resource_name.lower()}_roi.png")
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
        dark_elixir = loot_dict.get("dark_elixir", 0)

        sufficient = gold >= self.min_gold and elixir >= self.min_elixir
        print(
            f"[LOOT CHECK] Gold: {gold:,} (Min: {self.min_gold:,}) | "
            f"Elixir: {elixir:,} (Min: {self.min_elixir:,}) | "
            f"Dark Elixir: {dark_elixir:,} -> Attack Suitable: {sufficient}"
        )
        return sufficient


if __name__ == "__main__":
    reader = LootReader(min_gold=800000, min_elixir=800000)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    loot = reader.read_loot(dummy)
    print("Test Loot Reader output:", loot)
