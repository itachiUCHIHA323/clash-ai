"""
Loot OCR & Template Reader (Precision Digit Template Matching + Multi-Mode Rapid OCR)
-------------------------------------------------------------------------------------
Scans the enemy village scout screen in the UPPER-LEFT corner using 'avail_loot.PNG',
'gold.PNG', 'elixir.PNG', and 'dark_exlixir.PNG'.

1. Slices digits starting 2px inside the right edge of each icon (X = icon_x + tw - 2 to icon_x + tw + 220)
   so the first digit is NEVER cut off.
2. Uses COC-FARMER Digit Template Matching ('templates/digits/enemy/0.png'..'9.png') as the #1 primary engine.
   - Slides all 10 digit templates across the Gold and Elixir digit boxes.
   - Sorts matched digit bounding boxes left-to-right to construct the exact integer (e.g., '1', '2', '5', '0', '0', '0' -> 1,250,000).
3. Uses ONNX RapidOCR / EasyOCR / Tesseract across Bright-Pixel, Otsu, and Raw BGR modes as a fallback.
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

# Try importing RapidOCR (ultra-fast ONNX runtime OCR)
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
    RAPIDOCR_ENGINE = None
except ImportError:
    RAPIDOCR_AVAILABLE = False

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
    of battle scout screenshots using COC-FARMER digit templates and hybrid OCR.
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

        self.digit_templates: Dict[int, np.ndarray] = {}
        self._load_digit_templates()

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

    def _load_digit_templates(self) -> None:
        """Load digit templates 0.png..9.png from COC-FARMER template library."""
        for d in range(10):
            path = os.path.join(self.digits_dir, f"{d}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.digit_templates[d] = img
        if self.digit_templates:
            print(f"[INFO] Loaded {len(self.digit_templates)} digit templates (0..9) from '{self.digits_dir}'")

    def read_loot(self, frame: np.ndarray, save_debug_roi: bool = True) -> Dict[str, int]:
        """
        Extract Available Gold, Elixir, and Dark Elixir amounts from the UPPER-LEFT corner of a scout frame.
        """
        h, w, _ = frame.shape

        # Search ONLY in the upper-left corner (Y = 10 to 280 px, X = 10 to 380 px at 1280x720)
        y1, y2 = int(h * 0.01), int(h * 0.38)
        x1, x2 = int(w * 0.01), int(w * 0.30)
        upper_left_roi = frame[y1:y2, x1:x2]

        anchor_found, anchor_coords = self._find_avail_loot_anchor(upper_left_roi)
        gold_crop, elixir_crop, dark_crop = None, None, None

        if anchor_found and anchor_coords:
            ax, ay, a_w, a_h = anchor_coords
            roi_h, roi_w, _ = upper_left_roi.shape
            # Slicing from ax + 20 to ax + 235 so leftmost digits are never cut off
            gy1 = min(roi_h, ay + a_h - 2)
            gy2 = min(roi_h, gy1 + 38)
            gx1 = min(roi_w, max(0, ax + 20))
            gx2 = min(roi_w, gx1 + 215)
            gold_crop = upper_left_roi[gy1:gy2, gx1:gx2]

            ey1 = min(roi_h, gy2 - 2)
            ey2 = min(roi_h, ey1 + 38)
            elixir_crop = upper_left_roi[ey1:ey2, gx1:gx2]

            dy1 = min(roi_h, ey2 - 2)
            dy2 = min(roi_h, dy1 + 38)
            dark_crop = upper_left_roi[dy1:dy2, gx1:gx2]

        if gold_crop is None or gold_crop.size == 0:
            gold_crop = self._slice_icon_right(upper_left_roi, "gold", default_y=(60, 98), default_x=(85, 260))
        if elixir_crop is None or elixir_crop.size == 0:
            elixir_crop = self._slice_icon_right(upper_left_roi, "elixir", default_y=(98, 136), default_x=(85, 260))
        if dark_crop is None or dark_crop.size == 0:
            dark_crop = self._slice_icon_right(upper_left_roi, "dark_elixir", default_y=(136, 175), default_x=(85, 230))

        if save_debug_roi:
            for name, crop in [("gold", gold_crop), ("elixir", elixir_crop), ("dark_elixir", dark_crop)]:
                if crop is not None and crop.size > 0:
                    try:
                        cv2.imwrite(f"debug_loot_{name}_roi.png", crop)
                    except Exception:
                        pass

        gold_val = self._run_hybrid_reader(gold_crop, "GOLD") if gold_crop is not None else 0
        elixir_val = self._run_hybrid_reader(elixir_crop, "ELIXIR") if elixir_crop is not None else 0
        dark_val = self._run_hybrid_reader(dark_crop, "DARK_ELIXIR") if dark_crop is not None else 0

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
        """
        Locate resource icon and slice digits starting 2px inside its right edge
        (X = icon_x + icon_width - 2) all the way to +220px to prevent first-digit truncation.
        """
        roi_h, roi_w, _ = upper_left_roi.shape
        if resource_name in self.icon_templates:
            tmpl = self.icon_templates[resource_name]
            th, tw, _ = tmpl.shape
            if tw <= roi_w and th <= roi_h:
                res = cv2.matchTemplate(upper_left_roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.55:
                    ix, iy = max_loc
                    # Slicing from ix + tw - 2 so the leftmost digit is never cut off
                    crop_x1 = max(0, ix + tw - 2)
                    crop_x2 = min(roi_w, crop_x1 + 220)
                    crop_y1 = max(0, iy - 2)
                    crop_y2 = min(roi_h, iy + th + 4)
                    return upper_left_roi[crop_y1:crop_y2, crop_x1:crop_x2]

        dy1, dy2 = default_y
        dx1, dx2 = default_x
        return upper_left_roi[dy1:dy2, dx1:dx2]

    def _run_hybrid_reader(self, crop: Optional[np.ndarray], resource_name: str) -> int:
        """
        Runs COC-FARMER Digit Template Matching as priority #1.
        If template matching confidence is low, runs Multi-Mode Rapid OCR (ONNX / EasyOCR / Tesseract).
        """
        if crop is None or crop.size == 0:
            return 0

        # 1. Primary Engine: COC-FARMER Digit Template Matching (0.png..9.png)
        tmpl_val = self._match_digit_templates(crop)
        if tmpl_val >= 1000:  # Clash of Clans loot is typically >= 1,000
            print(f"[LOOT SCAN] {resource_name} (Digit Templates): {tmpl_val:,}")
            return tmpl_val

        # 2. Secondary Engine: Multi-Mode Rapid OCR (Bright-Pixel, Otsu, Raw BGR)
        ocr_val = self._run_rapid_ocr(crop, resource_name)
        if ocr_val >= 1000:
            return ocr_val

        return tmpl_val if tmpl_val > 0 else ocr_val

    def _match_digit_templates(self, crop: np.ndarray, threshold: float = 0.74) -> int:
        """
        Slide digit templates (0.png..9.png) across crop and sort matched positions
        left-to-right to construct the exact integer.
        """
        if not self.digit_templates:
            return 0

        ch, cw, _ = crop.shape
        matches = []  # List of tuples: (x_coord, digit_char, confidence)

        for d, tmpl in self.digit_templates.items():
            th, tw, _ = tmpl.shape
            if tw > cw or th > ch:
                continue

            res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= threshold)
            for pt_y, pt_x in zip(*locs):
                conf = float(res[pt_y, pt_x])
                matches.append((int(pt_x), str(d), conf, tw))

        if not matches:
            return 0

        # Sort matches by X coordinate
        matches.sort(key=lambda item: item[0])

        # Non-Maximum Suppression horizontally: remove overlapping duplicate hits
        filtered = []
        for match in matches:
            x, d, conf, tw = match
            overlap = False
            for prev in filtered:
                prev_x, prev_d, prev_conf, prev_tw = prev
                if abs(x - prev_x) < max(4, prev_tw // 2):
                    overlap = True
                    # If this match has higher confidence than prev, replace it
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

    def _run_rapid_ocr(self, crop: np.ndarray, resource_name: str) -> int:
        """Fallback Multi-Mode OCR across ONNX RapidOCR, EasyOCR, and Tesseract."""
        scaled_bgr = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2GRAY)

        _, thresh_bright = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = []

        if RAPIDOCR_AVAILABLE:
            global RAPIDOCR_ENGINE
            try:
                if RAPIDOCR_ENGINE is None:
                    RAPIDOCR_ENGINE = RapidOCR()
                for mode_name, img in [
                    ("BrightBin", thresh_bright),
                    ("BGR", scaled_bgr),
                    ("Gray", gray),
                    ("Otsu", thresh_otsu),
                ]:
                    result, _ = RAPIDOCR_ENGINE(img)
                    if result:
                        for box, text, score in result:
                            digits = re.sub(r"\D", "", text)
                            if digits and len(digits) >= 4:
                                candidates.append(int(digits))
            except Exception:
                pass

        if EASYOCR_AVAILABLE and not candidates:
            global EASYOCR_READER
            try:
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                for mode_name, img in [("BrightBin", thresh_bright), ("BGR", scaled_bgr), ("Gray", gray)]:
                    results = EASYOCR_READER.readtext(img, allowlist="0123456789")
                    for _, text, conf in results:
                        digits = re.sub(r"\D", "", text)
                        if digits and len(digits) >= 4:
                            candidates.append(int(digits))
            except Exception:
                pass

        if PYTESSERACT_AVAILABLE and not candidates:
            for mode_name, th_img in [("BrightBin", thresh_bright), ("Otsu", thresh_otsu), ("Gray", gray)]:
                try:
                    text = pytesseract.image_to_string(th_img, config="--psm 7 -c tessedit_char_whitelist=0123456789")
                    digits = re.sub(r"\D", "", text)
                    if digits and len(digits) >= 4:
                        candidates.append(int(digits))
                except Exception:
                    pass

        return max(candidates) if candidates else 0

    def is_loot_sufficient(self, loot_dict: Dict[str, int], force_attack: bool = False) -> bool:
        """
        Return True if both Gold and Elixir are >= the minimum thresholds (default: 800,000).
        Dark Elixir requirement is 0 by default.
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
