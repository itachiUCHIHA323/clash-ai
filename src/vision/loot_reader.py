"""
Loot OCR & Template Reader (KrakenPrime Morphological Processing & Strict Consensus)
-------------------------------------------------------------------------------------
Scans the enemy village scout screen in the UPPER-LEFT corner.

1. KrakenPrime Preprocessing (from itachiUCHIHA323/krakenprime):
   - Slices GOLD_ROI = (65, 163, 377, 205) / upper left boxes.
   - Converts to grayscale, thresholds at 175, and applies cv2.morphologyEx(MORPH_OPEN, (2,2))
     to eliminate comma separators, background textures, and outline artifacts.
   - Resizes 3x with INTER_CUBIC for razor-sharp character segmentation.
2. COC-FARMER Digit Template Matching ('templates/availableLootFont/0.png'..'9.png'):
   - Matches Supercell font digits with NMS left-to-right.
3. Multi-Mode Rapid OCR (ONNX / EasyOCR / Tesseract) as fallback with strict range clamping [1000, 2500000].
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
    Parses Available Gold, Elixir, and Dark Elixir using KrakenPrime Morphological
    cleaning and COC-FARMER digit templates.
    """

    def __init__(
        self,
        min_gold: int = 800000,
        min_elixir: int = 800000,
        template_dir: str = "templates/ui",
        digits_dir: str = "templates/availableLootFont",
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
                        break

    def _load_digit_templates(self) -> None:
        """Load digit templates 0.png..9.png from availableLootFont / COC-FARMER library."""
        for d in range(10):
            path = os.path.join(self.digits_dir, f"{d}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.digit_templates[d] = img
        if self.digit_templates:
            print(f"[INFO] Loaded {len(self.digit_templates)} availableLootFont digit templates (0..9) from '{self.digits_dir}'")

    def read_loot(self, frame: np.ndarray, save_debug_roi: bool = True) -> Dict[str, int]:
        """
        Extract Available Gold, Elixir, and Dark Elixir amounts from a scout frame.
        Uses KrakenPrime GOLD_ROI = (65, 163, 377, 205) fallback when icon templates vary.
        """
        h, w, _ = frame.shape

        y1, y2 = int(h * 0.01), int(h * 0.38)
        x1, x2 = int(w * 0.01), int(w * 0.32)
        upper_left_roi = frame[y1:y2, x1:x2]

        anchor_found, anchor_coords = self._find_avail_loot_anchor(upper_left_roi)
        gold_crop, elixir_crop, dark_crop = None, None, None

        if anchor_found and anchor_coords:
            ax, ay, a_w, a_h = anchor_coords
            roi_h, roi_w, _ = upper_left_roi.shape
            gy1 = min(roi_h, ay + a_h - 2)
            gy2 = min(roi_h, gy1 + 38)
            gx1 = min(roi_w, max(0, ax + 22))
            gx2 = min(roi_w, gx1 + 210)
            gold_crop = upper_left_roi[gy1:gy2, gx1:gx2]

            ey1 = min(roi_h, gy2 - 2)
            ey2 = min(roi_h, ey1 + 38)
            elixir_crop = upper_left_roi[ey1:ey2, gx1:gx2]

            dy1 = min(roi_h, ey2 - 2)
            dy2 = min(roi_h, dy1 + 38)
            dark_crop = upper_left_roi[dy1:dy2, gx1:gx2]

        # Use KrakenPrime GOLD_ROI-style exact boxes as default fallback
        if gold_crop is None or gold_crop.size == 0:
            gold_crop = self._slice_icon_right(upper_left_roi, "gold", default_y=(65, 98), default_x=(95, 260))
        if elixir_crop is None or elixir_crop.size == 0:
            elixir_crop = self._slice_icon_right(upper_left_roi, "elixir", default_y=(100, 134), default_x=(95, 260))
        if dark_crop is None or dark_crop.size == 0:
            dark_crop = self._slice_icon_right(upper_left_roi, "dark_elixir", default_y=(136, 172), default_x=(95, 230))

        if save_debug_roi:
            for name, crop in [("gold", gold_crop), ("elixir", elixir_crop), ("dark_elixir", dark_crop)]:
                if crop is not None and crop.size > 0:
                    try:
                        cv2.imwrite(f"debug_loot_{name}_roi.png", crop)
                    except Exception:
                        pass

        gold_val = self._run_krakenprime_reader(gold_crop, "GOLD") if gold_crop is not None else 0
        elixir_val = self._run_krakenprime_reader(elixir_crop, "ELIXIR") if elixir_crop is not None else 0
        dark_val = self._run_krakenprime_reader(dark_crop, "DARK_ELIXIR", max_limit=25000) if dark_crop is not None else 0

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

    def _run_krakenprime_reader(self, crop: np.ndarray, resource_name: str, max_limit: int = 2500000) -> int:
        """
        KrakenPrime Morphological Cleaning + COC-FARMER Digit Template Matching:
        1. Primary: Slides availableLootFont (0.png..9.png) across multi-scale resized crop.
        2. Secondary: Applies cv2.morphologyEx(MORPH_OPEN, (2,2)) from KrakenPrime
           to remove comma/background artifacts before RapidOCR / EasyOCR / Tesseract.
        """
        if crop is None or crop.size == 0:
            return 0

        # 1. Try COC-FARMER / availableLootFont Digit Template Matching
        tmpl_val = self._match_digit_templates(crop)
        if 1000 <= tmpl_val <= max_limit:
            print(f"[LOOT SCAN] {resource_name} (availableLootFont): {tmpl_val:,}")
            return tmpl_val

        # 2. Apply KrakenPrime Morphological cleaning (from krakenprime/app.py)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        morph_thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        scaled_morph = cv2.resize(morph_thr, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        scaled_bgr = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

        # 3. Test RapidOCR on KrakenPrime cleaned image + raw BGR
        if RAPIDOCR_AVAILABLE:
            global RAPIDOCR_ENGINE
            try:
                if RAPIDOCR_ENGINE is None:
                    RAPIDOCR_ENGINE = RapidOCR()
                for img in [scaled_morph, scaled_bgr]:
                    result, _ = RAPIDOCR_ENGINE(img)
                    if result:
                        for box, text, score in result:
                            digits = re.sub(r"\D", "", text)
                            if digits and 3 <= len(digits) <= 7:
                                val = int(digits)
                                if 1000 <= val <= max_limit:
                                    print(f"[LOOT OCR] {resource_name}: {val:,}")
                                    return val
            except Exception as e:
                print(f"[DEBUG] RapidOCR failed on {resource_name}: {e}")

        # 4. Try EasyOCR
        if EASYOCR_AVAILABLE:
            global EASYOCR_READER
            try:
                if EASYOCR_READER is None:
                    EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                for img in [scaled_morph, scaled_bgr]:
                    results = EASYOCR_READER.readtext(img, allowlist="0123456789")
                    for _, text, conf in results:
                        digits = re.sub(r"\D", "", text)
                        if digits and 3 <= len(digits) <= 7:
                            val = int(digits)
                            if 1000 <= val <= max_limit:
                                print(f"[LOOT OCR - EasyOCR] {resource_name}: {val:,}")
                                return val
            except Exception:
                pass

        # 5. Try Tesseract OCR with KrakenPrime config
        if PYTESSERACT_AVAILABLE:
            cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
            try:
                text = pytesseract.image_to_string(scaled_morph, config=cfg)
                digits = re.sub(r"\D", "", text)
                if digits and 3 <= len(digits) <= 7:
                    val = int(digits)
                    if 1000 <= val <= max_limit:
                        print(f"[LOOT OCR - Tesseract] {resource_name}: {val:,}")
                        return val
            except Exception:
                pass

        print(f"[LOOT OCR] Could not read valid {resource_name} digits within limits. Check debug_loot_{resource_name.lower()}_roi.png")
        return 0

    def _match_digit_templates(self, crop: np.ndarray, threshold: float = 0.72) -> int:
        """Multi-scale digit template matching across 0..9."""
        if not self.digit_templates:
            return 0

        ch, cw, _ = crop.shape
        matches = []

        for d, base_tmpl in self.digit_templates.items():
            for scale in [0.90, 0.95, 1.0, 1.05, 1.10]:
                th, tw = int(base_tmpl.shape[0] * scale), int(base_tmpl.shape[1] * scale)
                if tw > cw or th > ch or th < 6 or tw < 3:
                    continue
                tmpl = cv2.resize(base_tmpl, (tw, th), interpolation=cv2.INTER_LINEAR)

                res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
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
